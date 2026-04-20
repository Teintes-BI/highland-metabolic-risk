import argparse
import json
from pathlib import Path

from .config import load_specs
from .qc import run_qc
from .run import run_pipeline
from .report import write_report
from .deploy import fit_models_from_picks, predict_from_models


def main():
    parser = argparse.ArgumentParser(prog="ncdpipe")
    sub = parser.add_subparsers(dest="command", required=True)

    qc_p = sub.add_parser("qc", help="Run data QC")
    qc_p.add_argument("--specs", default="specs", help="Specs directory")

    run_p = sub.add_parser("run", help="Run end-to-end pipeline")
    run_p.add_argument("--specs", default="specs", help="Specs directory")
    run_p.add_argument("--mode", choices=["cv", "dev_val"], default=None)
    run_p.add_argument("--run-id", default=None)

    train_p = sub.add_parser("train", help="Train models only (uses run pipeline)")
    train_p.add_argument("--specs", default="specs", help="Specs directory")

    report_p = sub.add_parser("report", help="Generate report from summary.json")
    report_p.add_argument("--project-root", default=".", help="Project root directory")

    fit_p = sub.add_parser("fit", help="Fit deployable models from picks TSV")
    fit_p.add_argument("--specs", default="specs", help="Specs directory")
    fit_p.add_argument("--picks", required=True, help="TSV with columns: outcome,featureset,model (e.g. outputs/qc_qonly/best_models.tsv)")
    fit_p.add_argument("--out-dir", default="outputs/models_final", help="Output folder for trained pipelines")
    fit_p.add_argument("--tag", default="primary", help="Model tag namespace (e.g. primary/enhanced)")

    pred_p = sub.add_parser("predict", help="Predict on a new Excel using fitted models")
    pred_p.add_argument("--specs", default="specs", help="Specs directory")
    pred_p.add_argument("--models-dir", default="outputs/models_final", help="Folder produced by `ncdpipe fit`")
    pred_p.add_argument("--input", required=True, help="Input Excel path (.xlsx)")
    pred_p.add_argument("--output", default="outputs/predictions.tsv", help="Output TSV path (long format)")
    pred_p.add_argument("--mapping", default=None, help="Optional column mapping YAML/CSV to override specs validation.external_mapping_path")
    pred_p.add_argument("--no-tiers", action="store_true", help="Do not compute T0–T3 tiers")
    pred_p.add_argument("--htn_hi", type=float, default=0.8)
    pred_p.add_argument("--hyper_hi", type=float, default=0.7)
    pred_p.add_argument("--lip_hi", type=float, default=0.7)
    pred_p.add_argument("--any_mid", type=float, default=0.6)

    args = parser.parse_args()
    specs_dir = Path(args.specs)
    specs = load_specs(specs_dir)

    if args.command == "qc":
        run_qc(specs, specs_dir.parent)
        return

    if args.command == "run":
        if args.mode:
            specs["validation"]["mode"] = args.mode
        run_pipeline(specs, specs_dir.parent, run_id=args.run_id)
        return

    if args.command == "train":
        run_pipeline(specs, specs_dir.parent, run_id=None)
        return

    if args.command == "report":
        project_root = Path(args.project_root)
        summary_path = project_root / "outputs" / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        write_report(project_root / "outputs" / "report", summary)
        return

    if args.command == "fit":
        project_root = specs_dir.parent
        fit_models_from_picks(
            specs=specs,
            project_root=project_root,
            picks_path=Path(args.picks),
            out_dir=Path(args.out_dir),
            tag=args.tag,
        )
        return

    if args.command == "predict":
        project_root = specs_dir.parent
        tier_thresholds = {"htn_hi": args.htn_hi, "hyper_hi": args.hyper_hi, "lip_hi": args.lip_hi, "any_mid": args.any_mid}
        predict_from_models(
            specs=specs,
            project_root=project_root,
            models_root=Path(args.models_dir),
            input_path=Path(args.input),
            output_path=Path(args.output),
            mapping_path=Path(args.mapping) if args.mapping else None,
            include_tiers=not args.no_tiers,
            tier_thresholds=tier_thresholds,
        )
        return


if __name__ == "__main__":
    main()
