import json
from pathlib import Path


def _load_yaml(path):
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required. Install pyyaml.") from exc

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_specs(specs_dir: Path):
    specs = {}
    specs["data"] = _load_yaml(specs_dir / "data_spec.yaml")["data"]
    specs["outcomes"] = _load_yaml(specs_dir / "outcomes_spec.yaml")["outcomes"]
    specs["features"] = _load_yaml(specs_dir / "features_spec.yaml")
    specs["model"] = _load_yaml(specs_dir / "model_spec.yaml")
    specs["stability"] = _load_yaml(specs_dir / "stability_spec.yaml")["stability"]
    specs["orthogonal"] = _load_yaml(specs_dir / "orthogonal_tables_spec.yaml")["orthogonal_tables"]
    specs["validation"] = _load_yaml(specs_dir / "validation_spec.yaml")["validation"]
    specs["external_mapping"] = _load_yaml(specs_dir / "external_mapping_template.yaml")["external_mapping"]
    return specs


def save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
