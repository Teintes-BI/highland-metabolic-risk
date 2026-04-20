from pathlib import Path


def write_report(output_dir: Path, summary: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    html = """<html><head><meta charset='utf-8'></head><body>
<h1>NCDPipe Report</h1>
<pre>{}</pre>
</body></html>""".format(summary)

    (output_dir / "report.html").write_text(html, encoding="utf-8")
