from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Synthetic Data Report - {title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f7f7f7; }}
    header {{ background-color: #111827; color: white; padding: 20px; }}
    h1 {{ margin: 0; font-size: 24px; }}
    main {{ padding: 20px; }}
    section {{ background-color: white; border-radius: 8px; padding: 16px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
    th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid #e5e7eb; font-size: 14px; }}
    th {{ background-color: #f3f4f6; }}
    .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
    .image-card {{ background-color: white; border-radius: 8px; padding: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
    .image-card img {{ width: 100%; border-radius: 4px; }}
    .timestamp {{ font-size: 14px; color: #9ca3af; }}
  </style>
</head>
<body>
  <header>
    <h1>Synthetic Data Evaluation — {title}</h1>
    <div class="timestamp">Generated on {timestamp}</div>
  </header>
  <main>
    {sections}
    {images}
  </main>
</body>
</html>
"""

SECTION_TEMPLATE = """<section>
  <h2>{heading}</h2>
  <table>
    <tbody>
      {rows}
    </tbody>
  </table>
</section>
"""

ROW_TEMPLATE = "<tr><th>{key}</th><td>{value}</td></tr>"

IMAGE_CARD_TEMPLATE = """<div class="image-card">
  <h3>{title}</h3>
  <img src="{data_url}" alt="{title}"/>
</div>"""


KNOWN_IMAGE_TITLES = {
    "distributions.png": "Distributions",
    "qq_plots.png": "QQ Plots",
    "correlations.png": "Correlation Heatmaps",
    "statistical_summary.png": "Statistical Summary",
    "constraint_violations.png": "Constraint Violations",
}


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (float, int)):
        if isinstance(value, float) and (value != value or value == float("inf") or value == float("-inf")):
            return "-"
        if abs(value) >= 1000 or abs(value) < 1e-4 and value != 0:
            return f"{value:.2e}"
        return f"{value:.4f}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_format_value(v) for v in value)
    return str(value)


def _build_section(heading: str, data: Dict[str, Any]) -> str:
    if not data:
        return ""
    rows = "\n".join(ROW_TEMPLATE.format(key=key, value=_format_value(value)) for key, value in sorted(data.items()))
    return SECTION_TEMPLATE.format(heading=heading, rows=rows)


def _encode_image(path: Path) -> str | None:
    if not path.exists():
        return None
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _collect_image_cards(image_paths: Iterable[Path]) -> str:
    cards: List[str] = []
    seen: set[Path] = set()
    for path in image_paths:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        title = KNOWN_IMAGE_TITLES.get(path.name, path.stem.replace("_", " ").title())
        data_url = _encode_image(path)
        if not data_url:
            continue
        cards.append(IMAGE_CARD_TEMPLATE.format(title=title, data_url=data_url))
    if not cards:
        return ""
    return f'<section><h2>Visual Diagnostics</h2><div class="image-grid">{"".join(cards)}</div></section>'


def generate_html_report(
    output_dir: str,
    synthetic_name: str,
    metrics: Dict[str, Dict[str, Any]],
    image_paths: List[Path],
) -> str:
    """
    Generate an HTML report summarizing metrics and optional visuals.

    Returns the path to the generated HTML file.
    """
    sections: List[str] = []

    ordering: List[Tuple[str, str]] = [
        ("statistical", "Statistical Metrics"),
        ("coverage", "Coverage Metrics"),
        ("privacy", "Privacy Metrics"),
        ("constraints", "Constraint Metrics"),
        ("ml_efficacy", "ML Efficacy"),
        ("plausibility", "Plausibility"),
        ("dp", "Differential Privacy"),
    ]
    for key, heading in ordering:
        section_data = metrics.get(key)
        if isinstance(section_data, dict):
            section_html = _build_section(heading, section_data)
            if section_html:
                sections.append(section_html)

    images_html = _collect_image_cards(image_paths)

    report_dir = Path(output_dir) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{synthetic_name}.html"

    html = HTML_TEMPLATE.format(
        title=synthetic_name,
        timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        sections="\n".join(sections) if sections else "<section><p>No metrics available.</p></section>",
        images=images_html,
    )
    report_path.write_text(html, encoding="utf-8")
    return str(report_path)
