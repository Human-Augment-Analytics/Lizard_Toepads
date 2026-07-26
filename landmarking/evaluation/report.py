"""HTML report generation for evaluation results.

Generates a self-contained HTML file with per-model results,
comparison tables, and summary statistics.
"""

import json
from pathlib import Path
from typing import Dict, Optional


def generate_html_report(
    results: Dict[str, dict],
    output_path: str,
    title: str = "Landmark Detection Evaluation Report",
) -> None:
    """Generate an HTML report from evaluation results.

    Args:
        results: Dict mapping model names to their evaluation result dicts.
        output_path: Path for the output HTML file.
        title: Report title.
    """
    sections = []

    # Header
    sections.append(f"<html><head><title>{title}</title>")
    sections.append("<style>")
    sections.append("body { font-family: sans-serif; margin: 20px; }")
    sections.append("table { border-collapse: collapse; margin: 10px 0; }")
    sections.append("th, td { border: 1px solid #ccc; padding: 6px 12px; text-align: right; }")
    sections.append("th { background: #f0f0f0; }")
    sections.append(".best { font-weight: bold; color: #006600; }")
    sections.append("</style></head><body>")
    sections.append(f"<h1>{title}</h1>")

    # Detect dataset type from first result
    first_result = next(iter(results.values()), {}) if results else {}
    is_wflw = "nme" in first_result

    if is_wflw:
        sections.append(_build_wflw_table(results))
    else:
        sections.append(_build_lizard_table(results))

    # Raw JSON dump
    sections.append("<h2>Raw Results (JSON)</h2>")
    sections.append("<pre>")
    sections.append(_safe_json(results))
    sections.append("</pre>")

    sections.append("</body></html>")

    # Write file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(sections))


def _build_lizard_table(results: Dict[str, dict]) -> str:
    """Build HTML table for Lizard pixel error results."""
    lines = []
    lines.append("<h2>Per-Model Pixel Error</h2>")
    lines.append("<table>")
    lines.append("<tr><th>Model</th><th>Mean (px)</th><th>Median (px)</th><th>Mean (mm)</th><th>N</th></tr>")

    for name, metrics in results.items():
        mean_px = _fmt(metrics.get("mean_px_error"))
        median_px = _fmt(metrics.get("median_px_error"))
        mean_mm = _fmt(metrics.get("mean_mm"))
        n = metrics.get("n_evaluated", "?")
        lines.append(
            f"<tr><td>{name}</td><td>{mean_px}</td>"
            f"<td>{median_px}</td><td>{mean_mm}</td><td>{n}</td></tr>"
        )

    lines.append("</table>")
    return "\n".join(lines)


def _build_wflw_table(results: Dict[str, dict]) -> str:
    """Build HTML table for WFLW NME/FR/AUC results."""
    lines = []
    lines.append("<h2>WFLW Evaluation Metrics</h2>")
    lines.append("<table>")
    lines.append("<tr><th>Model</th><th>NME</th><th>FR@0.1</th><th>AUC@0.1</th><th>N</th></tr>")

    for name, metrics in results.items():
        nme_dict = metrics.get("nme", {})
        fr_dict = metrics.get("fr", {})
        auc_dict = metrics.get("auc", {})
        counts = metrics.get("counts", {})

        nme = _fmt(nme_dict.get("full"), 4)
        fr = _fmt(fr_dict.get("full"), 4)
        auc = _fmt(auc_dict.get("full"), 4)
        n = counts.get("full", "?")

        lines.append(
            f"<tr><td>{name}</td><td>{nme}</td>"
            f"<td>{fr}</td><td>{auc}</td><td>{n}</td></tr>"
        )

    lines.append("</table>")
    return "\n".join(lines)


def _fmt(value, decimals: int = 2) -> str:
    """Format a numeric value or return N/A."""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def _safe_json(obj) -> str:
    """JSON serialize with fallback for non-serializable types."""
    try:
        return json.dumps(obj, indent=2, default=str)
    except (TypeError, ValueError):
        return str(obj)
