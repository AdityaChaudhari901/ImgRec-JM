"""Render an eval Metrics object as a readable Markdown report."""

from __future__ import annotations

from eval_harness.metrics import Metrics

_DECISIONS = ("approve", "reject", "agent")


def format_report(metrics: Metrics, title: str = "Dispute Eval Report") -> str:
    lines = [f"# {title}", ""]
    lines.append(f"- Cases: **{metrics.total}**")
    lines.append(f"- Decision accuracy: **{metrics.decision_accuracy:.2%}**")
    lines.append(f"- Category accuracy: **{metrics.category_accuracy:.2%}**")
    lines.append(
        f"- Approve precision: **{metrics.approve_precision:.2%}** · "
        f"recall: **{metrics.approve_recall:.2%}**"
    )
    lines.append("")

    lines.append("## Per-category decision accuracy")
    lines.append("")
    lines.append("| category | n | decision accuracy |")
    lines.append("|---|---|---|")
    for cat, stats in metrics.per_category.items():
        lines.append(f"| {cat} | {stats['n']} | {stats['decision_accuracy']:.2%} |")
    lines.append("")

    lines.append("## Confusion (expected → predicted)")
    lines.append("")
    lines.append("| expected ↓ / predicted → | " + " | ".join(_DECISIONS) + " |")
    lines.append("|---|" + "|".join(["---"] * len(_DECISIONS)) + "|")
    for exp in _DECISIONS:
        row = metrics.decision_confusion.get(exp, {})
        cells = " | ".join(str(row.get(p, 0)) for p in _DECISIONS)
        lines.append(f"| {exp} | {cells} |")
    lines.append("")

    if metrics.mismatches:
        lines.append("## Mismatches")
        lines.append("")
        for m in metrics.mismatches:
            lines.append(f"- {m}")
        lines.append("")

    return "\n".join(lines)
