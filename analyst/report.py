"""Export the conversation as a shareable Markdown or self-contained HTML report."""
from __future__ import annotations

import html
import re
from datetime import datetime

from .data_store import DataStore
from .models import Turn


def build_markdown(turns: list[Turn], store: DataStore) -> str:
    lines = ["# AI Data Analyst Report", f"_Generated {datetime.now():%Y-%m-%d %H:%M}_", "", "## Datasets"]
    for t in store.tables.values():
        lines.append(f"- **{t.name}** ({t.source}): {t.rows:,} rows, {len(t.columns)} columns")
    lines.append("")
    for i, turn in enumerate(turns, 1):
        lines += [f"## Q{i}. {turn.question}", "", turn.answer, ""]
        resp = turn.response
        if resp:
            for c in resp.artifacts.code:
                lines += [f"```{'sql' if c['language'] == 'sql' else 'python'}", c["code"], "```", ""]
            for tb in resp.artifacts.tables[:2]:
                lines += [f"**{tb['title']}**", "", tb["df"].head(15).to_csv(index=False).strip().replace(",", " | "), ""]
    return "\n".join(lines)


def _md_inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", text)


def _md_to_html(md: str) -> str:
    out, in_list, in_code = [], False, False
    for line in md.splitlines():
        if line.strip().startswith("```"):
            out.append("</pre>" if in_code else "<pre>")
            in_code = not in_code
            continue
        if in_code:
            out.append(html.escape(line))
            continue
        if line.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{_md_inline(line[2:])}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        if line.strip():
            out.append(f"<p>{_md_inline(line)}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


_CSS = """body{font-family:system-ui,Segoe UI,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem;color:#1f2937}
h1{border-bottom:3px solid #4f46e5;padding-bottom:.4rem}h2{margin-top:2.2rem;color:#312e81}
pre{background:#0f172a;color:#e2e8f0;padding:12px;border-radius:8px;overflow:auto}code{background:#eef2ff;padding:1px 5px;border-radius:4px}
pre code{background:none}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #d1d5db;padding:4px 8px}
th{background:#eef2ff}.meta{color:#6b7280;font-size:13px}.card{border:1px solid #e5e7eb;border-radius:10px;padding:1rem 1.2rem;margin:1rem 0}"""


def build_html(turns: list[Turn], store: DataStore) -> str:
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>AI Data Analyst Report</title><style>{_CSS}</style></head><body>",
             "<h1>AI Data Analyst Report</h1>",
             f"<p class='meta'>Generated {datetime.now():%Y-%m-%d %H:%M}</p><h2>Datasets</h2><ul>"]
    parts += [f"<li><strong>{html.escape(t.name)}</strong> ({html.escape(t.source)}): {t.rows:,} rows, {len(t.columns)} columns</li>"
              for t in store.tables.values()]
    parts.append("</ul>")
    first_plot = True
    for i, turn in enumerate(turns, 1):
        parts.append(f"<div class='card'><h2>Q{i}. {html.escape(turn.question)}</h2>{_md_to_html(turn.answer)}")
        resp = turn.response
        if resp:
            for fig in resp.artifacts.figures:
                parts.append(fig["fig"].to_html(full_html=False, include_plotlyjs="cdn" if first_plot else False))
                first_plot = False
            for c in resp.artifacts.code:
                parts.append(f"<details><summary>{html.escape(c['purpose'] or c['language'].upper())}</summary>"
                             f"<pre>{html.escape(c['code'])}</pre></details>")
            for tb in resp.artifacts.tables[:2]:
                parts.append(f"<details><summary>{html.escape(tb['title'])}</summary>{tb['df'].head(30).to_html(index=False, border=0)}</details>")
        parts.append("</div>")
    parts.append("</body></html>")
    return "\n".join(parts)
