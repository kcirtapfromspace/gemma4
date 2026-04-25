#!/usr/bin/env python3
"""Sync duckdb experiment rows into markdown notes.

Writes one note per experiment to TWO destinations:
  1. `docs/experiments/{experiment_name}.md` — always; repo-tracked, simple
     YAML frontmatter. Audience: PR reviewers + dev team.
  2. The user's Obsidian vault via the Local REST API, at path
     `OBSIDIAN_NOTES_PATH/{experiment_name}.md` (default
     `the_archives/archives` — the source the kcirtap.io blog pulls from).
     Hugo-style YAML frontmatter (`title`, `categories`, `tags`, `draft: false`,
     etc.) matching the existing blog post convention. Audience: blog readers.
     Iff `OBSIDIAN_API_KEY` is set in the env (typically via the gitignored
     `.claude/settings.local.json`).

Both destinations are merged independently — frontmatter and the
`<!-- METRICS:BEGIN/END -->` block are rewritten from the DB; everything
else (Hypothesis, Method, Result, Decision, Links) is preserved per-side
so notes edited in Obsidian don't overwrite the repo, and vice versa.

Fires automatically via the PostToolUse hook in `.claude/settings.json`
after any run of `scripts/benchmark.py` or `scripts/publish_benchmarks.py`.
Also safe to invoke manually: `python3 scripts/sync_experiment_notes.py`.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

import duckdb
import requests
import urllib3

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
DB = SCRIPT_DIR / "benchmarks.duckdb"
NOTES_DIR = REPO_ROOT / "docs" / "experiments"

OBSIDIAN_API_URL = os.environ.get("OBSIDIAN_API_URL", "https://127.0.0.1:27124")
OBSIDIAN_API_KEY = os.environ.get("OBSIDIAN_API_KEY")
OBSIDIAN_NOTES_PATH = os.environ.get("OBSIDIAN_NOTES_PATH", "the_archives/archives")
BLOG_AUTHOR = os.environ.get("BLOG_AUTHOR", "Patrick Deutsch")
BLOG_TZ_OFFSET = os.environ.get("BLOG_TZ_OFFSET", "-06:00")
BLOG_CATEGORY = os.environ.get("BLOG_CATEGORY", "experiments")
BLOG_ARTICLE_SLUG = os.environ.get("BLOG_ARTICLE_SLUG", "gemma4-experiments")
BLOG_ARTICLE_TITLE = os.environ.get(
    "BLOG_ARTICLE_TITLE",
    "Gemma 4 Good — Experiment Log",
)
BLOG_ARTICLE_URL = os.environ.get("BLOG_ARTICLE_URL", "/experiments/gemma4-hackathon-log")
BLOG_ARTICLE_EXCERPT = os.environ.get(
    "BLOG_ARTICLE_EXCERPT",
    "Live log of every benchmark and validation run during the Gemma 4 Good hackathon — auto-updated from the duckdb experiment store.",
)

METRICS_BEGIN = "<!-- METRICS:BEGIN (auto-generated from duckdb — edits here are overwritten) -->"
METRICS_END = "<!-- METRICS:END -->"


def fmt_float(v, digits: int = 2) -> str:
    return "—" if v is None else f"{float(v):.{digits}f}"


def fmt_pct(v) -> str:
    return "—" if v is None else f"{float(v) * 100:.1f}%"


def as_iso(v) -> str:
    if v is None:
        return ""
    return v.isoformat(sep=" ", timespec="seconds") if hasattr(v, "isoformat") else str(v)


def query_experiments(conn: duckdb.DuckDBPyConnection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT experiment_id, experiment_name, team_tag, created_at,
               backend, device, runtime, model_variant, model_format, data_source,
               model_file,
               avg_gen_tok_s, p50_gen_tok_s, avg_ttft_ms, avg_prompt_tok_s,
               success_rate, avg_extraction_score, extraction_pass_rate,
               total_runs, speedup_pct, notes
        FROM experiments
        WHERE team_tag IS NOT NULL
        ORDER BY created_at DESC, experiment_name
        """
    ).fetchall()
    cols = [d[0] for d in conn.description]
    return [dict(zip(cols, r)) for r in rows]


def query_per_case(conn: duckdb.DuckDBPyConnection, experiment_id: str) -> list[dict]:
    rows = conn.execute(
        """
        SELECT case_id, gen_tok_per_sec, extraction_score,
               completion_tokens, total_ms, valid_json
        FROM benchmark_runs
        WHERE experiment_id = ?
        ORDER BY case_id, run_number
        """,
        [experiment_id],
    ).fetchall()
    cols = [d[0] for d in conn.description]
    return [dict(zip(cols, r)) for r in rows]


def build_frontmatter(exp: dict) -> str:
    fields: list[tuple[str, str]] = [
        ("experiment", exp["experiment_name"]),
        ("team_tag", exp["team_tag"] or ""),
        ("backend", exp["backend"] or ""),
        ("device", exp["device"] or ""),
        ("runtime", exp["runtime"] or ""),
        ("model_variant", exp["model_variant"] or ""),
        ("model_format", exp["model_format"] or ""),
        ("data_source", exp["data_source"] or ""),
        ("created_at", as_iso(exp["created_at"])),
        ("status", "done"),
    ]
    body = "\n".join(f"{k}: {v}" for k, v in fields)
    return f"---\n{body}\n---"


def build_metrics_block(exp: dict, per_case: list[dict]) -> str:
    lines: list[str] = [METRICS_BEGIN, "## Metrics", ""]
    lines.append(
        f"- **avg_gen_tok_s**: {fmt_float(exp['avg_gen_tok_s'])}  •  "
        f"**p50_gen_tok_s**: {fmt_float(exp['p50_gen_tok_s'])}"
    )
    lines.append(
        f"- **avg_ttft_ms**: {fmt_float(exp['avg_ttft_ms'], 0)}  •  "
        f"**avg_prompt_tok_s**: {fmt_float(exp['avg_prompt_tok_s'])}"
    )
    lines.append(f"- **extraction_pass_rate**: {fmt_pct(exp['extraction_pass_rate'])}")
    lines.append(f"- **avg_extraction_score**: {fmt_float(exp['avg_extraction_score'], 3)}")
    lines.append(f"- **success_rate (valid JSON)**: {fmt_pct(exp['success_rate'])}")
    lines.append(f"- **total_runs**: {exp['total_runs'] if exp['total_runs'] is not None else '—'}")
    if exp.get("model_file"):
        lines.append(f"- **model_file**: `{exp['model_file']}`")
    if exp.get("notes"):
        lines.append("")
        for para in str(exp["notes"]).split("\n"):
            lines.append(f"> {para}" if para.strip() else ">")

    if per_case:
        lines += ["", "### Per-case runs", "", "| case_id | tok/s | extraction_score | tokens | valid_json |", "|---|---:|---:|---:|:---:|"]
        for r in per_case:
            lines.append(
                f"| {r['case_id']} "
                f"| {fmt_float(r['gen_tok_per_sec'])} "
                f"| {fmt_float(r['extraction_score'], 3)} "
                f"| {r['completion_tokens'] if r['completion_tokens'] is not None else '—'} "
                f"| {'✓' if r['valid_json'] else '✗'} |"
            )

    lines.append("")
    lines.append(METRICS_END)
    return "\n".join(lines)


HUMAN_SCAFFOLD = """## Hypothesis
_TODO: what did we expect going in? What would make this experiment interesting?_

## Method
_TODO: hardware, workload, scorer, anything non-obvious about how this was measured._

## Result
_TODO: narrative interpretation of the Metrics block above. Surprises? Regressions?_

## Decision
_TODO: what are we doing with this? What's the next experiment?_

## Links
- duckdb: `SELECT * FROM experiments WHERE experiment_name = '{name}';`
- per-case: `SELECT * FROM benchmark_runs WHERE experiment_name = '{name}';`
- commit: _TODO_
- raw artifacts: _TODO_
"""


FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
METRICS_RE = re.compile(
    re.escape(METRICS_BEGIN) + r".*?" + re.escape(METRICS_END),
    re.DOTALL,
)


def merge_note_content(existing: str | None, exp: dict, per_case: list[dict]) -> str:
    """Produce note content, preserving human sections in `existing` if any."""
    name = exp["experiment_name"]
    frontmatter = build_frontmatter(exp)
    metrics = build_metrics_block(exp, per_case)

    if existing is None:
        return (
            frontmatter + "\n\n"
            + f"# {name}\n\n"
            + metrics + "\n\n"
            + HUMAN_SCAFFOLD.format(name=name)
        )

    content = existing
    if FRONTMATTER_RE.match(content):
        content = FRONTMATTER_RE.sub(frontmatter + "\n", content, count=1)
    else:
        content = frontmatter + "\n\n" + content

    if METRICS_RE.search(content):
        content = METRICS_RE.sub(metrics, content, count=1)
    else:
        content = FRONTMATTER_RE.sub(
            lambda m: m.group(0) + "\n" + metrics + "\n\n",
            content,
            count=1,
        )
    return content


def sync_local_note(path: Path, exp: dict, per_case: list[dict]) -> str:
    existing = path.read_text() if path.exists() else None
    content = merge_note_content(existing, exp, per_case)
    path.write_text(content)
    return "updated" if existing else "created"


def _obsidian_session() -> requests.Session | None:
    if not OBSIDIAN_API_KEY:
        return None
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    s = requests.Session()
    s.verify = False
    s.headers.update({
        "Authorization": f"Bearer {OBSIDIAN_API_KEY}",
        "Content-Type": "text/markdown",
    })
    return s


def _yaml_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


SECTION_RE = re.compile(
    r"<!-- EXP:([\w.-]+):BEGIN -->(.*?)<!-- EXP:\1:END -->",
    re.DOTALL,
)
TABLE_BEGIN = "<!-- TABLE:BEGIN (auto-generated; edits here are overwritten) -->"
TABLE_END = "<!-- TABLE:END -->"
TABLE_RE = re.compile(re.escape(TABLE_BEGIN) + r".*?" + re.escape(TABLE_END), re.DOTALL)


def build_article_frontmatter(experiments: list[dict]) -> str:
    """Hugo-style YAML for the single experiment-log post."""
    earliest = min((e["created_at"] for e in experiments if e["created_at"]), default=datetime.now())
    date_str = earliest.strftime("%Y-%m-%dT%H:%M:%S") + BLOG_TZ_OFFSET
    lastmod_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S") + BLOG_TZ_OFFSET

    backends = sorted({e["backend"] for e in experiments if e["backend"]})
    devices = sorted({e["device"] for e in experiments if e["device"]})

    lines = [
        "---",
        f'title: "{_yaml_escape(BLOG_ARTICLE_TITLE)}"',
        f"author: {_yaml_escape(BLOG_AUTHOR)}",
        "type:",
        "  - post",
        f"date: {date_str}",
        f"lastmod: {lastmod_str}",
        f"url: {BLOG_ARTICLE_URL}",
        f'excerpt: "{_yaml_escape(BLOG_ARTICLE_EXCERPT)}"',
        "categories:",
        f"  - {BLOG_CATEGORY}",
        "tags:",
        "  - gemma-4",
        "  - hackathon",
        "  - edge-inference",
    ]
    for b in backends:
        lines.append(f"  - {_yaml_escape(b)}")
    for d in devices[:4]:  # cap: don't blow up the tag list
        lines.append(f"  - {_yaml_escape(d)}")
    lines += ["draft: false", "---"]
    return "\n".join(lines)


def build_summary_table(experiments: list[dict]) -> str:
    lines = [
        TABLE_BEGIN,
        "| team | experiment | backend | device | runtime | tok/s | extraction | source | date |",
        "|---|---|---|---|---|---:|---:|---|---|",
    ]
    for e in experiments:
        team = e["team_tag"] or "—"
        name = e["experiment_name"]
        backend = e["backend"] or "—"
        device = e["device"] or "—"
        runtime = e["runtime"] or "—"
        tok = fmt_float(e["avg_gen_tok_s"])
        ext = fmt_pct(e["extraction_pass_rate"])
        src = e["data_source"] or "—"
        date = e["created_at"].strftime("%Y-%m-%d") if e["created_at"] else "—"
        anchor = name.lower().replace("_", "-")
        lines.append(
            f"| {team} | [{name}](#{anchor}) | {backend} | {device} | "
            f"{runtime} | {tok} | {ext} | {src} | {date} |"
        )
    lines.append(TABLE_END)
    return "\n".join(lines)


def build_section(exp: dict, per_case: list[dict], existing: str | None) -> str:
    """Build one experiment section. If `existing` is provided, only the metrics
    sub-block is replaced — the rest of the section (your narrative) is preserved.
    """
    name = exp["experiment_name"]
    team = exp["team_tag"] or ""
    short_name = name[len(team) + 1:] if team and name.startswith(f"{team}-") else name
    header = f"## {team} — {short_name}" if team else f"## {name}"
    meta_bits = [exp.get("data_source") or "measured"]
    for k in ("backend", "device", "runtime", "model_variant", "model_format"):
        v = exp.get(k)
        if v:
            meta_bits.append(v)
    if exp["created_at"]:
        meta_bits.append(exp["created_at"].strftime("%Y-%m-%d"))
    meta_line = "*" + " • ".join(meta_bits) + "*"

    metrics = build_metrics_block(exp, per_case)
    begin = f"<!-- EXP:{name}:BEGIN -->"
    end = f"<!-- EXP:{name}:END -->"

    if existing is None:
        return (
            f"{begin}\n\n"
            f"{header}\n\n"
            f"{meta_line}\n\n"
            f"{metrics}\n\n"
            f"### Hypothesis\n_TODO: what did we expect going in?_\n\n"
            f"### Method\n_TODO: how was this measured?_\n\n"
            f"### Result\n_TODO: surprises, regressions, what the numbers mean._\n\n"
            f"### Decision\n_TODO: what's the next experiment?_\n\n"
            f"{end}"
        )

    inner = existing
    if METRICS_RE.search(inner):
        inner = METRICS_RE.sub(metrics, inner, count=1)
    else:
        inner = inner.rstrip() + "\n\n" + metrics + "\n\n"
    return inner


def build_article_content(
    existing: str | None,
    experiments: list[dict],
    per_case_map: dict[str, list[dict]],
) -> str:
    frontmatter = build_article_frontmatter(experiments)
    intro = (
        f"# {BLOG_ARTICLE_TITLE}\n\n"
        "_This article auto-updates from the project's duckdb experiment store_ "
        "(`scripts/benchmarks.duckdb`). _The summary table and per-experiment metrics blocks "
        "are regenerated on every sync; the prose sections (Hypothesis / Method / Result / Decision) "
        "are preserved across re-syncs so notes you write here stick._\n"
    )
    table = build_summary_table(experiments)

    existing_sections: dict[str, str] = {}
    if existing:
        for m in SECTION_RE.finditer(existing):
            existing_sections[m.group(1)] = m.group(2)  # inner content only

    section_blocks: list[str] = []
    for exp in experiments:
        name = exp["experiment_name"]
        per_case = per_case_map.get(exp["experiment_id"], [])
        existing_inner = existing_sections.get(name)
        if existing_inner is not None:
            inner = build_section(exp, per_case, existing_inner)
            section_blocks.append(f"<!-- EXP:{name}:BEGIN -->\n{inner}\n<!-- EXP:{name}:END -->")
        else:
            section_blocks.append(build_section(exp, per_case, None))

    return (
        frontmatter + "\n\n"
        + intro + "\n"
        + table + "\n\n"
        + "\n\n".join(section_blocks) + "\n"
    )


def sync_obsidian_article(
    session: requests.Session,
    experiments: list[dict],
    per_case_map: dict[str, list[dict]],
) -> tuple[str, int]:
    url = f"{OBSIDIAN_API_URL}/vault/{OBSIDIAN_NOTES_PATH}/{BLOG_ARTICLE_SLUG}.md"
    r = session.get(url, timeout=10)
    if r.status_code == 200:
        existing = r.text
        action = "updated"
    elif r.status_code == 404:
        existing = None
        action = "created"
    else:
        r.raise_for_status()
        existing = None
        action = "created"

    content = build_article_content(existing, experiments, per_case_map)
    p = session.put(url, data=content.encode("utf-8"), timeout=10)
    p.raise_for_status()
    return action, len(experiments)


def build_index(experiments: list[dict]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Experiments index",
        "",
        f"_Auto-generated from `scripts/benchmarks.duckdb` on {now}._",
        "_Edit individual notes directly; this index regenerates on every sync._",
        "",
        "| team | experiment | backend | device | runtime | tok/s | extraction | date |",
        "|---|---|---|---|---|---:|---:|---|",
    ]
    for e in experiments:
        name = e["experiment_name"]
        team = e["team_tag"] or "—"
        backend = e["backend"] or "—"
        device = e["device"] or "—"
        runtime = e["runtime"] or "—"
        tok = fmt_float(e["avg_gen_tok_s"])
        ext = fmt_pct(e["extraction_pass_rate"])
        date = e["created_at"].strftime("%Y-%m-%d") if e["created_at"] else "—"
        lines.append(
            f"| {team} | [{name}](./{name}.md) | {backend} | {device} | "
            f"{runtime} | {tok} | {ext} | {date} |"
        )
    lines.append("")
    return "\n".join(lines)


TEMPLATE_BODY = """# _template

_Reference shape for an experiment note. Actual notes are auto-generated by
`scripts/sync_experiment_notes.py` from `scripts/benchmarks.duckdb`._

---
experiment: c99-example
team_tag: c99
backend: litert-lm
device: iphone-17-pro
runtime: gpu
model_variant: cliniq-compact-lora-v2
model_format: litertlm-int4
data_source: measured
created_at: 2026-04-24 00:00:00
status: planned | running | done
---

# c99-example

<!-- METRICS:BEGIN -->
## Metrics

(auto-synced from duckdb — do not edit here)
<!-- METRICS:END -->

## Hypothesis
## Method
## Result
## Decision
## Links
"""


def main() -> None:
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    (NOTES_DIR / "_template.md").write_text(TEMPLATE_BODY)

    conn = duckdb.connect(str(DB), read_only=True)
    try:
        experiments = query_experiments(conn)
        local_stats = {"created": 0, "updated": 0}

        per_case_map: dict[str, list[dict]] = {}
        for exp in experiments:
            per_case = query_per_case(conn, exp["experiment_id"])
            per_case_map[exp["experiment_id"]] = per_case
            note_path = NOTES_DIR / f"{exp['experiment_name']}.md"
            local_stats[sync_local_note(note_path, exp, per_case)] += 1

        (NOTES_DIR / "README.md").write_text(build_index(experiments))

        article_status: tuple[str, int] | None = None
        article_error: str | None = None
        session = _obsidian_session()
        if session is not None:
            try:
                article_status = sync_obsidian_article(session, experiments, per_case_map)
            except requests.RequestException as e:
                article_error = str(e)
    finally:
        conn.close()

    print(
        f"local:   {len(experiments)} notes → docs/experiments/ "
        f"(created={local_stats['created']}, updated={local_stats['updated']})"
    )
    if session is None:
        print("article: skipped (OBSIDIAN_API_KEY not set)")
    elif article_error:
        print(f"article: FAILED — {article_error}")
    else:
        action, n = article_status  # type: ignore[misc]
        print(
            f"article: {action} → {OBSIDIAN_API_URL}/vault/{OBSIDIAN_NOTES_PATH}/{BLOG_ARTICLE_SLUG}.md "
            f"({n} experiment sections)"
        )


if __name__ == "__main__":
    main()
