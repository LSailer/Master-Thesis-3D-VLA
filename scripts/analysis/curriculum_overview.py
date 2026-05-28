"""Build a self-contained HTML overview of the 4 curriculum levels.

Reads data/curriculum/level{1..4}*.json, counts train+eval episodes,
breaks down per goal category and per house (scene), and writes the
report to docs/curriculum-overview.html.

Run from the repo root:
    python scripts/analysis/curriculum_overview.py
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date
from html import escape
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CURRICULUM_DIR = REPO / "data" / "curriculum"
OUTPUT = REPO / "docs" / "curriculum-overview.html"

LEVEL_FILES = [
    "level1_1house_1goal.json",
    "level2_1house_6goals.json",
    "level3_10houses_1goal.json",
    "level4_10houses_6goals.json",
]

# Distinct accent per goal category — chosen to stay readable on dark bg.
GOAL_COLORS = {
    "chair":      "#6366f1",
    "bed":        "#22d3ee",
    "plant":      "#34d399",
    "sofa":       "#fbbf24",
    "toilet":     "#f87171",
    "tv_monitor": "#a78bfa",
}
FALLBACK_COLOR = "#94a3b8"


def load_level(path: Path) -> dict:
    data = json.loads(path.read_text())
    train = data.get("train_episode_keys", [])
    evalk = data.get("eval_episode_keys", [])
    # tuples shaped (episode_id, goal, scene)
    by_goal_train = Counter(t[1] for t in train)
    by_goal_eval = Counter(t[1] for t in evalk)
    by_scene_train = Counter(t[2] for t in train)
    by_scene_eval = Counter(t[2] for t in evalk)
    pair_train = Counter((t[2], t[1]) for t in train)
    pair_eval = Counter((t[2], t[1]) for t in evalk)
    return {
        "name": data["name"],
        "description": data.get("description", ""),
        "scenes": data.get("scenes", []),
        "categories": data.get("categories", []),
        "train_total": len(train),
        "eval_total": len(evalk),
        "train_ratio": data.get("train_ratio"),
        "seed": data.get("seed"),
        "by_goal_train": by_goal_train,
        "by_goal_eval": by_goal_eval,
        "by_scene_train": by_scene_train,
        "by_scene_eval": by_scene_eval,
        "pair_train": pair_train,
        "pair_eval": pair_eval,
    }


def color_for(goal: str) -> str:
    return GOAL_COLORS.get(goal, FALLBACK_COLOR)


def fmt_int(n: int) -> str:
    return f"{n:,}"


def bar_row(label: str, train: int, eval_: int, max_total: int, color: str) -> str:
    total = train + eval_
    pct_total = (total / max_total * 100) if max_total else 0
    train_w = (train / max_total * 100) if max_total else 0
    eval_w = (eval_ / max_total * 100) if max_total else 0
    return f"""
    <div class="bar-row">
      <div class="bar-label"><code>{escape(label)}</code></div>
      <div class="bar-track">
        <div class="bar-fill bar-train" style="width:{train_w:.2f}%; background:{color};"
             title="train: {fmt_int(train)}"></div>
        <div class="bar-fill bar-eval"  style="width:{eval_w:.2f}%; background:{color};
             opacity:0.45;" title="eval: {fmt_int(eval_)}"></div>
      </div>
      <div class="bar-value">
        <span class="v-total">{fmt_int(total)}</span>
        <span class="v-split">{fmt_int(train)} train · {fmt_int(eval_)} eval</span>
      </div>
    </div>"""


def render_goal_distribution(level: dict) -> str:
    cats = level["categories"]
    if not cats:
        return ""
    totals = {c: level["by_goal_train"][c] + level["by_goal_eval"][c] for c in cats}
    max_total = max(totals.values()) if totals else 1
    rows = "".join(
        bar_row(
            c,
            level["by_goal_train"][c],
            level["by_goal_eval"][c],
            max_total,
            color_for(c),
        )
        for c in sorted(cats, key=lambda x: -totals[x])
    )
    return f"""
    <div class="dist-block">
      <h4>Goals × episodes</h4>
      <div class="bars">{rows}</div>
    </div>"""


def render_scene_distribution(level: dict) -> str:
    scenes = level["scenes"]
    if not scenes:
        return ""
    totals = {
        s: level["by_scene_train"][s] + level["by_scene_eval"][s] for s in scenes
    }
    max_total = max(totals.values()) if totals else 1
    # Single house cases benefit less from this view.
    if len(scenes) <= 1:
        return ""
    rows = "".join(
        bar_row(
            s,
            level["by_scene_train"][s],
            level["by_scene_eval"][s],
            max_total,
            "#818cf8",
        )
        for s in sorted(scenes, key=lambda x: -totals[x])
    )
    return f"""
    <div class="dist-block">
      <h4>Houses × episodes</h4>
      <div class="bars">{rows}</div>
    </div>"""


def render_pair_heatmap(level: dict) -> str:
    scenes = level["scenes"]
    cats = level["categories"]
    if len(scenes) <= 1 or len(cats) <= 1:
        return ""
    pair_total = {
        (s, c): level["pair_train"][(s, c)] + level["pair_eval"][(s, c)]
        for s in scenes
        for c in cats
    }
    max_val = max(pair_total.values()) if pair_total else 1
    min_val = min(v for v in pair_total.values() if v > 0) if pair_total else 0

    header = (
        "<thead><tr><th class='heat-corner'>house \\ goal</th>"
        + "".join(f"<th>{escape(c)}</th>" for c in cats)
        + "<th>total</th></tr></thead>"
    )
    rows = []
    scene_order = sorted(scenes, key=lambda s: -sum(pair_total[(s, c)] for c in cats))
    for s in scene_order:
        row_cells = [f"<th class='heat-row-label'><code>{escape(s)}</code></th>"]
        row_total = 0
        for c in cats:
            v = pair_total[(s, c)]
            row_total += v
            frac = v / max_val if max_val else 0
            base = color_for(c)
            row_cells.append(
                f"<td class='heat-cell' style='background:{base};opacity:{0.18 + 0.7 * frac:.2f};'>"
                f"<span>{fmt_int(v)}</span></td>"
            )
        row_cells.append(f"<td class='heat-total'>{fmt_int(row_total)}</td>")
        rows.append(f"<tr>{''.join(row_cells)}</tr>")

    # column totals
    col_totals = [sum(pair_total[(s, c)] for s in scenes) for c in cats]
    grand_total = sum(col_totals)
    footer_cells = (
        "<th class='heat-row-label'>total</th>"
        + "".join(f"<td class='heat-total'>{fmt_int(v)}</td>" for v in col_totals)
        + f"<td class='heat-total heat-grand'>{fmt_int(grand_total)}</td>"
    )
    return f"""
    <div class="dist-block dist-wide">
      <h4>House × goal heatmap (train + eval)</h4>
      <table class="heatmap">
        {header}
        <tbody>{''.join(rows)}</tbody>
        <tfoot><tr>{footer_cells}</tr></tfoot>
      </table>
      <p class="heat-note">Cell shade scales with episode count
        (min {fmt_int(min_val)}, max {fmt_int(max_val)}). Hue encodes goal category.</p>
    </div>"""


def render_level(level: dict) -> str:
    n_scenes = len(level["scenes"])
    n_cats = len(level["categories"])
    grand = level["train_total"] + level["eval_total"]
    eval_frac = level["eval_total"] / grand * 100 if grand else 0
    return f"""
    <section class="level-card">
      <header class="level-head">
        <h2><span class="level-tag">{escape(level['name'])}</span></h2>
        <p class="level-desc">{escape(level['description'])}</p>
      </header>

      <div class="stat-grid">
        <div class="stat"><div class="stat-num">{fmt_int(grand)}</div>
          <div class="stat-lbl">total episodes</div></div>
        <div class="stat"><div class="stat-num">{fmt_int(level['train_total'])}</div>
          <div class="stat-lbl">train</div></div>
        <div class="stat"><div class="stat-num">{fmt_int(level['eval_total'])}</div>
          <div class="stat-lbl">eval ({eval_frac:.1f}%)</div></div>
        <div class="stat"><div class="stat-num">{n_scenes}</div>
          <div class="stat-lbl">house{'s' if n_scenes != 1 else ''}</div></div>
        <div class="stat"><div class="stat-num">{n_cats}</div>
          <div class="stat-lbl">goal{'s' if n_cats != 1 else ''}</div></div>
      </div>

      <div class="dist-grid">
        {render_goal_distribution(level)}
        {render_scene_distribution(level)}
      </div>
      {render_pair_heatmap(level)}
    </section>"""


def render_summary_table(levels: list[dict]) -> str:
    rows = []
    for L in levels:
        n_s = len(L["scenes"])
        n_c = len(L["categories"])
        total = L["train_total"] + L["eval_total"]
        # how many train episodes per (house, goal) cell on average
        per_cell = total / max(1, n_s * n_c)
        rows.append(
            f"<tr>"
            f"<td><code>{escape(L['name'])}</code></td>"
            f"<td class='r'>{n_s}</td>"
            f"<td class='r'>{n_c}</td>"
            f"<td class='r'>{fmt_int(L['train_total'])}</td>"
            f"<td class='r'>{fmt_int(L['eval_total'])}</td>"
            f"<td class='r'><strong>{fmt_int(total)}</strong></td>"
            f"<td class='r'>{fmt_int(round(per_cell))}</td>"
            f"</tr>"
        )
    return f"""
    <table class="summary-table">
      <thead><tr>
        <th>level</th><th class='r'>houses</th><th class='r'>goals</th>
        <th class='r'>train</th><th class='r'>eval</th>
        <th class='r'>total</th><th class='r'>≈ episodes / (house × goal)</th>
      </tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>"""


def render_html(levels: list[dict]) -> str:
    today = date.today().isoformat()
    sections = "\n".join(render_level(L) for L in levels)
    summary = render_summary_table(levels)
    legend = " ".join(
        f"<span class='legend-pill' style='--c:{c}'><span class='dot'></span>"
        f"<code>{escape(g)}</code></span>"
        for g, c in GOAL_COLORS.items()
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Curriculum Overview — 4 levels, houses × goals × episodes</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  :root {{
    --bg: #0f0f13;
    --surface: #1a1a24;
    --surface-2: #22222e;
    --surface-3: #2d2d3d;
    --accent: #6366f1;
    --accent2: #818cf8;
    --good: #34d399;
    --warn: #fbbf24;
    --bad: #f87171;
    --text: #e2e8f0;
    --text-dim: #94a3b8;
    --text-dimmer: #64748b;
    --border: #2d2d3d;
  }}

  body {{
    font-family: 'Inter', -apple-system, sans-serif;
    background: var(--bg); color: var(--text);
    min-height: 100vh; padding: 48px 24px; line-height: 1.55;
  }}
  .container {{ max-width: 1100px; margin: 0 auto; }}

  header.page {{ margin-bottom: 28px; }}
  h1 {{ font-size: 1.85rem; font-weight: 700; margin-bottom: 6px; }}
  .subtitle {{ color: var(--text-dim); font-size: 0.9rem; }}
  .tag {{
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;
    background: var(--surface-3); color: var(--text-dim); margin-right: 6px;
  }}
  code, .mono {{ font-family: 'JetBrains Mono', monospace; font-size: 0.85em; }}
  code {{
    background: var(--surface-2); padding: 1px 5px; border-radius: 3px;
    color: #c4b5fd;
  }}

  h2 {{
    font-size: 1.15rem; font-weight: 600; margin: 0 0 4px;
  }}
  h3 {{
    font-size: 0.95rem; font-weight: 600; margin: 22px 0 10px;
    color: var(--accent2); text-transform: uppercase; letter-spacing: 1px;
  }}
  h4 {{
    font-size: 0.85rem; font-weight: 600; margin: 0 0 10px;
    color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.8px;
  }}

  .summary-card {{
    background: var(--surface); border: 1px solid var(--border);
    border-left: 3px solid var(--accent); border-radius: 6px;
    padding: 18px 22px; margin-bottom: 20px;
  }}
  .summary-card p {{ color: var(--text-dim); }}

  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th, td {{ padding: 8px 12px; border-bottom: 1px solid var(--border); }}
  th {{
    text-align: left; color: var(--accent2); font-weight: 600;
    background: var(--surface); border-bottom: 2px solid var(--border);
  }}
  td {{ color: var(--text-dim); }}
  td.r, th.r {{ text-align: right; font-variant-numeric: tabular-nums; }}
  tr:hover td {{ background: rgba(99,102,241,0.05); }}

  .level-card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 22px 26px; margin: 22px 0;
  }}
  .level-head {{ margin-bottom: 16px; }}
  .level-tag {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.95rem; color: var(--accent2);
  }}
  .level-desc {{ color: var(--text-dim); font-size: 0.88rem; margin-top: 4px; }}

  .stat-grid {{
    display: grid; gap: 10px;
    grid-template-columns: repeat(5, 1fr);
    margin: 8px 0 22px;
  }}
  .stat {{
    background: var(--surface-2); border: 1px solid var(--border);
    border-radius: 6px; padding: 12px 14px;
  }}
  .stat-num {{
    font-family: 'JetBrains Mono', monospace; font-weight: 600;
    font-size: 1.15rem; color: var(--text);
  }}
  .stat-lbl {{
    color: var(--text-dimmer); font-size: 0.7rem;
    text-transform: uppercase; letter-spacing: 1px; margin-top: 2px;
  }}

  .dist-grid {{
    display: grid; gap: 22px;
    grid-template-columns: 1fr 1fr;
    margin-top: 4px;
  }}
  .dist-block.dist-wide {{ grid-column: 1 / -1; margin-top: 18px; }}

  .bars {{ display: flex; flex-direction: column; gap: 6px; }}
  .bar-row {{
    display: grid;
    grid-template-columns: 78px 1fr 132px;
    gap: 10px; align-items: center;
    font-size: 0.78rem;
  }}
  .bar-label code {{ font-size: 0.75rem; }}
  .bar-track {{
    background: var(--surface-2); border-radius: 3px;
    height: 14px; position: relative; overflow: hidden;
    border: 1px solid var(--border);
  }}
  .bar-fill {{
    height: 100%; display: inline-block;
    transition: width 0.2s;
  }}
  .bar-eval {{ /* sits next to train */ }}
  .bar-value {{
    text-align: right; font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; color: var(--text-dim);
    display: flex; flex-direction: column; line-height: 1.25;
  }}
  .v-total {{ color: var(--text); font-weight: 600; }}
  .v-split {{ font-size: 0.66rem; color: var(--text-dimmer); }}

  /* Heatmap */
  .heatmap {{ table-layout: fixed; font-size: 0.78rem; }}
  .heatmap th, .heatmap td {{ text-align: center; padding: 6px 4px; }}
  .heatmap th {{ font-size: 0.72rem; }}
  .heat-corner {{ background: transparent !important; color: var(--text-dimmer); }}
  .heat-row-label {{
    text-align: left !important; background: var(--surface) !important;
    border-right: 1px solid var(--border);
  }}
  .heat-row-label code {{ font-size: 0.7rem; }}
  .heat-cell {{
    color: #0f0f13; font-weight: 600;
    font-family: 'JetBrains Mono', monospace; font-size: 0.74rem;
    border: 1px solid var(--border);
  }}
  .heat-cell span {{
    mix-blend-mode: normal;
    text-shadow: 0 1px 2px rgba(15,15,19,0.55);
    color: #0f0f13;
  }}
  .heat-total {{
    font-family: 'JetBrains Mono', monospace; font-weight: 600;
    color: var(--text); background: var(--surface-2);
  }}
  .heat-grand {{ color: var(--accent2); }}
  .heat-note {{ font-size: 0.72rem; color: var(--text-dimmer); margin-top: 8px; }}

  .legend {{
    display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 22px;
  }}
  .legend-pill {{
    display: inline-flex; align-items: center; gap: 6px;
    padding: 3px 10px; background: var(--surface);
    border: 1px solid var(--border); border-radius: 999px;
    font-size: 0.75rem;
  }}
  .legend-pill .dot {{
    width: 9px; height: 9px; border-radius: 50%; background: var(--c);
  }}

  footer.page {{
    margin-top: 36px; padding-top: 16px;
    border-top: 1px solid var(--border);
    font-size: 0.78rem; color: var(--text-dimmer);
  }}

  @media (max-width: 720px) {{
    .stat-grid {{ grid-template-columns: repeat(2, 1fr); }}
    .dist-grid {{ grid-template-columns: 1fr; }}
    .bar-row {{ grid-template-columns: 70px 1fr 110px; }}
  }}
</style>
</head>
<body>
<div class="container">

<header class="page">
  <span class="tag">curriculum</span><span class="tag">overview</span>
  <h1>Curriculum overview — houses × goals × episodes</h1>
  <p class="subtitle">
    Auto-generated from <code>data/curriculum/level{{1..4}}_*.json</code>.
    Each episode is a Habitat ObjectNav trajectory keyed by
    <code>(episode_id, goal_category, scene_id)</code>.
  </p>
</header>

<div class="summary-card">
  <h2>Summary</h2>
  <p style="margin-top:6px;">
    The curriculum is a 2&times;2 design varying <em>scene diversity</em>
    (1 vs 10 houses) and <em>goal diversity</em> (1 vs 6 object categories).
    All splits use <code>seed=42</code> with <code>train_ratio=0.9</code>,
    so the same trajectory always falls on the same side of the split
    across ablations.
  </p>
</div>

{summary}

<h3>Goal-category colour legend</h3>
<div class="legend">{legend}</div>

{sections}

<footer class="page">
  Generated {today} by
  <code>scripts/analysis/curriculum_overview.py</code>.
  Bar tracks: solid = train, faded = eval (the two are concatenated, not stacked).
</footer>

</div>
</body>
</html>"""


def main() -> None:
    levels = [load_level(CURRICULUM_DIR / fn) for fn in LEVEL_FILES]
    html = render_html(levels)
    OUTPUT.write_text(html)
    print(f"wrote {OUTPUT.relative_to(REPO)}  ({OUTPUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
