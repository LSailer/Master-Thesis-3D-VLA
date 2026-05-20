"""Build the curriculum-scaling HTML slide deck with embedded base64 images.

Generates docs/curriculum-scaling.html.
"""

import base64
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

OUTPUT = "docs/curriculum-scaling.html"
FIG_DIR = "output/methods/comparisons/figures"


def b64(filename):
    path = os.path.join(FIG_DIR, filename)
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def main():
    sr_comp = b64("curriculum-sr-comparison.png")
    spl_comp = b64("curriculum-spl-comparison.png")
    goal_sr = b64("l2-per-goal-sr.png")
    goal_bar = b64("l2-goal-bar.png")
    floorplan = b64("l2-semantic-floorplan.png")
    wm_losses = b64("curriculum-wm-losses.png")

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Curriculum Scaling — L1, L2, L3</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  :root {{
    --bg: #0f0f13; --surface: #1a1a24; --accent: #6366f1; --accent2: #818cf8;
    --accent-green: #34d399; --accent-amber: #fbbf24; --accent-red: #f87171;
    --accent-cyan: #22d3ee; --accent-orange: #fb923c; --accent-pink: #f472b6;
    --text: #e2e8f0; --text-dim: #94a3b8; --border: #2d2d3d;
  }}
  body {{ font-family: 'Inter', -apple-system, sans-serif; background: var(--bg); color: var(--text); overflow: hidden; height: 100vh; width: 100vw; }}
  .slide-container {{ width: 100vw; height: 100vh; display: flex; align-items: center; justify-content: center; position: relative; }}
  .slide {{ display: none; width: 100%; height: 100%; padding: 40px 60px; flex-direction: column; justify-content: center; position: relative; animation: fadeIn 0.4s ease; }}
  .slide.active {{ display: flex; }}
  @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: translateY(0); }} }}
  .slide-number {{ position: fixed; bottom: 20px; right: 30px; font-size: 14px; color: var(--text-dim); font-weight: 300; }}
  .nav-hint {{ position: fixed; bottom: 20px; left: 30px; font-size: 12px; color: var(--text-dim); opacity: 0.5; }}
  .progress-bar {{ position: fixed; top: 0; left: 0; height: 3px; background: linear-gradient(90deg, var(--accent), var(--accent-cyan)); transition: width 0.3s ease; z-index: 100; }}
  h1 {{ font-size: 2.8rem; font-weight: 700; line-height: 1.15; margin-bottom: 20px; background: linear-gradient(135deg, #fff 0%, var(--accent2) 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
  h2 {{ font-size: 1.8rem; font-weight: 600; margin-bottom: 20px; color: #fff; }}
  h3 {{ font-size: 1rem; font-weight: 500; color: var(--accent2); margin-bottom: 10px; text-transform: uppercase; letter-spacing: 1.5px; }}
  .subtitle {{ font-size: 1.2rem; color: var(--text-dim); font-weight: 300; line-height: 1.6; max-width: 800px; }}
  .tag {{ display: inline-block; padding: 4px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 500; margin: 4px 4px 4px 0; }}
  .tag-accent {{ background: rgba(99,102,241,0.15); color: var(--accent2); border: 1px solid rgba(99,102,241,0.3); }}
  .tag-green {{ background: rgba(52,211,153,0.12); color: var(--accent-green); border: 1px solid rgba(52,211,153,0.25); }}
  .tag-amber {{ background: rgba(251,191,36,0.12); color: var(--accent-amber); border: 1px solid rgba(251,191,36,0.25); }}
  .tag-red {{ background: rgba(248,113,113,0.12); color: var(--accent-red); border: 1px solid rgba(248,113,113,0.25); }}
  .tag-cyan {{ background: rgba(34,211,238,0.12); color: var(--accent-cyan); border: 1px solid rgba(34,211,238,0.25); }}
  .card-grid {{ display: grid; gap: 14px; margin-top: 12px; }}
  .grid-2 {{ grid-template-columns: 1fr 1fr; }}
  .grid-3 {{ grid-template-columns: 1fr 1fr 1fr; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 20px; }}
  .card h4 {{ font-size: 1rem; font-weight: 600; margin-bottom: 8px; color: #fff; }}
  .card p, .card li {{ font-size: 0.88rem; color: var(--text-dim); line-height: 1.55; }}
  .card ul {{ list-style: none; padding: 0; }}
  .card ul li {{ padding: 2px 0; padding-left: 16px; position: relative; }}
  .card ul li::before {{ content: '\\2192'; position: absolute; left: 0; color: var(--accent2); }}
  .highlight-box {{ background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(34,211,238,0.05)); border: 1px solid rgba(99,102,241,0.2); border-radius: 14px; padding: 18px 24px; margin: 10px 0; }}
  .highlight-box.green {{ background: linear-gradient(135deg, rgba(52,211,153,0.08), rgba(34,211,238,0.05)); border-color: rgba(52,211,153,0.25); }}
  .highlight-box.amber {{ background: linear-gradient(135deg, rgba(251,191,36,0.08), rgba(248,113,113,0.05)); border-color: rgba(251,191,36,0.25); }}
  .highlight-box.red {{ background: linear-gradient(135deg, rgba(248,113,113,0.08), rgba(251,191,36,0.05)); border-color: rgba(248,113,113,0.25); }}
  .metric-big {{ font-size: 2.8rem; font-weight: 700; line-height: 1; }}
  .metric-label {{ font-size: 0.85rem; color: var(--text-dim); margin-top: 4px; }}
  .plot-img {{ max-width: 100%; max-height: 55vh; border-radius: 10px; border: 1px solid var(--border); margin: 8px auto; display: block; }}
  .plot-img.large {{ max-height: 65vh; }}
  table {{ width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 0.9rem; }}
  th {{ text-align: left; padding: 8px 12px; background: var(--surface); color: var(--accent2); font-weight: 600; border-bottom: 2px solid var(--border); }}
  td {{ padding: 7px 12px; border-bottom: 1px solid var(--border); color: var(--text-dim); }}
  td.highlight {{ color: #fff; font-weight: 600; }}
</style>
</head>
<body>

<div class="progress-bar" id="progress"></div>
<div class="slide-container">

<!-- Slide 1: Title -->
<div class="slide active">
  <h3>Curriculum Scaling Experiments</h3>
  <h1>Scaling Complexity Axes<br>in World-Model Navigation</h1>
  <p class="subtitle">
    How does R2-Dreamer perform as we independently scale<br>
    the number of goals (L2) and the number of scenes (L3)?
  </p>
  <div style="margin-top: 20px;">
    <span class="tag tag-accent">L1: 1 house, 1 goal &rarr; 75% SR</span>
    <span class="tag tag-amber">L2: 1 house, 6 goals &rarr; 36% SR</span>
    <span class="tag tag-green">L3: 10 houses, 1 goal &rarr; 32% SR</span>
  </div>
  <div style="margin-top: 20px;">
    <span class="tag tag-cyan">R2-Dreamer</span>
    <span class="tag tag-cyan">HM3D ObjectNav</span>
    <span class="tag tag-cyan">2.4M steps each</span>
  </div>
  <p style="color: var(--text-dim); margin-top: 30px; font-size: 0.85rem;">
    WandB: y5a0upzd, flky9ybh, rsopsua1 &nbsp;|&nbsp; 2026-04-16
  </p>
</div>

<!-- Slide 2: Experiment Design -->
<div class="slide">
  <h3>Experiment Design</h3>
  <h2>Two Axes of Complexity</h2>
  <p class="subtitle" style="margin-bottom: 16px;">
    Starting from L1 (1 house, chair only), we scale one axis at a time to isolate effects.
  </p>
  <div class="card-grid grid-3">
    <div class="card">
      <h4 style="color: var(--accent2);">L1 Rerun (Anchor)</h4>
      <ul>
        <li>1 house (fK2vEV32Lag)</li>
        <li>1 goal (chair)</li>
        <li>2.4M steps</li>
        <li>Buffer fix + step penalty</li>
      </ul>
      <div style="margin-top: 12px;">
        <span class="metric-big" style="color: var(--accent2);">75%</span>
        <div class="metric-label">Success Rate</div>
      </div>
    </div>
    <div class="card">
      <h4 style="color: var(--accent-amber);">L2 (Scale Goals)</h4>
      <ul>
        <li>1 house (same)</li>
        <li><strong>6 goals</strong> (no conditioning)</li>
        <li>2.4M steps</li>
        <li>Uniform episode distribution</li>
      </ul>
      <div style="margin-top: 12px;">
        <span class="metric-big" style="color: var(--accent-amber);">36%</span>
        <div class="metric-label">Average SR</div>
      </div>
    </div>
    <div class="card">
      <h4 style="color: var(--accent-green);">L3 (Scale Scenes)</h4>
      <ul>
        <li><strong>10 houses</strong> (4 easy, 4 med, 2 hard)</li>
        <li>1 goal (chair)</li>
        <li>2.4M steps</li>
        <li>Data spread across environments</li>
      </ul>
      <div style="margin-top: 12px;">
        <span class="metric-big" style="color: var(--accent-green);">32%</span>
        <div class="metric-label">Success Rate</div>
      </div>
    </div>
  </div>
</div>

<!-- Slide 3: SR Comparison -->
<div class="slide">
  <h3>Training Curves</h3>
  <h2>Success Rate Across Curriculum Levels</h2>
  <img class="plot-img" src="data:image/png;base64,{sr_comp}" alt="SR comparison">
  <div class="highlight-box" style="margin-top: 10px;">
    <p style="font-size: 0.9rem; color: var(--text-dim);">
      <strong style="color: #fff;">Both L2 and L3 plateau around 30-40% SR</strong> &mdash;
      scaling goals and scaling scenes impose similar performance costs (~40pp below L1).
      All three experiments learn well above the random baseline (3.8%).
    </p>
  </div>
</div>

<!-- Slide 4: L2 Goal Hierarchy -->
<div class="slide">
  <h3>L2 Deep Dive</h3>
  <h2>Goal Difficulty Hierarchy</h2>
  <div class="card-grid grid-2">
    <div>
      <img style="width:100%; border-radius:10px; border:1px solid var(--border);" src="data:image/png;base64,{goal_sr}" alt="Per-goal SR">
    </div>
    <div>
      <img style="width:100%; border-radius:10px; border:1px solid var(--border);" src="data:image/png;base64,{goal_bar}" alt="Goal bar chart">
    </div>
  </div>
  <div class="highlight-box amber" style="margin-top: 10px;">
    <p style="font-size: 0.9rem; color: var(--text-dim);">
      <strong style="color: #fff;">Navigation complexity (Geo/Euc ratio) predicts SR</strong>, not distance or instance count.
      Plant (66% SR, 1 instance) succeeds because paths are direct (ratio 1.18).
      TV monitor (3% SR, 2 instances) fails because paths detour 77% around walls.
    </p>
  </div>
</div>

<!-- Slide 5: Semantic Floor Plan -->
<div class="slide">
  <h3>L2 Spatial Analysis</h3>
  <h2>Why Some Goals Are Harder</h2>
  <img class="plot-img large" src="data:image/png;base64,{floorplan}" alt="Semantic floor plan">
  <div class="highlight-box green" style="margin-top: 6px;">
    <p style="font-size: 0.9rem; color: var(--text-dim);">
      <strong style="color: #fff;">The house layout explains everything.</strong>
      Plant sits in an open area with direct paths from most spawn points.
      Toilet and TV monitor are tucked behind walls and doorways &mdash;
      the agent must navigate complex corridors to reach them.
    </p>
  </div>
</div>

<!-- Slide 6: World Model Losses -->
<div class="slide">
  <h3>World Model</h3>
  <h2>Dynamics Loss &mdash; Overfitting vs Generalization</h2>
  <img class="plot-img" src="data:image/png;base64,{wm_losses}" alt="WM losses comparison">
  <div class="card-grid grid-3" style="margin-top: 10px;">
    <div class="card">
      <h4 style="color: var(--accent2);">L1: Heavy overfit</h4>
      <p>Val dyn 40.2 vs train 5.9. Single house &rarr; world model memorizes.</p>
    </div>
    <div class="card">
      <h4 style="color: var(--accent-amber);">L2: Similar overfit</h4>
      <p>Val dyn 35.0 vs train 6.7. Same house, different goals &mdash; still memorizes.</p>
    </div>
    <div class="card">
      <h4 style="color: var(--accent-green);">L3: Less overfit</h4>
      <p>Val dyn 27.5 vs train 6.9. 10 houses force generalization &mdash; lower gap.</p>
    </div>
  </div>
</div>

<!-- Slide 7: Findings & Next -->
<div class="slide">
  <h3>Takeaways</h3>
  <h2>What We Learned</h2>
  <div class="card-grid grid-2">
    <div>
      <div class="highlight-box">
        <h4 style="color: #fff; margin-bottom: 8px;">Findings</h4>
        <ul style="list-style: none; padding: 0;">
          <li style="padding: 4px 0; color: var(--text-dim); padding-left: 18px; position: relative;"><span style="position:absolute;left:0;color:var(--accent-green);">1.</span> Buffer fix lifts L1 from 67% &rarr; 75% SR</li>
          <li style="padding: 4px 0; color: var(--text-dim); padding-left: 18px; position: relative;"><span style="position:absolute;left:0;color:var(--accent-green);">2.</span> Scaling goals OR scenes costs ~40pp SR vs L1</li>
          <li style="padding: 4px 0; color: var(--text-dim); padding-left: 18px; position: relative;"><span style="position:absolute;left:0;color:var(--accent-green);">3.</span> Navigation complexity (Geo/Euc) &mdash; not distance &mdash; drives goal difficulty</li>
          <li style="padding: 4px 0; color: var(--text-dim); padding-left: 18px; position: relative;"><span style="position:absolute;left:0;color:var(--accent-green);">4.</span> More scenes &rarr; less overfitting (L3 val loss lower than L1)</li>
          <li style="padding: 4px 0; color: var(--text-dim); padding-left: 18px; position: relative;"><span style="position:absolute;left:0;color:var(--accent-green);">5.</span> Semantic floor plan confirms: object accessibility = SR</li>
        </ul>
      </div>
    </div>
    <div>
      <div class="highlight-box amber">
        <h4 style="color: #fff; margin-bottom: 8px;">Next Steps</h4>
        <ul style="list-style: none; padding: 0;">
          <li style="padding: 4px 0; color: var(--text-dim); padding-left: 18px; position: relative;"><span style="position:absolute;left:0;color:var(--accent-amber);">&rarr;</span> Add <strong>goal conditioning</strong> &mdash; the agent currently can't distinguish which object to find</li>
          <li style="padding: 4px 0; color: var(--text-dim); padding-left: 18px; position: relative;"><span style="position:absolute;left:0;color:var(--accent-amber);">&rarr;</span> <strong>L4: 10 houses + 6 goals</strong> &mdash; both axes scaled simultaneously</li>
          <li style="padding: 4px 0; color: var(--text-dim); padding-left: 18px; position: relative;"><span style="position:absolute;left:0;color:var(--accent-amber);">&rarr;</span> Inject <strong>3D features (UNITE/VGGT)</strong> &mdash; test if 3D scene understanding helps navigate complex layouts</li>
          <li style="padding: 4px 0; color: var(--text-dim); padding-left: 18px; position: relative;"><span style="position:absolute;left:0;color:var(--accent-amber);">&rarr;</span> Scale training budget beyond 2.4M steps</li>
        </ul>
      </div>
      <div class="highlight-box red" style="margin-top: 10px;">
        <p style="font-size: 0.9rem; color: var(--text-dim);">
          <strong style="color: #fff;">Research question:</strong>
          Will 3D semantic features (UNITE) help the world model navigate complex layouts
          that defeat the 2D-only agent &mdash; especially high Geo/Euc goals like toilet and tv_monitor?
        </p>
      </div>
    </div>
  </div>
</div>

</div><!-- end slide-container -->

<div class="slide-number" id="slideNum">1 / 7</div>
<div class="nav-hint">&larr; &rarr; arrow keys to navigate</div>

<script>
  const slides = document.querySelectorAll('.slide');
  let current = 0;
  const total = slides.length;
  function showSlide(n) {{
    slides[current].classList.remove('active');
    current = Math.max(0, Math.min(n, total - 1));
    slides[current].classList.add('active');
    document.getElementById('slideNum').textContent = `${{current + 1}} / ${{total}}`;
    document.getElementById('progress').style.width = `${{((current + 1) / total) * 100}}%`;
  }}
  document.addEventListener('keydown', (e) => {{
    if (e.key === 'ArrowRight' || e.key === ' ') {{ e.preventDefault(); showSlide(current + 1); }}
    if (e.key === 'ArrowLeft') {{ e.preventDefault(); showSlide(current - 1); }}
    if (e.key === 'Home') showSlide(0);
    if (e.key === 'End') showSlide(total - 1);
  }});
  let touchStartX = 0;
  document.addEventListener('touchstart', (e) => {{ touchStartX = e.touches[0].clientX; }});
  document.addEventListener('touchend', (e) => {{
    const diff = touchStartX - e.changedTouches[0].clientX;
    if (Math.abs(diff) > 50) {{ diff > 0 ? showSlide(current + 1) : showSlide(current - 1); }}
  }});
  showSlide(0);
</script>

</body>
</html>'''

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        f.write(html)
    print(f"Saved: {OUTPUT} ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
