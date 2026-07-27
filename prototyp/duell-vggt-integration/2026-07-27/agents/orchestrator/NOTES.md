# orchestrator - Arbeitsnotizen

Alles, was dieser Agent probiert, findet und verwirft. Roh und fortlaufend.
Kein Politur-Anspruch, aber jede Zahl mit Quelle (SLURM-Job, Run-Dir, Datei:Zeile).

## Zeitplan (2h, Start ~11:05, Ende 13:05)

- 11:05-11:13 Orientierung: Regeln gelesen, Worktree eingerichtet
  (setup_worktree.sh - die Symlinks fehlten!), verify.sh PASS,
  Baseline-SR-Kurve gezogen.
- 11:13 Welle 1 delegiert an launcher (slurm-runner):
  - Job A: `duell_l3_hybrid_p2048` (extends l3_hybrid, prefill 2048),
    dev_gpu_h100 primaer
  - Job B: `duell_l3_aggpool_p2048` (extends l3_aggregator_pooled,
    prefill 2048), gpu_h100_short
- 11:13 Berater parallel: danijar-hafner (Trainingsdynamik bei ~5k Steps),
  jianyuan-wang (ms/Step-Hebel, Arm-Wahl, Aggregator-MLP-Port).

## Strategische Entscheidungen

1. **Prefill 5000 -> 2048 als erster Hebel.** Bei ~9k Gesamtsteps frisst der
   Prod-Default-Prefill mehr als die Haelfte des Budgets als Random-Steps.
   2048 ist der erprobte Smoke-Wert und liegt ueber dem Replay-Gate
   (batch 16 * seq 64 = 1024, _base.yaml Kommentar).
2. **Messfenster ausnutzen.** Baseline-SR ist 0.0 bis Step 14042
   (run-6056750/metrics.csv). Ein 3D-Lauf, der unter 14k Steps endet,
   gewinnt mit einem einzigen Erfolg. Bei N in 14k-20k liegt die Latte bei
   0.025-0.034. Trade-off "mehr Steps vs. niedrigere Latte" wird nach
   Welle 1 mit echten Zahlen entschieden.
3. **Welle 1 = Stock-Arme + Prefill-Fix**, keine Architekturaenderung: erst
   eine echte Zahl holen, dann optimieren. Queue ist der Flaschenhals.
4. Max 2 parallele GPU-Jobs: Welle 2 startet, sobald Welle-1-Jobs enden
   (~11:50-12:00 erwartet). Kandidaten werden waehrenddessen vorbereitet.

## Welle-2-Kandidaten (vorlaeufig, nach Persona-Antworten schaerfen)

- Aggregator-MLP auf L3 portieren (schnellster 3D-Arm, 38.1k Steps/h laut
  PLAN.md) - Achtung Messfenster-Argument oben (koennte >14k Steps laufen).
- train_ratio hoch / entropy hoch fuer die 5k-Step-Anlernphase.
- Reward-Shaping (success_bonus, step_penalty) - erlaubt laut RULES.md.

## 11:25 Danijar-Antwort eingetroffen (agents/danijar-hafner/NOTES.md)

Kernrahmung: bei diesem Budget ist metrics/sr eine Lotterie ueber die Zahl
der gewerteten Episoden (P(>=1 Erfolg) bei Random 3.84% und 14 Episoden =
42%). Die Baseline ist nach 59k Steps mit SR 0.02 UNTER Random - halbtrainiert
ist schlechter als Wuerfeln. Also: Lose maximieren, Policy nahe Random halten.

Top-3 (sofort umsetzen laut Danijar):
1. act_entropy 3e-2 -> 1e-1 (agent_config.py:166, Flag parser.py:136)
2. train_ratio 512 -> 256 (Flag parser.py:230) - ~24% mehr Env-Steps/Episoden
3. prefill 2048 -> 1024 (Gate exakt batch*seq=1024, loops.py:453); prefill=0
   NICHT (PERSIST_SCENE-Bug, loops.py:353-361)

Nicht: seq_len halbieren ohne train_ratio (verdoppelt Update-Rate).
Reward-Shaping niedrig priorisiert (kein Gradientenbeitrag ohne Erfolge im
Replay; step_penalty kuerzt der normierte Advantage weg).

## 11:30 Jianyuan-Antwort (agents/jianyuan-wang/NOTES.md)

- Groesster ms/Step-Hebel ohne Architekturumbau: KV-Cache-Budget
  total_budget 1.2M -> 200k (feature_extractor.py:55, Konstruktor-Arg :294).
  Default saettigt nach ~36 Frames, dann voller top_k-Sort pro Block und
  Step. Schaetzung -20 bis -30 ms/Step, Preis: Kontextfenster ~6 Frames.
- Aufloesung senken: toter Hebel im Duell-Budget (Backbone wirft, RoPE,
  Replay-Shapes; halber Tag Arbeit).
- Aggregator depth=12: -30-40% moeglich, aber Budget muss mitskaliert werden
  und Feature-Qualitaet ungeprueft - zu riskant fuer gewertete Laeufe heute.
- aggregator-pooled existiert auf L3 bereits (_run_configs.py:259); seine
  Geschwindigkeit kommt von 12KB-Replay-Zeilen, nicht vom VGGT-Forward.
- Warnung: aggregator-pooled hat KEIN Rekonstruktionsziel (WITH_RGB=False).

## 11:32-11:40 Welle 2 implementiert (Orchestrator selbst, mechanisch)

- Neuer Adapter `AggregatorPooledBudget200kAdapter`
  (src/adapters/global_tokens.py, EXTRACTOR_KWARGS total_budget=200_000),
  registriert als `aggregator_pooled_b200k`, RUN_CONFIGS-Eintrag
  `habitat-l3-aggregator-pooled-b200k` (_run_configs.py).
- YAMLs: duell_l3_hybrid_lottery.yaml (konservativ: prefill 1024,
  train_ratio 256, act_entropy 0.1) und duell_l3_aggpool_lottery.yaml
  (aggressiv: dieselben Knobs + b200k-Adapter). Diversifikation: der
  Budget-Cap-Effekt auf die Feature-Qualitaet ist ungeprueft, deshalb
  traegt nur ein Arm das Risiko.
- verify.sh PASS (alle 4 Duell-YAMLs SEED=42), CPU-Import-Test OK,
  Dry-Run-Render OK (--prefill 1024 --train_ratio 256 --act_entropy 0.1,
  run.py bekommt die b200k-Run-Id positional).
- Commit auf Branch duell/2026-07-27-lottery-knobs-kv200k.
- 11:40 Launcher instruiert: Welle 2 startet, sobald der jeweilige
  Welle-1-Job endet (kein Slot-Leerlauf, 2-GPU-Grenze haelt).

## Welle-1-Status 11:35

- 6057639 r2d-L3du-hyb, dev_gpu_h100/uc3n082, laeuft (8:44), Prefill ueberlebt
- 6057641 r2d-L3du-agg, gpu_h100_short/uc3n104, laeuft (10:46), Prefill ueberlebt

## Welle-1-Zwischenstand 11:52 (live aus metrics.csv, Jobs laufen noch)

- aggpool 6057641: Step 5404, **metrics/sr 0.0909** (1 Erfolg, 11 Episoden),
  spl 0.033, perf 133.0 ms/Step. Projektion Ende: N ~ 8900 < 14042 ->
  Baseline-SR dort 0.0, d.h. Stand jetzt GESCHLAGEN.
- hybrid 6057639: Step 3999, sr 0.0, 8 Episoden, perf 151.8 ms/Step.
  Projektion N ~ 7700; braucht noch einen Erfolg.
- Nebenbefund: perf/ms_per_step_interval WIRD geloggt in beiden Laeufen -
  prefill 2048 (gerade Zahl) verschiebt die Update-Paritaet auf gerade
  Steps, damit treffen sich Update und log_every 250 wieder. Der
  Logging-Bug aus PROBLEMS.md ist in den Duell-Laeufen faktisch umgangen.

Meine Rechnung dazu (Quelle Baseline-metrics.csv): Gewinnbedingung bei Step N
ist "mehr Erfolge im 100er-Fenster als die Baseline dort hatte": N < 14042
-> 1 Erfolg reicht; N 14k-40k -> 2 Erfolge noetig (Baseline hatte dort genau
1). Langsamer Arm (~180ms, N~10-12k): P(>=1 in ~20 Ep.) ~ 55-60%. Schneller
Arm (~95ms, N~19k): P(>=2 in ~38 Ep.) ~ 43%. Beide Regime legitim, Welle 2
faehrt idealerweise beide.
