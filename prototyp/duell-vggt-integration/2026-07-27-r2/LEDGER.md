# Duell-2-Ledger 2026-07-27-r2 (Agent)

Gepflegt vom Orchestrator. Eine Zeile pro Versuch, sofort nach der Auswertung.

## Rahmen

- Start (Wall-Clock): 2026-07-27 12:37 UTC (erster Tool-Call, setup_worktree)
- Ende (Wall-Clock): 14:59 UTC, Ledger final (innerhalb der 3h-Deadline 15:37)
- Deadline: 15:37 UTC. Letzte Welle spaetestens 14:22 UTC (T+1:45).
- Eingefrorener Seed: 42 (Bestaetigung des Fuehrenden: SEED=43 per CLI)
- verify.sh beim Start: PASS (12:41), nach Welle-1-YAMLs: PASS (12:50)

## Referenz (die Latte)

SLURM 6057641, aggregator_pooled + prefill 2048, 30 min --prod, Seed 42:
Treffer 1, sr 0.0556, spl 0.0201, softspl 0.0605, dtg 5.193,
134.1 ms/Step (steady), 18 Episoden, N=9001.
Quelle: prototyp/duell-vggt-integration/2026-07-27/runs/6057641-aggpool-p2048/metrics.csv

## Wertungsmatrix (aus GOAL.md, hier nur als Gedaechtnisstuetze)

Score = 0.45*rel(Treffer, hoch, Kappung +200%) + 0.15*rel(softspl, hoch)
      + 0.15*rel(dtg, niedrig) + 0.10*rel(spl, hoch) + 0.10*rel(ms/Step, niedrig)
      + 0.05*rel(Episoden, hoch); alle ausser Treffer auf +/-100% gekappt.
Ablesung: Treffer = Zeilen mit episode/success==1; Rest = letzter geloggter Wert.

## Welle 1 (submittet 13:21 UTC, T+0:44 - Hook-Timeouts des Clients kosteten ~35 min Wall-Clock, siehe agents/orchestrator/NOTES.md)

| Slot | Config | Aenderung ggue. Referenz | SLURM | Status |
|---|---|---|---|---|
| A | duell_l3_aggpool_lottery | kv200k + prefill 1024 + tr256 + ent 0.1 (Duell-1 #4 Rerun) | 6060402 | TIMEOUT (ok) |
| B | duell2_l3_aggpool_b200k_p2048 | nur kv200k, sonst identisch zur Referenz | 6060403 | gescheitert: uv-sync-Race der parallel startenden Jobs zerlegte die shared .venv (slurm-6060403.err), exit 2 nach 31s |
| C | duell2_l3_aggpool_b200k_tr128 | kv200k + prefill 1024 + tr128 + ent 0.1 | 6060404 | TIMEOUT (ok) |
| D | duell2_l3_pointmap_p2048 | Geometrie-Arm pointmap_pose + prefill 2048 | 6060405 | gescheitert: exit 134 (habitat-GL-SIGABRT, PROBLEMS.md) nach 10:04 |

## Finale Welle (submittet 14:12 UTC, T+1:35)

| Slot | Config | Aenderung | SLURM | Status |
|---|---|---|---|---|
| E | duell2_l3_aggpool_b200k_tr128 --env SEED=43 | Seed-43-Bestaetigung des Fuehrenden C | 6061173 | TIMEOUT (ok) |
| F | duell2_l3_b200k_tr128_ent3em4 | C + act_entropy 3e-4 (Danijar: Aktionsverteilung war uniform, 0.1 haelt den Actor auf uniform) | 6061174 | TIMEOUT (ok) |
| G | duell2_l3_b200k_tr128_ent3em3 | C + act_entropy 3e-3 (konservative Klammer) | 6061175 | TIMEOUT (ok) |
| H | duell2_l3_pointmap_p2048 (Retry) | D-Retry nach GL-Abort | 6061176 | TIMEOUT (ok) |

Hypothesist-Kandidaten K1 (b100k-Adapter) und K2 (depth 12) kamen erst 14:13 an -
nach Submit der finalen Welle und bei vollem 4-Job-Limit nicht mehr startbar.
Als offene Faeden vermerkt.

Bewusst nicht gestartet: l3_global_tokens (254 ms/Step laut Config-Kommentar,
in einer 30-min-Matrix mit Speed- und Episodengewicht chancenlos).

## Versuche (Scores)

| # | Config | SLURM | Treffer | softspl | dtg | spl | ms/Step | Ep. | N | Score | Verdikt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | aggpool_lottery (kv200k+tr256+ent0.1) | 6060402 | 1 | 0.0804 | 5.957 | 0.0035 | 72.2 | 39 | 19175 | +0.0407 | besser |
| C | b200k_tr128 (kv200k+tr128+ent0.1) | 6060404 | 1 | 0.0866 | 6.379 | 0.0119 | 66.8 | 44 | 21751 | **+0.0898** | **besser, Fuehrender W1** |
| E | = C, Seed 43 (Bestaetigung) | 6061173 | 1 | 0.0539 | 4.975 | 0.0062 | 69.1 | 41 | 20267 | +0.0194 | besser (s43) |
| F | C + act_entropy 3e-4 | 6061174 | 0 | 0.0711 | 6.251 | 0.0000 | 68.6 | 39 | 19751 | -0.4555 | schlechter |
| G | C + act_entropy 3e-3 | 6061175 | 1 | 0.0958 | 6.488 | 0.0016 | 69.5 | 41 | 20317 | +0.0564 | besser, unbestaetigt |
| H | pointmap_pose + prefill 2048 | 6061176 | 0 | 0.0551 | 5.362 | 0.0000 | 146.2 | 16 | 8001 | -0.5829 | schlechter |

Quellen: output/runs/duell-l3-aggpool-lottery/run-6060402/metrics.csv bzw.
output/runs/duell2-l3-aggpool-b200k-tr128/run-6060404/metrics.csv, Score-Formel
oben, Einzelbeitraege in agents/orchestrator/NOTES.md. Beide Scores > 0, beide
unter PR-Schwelle +0.10. spl-Verduennung bestaetigt (Hypothesist): mehr Episoden
bei gleicher Trefferzahl druecken den Rolling-Mean von spl mechanisch.

## Erkenntnisse

1. **Ziehungsvarianz identischer Configs ist ~+/-0.04 Score.** Slot A (6060402,
   +0.0407) und sein config-identischer Duell-1-Zwilling 6057871 (-0.0040,
   nachgerechnet vom Analyst) unterscheiden sich nur durch die Ziehung. Ein
   einzelner Score unter ~0.05 Abstand ist damit kein belastbares Ranking.
2. **Speed allein ist score-neutral** (Analyst + Hypothesist unabhaengig):
   ms/Step- und Episoden-Gewinn (+0.10 zusammen) werden vom mechanischen
   spl-Rolling-Mean-Einbruch (mehr Episoden, gleiche Trefferzahl) und dtg
   aufgezehrt. Speed zahlt nur ueber mehr Lotterie-Lose (2. Treffer = +0.45).
3. **Kein bisheriger Arm hat eine Aktionspraeferenz gelernt** (Danijar:
   action/forward_pct ~0.25 = uniform in allen vier Duell-1-Laeufen;
   act_entropy 0.1 haelt den Actor bei ln(4)-Entropie fest). Deshalb testet
   die finale Welle das Entropie-Bracket 3e-4 / 3e-3.
4. **Ergebnis des Duells: C (kv200k + prefill 1024 + train_ratio 128 +
   act_entropy 0.1) ist der bestaetigte Beste.** Seed 42 +0.0898, Seed 43
   +0.0194, Mittel +0.0546. Auf beiden Seeds besser als die Referenz, aber
   unter der PR-Schwelle +0.10 -> kein PR (RULES.md Abschnitt 6).
5. **Das Entropie-Bracket hat eine Richtung, aber kein Optimum gefunden:**
   3e-4 kollabiert den Actor (action/forward_pct 0.048, 0 Treffer, -0.4555);
   3e-3 ist gesund und holt das beste softspl des Tages (0.0958, +0.0564,
   fwd_pct 0.236); 0.1 (C/E) bleibt nahe uniform (0.262/0.284), scored aber
   auf beiden Seeds positiv. Das Optimum liegt vermutlich zwischen 3e-3 und
   0.1 - offen.
6. **Der Geometrie-Arm pointmap_pose zahlt im 30-min-Fenster nicht** (H:
   146.2 ms/Step, 0 Treffer, dtg 5.362 nur neutral, -0.5829). Die Hypothese
   "Geometrie-Prior hebt dtg/softspl frueh" ist fuer dieses Fenster widerlegt.
7. **Beide Hybrid-Arme scoren katastrophal** (-0.59 / -0.47, Analyst): spl=0
   ohne Treffer plus Tempo-Malus. Der pooled Arm bleibt die richtige Basis.

## Sackgassen

(folgen)

## Offene Faeden

- K1 (Hypothesist, b100k allein) ist durch Jianyuans Kostenzerlegung WIDERLEGT:
  top_k laeuft gepolstert immer ueber MAX-1374 Slots, Kosten linear im Budget;
  gemessen 1.50e-3 ms/Slot, Cache-Term bei 200k nur 14.6 ms von 71.5, fixer
  Sockel 56.9 ms. 100k braechte ~-6 ms, 50k ~-9 ms; 200k hat schon 83% des
  moeglichen Cache-Gewinns. Haette einen Slot verschwendet.
- K2 praezisiert (Jianyuan): adapter aggregator_pooled_d12b100k mit
  EXTRACTOR_KWARGS compute_heads=False, total_budget=100_000, agg_depth=12.
  uniform = total_budget // agg_depth: depth 12 + 100k = 8333 Slots/Block =
  exakt heutige Cache-Geometrie bei halber Blockzahl, saubere
  Ein-Variablen-Aenderung, est. ~55-60 ms/Step. Drei Edits im Extractor
  (feature_extractor.py:294ff Signatur, :349-354 _configure_runtime_options,
  :358 Aggregator(depth=...)) + Subklasse in src/adapters/global_tokens.py:160.
  Risiko: Checkpoint hat frame_blocks_0-23, depth 12 instanziiert 0-11 -
  vorher lokal konstruieren (_warmup schlaegt im __init__ zu), Fallback
  Dict-Filter auf _agg_params. Vorab benchmark_streaming.py
  --jax-static-budgets fuer die ms/Frame-Kurve (~5 min).
- Pose-Delta-Feature (Jianyuan): Token 0 des pooled Vektors ist der
  Kamera-Token (latent); cam_t - cam_0 als vierter 1024er-Block (3072->4096,
  0 ms VGGT-Kosten, Frame 0 ist permanenter Cache-Anchor) waere die metrische
  Form fuer dtg/softspl. Nie ablatiert, reines Los.
- Danijar: reward_scale-Knob an experience.py:257 (3 Zeilen) statt
  Reward-Wrapper; success_bonus wirkt ueber Return-EMA ~100 Updates lang
  signalunterdrueckend; ret_scale aus return_ema.get_stats loggen.
- Danijar: batch 32 x seq 32 statt 16 x 64 (batch_steps konstant 1024, mehr
  Imagination-Startzustaende) - nicht mehr getestet.
- uv-sync-Race bei parallel startenden Jobs aus demselben Worktree (B,
  6060403): Launcher sollte Jobs um ~1 min staggern oder uv frozen syncen.

## Pull Request

Keiner. C ist auf beiden Seeds besser als die Referenz (Mittel +0.0546),
verfehlt aber die PR-Schwelle von +0.10 (RULES.md Abschnitt 6). G (+0.0564,
Seed 42) liegt ebenfalls unter der Schwelle und ist unbestaetigt.

Quellen aller Score-Zeilen: output/runs/<config>/run-<jobid>/metrics.csv,
kopiert nach runs/<jobid>-<slot>/; Score-Formel siehe Wertungsmatrix oben;
Einzelbeitraege in agents/orchestrator/NOTES.md.
