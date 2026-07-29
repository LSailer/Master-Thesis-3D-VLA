# Duell-3-Ledger 2026-07-29-r3 (Agent)

Gepflegt vom Orchestrator. Eine Zeile pro Versuch, sofort nach der Auswertung.

## Rahmen

- Start (Wall-Clock): 2026-07-29 06:10 UTC (erster Tool-Call)
- Deadline: 09:10 UTC. Letzte Welle spaetestens 07:55 UTC (T+1:45).
- Eingefrorener Seed: 42 (Bestaetigung des Fuehrenden: SEED=43 per CLI)
- verify.sh beim Start: PASS (06:10 UTC); nach den Welle-1-YAMLs: PASS
  (06:22 UTC, alle vier neuen Configs SEED=42)
- Branch: duell/2026-07-29-r3-frame-camera-tokens
- CPU-Gate vor Welle 1: tests/adapters 94 passed (06:22 UTC, deckt die vier
  neuen Registry-Eintraege automatisch mit ab)

## Referenz (die Latte, paarweise gewertet)

Duell-2-Sieger C: `aggregator_pooled_b200k` + prefill 1024 + train_ratio 128 +
act_entropy 0.1, 30 min --prod.

| Referenz | Seed | Treffer | sr | spl | softspl | dtg | ms/Step | Ep. | N |
|---|---|---|---|---|---|---|---|---|---|
| 6060404 | 42 | 1 | 0.0227 | 0.0119 | 0.0866 | 6.379 | 66.8 | 44 | 21751 |
| 6061173 | 43 | 1 | 0.0244 | 0.0062 | 0.0539 | 4.975 | 69.1 | 41 | 20267 |

Quellen: `../2026-07-27-r2/runs/6060404-aggpool-b200k-tr128/metrics.csv`,
`../2026-07-27-r2/runs/6061173-aggpool-b200k-tr128-s43/metrics.csv`.

## Wertungsmatrix (aus GOAL.md, hier nur als Gedaechtnisstuetze)

Score = 0.45*rel(Treffer, hoch, Kappung +200%) + 0.15*rel(softspl, hoch)
      + 0.15*rel(dtg, niedrig) + 0.10*rel(spl, hoch) + 0.10*rel(ms/Step, niedrig)
      + 0.05*rel(Episoden, hoch); alle ausser Treffer auf +/-100% gekappt.
Seed 42 gegen 6060404, Seed 43 gegen 6061173.
Ablesung: Treffer = Zeilen mit `episode/success == 1`; Rest = letzter
geloggter Wert. `metrics.csv` ist Langformat `step,metric,value`.

## Welle 1 (Submit 06:23-06:27 UTC, T+0:14; Resubmit 06:33-06:35, s. u.)

| Slot | Config | Adapter / Encoder | Zeile / Kapazitaet | SLURM | Status |
|---|---|---|---|---|---|
| A | duell3_l3_p1_full | P1 - `aggregator_pooled_full` [cam_full, mean, max] = 6144, MLP wie C | 24 KB / 500 000 (12 GB) | 6087073 | pending |
| B | duell3_l3_p2_meanf | P2 - `aggregator_pooled_meanf` [cam_g, mean_g, max_g, mean_f] = 4096, MLP wie C | 16 KB / 500 000 (8 GB) | 6087075 | pending |
| C | duell3_l3_p3_delta | P3 - `aggregator_pooled_full_delta` P1 + (cam_t - cam_0) = 8192, MLP wie C | 32 KB / 500 000 (16 GB) | 6087077 | pending |
| D | duell3_l3_p5_split | P5 - `aggregator_pooled_full_split` 3 Felder a 2048, je MLP-Zweig + Fusion-Dense | 3x8 KB / 500 000 (12 GB) | 6087078 | pending |

Alle vier Configs extends duell2_l3_aggpool_b200k_tr128 (Knobs eingefroren),
Code src/adapters/global_tokens.py, Commit 54d0d9b.

Erst-Submit 6087059/6087060/6087061/6087064 um 06:33 gecancelt und neu
abgesetzt: der Worktree hatte kein uv.lock, und launch.py prod rendert
`uv run python` (mit Sync). Bei gleichzeitigem Start aller vier pending Jobs
haette das uv-sync-Race die geteilte .venv zerlegt (r2, Slot B/6060403).
Fix: uv.lock aus dem Main-Checkout kopiert + launch.py rendert immer
`uv run --no-sync python` (Commit auf dem Duell-Branch). Kein Laufzeitverlust,
die Jobs waren noch pending.

## Welle 2 (submittet <UTC>, T+<h:mm>)

| Slot | Config | Zweck | SLURM | Status |
|---|---|---|---|---|
| E | | Seed-43-Bestaetigung des Fuehrenden | | |
| F | | Kontrolllauf C, Seed 42 (Ziehungsvarianz) | | |
| G | | | | |
| H | | | | |

## Versuche (Scores gegen C)

| # | Config | SLURM | Seed | Treffer | softspl | dtg | spl | ms/Step | Ep. | N | Score | Verdikt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |

Verdikt-Vokabular: `besser` / `schlechter` / `neutral` / `gescheitert`
(gescheitert = Job kam nicht durch, keine verwertbare Zahl).

## Headline: P1 gegen C - bringen Frame-Tokens etwas?

<Score von P1 gegen C, Einzelbeitraege, und die Antwort in einem Satz.>

## Interne Rangliste gegen P1

| Arm | Aenderung gegenueber P1 | Delta zu P1 | Verdikt |
|---|---|---|---|
| | | | |

## Kontrolllauf

<Cs frische Ziehung auf Seed 42 gegen 6060404. Beziffert, wie viel des
beobachteten Abstands blosse Ziehungsvarianz ist. r2-Erwartung: ~+/-0.04.>

## Erkenntnisse

(folgen)

## Sackgassen

(folgen)

## Offene Faeden

(folgen)

## Pull Request

<PR-Nummer und Score, oder: keiner, mit Begruendung an der Schwelle.>
