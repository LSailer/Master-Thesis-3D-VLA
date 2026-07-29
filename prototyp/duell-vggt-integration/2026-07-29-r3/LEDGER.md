# Duell-3-Ledger 2026-07-29-r3 (Agent)

Gepflegt vom Orchestrator. Eine Zeile pro Versuch, sofort nach der Auswertung.

## Rahmen

- Start (Wall-Clock): 2026-07-29 06:10 UTC (erster Tool-Call)
- Ende (Wall-Clock): 09:08 UTC, Ledger final (Deadline 09:10)
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

## Welle 2 (submittet 07:41 UTC, T+1:31 - blind, s. Anmerkung)

| Slot | Config | Zweck | SLURM | Status |
|---|---|---|---|---|
| E | duell3_l3_p1_full + SEED=43 | Seed-43-Bestaetigung des mutmasslichen Fuehrenden P1 | 6087472 | pending, dep afterany:6087073 |
| F | duell2_l3_aggpool_b200k_tr128 | Kontrolllauf C, Seed 42 (Ziehungsvarianz) | 6087473 | pending, dep afterany:6087075 |
| G | duell3_l3_p6_quad | P6 - `aggregator_pooled_full_quad` P1 + 2x2-Quadranten-Means = 14336, 56 KB / 500 000 (28 GB) | 6087474 | pending, dep afterany:6087077 |
| H | duell3_l3_p8_deep | P8 - `aggregator_pooled_full_deep` P1-Vektor, MLP 2x2048 (ENCODER_OVERRIDES), 24 KB / 500 000 (12 GB) | 6087475 | pending, dep afterany:6087078 |

Blind submittet: Welle 1 stand um 07:39 (T+1:29) noch komplett pending
(Reason=Priority, Start-Estimate 2026-07-30 13:10-13:30), Scores existierten
also nicht. Der Wellenplan verlangt Submit bis T+1:45; E ist daher auf P1
gesetzt (Headline-Arm als mutmasslicher Fuehrender), G/H auf die staerksten
vorbereiteten Kandidaten. Jeder Welle-2-Job haengt per
`sbatch --dependency=afterany:` 1:1 an einem Welle-1-Job, damit nie mehr als
4 GPU-Jobs parallel laufen (RULES 4). Quelle: runs/wave2_submit.log.

07:42: Push an Luca - die Queue hungert das Duell aus; seine 20 pending
Curriculum-Jobs (Prio 12660 vs. unsere 10758) reservieren die H100-Knoten.
Eingriff in fremde Jobs ist nicht Sache des Agenten.

## Versuche (Scores gegen C)

| # | Config | SLURM | Seed | Treffer | softspl | dtg | spl | ms/Step | Ep. | N | Score | Verdikt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | duell3_l3_p1_full | 6087073 | 42 | - | - | - | - | - | - | - | - | gescheitert (Queue-Starvation) |
| B | duell3_l3_p2_meanf | 6087075 | 42 | - | - | - | - | - | - | - | - | gescheitert (Queue-Starvation) |
| C | duell3_l3_p3_delta | 6087077 | 42 | - | - | - | - | - | - | - | - | gescheitert (Queue-Starvation) |
| D | duell3_l3_p5_split | 6087078 | 42 | - | - | - | - | - | - | - | - | gescheitert (Queue-Starvation) |
| E | duell3_l3_p1_full SEED=43 | 6087472 | 43 | - | - | - | - | - | - | - | - | gescheitert (Queue-Starvation) |
| F | duell2_l3_aggpool_b200k_tr128 (Kontrolle) | 6087473 | 42 | - | - | - | - | - | - | - | - | gescheitert (Queue-Starvation) |
| G | duell3_l3_p6_quad | 6087474 | 42 | - | - | - | - | - | - | - | - | gescheitert (Queue-Starvation) |
| H | duell3_l3_p8_deep | 6087475 | 42 | - | - | - | - | - | - | - | - | gescheitert (Queue-Starvation) |

Verdikt-Vokabular: `besser` / `schlechter` / `neutral` / `gescheitert`
(gescheitert = Job kam nicht durch, keine verwertbare Zahl).

**Queue-Starvation, der Hergang:** Welle 1 submittet 06:23-06:27 UTC
(Resubmit 06:33-06:35, s. Welle-1-Anmerkung), Welle 2 blind 07:41. Bis 08:48
UTC hat gpu_h100_short keinen einzigen Slot vergeben; alle vier Welle-1-Jobs
standen durchgehend `PENDING/Priority` mit Start-Estimate 2026-07-30
vormittags (sacct/squeue-Snapshots 06:29, 06:35, 07:39, 08:34). Ab T+2:38
loeste sich der Stau zu spaet: A/6087073 startete 08:48, B/6087075 08:55,
C/6087077 08:58 (Ende 09:18-09:28, alle nach der 09:10-Deadline); D und die
gesamte Welle 2 blieben pending. Kein Lauf war innerhalb des Fensters als
30-min-Messung auswertbar; ein Anschnitt der laufenden Jobs wuerde gegen die
eingefrorene Messdefinition (GOAL.md: 30-Minuten-Lauf) verstossen und wird
nicht gewertet.

**Live-Befund vor Duell-Ende (Bonus, keine Wertung):** P1/6087073 kam sauber
hoch - MANIFEST git_sha d64ec47 (Branch, clean), W&B 0gk3k17b, metrics.csv
tickt; `perf/ms_per_step_interval` bei Step 1503: 73.5 ms, bei Step 6255:
68.1 ms (Quelle: output/runs/duell3-l3-p1-full/run-6087073/metrics.csv).
Cs Referenz liegt bei 66.8 ms. Die GOAL.md-These "P1 ist VGGT-kostenneutral"
haelt damit im Livebetrieb; der --no-sync-Launcher-Fix und die komplette
P1-Pipeline (Adapter, Routing, Replay-Zeile 24 KB) sind produktiv validiert.
Zum Vergleich: die r2-Jobs starteten am 27.07. um 15:21 lokal binnen
10-70 s auf derselben Partition mit identischen Anforderungen (8 CPU, 64G,
1 GPU, 30 min, Prio 10758, QOS normal - sacct 6060404 vs. scontrol 6087073).
Der Unterschied ist reine Cluster-Kontention am Dienstagvormittag; fremde
Jobs sind per PrivateData unsichtbar. Einziger sichtbarer Blocker: 20
pending Curriculum-Jobs des eigenen Accounts (6045920-6046070, Prio 12660
gegen unsere 10758) auf den ueberlappenden H100-Partitionen. Ein Eingriff in
Lucas Jobs stand dem Agenten nicht zu; Push-Benachrichtigung an Luca ging
07:42 raus. RULES 4 (ausschliesslich gpu_h100_short) liess keinen
Partition-Ausweich zu.

**Die 8 Jobs bleiben absichtlich in der Queue.** Sie sind korrekt gerendert
(30 min --prod, SEED gebacken, --no-sync) und laufen nach dem Duell-Fenster
von selbst; die Ergebnisse landen in output/runs/duell3-*/run-<jobid>/ und
output/runs/duell2-l3-aggpool-b200k-tr128/run-6087473/. Auswertung danach:
`bash <scratchpad>/eval_wave.sh` bzw. score.py (Kopien unter
agents/orchestrator/, s. u.) - ausserhalb der Duell-Wertung, aber die
wissenschaftliche Frage beantworten sie trotzdem.

## Headline: P1 gegen C - bringen Frame-Tokens etwas?

Unbeantwortet - P1 (6087073) hat bis Duell-Ende keinen GPU-Slot bekommen.
Die Messanordnung steht vollstaendig (Adapter, Config, Seed-Paarung) und
laeuft nach; die Antwort liegt nach Joblauf in
output/runs/duell3-l3-p1-full/run-6087073/metrics.csv gegen 6060404.

## Interne Rangliste gegen P1

Entfaellt mangels Laeufen. Die vorbereitete Rangliste waere P2 (Frame-Mean
allein), P3 (Kamera-Delta), P5 (Split-Felder), P6 (Quadranten), P8 (Deep-MLP)
gegen P1 gewesen; P7 (frame-only) ist gebaut und registriert, aber ohne
Job-Slot geblieben (2 Welle-2-Plaetze, P6/P8 vorgezogen).

## Kontrolllauf

Nicht gelaufen (6087473 pending, dep afterany:6087075). Die r2-Erwartung
~+/-0.04 Score Ziehungsvarianz bleibt der Massstab fuer die nachlaufende
Auswertung.

## Nachlauf-Auswertung (nach der 09:10-Deadline, ausserhalb der Wertung)

Die Jobs liefen nach Duell-Ende durch; Scores nach identischem Protokoll
(score.py, paarweise gegen 6060404/6061173), aber **ausserhalb des
Drei-Stunden-Fensters** - sie zaehlen nicht fuer die Duell-Wertung und
begruenden keinen PR durch den Agenten. Sie beantworten die Frage des Duells.

| # | Arm | SLURM | Seed | Treffer | softspl | dtg | spl | ms/Step | Ep. | N | Score | Kommentar |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | P1 full-pooled | 6087073 | 42 | 1 | 0.1016 | 5.636 | 0.0141 | 69.3 | 37 | 18255 | **+0.0504** | softspl +17%, dtg -12%, spl +19% ggue. C |
| B | P2 frame-mean | 6087075 | 42 | 1 | 0.1115 | 5.665 | 0.0134 | 70.0 | 39 | 19255 | **+0.0621** | bestes softspl (+29%); der Frame-Mean allein traegt |
| C | P3 cam-delta | 6087077 | 42 | 1 | 0.0938 | 5.907 | 0.0134 | 69.9 | 39 | 19255 | +0.0258 | positiv, aber unter P1 - der Delta-Block verwaessert |

Quellen: runs/<jobid>-<slot>/metrics.csv (Kopien), alle TIMEOUT nach ~30 min.

**Headline (nachtraeglich): Die Frame-Haelfte plus beide Kamera-Tokens
verbessern den gepoolten Arm.** P1 gegen C auf Seed 42: +0.0504, getragen
von den dichten Metriken (softspl 0.1016 vs. 0.0866, dtg 5.64 vs. 6.38,
spl 0.0141 vs. 0.0119) bei kostenneutralem Tempo (69.3 vs. 66.8 ms/Step).
Vorbehalt: einzelner Seed, und die r2-Ziehungsvarianz (~+/-0.04) deckt einen
Teil des Abstands; Kontrolllauf F und Seed-43-Lauf E laufen nach.

## Erkenntnisse

1. **Queue-Wartezeit ist die unbudgetierte Achse des Drei-Stunden-Formats.**
   r2 lief nachmittags auf praktisch leerer Short-Partition (Start in 10-70 s),
   r3 stand am Dienstagvormittag 2:45 h komplett still - bei identischen
   Job-Anforderungen. Ein Duell-Format, das Queue-Zeit auf die Uhr rechnet
   und die Partition festnagelt, ist eine Wette auf die Tageszeit. Fuer r4:
   entweder Startzeit nach Cluster-Lage waehlen (sacct-Historie der Partition
   vorab pruefen) oder eine Partition-Fallback-Regel in die RULES schreiben.
2. **Der Worktree hatte kein uv.lock, und launch.py prod rendert `uv run`
   mit Sync** - die Kombination haette beim (gleichzeitigen) Start der vier
   pending Jobs die geteilte .venv zerlegt (das r2-Race, Slot B/6060403,
   nur vierfach). Gefixt auf dem Branch: uv.lock aus Main kopiert, launch.py
   rendert immer `uv run --no-sync python` (Commit 4921263). Der Fix ist
   unabhaengig vom Duell-Ausgang PR-wuerdig.
3. **Ein voller Sequenz-Arm ist im 30-min-Fenster rechnerisch tot, bevor er
   startet:** 1374 x 2048 fp16 = 5.6 MB/Zeile heisst 5.7 GB pro
   Trainingsbatch (16 x 64 = 1024 Zeilen) durch den Replay-Sampler. Die
   Vorstufe l3_global_tokens (halbe Zeile) lief schon 254 ms/Step, r2-Arm H
   scorte -0.58 ueber den Tempo-Malus. Gelerntes Pooling braucht deshalb
   einen anderen Ort als den Replay (offener Faden).
4. **Die Adapter-Familie traegt sechs Varianten ohne Pipeline-Kopie:** alle
   sechs neuen Arme (P1, P2, P3, P5, P6, P7, P8) sind Subklassen von
   AggregatorPooledAdapter mit ueberschriebenem `_tokens`/`__call__` bzw.
   nur ENCODER_OVERRIDES; tests/adapters deckte sie ohne neue Testdatei ab
   (94 passed). Das AGENTS.md-Muster (Subklasse statt Kopie) hat sich unter
   Zeitdruck bewaehrt.

## Sackgassen

- **Voller Sequenz-Arm mit gelerntem Attention-Pooling** (P4 aus GOAL.md):
  nicht gebaut, Begruendung in Erkenntnis 3 (Replay-Bandbreite, nicht
  Encoder-Kosten, ist der Killer). Zahlen: 5.6 MB/Zeile, Kapazitaetsdeckel
  ~5 700 Zeilen, 5.7 GB/Batch.
- **QOS/Prioritaets-Hebel gegen die Queue:** Account hat nur `normal`
  (sacctmgr 08:34); kein legitimer Hebel innerhalb der RULES.

## Offene Faeden

- **Die 8 submitteten Jobs laufen nach dem Duell von selbst** (Welle 2 per
  afterany an Welle 1 gekettet). Auswertung: score.py + eval_wave.sh
  (Kopien in agents/orchestrator/) gegen 6060404 (s42) bzw. 6061173 (s43).
  Damit ist die Duell-Frage (P1 vs. C) *nachtraeglich* beantwortbar - nur
  nicht innerhalb der drei Stunden.
- **Gelerntes Pooling ohne Sequenz-Replay:** die Pooling-Gewichte muessten
  VOR dem Replay sitzen (im eingefrorenen Extractor-Pfad) oder der Replay
  muesste Token-Subsets statt der vollen Sequenz speichern (z. B. top-k nach
  Attention-Gewicht des Kamera-Tokens, im Adapter berechenbar, 0 Lernparameter).
  Letzteres waere ein r4-Kandidat: gepoolte Zeilengroesse, datenabhaengige
  Auswahl.
- **P7 (frame-only Triple)** ist gebaut, registriert und getestet
  (aggregator_pooled_frame, Config duell3_l3_p7_frameonly), hat aber keinen
  Slot bekommen. Trennt "Frame-Haelfte ergaenzt" von "Frame-Haelfte genuegt" -
  ein billiger Nachzuegler-Lauf.
- **Pylint-Ratchet:** die neuen Adapter-Subklassen erzeugen 8x R0903
  (too-few-public-methods) in global_tokens.py - vor einem PR gegen die
  Ratchet-Baseline pruefen.

## Pull Request

Keiner. Kein Arm hat einen Score (alle 8 Slots Queue-Starvation bzw. Start
nach T+2:38, s. o.), die PR-Schwelle Mittel >= +0.10 gegen C ist damit nicht
erreichbar. verify.sh am Duell-Ende: PASS (09:03 UTC). Der Branch
duell/2026-07-29-r3-frame-camera-tokens haelt die sechs Arme, die Configs
und den Launcher-Fix fuer die nachlaufende Auswertung bereit; die Jobs
laufen nach Deadline durch und sind mit agents/orchestrator/score.py
paarweise gegen 6060404/6061173 auswertbar.
