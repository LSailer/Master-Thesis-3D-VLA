# Duell-Ledger 2026-07-27 (Agent)

Gepflegt vom Orchestrator. Eine Zeile pro Versuch, sofort nach der Auswertung.
Grundlage fuer die Konsolidierung mit Luca nach Ablauf der zwei Stunden.

## Rahmen

- Start (Wall-Clock): 2026-07-27 ~11:05 (erster Tool-Call)
- Ende (Wall-Clock): spaetestens 13:05
- Eingefrorener Seed: **42**
- `verify.sh` beim Start: PASS (11:10)

## Baseline-SR-Kurve bei kleinen Steps (abgelesen aus run-6056750/metrics.csv, 11:10)

- `metrics/sr` ist **0.0 bis Step 14042**. Erster Erfolg bei 14042 (SR 0.0345),
  danach abklingend: 0.025 bei 19542, 0.02 am Ende (59322).
- Konsequenz: Endet der eigene Lauf bei N < 14042, gewinnt jeder Lauf mit
  mindestens einem Erfolg im SR-Fenster. Bei N zwischen 14k und 20k liegt die
  Latte bei ~0.025-0.034.

## Baseline (CNN, von Luca vor dem Duell gelaufen)

- Run-Dir: `output/runs/r2dreamer-curriculum-l3/run-6056750`
- SLURM 6056750, Node `uc3n105`, Partition `gpu_h100_short`, Seed 42
- 30 min Walltime, `--prod`, endete als TIMEOUT (erwartet)

| Groesse | Wert |
|---|---|
| erreichte Steps | 59 322 |
| `metrics/sr` | 0.02 |
| `metrics/sr_mean` | 0.0167 |
| `metrics/spl` | 0.0083 |
| Episoden | 120 |
| `episode/steps` (letzte) | 500 (Cap) |
| ms/Step (gerechnet) | 30.3 |
| GPU-Speicher (Peak) | 75 372 von 95 830 MiB |

**Zur Einordnung:** der Random-Agent erreicht auf L3 3.84 % SR. Die Baseline
liegt nach 30 Minuten mit 2 % darunter. Bei der Step-Zahl, die ein 3D-Arm in
derselben Zeit schafft (Groessenordnung 10 000), ist die Baseline-SR
voraussichtlich 0. Die Zahl misst in diesem Durchlauf also eher, ob ein Lauf
ueberhaupt durchkommt, als welche Integration besser ist.

**Die Baseline ist gueltig.** In der `metrics.csv` fehlen zwar saemtliche
Trainingsmetriken (`total_loss`, `loss/*`, `perf/*`), aber der Lauf hat
trainiert: rund 29 660 Gradientenschritte bei 59 322 Env-Steps. Es ist ein
Logging-Bug, kein Trainingsausfall (siehe `PROBLEMS.md`).

## Versuche

| # | Hypothese | Aenderung | SLURM | Steps N | SR | Baseline-SR @ N | Delta | ms/Step | ep/steps | Verdikt |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Prefill halbiert = mehr Training im Fenster | l3_hybrid + prefill 2048 | 6057639 | 8001 | 0.0 | 0.0 (@7999) | 0 | 228.6 wall / ~149 steady | 500.0 (16 Ep.) | neutral |
| 2 | dito, pooled Arm | l3_aggregator_pooled + prefill 2048 | 6057641 | 9001 | **0.0556** | 0.0 (@8999) | **+5.56 pp** | 203.4 wall / 134.1 steady | 494.7 (18 Ep.) | **besser** |
| 3 | Lotterie-Knobs auf hybrid | + prefill 1024, train_ratio 256, act_entropy 0.1 | 6057877 | 9503 | 0.0 | 0.0 (@9499) | 0 | 191.4 wall / ~137 steady | ~500 (19 Ep.) | neutral |
| 4 | KV-Budget 200k + Lotterie-Knobs auf pooled | adapter aggregator_pooled_b200k (total_budget 200k) + Knobs aus #3 | 6057871 | 19675 | 0.025 | 0.025 (@19542) | ~0 | 92.6 wall / ~71.5 steady | ~493 (40 Ep.) | neutral |

Quellen: prototyp/duell-vggt-integration/2026-07-27/runs/<jobid>-<arm>/metrics.csv,
sacct Elapsed (6057641: 30:31). "wall" = Elapsed_s/N*1000, "steady" = letzte
perf/ms_per_step_interval-Logs. Baseline-SR aus run-6056750/metrics.csv am
naechstgelegenen geloggten Step.

Verdikt-Vokabular: `besser` / `schlechter` / `neutral` / `gescheitert`
(gescheitert = Job kam nicht durch, keine verwertbare Zahl).

## Erkenntnisse

Was hat gewirkt, was nicht, und warum. Keine Spekulation ohne Zahl.

1. **KV-Cache-Budget 200k ist der staerkste ms/Step-Hebel des Tages.**
   Steady-State 133-138 -> ~71.5 ms/Step (Lauf #2 vs #4, beide pooled),
   N 9001 -> 19675 im selben 30-min-Fenster. Jianyuans Schaetzung (-20 bis
   -30 ms) wurde deutlich uebertroffen (~-63 ms). Der Default 1.2M saettigt
   nach ~36 Frames und zahlt dann jeden Step einen vollen top_k-Sort ueber
   ~50k Kandidaten pro Block (feature_extractor.py:55,365-370).
2. **Mehr Steps haben die SR nicht gehoben.** #4 hatte 2.2x so viele Steps
   und Episoden wie #2, aber genauso genau 1 Erfolg (1/40 statt 1/18).
   Danijars Lotterie-Rahmung haelt: bei diesem Budget ist metrics/sr
   Rauschen ueber der Episodenzahl, kein Lernsignal.
3. **Das Messfenster entscheidet, nicht die Lernkurve.** #2 endete bei
   N=9001, wo die Baseline noch 0.0 hat -> +5.56 pp. #4 endete bei
   N=19675, wo die Baseline schon ihren Zufallserfolg im Fenster hat
   (0.025) -> Delta ~0 trotz identischer Erfolgszahl. Ein schnellerer Arm
   hebt die Latte gegen sich selbst.
4. **Beide pooled-Laeufe erzielten Erfolge, beide hybrid-Laeufe keine**
   (0 Erfolge in 35 Episoden). Bei n=2 vs n=2 nicht signifikant, aber als
   Beobachtung notiert.
5. **Der Trainingsmetriken-Logging-Bug ist in allen vier Duell-Laeufen
   faktisch umgangen**: gerade prefill-Werte (2048/1024) verschieben die
   Update-Paritaet auf gerade Steps, damit treffen sich Update und
   log_every 250 wieder; perf/* und total_loss stehen in den metrics.csv.

## Sackgassen

- **VGGT-Eingangsaufloesung senken**: Backbone wirft bei fremden Shapes,
  RoPE-Tabellen, Pooler und Replay-Shapes haengen daran (backbone.py:96-147).
  Halber Tag Arbeit, kein Duell-Hebel. (Jianyuan, agents/jianyuan-wang/NOTES.md)
- **prefill 0**: PERSIST_SCENE-Bug dokumentiert in loops.py:353-361, nicht
  anfassen. (Danijar)
- **Reward-Shaping**: bei 0-1 Erfolgen im Replay hat success_bonus keinen
  Gradientenbeitrag, step_penalty kuerzt der normierte Advantage weg.
  (Danijar, nicht empirisch geprueft)

## Offene Faeden

- kv200k auf den hybrid-Arm uebertragen (dort ungeprueft; wenn die
  Feature-Qualitaet haelt, waere hybrid bei ~90 ms/Step).
- Aggregator depth=12 (Schaetzung -30-40%, Feature-Qualitaet vorher per
  Punktwolken-Render pruefen; total_budget muss mitskaliert werden).
- Lotterie-Rauschen quantifizieren: dieselbe Config mit 3-5 Seeds ausserhalb
  der Duell-Wertung (Seed ist im Duell auf 42 eingefroren).
- steps-Cap knapp unter 14042 fuer schnelle Arme als Messfenster-Taktik -
  im Duell bewusst nicht gemacht (riecht nach Metrik-Gaming), fuer die
  Konsolidierung mit Luca als Diskussionspunkt notiert.
- Ob #2 (5.56%) echtes Anlernen oder Lotterie war: total_loss faellt
  (metrics.csv), aber 1 Erfolg traegt keine Aussage.

## Geoeffnete Pull Requests

| PR | Branch | SR | Delta | Status |
|---|---|---|---|---|
| | | | | |
