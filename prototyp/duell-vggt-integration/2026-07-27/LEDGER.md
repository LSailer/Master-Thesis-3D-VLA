# Duell-Ledger 2026-07-27 (Agent)

Gepflegt vom Orchestrator. Eine Zeile pro Versuch, sofort nach der Auswertung.
Grundlage fuer die Konsolidierung mit Luca nach Ablauf der zwei Stunden.

## Rahmen

- Start (Wall-Clock):
- Ende (Wall-Clock):
- Eingefrorener Seed: **42**
- `verify.sh` beim Start: PASS / FAIL

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
| 1 | | | | | | | | | | |

Verdikt-Vokabular: `besser` / `schlechter` / `neutral` / `gescheitert`
(gescheitert = Job kam nicht durch, keine verwertbare Zahl).

## Erkenntnisse

Was hat gewirkt, was nicht, und warum. Keine Spekulation ohne Zahl.

## Sackgassen

Was nicht noch einmal probiert werden sollte, mit Begruendung.

## Offene Faeden

Was mit mehr Zeit als naechstes dran waere.

## Geoeffnete Pull Requests

| PR | Branch | SR | Delta | Status |
|---|---|---|---|---|
| | | | | |
