# Duell-Ledger 2026-07-27 (Agent)

Gepflegt vom Orchestrator. Eine Zeile pro Versuch, sofort nach der Auswertung.
Grundlage fuer die Konsolidierung mit Luca nach Ablauf der zwei Stunden.

## Rahmen

- Start (Wall-Clock): 2026-07-27 08:00:13Z
- Ende (Wall-Clock): 2026-07-27 10:00:13Z
- Eingefrorener Seed: **42**
- `verify.sh` beim Start: PASS (auf dem Cluster inkl. Curriculum-Pruefsumme)
- Branch: `duell/2026-07-27-3d-short-horizon`
- Cluster-Worktree: `~/duell-agent` (nicht `/tmp`, siehe Sackgassen)

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

### Die Baseline-SR-Kurve, Step fuer Step

Das ist die wichtigste Tabelle des Durchlaufs, weil sie die Latte definiert.
Quelle: `run-6056750/metrics.csv`, Spalte `metrics/sr`.

| Step | `metrics/sr` |
|---|---|
| 499 .. 13999 | **0.0**, jede einzelne der 28 geloggten Zeilen |
| 14042 | 0.03448 = 1/29 |
| 14542 .. 20542 | faellt monoton 1/30, 1/31, ... 1/42 |
| 56042 .. 58042 | 0.01 |
| 58322 .. 59322 | 0.02 |

Ein VGGT-Arm laeuft mit 171-254 ms/Step und erreicht in 30 Minuten
N = 7000-11000. **Dort steht die Baseline auf exakt 0.0.** Ein einziger
Erfolg im eigenen Lauf entscheidet das Duell. Umgekehrt gilt: haette der
eigene Lauf N > 14042 erreicht, waere die Latte schlagartig auf ~0.034
gesprungen. Das ist eine Eigenschaft des Messprotokolls, kein Verdienst -
gehoert in die Konsolidierung.

### Der eigentliche Befund: der Engpass ist Fortbewegung, nicht Wahrnehmung

Ebenfalls aus `run-6056750/metrics.csv`, und aus meiner Sicht das Wertvollste,
was dieser Durchlauf hergibt:

| Groesse | Anfang (Step 499-1499) | Ende (Step 58322-59322) |
|---|---|---|
| `episode/steps` | 500 (Cap) | 500 (Cap) |
| `episode/path_length` | 2.93 / 5.30 / 1.96 m | 5.85 / 0.21 / 2.54 m |
| `episode/shortest_path` | 8.81 / 4.52 / 6.35 m | 2.92 / 5.83 / 3.45 m |
| `metrics/dtg` (rollend) | 5.36 m | 6.14 m |
| Aktionen | stop 0.264 / forward 0.252 | stop 0.220 / forward 0.256 / left 0.318 / right 0.206 |

Zu lesen als: das Ziel liegt geodaetisch **3 bis 9 Meter** entfernt. Der Agent
legt in **500 Aktionen** eine Pfadlaenge von **2 bis 6 Metern** zurueck. Bei
0.25 m Schrittweite und 25 % Forward-Anteil waeren rein rechnerisch ~31 m
Pfad zu erwarten; gemessen ist ein Bruchteil davon. Der Agent dreht sich also
auf der Stelle und laeuft in Waende. Die rollende Distanz zum Ziel ist am Ende
des Laufs **groesser** als am Anfang: ueber 59 322 Steps kein Netto-Fortschritt.

Die Aktionsverteilung bleibt dabei ueber den gesamten Lauf uniform. Zwei
Erfolge auf 120 Episoden sind 1.67 %, also **unter** den 3.84 % des
Random-Agents (GOAL.md:94).

Daraus folgt fuer dieses Zeitbudget: kein Encoder-Routing kann die SR heben,
solange der Agent sich nicht bewegt. Was zaehlt, ist (a) wie viele Episoden
der Lauf ueberhaupt sieht und (b) ob die Policy sich ueberhaupt von uniform
loest. Beides ist unabhaengig davon, welche 3D-Features anliegen.

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
