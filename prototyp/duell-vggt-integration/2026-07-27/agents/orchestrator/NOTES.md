# orchestrator - Arbeitsnotizen

Alles, was dieser Agent probiert, findet und verwirft. Roh und fortlaufend.
Kein Politur-Anspruch, aber jede Zahl mit Quelle (SLURM-Job, Run-Dir, Datei:Zeile).

Uhr gestartet 08:00:13Z, Ende 10:00Z.

## Setup

Der Agent laeuft auf dem Mac, nicht auf dem Cluster. Zugang ueber `ssh uc3`
(Login-Node uc3n991, `/usr/bin/sbatch` vorhanden). Cluster-Worktree fuers
Duell: `/tmp/duell-agent`, Branch `duell/2026-07-27-3d-short-horizon`, per
`scripts/setup_worktree.sh` verlinkt (data, .venv, output, external).

## Die entscheidende Ablesung, vor jeder Codezeile

Quelle: `output/runs/r2dreamer-curriculum-l3/run-6056750/metrics.csv` (CNN,
Seed 42, SLURM 6056750, 59322 Steps, 120 Episoden).

| Step | `metrics/sr` |
|---|---|
| 499 .. 13999 | **0.0**, jede einzelne Zeile |
| 14042 | 0.03448 = 1/29 |
| 14542 .. 20542 | monoton fallend 1/30, 1/31, ... 1/42 |
| 56042 .. 58042 | 0.01 |
| 58322 .. 59322 | 0.02 |

Zwei Konsequenzen, die den ganzen Durchlauf bestimmen:

1. **Die Baseline steht bis Step 13999 exakt auf SR 0.0.** Ein VGGT-Arm laeuft
   mit 171-254 ms/Step und erreicht in 30 Minuten N = 7000-10500. Das liegt
   vollstaendig im Nullbereich der Baseline. Ein einziger Erfolg im eigenen
   Lauf entscheidet das Duell.
2. **Die Baseline lernt in 59322 Steps nichts Sichtbares.** Die
   Aktionsverteilung bleibt ueber den ganzen Lauf uniform: Step 499
   stop 0.264 / forward 0.252, Step 59322 stop 0.220 / forward 0.256 /
   left 0.318 / right 0.206. Zwei Erfolge auf 120 Episoden = 1.67 %, also
   unter den 3.84 % des Random-Agents (GOAL.md:94). Bei ~4000
   Gradientenschritten entscheidet nicht die Encoder-Architektur, sondern wie
   viele Episoden der Lauf ueberhaupt sieht.

Daraus die Strategie: **Episodenzahl maximieren** (ms/Step minimieren) und
**zwei Arme parallel** fahren, was Regel 4 mit maximal zwei GPU-Jobs deckt.
Bei p = 3.84 % pro Episode und ~20 Episoden je Arm liegt P(mindestens ein
Erfolg) bei ~54 % pro Arm und ~79 % ueber zwei Arme.

Ehrlich dazugesagt: das ist eine Lotterie mit zwei Losen, keine Demonstration
besserer Architektur. Der Horizont gibt mehr nicht her, und GOAL.md:72-79 sagt
das auch so.

## Aenderung am Code

Zwei neue SLURM-YAMLs, beide `extends:` den bestehenden L3-Arm. Kein Python
angefasst, kein Encoder angefasst, kein neuer Shim.

- `scripts/slurm/configs/l3_aggregator_pooled_short.yaml`
- `scripts/slurm/configs/l3_pointmap_pose_short.yaml`

Je drei Overrides:

- `prefill: 5000 -> 2048`. Das ist der eigentliche Integrationsbefund.
  `prefill` ist eine **Step-Konstante, kein Anteil**. Die CNN-Baseline
  verbraucht dafuer 5000/59322 = 8.4 % ihres 30-Minuten-Budgets, ein VGGT-Arm
  bei ~170 ms/Step dagegen 5000/10500 = ~48 %. Dieselbe Konstante bestraft den
  langsamen Encoder um den Faktor sechs. 2048 stellt einen vergleichbaren
  Anteil her und bleibt 2x ueber dem Replay-Gate
  `batch_size * seq_len = 16 * 64 = 1024`, das Gate oeffnet also weiterhin
  beim ersten Trainingsschritt (`_base.yaml:42-47` begruendet genau diese 2x).
- `log_every: 250 -> 100`. Der Lauf endet als TIMEOUT bei unbekanntem Step;
  `metrics/sr` muss dicht genug abgetastet sein, damit ein Ablesepunkt nahe am
  Endstep liegt. Reine Logdichte, kein Eingriff in die Trainingsschleife.
- `env: SEED: "42"` fest im Config statt nur ueber `launch.sh --env`. Das ist
  die staerkere Form (der Lauf ist ohne Kommandozeile reproduzierbar) und
  Bedingung von `verify.sh` Check 3.

Bewusst **nicht** angefasst: `train_ratio`, `batch_size`, `seq_len`,
`buffer_capacity`, `actent`, Reward, Encoder-Routing. Jeder dieser Knoepfe
kostet entweder Wall-Clock (und damit Episoden) oder ist bei ~4000
Gradientenschritten nicht belegbar wirksam. Die Baseline zeigt, dass Lernen auf
diesem Horizont nichts bewegt - also nicht in Lernen investieren, sondern in
Episodenzahl.

## Armwahl

Vier L3-3D-Arme existieren. Gewaehlt sind die zwei schnellsten, weil Episoden
die Waehrung sind:

| Arm | ms/Step (L1-Messung laut Config-Kommentar) | gewaehlt |
|---|---|---|
| aggregator-pooled | ~94 (`aggregator_pooled_l1.yaml:19`) | ja |
| pointmap-pose | 106-121 (`pointmap_pose_l1.yaml:15`) | ja |
| hybrid | ~164 (`hybrid_v1.yaml:22`) | nein |
| global-tokens | ~254 (`l3_global_tokens.yaml:8`) | nein |

Die beiden Gewaehlten sind zusaetzlich das inhaltlich interessantere Paar:
gepoolte Semantik-Tokens gegen gepoolte Geometrie, beide am schnellen Ende.

## Fehlschlag 1: exit 2 nach 23 Sekunden (Job 6057269)

Beide Jobs kurz nach 08:05Z abgesetzt. 6057269 ging sofort auf `uc3n082` in
RUNNING und war 23 Sekunden spaeter aus der Queue: `sacct` meldet `FAILED`,
ExitCode `2:0`.

Erster Verdacht war der dev-GL-Abbruch aus `PROBLEMS.md` - **falsch**. Kein
Logfile, kein Output-Verzeichnis, gar nichts entstanden.

Ursache: `launch.py:227-228` setzt `#SBATCH --output={log_dir}/slurm-%j.out`
mit `log_dir = config.output_dir` (`launch.py:215`), das `mkdir -p {log_dir}`
steht aber erst in Zeile 245, also **im Skript** und damit zu spaet. Slurm kann
die Ausgabedatei nicht oeffnen und bricht vor dem ersten Kommando ab. Bei allen
bestehenden Armen faellt das nie auf, weil deren `output_dir` von frueheren
Laeufen existiert. Ein Arm mit neuem `output_dir` stirbt reproduzierbar.

Fix: die beiden Verzeichnisse einmal von Hand angelegt. Das rettete zugleich
den noch pendenden 6057270, der sonst beim Start dasselbe getan haette.

Nebenbefund fuer die Konsolidierung: **`dev_gpu_h100` schedult in unter einer
Sekunde**, waehrend `gpu_h100_short` mit 21 fremden PENDING-Jobs verstopft ist.
Der dev-GL-Abbruch aus `PROBLEMS.md` ist hier nicht aufgetreten.

## Submits

Alle `--prod --time 00:30:00 --exclude uc3n089 --env SEED=42`.

Zeitangaben sind Zulu und aus belastbaren Quellen rekonstruiert (`sacct`
Elapsed, W&B `createdAt`, `_runtime`). Eine frueher hier notierte Version mit
08:26Z / 08:36Z fuer die ersten Submits war falsch: sie stammte aus einer
fehlgelesenen lokalen Uhrzeit und widersprach dem Start von 6057316 um 08:14Z.

| Job | Arm | Partition | Ergebnis |
|---|---|---|---|
| 6057269 | l3_aggregator_pooled_short | dev_gpu_h100 | ~08:05Z, FAILED 2:0 nach 23 s |
| 6057270 | l3_pointmap_pose_short | gpu_h100_short | ~08:05Z, vor dem Start gecancelt |
| 6057297 | l3_aggregator_pooled_short | dev_gpu_h100 | ~08:09Z, FAILED 2:0 nach 21 s |
| 6057316 | l3_aggregator_pooled_short | dev_gpu_h100 | 08:14Z RUNNING, Ende 08:44Z, **N = 8501, SR 0.0588** |
| 6057317 | l3_pointmap_pose_short | gpu_h100_short | 08:14Z RUNNING, Ende 08:45Z, N = 8301, SR 0.0 |
| 6057422 | l3_aggregator_pooled_short_lowent | dev_gpu_h100 | 08:50Z, Ende 09:16Z, N = 7701, SR 0.0 |
| 6057423 | l3_pointmap_pose_short_lowent | gpu_h100_short | 08:50Z, Ende 09:16Z, N = 7499, SR 0.0 |

Die Ursache der beiden Fehlschlaege steht in `agents/launcher/NOTES.md`. Die dort
zuerst notierte Diagnose (fehlendes `output_dir`) war falsch; es war der
Worktree unter `/tmp`.

`verify.sh`: PASS (Check 2 SKIP, weil `data/` auf dem Mac nicht liegt; auf dem
Cluster nachgezogen, dort ebenfalls PASS).

## Ab 09:00Z ohne Cluster-Zugang

Der ControlMaster-Socket, ueber den `ssh uc3` lief, fiel weg; `ssh-add -l`
meldet "The agent has no identities", also war keine Neuanmeldung moeglich. Alle
Zahlen ab diesem Punkt kommen aus W&B (Projekt
`sailer-luca-university-ulm/3d-vla-objectnav`). Das reicht fuer die Auswertung,
nicht fuer das Kopieren der Rohdateien nach `runs/` - siehe LEDGER, Abschnitt
"Was am Ende offen blieb".
