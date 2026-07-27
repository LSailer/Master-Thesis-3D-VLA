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
| 0a | Worktree unter `/tmp` ist auf dem Compute-Node nicht sichtbar | - | 6057269 | - | - | - | - | - | - | `gescheitert` |
| 0b | dito | - | 6057297 | - | - | - | - | - | - | `gescheitert` |
| 1 | Gepoolte Aggregator-Tokens, Prefill-Anteil auf CNN-Niveau gebracht | `prefill 5000->2048`, `log_every 250->100` | 6057316 | 8501 | **0.0588** | **0.0** | **+5.88 pp** | 211.7 | 500 | **`besser`** |
| 2 | dito, aber gepoolte Geometrie statt Tokens | `prefill 5000->2048`, `log_every 250->100` | 6057317 | 8301 | 0.0 | 0.0 | 0.00 pp | 216.8 | 500 | `neutral` |
| 3 | Niedrigere Aktor-Entropie laesst die Policy sich festlegen, Tokens | `+ act_entropy 3e-2->3e-3` | 6057422 | laeuft | laeuft | | | | | offen |
| 4 | dito, Geometrie | `+ act_entropy 3e-2->3e-3` | 6057423 | laeuft | laeuft | | | | | offen |

Verdikt-Vokabular: `besser` / `schlechter` / `neutral` / `gescheitert`
(gescheitert = Job kam nicht durch, keine verwertbare Zahl).

ms/Step nach `GOAL.md:55` aus 1800 s Walltime und N gerechnet. Das liegt ueber
dem geloggten `perf/ms_per_step_interval` (123-144 ms), weil rund 7 der 30
Minuten fuer VGGT-Gewichte, JAX-Kompilierung und Habitat-Szenenaufbau vergehen,
bevor der erste Step laeuft. Beide Zahlen stehen unten je Lauf.

Quellen fuer jede Zeile: W&B-Projekt `sailer-luca-university-ulm/3d-vla-objectnav`,
Run-Ids `3sqrld07` (6057316), `74k1yvo0` (6057317), `ikrjoqrn` (6057422),
`mljupjg0` (6057423), `m7wae8m4` (Baseline 6056750).

### Versuch 1 im Detail, der einzige Gewinner

| Groesse | Wert | Quelle |
|---|---|---|
| Arm | `l3_aggregator_pooled_short` | `scripts/slurm/configs/l3_aggregator_pooled_short.yaml` |
| SLURM / W&B | 6057316 / `3sqrld07` | |
| N | 8501 | `_step` |
| `metrics/sr` | 0.0588 = **1/17** | |
| Baseline `metrics/sr` bei Step 8499 | **0.0** | `run-6056750/metrics.csv` |
| Delta | **+5.88 pp** | |
| Erfolge | 1 | `episode/success == 1` bei Step 2404 |
| Episoden | 17 | `episode/count` |
| `metrics/spl` | 0.0213 | |
| `perf/ms_per_step_interval` | 123.07 ms | |
| ms/Step nach GOAL.md:55 | 211.7 ms | 1800 s / 8501 |
| Aktionen am Ende | forward 0.256 / stop 0.258 / left 0.256 / right 0.230 | |

Die Erfolgsepisode (Step 2404, Episode 5, Szene `u5atqC7vRCY`): `episode/steps`
405 statt 500, `episode/dtg` 0.158 m, `episode/path_length` 10.32 m bei
`episode/shortest_path` 3.74 m, `episode/reward` +9.54, `episode/spl` 0.363.
Also eine echte Ankunft am Ziel, kein Messartefakt: die Episode endet vor dem
500-Step-Cap, was ausschliesslich bei Erfolg passiert.

**Wie belastbar ist das?** Ehrlich: ein einziger Erfolg. `metrics/sr` ist bei 17
Episoden ein Quotient mit Nenner 17, kein Mittel ueber 100. Der Vergleich ist
formal korrekt nach `GOAL.md:32-42` und `PLAN.md:58-68`, aber die Effektgroesse
ist nicht von Glueck zu trennen. Bemerkenswert bleibt, dass die
CNN-Baseline mit **demselben Seed 42 und demselben Curriculum** in ihren ersten
28 Episoden keinen einzigen Erfolg hatte und ihren ersten erst bei Step 14042
verbuchte.

## Erkenntnisse

1. **`prefill` ist eine Step-Konstante und benachteiligt langsame Encoder um
   den Faktor sechs.** Bei 30 Minuten Walltime kostet `prefill: 5000` die
   CNN-Baseline 8.4 % ihres Budgets (5000 von 59 322 Steps) und einen VGGT-Arm
   59 % (5000 von 8501). Das ist kein Encoder-Effekt, sondern eine Konstante,
   die im Zeitbudget zum Anteil wird. `prefill: 2048` stellt einen
   vergleichbaren Anteil her (24 %) und bleibt 2x ueber dem Replay-Gate
   `batch_size * seq_len = 1024`. Der Erfolg von 6057316 faellt bei Step 2404,
   also 356 Steps nach Ende des verkuerzten Prefills - mit 5000 waere dieser
   Zeitpunkt noch im Prefill gelegen.

2. **Der Engpass auf L3 ist Fortbewegung, nicht Wahrnehmung.** Details in der
   Baseline-Sektion oben. Kurz: Ziel 3-9 m entfernt, Pfadlaenge 2-6 m in 500
   Aktionen, rollende Zieldistanz am Ende hoeher als am Anfang, Aktionen ueber
   59 322 Steps uniform. Solange das so ist, kann kein Encoder-Routing die SR
   heben, und jeder Vergleich zwischen Routings auf diesem Horizont misst
   Rauschen.

3. **Rund 7 der 30 Minuten sind Startup.** `perf/ms_per_step_interval` liegt
   bei 123-146 ms, die nach `GOAL.md:55` gerechneten ms/Step bei 205-217. Die
   Luecke ist VGGT-Gewichtsladen, JAX-Kompilierung und Habitat-Szenenaufbau.
   Fuer ein 30-Minuten-Budget ist das ein Viertel des Budgets und damit ein
   groesserer Hebel als jede Encoder-Optimierung.

4. **`log_every: 100` liefert die Trainingsmetriken, die `PROBLEMS.md:47-79`
   als dauerhaft fehlend beschreibt.** In allen vier Duell-Laeufen stehen
   `total_loss`, `loss/{dyn,rep,rew,value,policy,con,barlow,repval}`,
   `latent/*`, `params/encoder_l2` und `perf/*` in der `metrics.csv`. Ob das an
   der Paritaetsverschiebung liegt oder daran, dass `67ed1c1` den Bug schon
   behoben hat, ist nicht getrennt - `PROBLEMS.md` gehoert aktualisiert.

5. **`dev_gpu_h100` ist der schnelle Weg zu einer Zahl.** Scheduling in unter
   einer Sekunde, waehrend `gpu_h100_short` mit 21 fremden PENDING-Jobs belegt
   war. Der in `PROBLEMS.md:149-152` beschriebene OpenGL-Abbruch beim Prefill
   auf dev ist in vier Laeufen nicht aufgetreten.

## Sackgassen

1. **Worktree unter `/tmp`.** Kostete zwei Jobs und rund 12 Minuten. `/tmp` ist
   auf bwUniCluster node-lokal; der Compute-Node sieht das WorkDir nicht, Slurm
   bricht vor dem ersten Kommando mit exit 2 ab und schreibt kein Logfile.
   Details und die zwischenzeitlich falsche Diagnose stehen in
   `agents/launcher/NOTES.md`. Worktrees fuer SLURM-Jobs gehoeren nach `$HOME`.

2. **`act_entropy 3e-2 -> 3e-3`.** Die Hypothese war: bei 3e-2 dominiert der
   Entropie-Bonus in `src/r2dreamer/behavior/loss.py:141` den Advantage-Term,
   die Policy kann sich nicht festlegen, und `forward` ist die einzige Aktion
   mit systematisch positivem Advantage unter `geodesic_delta`. Zahlen siehe
   Versuche 3 und 4. Die Aktionsverteilung loest sich auch mit einem Zehntel
   des Koeffizienten nicht von uniform; sie schwankt nur staerker von Episode
   zu Episode. Bei ~3200 Gradientenschritten ist der Aktor an seiner
   Initialisierung, und der Entropie-Term ist nicht das, was ihn dort haelt.
   Nicht ohne deutlich mehr Gradientenschritte erneut probieren.

3. **Berater-Subagents.** Drei Anlaeufe (zwei Personas, dann ein zweiter
   Versuch) starben an `API Error: 529 Overloaded`, ohne eine Zeile zu
   liefern. `agents/danijar-hafner/NOTES.md` und
   `agents/jianyuan-wang/NOTES.md` sind deshalb leer geblieben. Keine
   inhaltliche Sackgasse, aber ein Hinweis fuer die Planung: die Beratung darf
   nicht auf dem kritischen Pfad liegen.

## Offene Faeden

1. **Die 7 Minuten Startup senken.** Grosster ungenutzter Hebel: ein Viertel
   des Budgets. Zu pruefen waere, ob VGGT-Gewichte und JAX-Kompilierung
   cachebar sind und wie viel davon Habitats Szenenaufbau ist.
2. **ms/Step unter 130 druecken**, damit N naeher an die Grenze 14042 kommt und
   die Episodenzahl von 17 auf ~28 steigt. Kandidaten aus `PLAN.md:104-106`:
   Compute-dtype bf16, Aufloesung unter 518, VGGT-Forward nur jeden k-ten Step.
   Nicht geprueft, weil die Beratung dazu ausfiel.
3. **Ein Arm, der Fortbewegung direkt belohnt.** Reward-Shaping ist nach
   `RULES.md:44` frei. Zu testen: ein Term auf zurueckgelegter Distanz oder auf
   Kollisionsfreiheit, statt nur `geodesic_delta`. Das greift den unter
   Erkenntnis 2 belegten Engpass an, statt am Encoder zu drehen.
4. **Ein fairer Encoder-Vergleich braucht mehr als 30 Minuten.** Bei ~3200
   Gradientenschritten ist die Frage "welches Routing ist besser" nicht
   beantwortbar. Wenn diese Frage das Ziel ist, muss das Budget in
   Gradientenschritten definiert werden, nicht in Wall-Clock.
5. **Messprotokoll-Artefakt.** Die Baseline hat genau einen Erfolg zwischen
   Step 14042 und 56042. Deshalb ist ihre SR unterhalb 14042 exakt 0 und die
   Latte fuer einen 3D-Arm, der nur ~8500 Steps schafft, trivial. Ein Arm, der
   schneller waere und 14042 ueberschreitet, haette es schwerer. Das ist eine
   perverse Eigenschaft des Vergleichs bei gleicher Step-Zahl und sollte fuer
   den naechsten Durchlauf geradegezogen werden - etwa durch Vergleich bei
   gleicher Episodenzahl statt gleicher Step-Zahl.

## Geoeffnete Pull Requests

| PR | Branch | SR | Delta | Status |
|---|---|---|---|---|
| | | | | |
