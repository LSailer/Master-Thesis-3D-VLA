# Goal - Duell 2: VGGT-Features optimal in DreamerV3 integrieren

## Ziel

Innerhalb von **drei Stunden Gesamtzeit** den besten Arm aus Duell 1 schlagen,
indem 3D-Features (VGGT / UNITE) besser in DreamerV3 integriert werden.
Gemessen auf **Curriculum Level 3**.

Gemessen wird an einem **30-Minuten-Trainingslauf**, nicht an einem vollen
2M-Step-Lauf. Der Vergleichsmassstab ist **nicht mehr die CNN-Baseline**,
sondern der Gewinner von Duell 1 (siehe Referenz unten).

Ein gewerteter Lauf ist immer ein **`--prod`-Lauf mit 30 Minuten Walltime**,
nie `--smoke`. `--smoke` ist kein Zeitbudget, sondern ein Step-Budget: es
deckelt bei 1500 Steps (`_base.yaml:40-45`) und waere nach gut einer Minute
Training vorbei. Der gewertete Lauf laeuft stattdessen in die Walltime und
endet als SLURM `TIMEOUT`. Das ist erwartetes Verhalten, kein Fehler; die
`metrics.csv` wird fortlaufend geschrieben und verliert dadurch nichts.

## Format

Der Agent laeuft **allein**. Es gibt in diesem Durchlauf keine Gegenseite und
keine Blindheit. Der Agent meldet nach jeder Welle in drei Zeilen den Stand an
Luca.

## Referenz - die Latte

SLURM **6057641**, Arm `aggregator_pooled` mit `prefill 2048`, 30 Minuten
`--prod`, Seed 42. Quelle:
`prototyp/duell-vggt-integration/2026-07-27/runs/6057641-aggpool-p2048/metrics.csv`.

| Treffer | sr | spl | softspl | dtg | ms/Step | Episoden | N |
|---|---|---|---|---|---|---|---|
| 1 | 0.0556 | 0.0201 | 0.0605 | 5.193 | 134.1 | 18 | 9001 |

`ms/Step` ist der Steady-State-Wert aus dem letzten geloggten
`perf/ms_per_step_interval`, nicht der aus der Walltime gerechnete Wert 203.4.

## Erfolgskriterien - die Wertungsmatrix

Jede Metrik geht als **relative Aenderung zur Referenz** ein,
richtungskorrigiert, gekappt, gewichtet summiert:

```
rel = (wert - referenz) / |referenz|          fuer "hoeher ist besser"
rel = (referenz - wert) / |referenz|          fuer "niedriger ist besser"

Score = Summe ueber alle Metriken von (Gewicht * rel)
```

| Metrik | Richtung | Referenz | Gewicht | Kappung |
|---|---|---|---|---|
| **Treffer** | hoch | 1 | **0.45** | +200 % |
| `softspl` | hoch | 0.0605 | 0.15 | +/-100 % |
| `dtg` | niedrig | 5.193 | 0.15 | +/-100 % |
| `spl` | hoch | 0.0201 | 0.10 | +/-100 % |
| `ms/Step` | niedrig | 134.1 | 0.10 | +/-100 % |
| Episoden | hoch | 18 | 0.05 | +/-100 % |
| (`sr`) | hoch | 0.0556 | nur Bericht | - |

- **Score > 0** heisst besser und kommt so ins Ledger.
- **Score >= +0.10** ist die PR-Schwelle.
- Der Arm mit dem hoechsten Score wird auf **Seed 43** bestaetigt. Bei
  Gleichstand bekommen beide einen Bestaetigungslauf. Ueber den PR entscheidet
  der **Mittelwert beider Seeds**.
- Ein Job, der nicht durchkommt, ist `gescheitert`: kein Score, zaehlt nicht
  gegen die Variante.

## Ableseprotokoll

- **Treffer** = Anzahl der Zeilen mit `episode/success == 1` in der
  `metrics.csv`. Nicht aus `sr * Episoden` gerechnet.
- Alle uebrigen Werte = **letzter geloggter Wert innerhalb des Slots**
  (`metrics/spl`, `metrics/softspl`, `metrics/dtg`, `episode/count`,
  `perf/ms_per_step_interval`).
- Fehlt `perf/ms_per_step_interval` in einem Lauf, gilt ersatzweise
  `Elapsed_s / N * 1000` mit Vermerk im Ledger.

## Was sich gegenueber Duell 1 geaendert hat, und warum

**Der feste 30-Minuten-Slot ersetzt den Abgleich bei gleicher Step-Zahl.**
Duell 1 las die Baseline-SR bei genau der Step-Zahl `N` ab, die der 3D-Lauf
erreicht hatte. Das hat sich gerecht, aber es hat den schnellen Arm bestraft:
Lauf #4 war 2.2x so schnell wie #2, landete damit in einem Fenster, in dem die
Baseline ihren eigenen Zufallserfolg schon hatte, und kam trotz gleicher
Trefferzahl auf Delta 0. Jetzt zaehlt nur noch, wer in denselben 30 Minuten
weiter kommt. Geschwindigkeit ist damit ein echter Hebel und kein Eigentor.

**Treffer statt Rate im Score.** `metrics/sr` ist ein Mittelwert ueber die
abgeschlossenen Episoden. Bei wenigen Episoden blaeht ein einzelner
Zufallstreffer den Wert auf: ein Arm mit 6 Episoden und einem Treffer stuende
bei 16.7 % und wuerde einen Arm mit 40 Episoden und zwei Treffern schlagen.
Die Trefferzahl dreht das um. Mehr Episoden in denselben 30 Minuten heissen
mehr Chancen auf einen Treffer, und genau das soll belohnt werden. Deshalb
braucht es auch keine Untergrenze fuer die Episodenzahl.

**Reward ist aus der Wertung raus.** Reward-Shaping bleibt freies Spielfeld
(`RULES.md`, Abschnitt 3). Beides zusammen hiesse, der Agent darf seine eigene
Note schreiben. `sr`, `spl`, `softspl` und `dtg` kommen dagegen alle aus
`src/environments/habitat.py` und damit aus der eingefrorenen Zone.

**`ms/Step` bleibt bewusst in der Matrix**, obwohl der Zeit-Slot
Geschwindigkeit ohnehin schon ueber mehr Steps und mehr Episoden belohnt. Das
Gewicht 0.10 ist klein und die Doppelzaehlung gewollt: weniger Zeit pro Step
ist fuer diese Arbeit ein Ergebnis fuer sich.

## Hypothese

Die Art, wie VGGT-Features in den World Model Encoder geroutet werden
(Aggregator-Tokens, Global Tokens, Pointmaps, World-Points/Camera-Pose,
Hybrid-CNN-VGGT, FiLM-Konditionierung), bestimmt massgeblich, wie schnell der
Agent auf L3 anlernt. Eine bessere Integration schlaegt den bisher besten Arm
im selben 30-Minuten-Fenster.

## Ehrlicher Vorbehalt

Die Referenz steht bei **einem** Treffer in 18 Episoden. Zwei Treffer statt
einem sind statistisch kaum von einem Muenzwurf zu unterscheiden. Genau
deshalb tragen `dtg` und `softspl` zusammen 0.30 der Wertung: die sind dicht,
jede Episode liefert einen Wert, und sie zeigen Annaeherung an das Ziel auch
dann, wenn kein Erfolg zustande kommt. Und deshalb muss der Gewinner auf einem
zweiten Seed bestaetigt werden. Ein Score ohne Seed 43 ist ein Kandidat, kein
Ergebnis.

## Kontext zu Level 3

| | |
|---|---|
| Curriculum | `data/curriculum/level3_10houses_1goal.json`, 10 HM3D-Haeuser, nur `chair`, 74 997 Train-Episoden |
| Observation | ein RGB-Frame, 64x64 (CNN) bzw. 518x518 (VGGT), keine Goal-Conditioning |
| Episode | max. 500 Steps (`src/environments/habitat.py:49`) |
| Aktionen | 4 diskret, STOP ist ein No-op und beendet die Episode nicht |
| Success | geodaetische Distanz < `GOAL_RADIUS = 0.2` m (`habitat.py:36`) |
| Reward | `geodesic_delta` + `success_bonus 10.0` + `step_penalty -0.01` |

Historische L3-Zahlen (alle pre-migration, nicht als Baseline verwendbar):
CNN 32 % SR / 0.21 SPL (W&B `rsopsua1`), bester VGGT-Arm 22 % (W&B `6rrf50u3`,
SLURM TIMEOUT bei 48 h), Random 3.84 %.

Zur Einordnung des Duell-Fensters: die CNN-Baseline 6056750 erreichte in
30 Minuten 59 322 Steps bei 2 % SR, also unter dem Random-Agenten. In diesem
Fenster lernt kein Arm die Aufgabe. Gemessen wird, welche Integration am
schnellsten anfaengt, sich in die richtige Richtung zu bewegen.
