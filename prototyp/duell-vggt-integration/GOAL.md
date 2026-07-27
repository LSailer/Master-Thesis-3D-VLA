# Goal - Duell: VGGT-Features optimal in DreamerV3 integrieren

## Ziel

Innerhalb von **zwei Stunden Gesamtzeit** die bestmögliche Success Rate auf
**Curriculum Level 3** erreichen, indem 3D-Features (VGGT / UNITE) besser in
DreamerV3 integriert werden.

Gemessen wird an einem **30-Minuten-Trainingslauf**, nicht an einem vollen
2M-Step-Lauf. Der Vergleichsmassstab ist der CNN-Image-Encoder-Baseline.

## Format

Zwei Parteien treten unabhaengig gegeneinander an:

- **Luca** (Mensch)
- **Der Agent** (dieses Setup, laeuft direkt auf bwUniCluster)

Beide arbeiten **blind**: waehrend der zwei Stunden erfaehrt keine Seite, was
die andere versucht. Nach Ablauf treffen sich beide und konsolidieren ihr
Wissen: was wurde probiert, was hat gewirkt, was ist tot.

## Erfolgskriterien

**Primaer: Success Rate.**
`metrics/sr` (gleitender Mittelwert ueber 100 Episoden) am Ende des
30-Minuten-Laufs, verglichen mit dem CNN-Baseline-Lauf **bei gleicher
Step-Zahl**.

Das ist der Kern des Messprotokolls. Der CNN-Encoder laeuft mit ~28 ms/Step,
die VGGT-Arme mit 171-219 ms/Step. In derselben Wall-Clock-Zeit schafft der
CNN-Arm ~64 000 Steps, ein 3D-Arm nur ~8 000-10 000. Ein Vergleich bei
gleicher Zeit wuerde nur Encoder-Kosten messen, nicht Integrationsqualitaet.
Deshalb: `N` = die Step-Zahl, die der 3D-Lauf erreicht hat; die Baseline-SR
wird aus deren `metrics.csv` **bei genau Step N** abgelesen.

**Sekundaer: Effizienz.**
`episode/steps` (Mittel der letzten 100 Episoden) und
`perf/ms_per_step_interval`. Zaehlt **nur als Tie-Break**, wenn die
SR-Differenz unter 3 Prozentpunkten liegt.

Hinweis: ms/Step zu senken ist ein legitimer und erwuenschter Hebel. Weniger
Zeit pro Step bedeutet mehr Steps in denselben 30 Minuten und damit potenziell
eine hoehere SR. Das ist explizit Teil des Spielfelds.

## Hypothese

Die Art, wie VGGT-Features in den World Model Encoder geroutet werden
(Aggregator-Tokens, Global Tokens, Pointmaps, World-Points/Camera-Pose,
Hybrid-CNN-VGGT, FiLM-Konditionierung), bestimmt massgeblich, wie schnell der
Agent auf L3 anlernt. Eine bessere Integration schlaegt den reinen
CNN-Encoder bereits bei gleicher, kleiner Step-Zahl.

## Ehrlicher Vorbehalt

Bei ~8 000-10 000 Steps abzueglich 5 000 Steps Prefill bleiben etwa 3 000-5 000
echte Trainingsschritte. Die SR aller Arme liegt dort voraussichtlich nahe null
und die Unterschiede koennen im Rauschen liegen. Dieser erste Durchlauf misst
daher eher "laeuft es und lernt es ueberhaupt an" als "welche Integration ist
langfristig besser". Das ist fuer einen ersten Test des Duell-Formats
akzeptiert und bewusst so gewaehlt.

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
