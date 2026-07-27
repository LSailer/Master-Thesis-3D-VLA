# Duell-Ledger 2026-07-27 Runde 2 (Agent)

Gepflegt vom Orchestrator. Eine Zeile pro gewertetem Lauf, sofort nach der
Auswertung. Wertungsmatrix und Ableseprotokoll: `../GOAL.md`.

## Rahmen

- Start (Wall-Clock): TBD (erster Tool-Call)
- Ende (Wall-Clock): Start + 3:00
- Letzte Welle spaetestens: Start + 1:45
- Eingefrorener Seed: **42**, Bestaetigung auf **43**
- `verify.sh` beim Start: TBD

## Referenz (Latte)

SLURM **6057641**, `aggregator_pooled` + `prefill 2048`, 30 min `--prod`,
Seed 42. Quelle: `../2026-07-27/runs/6057641-aggpool-p2048/metrics.csv`.

| Treffer | sr | spl | softspl | dtg | ms/Step | Episoden | N |
|---|---|---|---|---|---|---|---|
| 1 | 0.0556 | 0.0201 | 0.0605 | 5.193 | 134.1 | 18 | 9001 |

## Gewertete Laeufe

| # | Hypothese | Aenderung | SLURM | Seed | Treffer | sr | spl | softspl | dtg | ms/Step | Ep. | Steps N | **Score** | Verdikt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | |

Verdikt-Vokabular: `besser` (Score > 0) / `schlechter` / `neutral` /
`gescheitert` (Job kam nicht durch, keine verwertbare Zahl) /
`ueber Schwelle, unbestaetigt` (Score >= +0.10, kein Seed-43-Lauf).

## Score-Aufschluesselung des Fuehrenden

| Metrik | Referenz | Seed 42 | Seed 43 | Gewicht | Beitrag |
|---|---|---|---|---|---|
| Treffer | 1 | | | 0.45 | |
| softspl | 0.0605 | | | 0.15 | |
| dtg | 5.193 | | | 0.15 | |
| spl | 0.0201 | | | 0.10 | |
| ms/Step | 134.1 | | | 0.10 | |
| Episoden | 18 | | | 0.05 | |
| (sr) | 0.0556 | | | Bericht | |

## Erkenntnisse

Was hat gewirkt, was nicht, und warum. Keine Spekulation ohne Zahl.

## Sackgassen

Aus Runde 1 uebernommen, nicht erneut probieren:

- **VGGT-Eingangsaufloesung senken**: Backbone wirft bei fremden Shapes,
  RoPE-Tabellen, Pooler und Replay-Shapes haengen daran (backbone.py:96-147).
- **prefill 0**: PERSIST_SCENE-Bug dokumentiert in loops.py:353-361.
- **Reward-Shaping**: bei 0-1 Erfolgen im Replay hat success_bonus keinen
  Gradientenbeitrag, step_penalty kuerzt der normierte Advantage weg. Nicht
  empirisch geprueft, und der Reward steht in Runde 2 nicht mehr in der
  Wertung.

## Offene Faeden

Aus Runde 1 uebernommen, als Startpunkte:

- kv200k auf den hybrid-Arm uebertragen (dort ungeprueft; wenn die
  Feature-Qualitaet haelt, waere hybrid bei ~90 ms/Step).
- Aggregator depth=12 (Schaetzung -30-40%, `total_budget` muss mitskalieren).
- Aggregator-MLP auf L3 portieren, schnellster 3D-Arm laut `../PLAN.md:104`.
- Ob der Referenzlauf echtes Anlernen oder Lotterie war: total_loss faellt,
  aber 1 Treffer traegt keine Aussage.

## Geoeffnete Pull Requests

| PR | Branch | Score s42 | Score s43 | Mittel | Status |
|---|---|---|---|---|---|
| | | | | | |
