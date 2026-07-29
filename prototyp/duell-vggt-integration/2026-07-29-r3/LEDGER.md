# Duell-3-Ledger 2026-07-29-r3 (Agent)

Gepflegt vom Orchestrator. Eine Zeile pro Versuch, sofort nach der Auswertung.

## Rahmen

- Start (Wall-Clock): <UTC, erster Tool-Call>
- Deadline: Start + 3:00. Letzte Welle spaetestens Start + 1:45.
- Eingefrorener Seed: 42 (Bestaetigung des Fuehrenden: SEED=43 per CLI)
- verify.sh beim Start: <PASS/FAIL, Uhrzeit>

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

## Welle 1 (submittet <UTC>, T+<h:mm>)

| Slot | Config | Adapter / Encoder | Zeile / Kapazitaet | SLURM | Status |
|---|---|---|---|---|---|
| A | | P1 - `aggregator_pooled_full`, MLP wie C | 24 KB / 500 000 | | |
| B | | | | | |
| C | | | | | |
| D | | | | | |

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
