# Goal - Duell 4: world_points als Injektion (Mensch gegen Agent)

## Ziel

Innerhalb von **fuenf Stunden Gesamtzeit** den besseren Weg finden, die pro
Frame vorhergesagten **`world_points` des VGGT-Point-Heads** in den
DreamerV3-Ansatz zu integrieren. Gemessen auf **Curriculum Level 3**, in
**30-Minuten-`--prod`-Laeufen**, gewertet gegen den frame-mean-Arm P2 aus
Duell 3.

Zwei Parteien treten gegeneinander an:

- **Luca** (Mensch)
- **Der Agent** (das bekannte Duell-Setup: Orchestrator mit Subagents,
  laeuft direkt auf dem bwUniCluster)

Ein gewerteter Lauf ist immer ein **`--prod`-Lauf mit 30 Minuten Walltime**,
nie `--smoke`. Er endet als SLURM `TIMEOUT`; das ist erwartetes Verhalten,
die `metrics.csv` wird fortlaufend geschrieben und verliert nichts.

## Format

- **Blind innerhalb des Fensters.** Waehrend der fuenf Stunden erfaehrt keine
  Seite, was die andere versucht. Kein Sync zwischen Wellen; der Austausch
  passiert geballt in der Nachbesprechung.
- **Geteiltes Vorwissen.** Beide Seiten kennen und nutzen ausdruecklich die
  vollstaendigen Ledger und Erkenntnisse aus Duell 1-3
  (`2026-07-27/`, `2026-07-27-r2/`, `2026-07-29-r3/`).
- **Wellen sind frei.** Jede Seite startet beliebig viele Arme, wann sie will,
  innerhalb der Compute-Regeln (`RULES.md` Abschnitt 5).
- **Der beste Lauf pro Seite zaehlt.** Der jeweils beste Seed-42-Arm einer
  Seite wird nach Ablauf auf Seed 43 bestaetigt; gewertet wird das paarweise
  Score-Mittel beider Seeds.

## Harte Bedingung: Points statt Tokens

**Jeder gewertete Arm erfuellt alle drei Punkte:**

1. Die pro Frame vorhergesagten **`world_points` des Point-Heads fliessen in
   die Beobachtung bzw. den Encoder** ein. Ein Arm ohne Points beantwortet die
   Frage dieses Duells nicht und wird nicht gewertet.
2. **`compute_heads=True`** fuer jeden gewerteten Arm. Der Camera-Head laeuft
   dadurch mit (der Flag schaltet beide Heads gemeinsam,
   `src/vggt/jax/feature_extractor.py:298`); seine Outputs duerfen genutzt
   werden, muessen aber nicht.
3. **Aggregator-Tokens sind verboten.** Weder Frame- noch globale Haelfte,
   weder Kamera-, Register- noch Patch-Tokens duerfen in Beobachtung oder
   Encoder einfliessen. Erlaubt ist zusaetzlich nur das **RGB-Bild** - als
   eigener CNN-Zweig (Muster `rgb_pointmap_pose`) oder als Farbwerte an den
   Punkten, freie Wahl.

## Referenz - die Latte, paarweise gewertet

Der beste Arm aus Duell 3: **P2 frame-mean** (`aggregator_pooled_meanf`),
Trainingsknobs wie Arm C (prefill 1024, train_ratio 128, act_entropy 0.1),
30 Minuten `--prod`.

| Referenz | Seed | Treffer | sr | spl | softspl | dtg | ms/Step | Episoden | N |
|---|---|---|---|---|---|---|---|---|---|
| **6087075** | 42 | 1 | 0.0256 | 0.0134 | 0.1115 | 5.665 | 70.0 | 39 | 19255 |
| **6089423** | 43 | 1 | 0.0256 | 0.0065 | 0.0438 | 5.070 | 66.9 | 39 | 19267 |

Quellen: `prototyp/duell-vggt-integration/2026-07-29-r3/runs/6087075-B-p2meanf/metrics.csv`
und `.../6089423-I-p2meanf-s43/metrics.csv`. Beide Laeufe existieren bereits;
es sind keine neuen Baseline-Laeufe noetig.

**Die Wertung ist paarweise pro Seed:** ein Seed-42-Lauf wird gegen 6087075
gewertet, der Seed-43-Bestaetigungslauf gegen 6089423. Der Grund ist derselbe
wie in r3: P2s Seeds unterscheiden sich in `softspl` um Faktor 2.5 (0.1115
gegen 0.0438); gegen einen festen Seed-42-Vektor waere die Bestaetigung
wertlos.

## Erfolgskriterien - die Wertungsmatrix

Unveraendert aus Duell 2/3, nur auf P2s Zahlen gestellt:

```
rel = (wert - referenz) / |referenz|          fuer "hoeher ist besser"
rel = (referenz - wert) / |referenz|          fuer "niedriger ist besser"

Score = Summe ueber alle Metriken von (Gewicht * rel)
```

| Metrik | Richtung | Referenz s42 | Referenz s43 | Gewicht | Kappung |
|---|---|---|---|---|---|
| **Treffer** | hoch | 1 | 1 | **0.45** | +200 % |
| `softspl` | hoch | 0.1115 | 0.0438 | 0.15 | +/-100 % |
| `dtg` | niedrig | 5.665 | 5.070 | 0.15 | +/-100 % |
| `spl` | hoch | 0.0134 | 0.0065 | 0.10 | +/-100 % |
| `ms/Step` | niedrig | 70.0 | 66.9 | 0.10 | +/-100 % |
| Episoden | hoch | 39 | 39 | 0.05 | +/-100 % |
| (`sr`) | hoch | 0.0256 | 0.0256 | nur Bericht | - |

- **Treffer** = Anzahl der Zeilen mit `episode/success == 1` in der
  `metrics.csv`. Nicht aus `sr * Episoden` gerechnet.
- Alle uebrigen Werte = **letzter geloggter Wert innerhalb des Slots**
  (`metrics/spl`, `metrics/softspl`, `metrics/dtg`, `episode/count`,
  `perf/ms_per_step_interval`).
- Die `metrics.csv` ist im Langformat `step,metric,value`.
- Fehlt `perf/ms_per_step_interval`, gilt ersatzweise `Elapsed_s / N * 1000`
  mit Vermerk im Ledger.

Der Scorer wird aus `2026-07-29-r3/agents/orchestrator/score.py` abgeleitet
und **vor Welle 1** auf die P2-Referenzen umgestellt; Validierung wie in r3:
P2 gegen sich selbst gescored muss ~0.00 ergeben.

## Siegbedingung

- **Sieger ist die Seite mit dem hoeheren Score-Mittel** aus Seed 42 und
  Seed 43 ihres besten Arms. Auch zwei negative Scores haben einen Sieger.
- **Kein PR aus dem Duell.** Das Duell produziert Erkenntnis, keine Merges.
  Was spaeter in main wandert, entscheidet Luca nach der Nachbesprechung auf
  dem normalen Weg.
- Deliverable nach den Bestaetigungslaeufen: eine **interaktive
  Ergebnis-Tabelle (Widget)** ueber alle Metriken und Laeufe beider Seiten,
  danach die gemeinsame Nachbesprechung fuers Paper.

## Ehrlicher Vorbehalt

- **Der Heads-Malus ist real und fuer beide gleich.** Der einzige bisherige
  Heads-on-Arm (`pointmap_pose`) lief mit 146.2 ms/Step gegen 66.8 des
  gepoolten Arms. Gegen P2s 70.0 ms laeuft `ms/Step` damit in die Kappung:
  rund **-0.10 Score-Handicap**, dazu weniger Steps und damit weniger
  Episoden (-0.05 moeglich). Ein Points-Arm muss das ueber Treffer, softspl
  und dtg zurueckverdienen. Das ist Absicht: beide Duellanten tragen dasselbe
  Handicap, die Latte bleibt der beste bekannte Arm.
- **`world_points` ist die semantikfreieste Groesse im Modell** (Wang-Notizen,
  r2/r3): Semantik lebt in den DINOv2-initialisierten Tokens, die hier
  verboten sind. Die Points muessen ihren Wert ueber Geometrie liefern -
  oder ueber den RGB-Zweig ergaenzt werden.
- **Ziehungsvarianz ~+/-0.04 Score** (r2/r3, identische Configs). Ein
  Abstand unter ~0.05 zwischen den Duellanten ist kein belastbares Ranking
  und wird im Fazit als solches benannt.
- **Der Flaschenhals von L3 ist Lokomotion, nicht Wahrnehmung** (Duell-1-Run-A-
  Ledger): in diesem Fenster lernt kein Arm die Aufgabe. Gemessen wird, welche
  Integration am schnellsten anfaengt, sich in die richtige Richtung zu
  bewegen.

## Kontext zu Level 3

| | |
|---|---|
| Curriculum | `data/curriculum/level3_10houses_1goal.json`, 10 HM3D-Haeuser, nur `chair`, 74 997 Train-Episoden |
| Observation | ein RGB-Frame, 518x518 (VGGT), keine Goal-Conditioning |
| Episode | max. 500 Steps (`src/environments/habitat.py:49`) |
| Aktionen | 4 diskret, STOP ist ein No-op und beendet die Episode nicht |
| Success | geodaetische Distanz < `GOAL_RADIUS = 0.2` m (`habitat.py:36`) |
| Reward | `geodesic_delta` + `success_bonus 10.0` + `step_penalty -0.01` |
