# Goal - Duell 3: Frame-Tokens und Kamera-Tokens in den gepoolten Arm

## Ziel

Innerhalb von **drei Stunden Gesamtzeit** den bestaetigten Sieger von Duell 2
schlagen, indem die **Frame-Haelfte** der VGGT-Aggregator-Tokens und **beide
Kamera-Tokens** in den Beobachtungsvektor aufgenommen werden. Gemessen auf
**Curriculum Level 3**, in einem **30-Minuten-Trainingslauf**.

Der heutige Arm `AggregatorPooledAdapter` (`src/adapters/global_tokens.py:127`)
liest ausschliesslich die **globale Haelfte** der Aggregator-Tokens und poolt
sie zu `[camera token, patch mean, patch max]` = 3 x 1024 = 3072. Die
Frame-Haelfte faellt im Extractor ohnehin an
(`src/vggt/jax/feature_extractor.py:992` splittet die 2048 Kanaele in
`frame_tokens` und `global_tokens`) und wird verworfen.

Ein gewerteter Lauf ist immer ein **`--prod`-Lauf mit 30 Minuten Walltime**,
nie `--smoke`. Der gewertete Lauf laeuft in die Walltime und endet als SLURM
`TIMEOUT`. Das ist erwartetes Verhalten, kein Fehler; die `metrics.csv` wird
fortlaufend geschrieben und verliert dadurch nichts.

## Format

Der Agent laeuft **allein**. Es gibt keine Gegenseite und keine Blindheit. Der
Agent meldet nach jeder Welle in drei Zeilen den Stand an Luca.

## Die zwei Vergleichsebenen

Dieses Duell beantwortet zwei getrennte Fragen und haelt sie deshalb
auseinander:

1. **Headline - bringen Frame-Tokens ueberhaupt etwas?**
   **P1 gegen C.** Genau eine Variable: Frame-Haelfte drin oder nicht. P1
   benutzt deshalb **denselben MLP-Encoder wie C**, unveraendert.
2. **Interne Rangliste - wie kodiert man sie am besten?**
   Alle weiteren Arme werden **gegen P1** verglichen und gegen P1 gerankt.

Die **PR-Entscheidung laeuft ausschliesslich gegen C**, damit die Schwelle
zwischen den Duellen vergleichbar bleibt. P1 ist PR-Kandidat wie jeder andere
Arm.

Faellt der Headline-Vergleich negativ aus, ist das ein Ergebnis und wird als
solches ins Ledger geschrieben. Die interne Rangliste laeuft trotzdem weiter.

## P1 - der Pflichtarm

```
tokens_full = concat([frame_tokens, global_tokens], axis=-1)   # (1374, 2048)
patches     = tokens_full[AGG_PATCH_START_IDX:]                # (1369, 2048)

P1 = concat([ tokens_full[AGG_CAMERA_TOKEN_IDX],   # 2048, beide Kamera-Tokens
              patches.mean(axis=0),                # 2048
              patches.max(axis=0) ])               # 2048
                                                   # = 6144 float32, 24 KB/Step
```

Das ist wortwoertlich das Muster der Referenz, nur auf voller Kanalbreite. Der
Kamera-Token kommt damit aus beiden Haelften; die Register-Tokens bleiben wie
in der Referenz draussen (`AGG_PATCH_START_IDX = 1 + AGG_REGISTER_TOKENS`,
`src/adapters/global_tokens.py:24`).

Adapter-Name in `ADAPTERS`: `aggregator_pooled_full`.

**P1 ist VGGT-kostenneutral.** Beide Token-Haelften fallen im Extractor
ohnehin an, das Pooling ist ein mean/max ueber 1369 Zeilen, und die
Replay-Zeile waechst von 12 auf 24 KB. Weicht `ms/Step` von P1 spuerbar von
Cs 66.8 ab, ist das ein Befund und gehoert ins Ledger.

## Startpunkte fuer die weiteren Arme

Vorschlaege, keine Vorschrift. Der Agent darf eigene Richtungen einschlagen,
solange die harte Bedingung aus `RULES.md` Abschnitt 3 gilt: **jeder gewertete
Arm enthaelt die Frame-Haelfte.**

| | Idee | Warum |
|---|---|---|
| P2 | `[cam_global, mean_g, max_g, mean_f]` = 4096 | isoliert, ob der Frame-Mittelwert allein schon traegt - die billige Halbstufe zu P1 |
| P3 | Pose-Delta: `cam_t - cam_0` als eigener Block | Jianyuans offener Faden aus dem r2-Ledger. Frame 0 ist permanenter Cache-Anker, also 0 ms VGGT-Kosten. Die metrische Form fuer `dtg`/`softspl`, nie ablatiert |
| P4 | gelerntes Attention-Pooling / Query-Token statt mean/max | mean/max sind die billigste, nicht die beste Reduktion |
| P5 | getrennte `AdapterField`s pro Block statt eines breiten Vektors | gibt jedem Block einen eigenen Encoder-Zweig statt einer 6144 -> 1024 Projektion |

## Referenz - die Latte, paarweise gewertet

Der bestaetigte Sieger von Duell 2: Arm C, `aggregator_pooled_b200k` mit
`prefill 1024`, `train_ratio 128`, `act_entropy 0.1`, 30 Minuten `--prod`.

| Referenz | Seed | Treffer | sr | spl | softspl | dtg | ms/Step | Episoden | N |
|---|---|---|---|---|---|---|---|---|---|
| **6060404** | 42 | 1 | 0.0227 | 0.0119 | 0.0866 | 6.379 | 66.8 | 44 | 21751 |
| **6061173** | 43 | 1 | 0.0244 | 0.0062 | 0.0539 | 4.975 | 69.1 | 41 | 20267 |

Quellen: `prototyp/duell-vggt-integration/2026-07-27-r2/runs/6060404-aggpool-b200k-tr128/metrics.csv`
und `.../6061173-aggpool-b200k-tr128-s43/metrics.csv`.

**Die Wertung ist paarweise pro Seed:** ein Seed-42-Lauf wird gegen 6060404
gewertet, der Seed-43-Bestaetigungslauf gegen 6061173. Der Grund steht in den
Zahlen: Cs beide Seeds unterscheiden sich in `softspl` um 60 % (0.0866 gegen
0.0539) und in `dtg` um 22 %. Gegen einen festen Seed-42-Vektor gewertet,
bekaeme der Bestaetigungslauf einen Grossteil davon geschenkt, und die
Bestaetigung waere wertlos.

## Erfolgskriterien - die Wertungsmatrix

Unveraendert aus Duell 2, nur auf Cs Zahlen gestellt. Jede Metrik geht als
**relative Aenderung zur Referenz** ein, richtungskorrigiert, gekappt,
gewichtet summiert:

```
rel = (wert - referenz) / |referenz|          fuer "hoeher ist besser"
rel = (referenz - wert) / |referenz|          fuer "niedriger ist besser"

Score = Summe ueber alle Metriken von (Gewicht * rel)
```

| Metrik | Richtung | Referenz s42 | Referenz s43 | Gewicht | Kappung |
|---|---|---|---|---|---|
| **Treffer** | hoch | 1 | 1 | **0.45** | +200 % |
| `softspl` | hoch | 0.0866 | 0.0539 | 0.15 | +/-100 % |
| `dtg` | niedrig | 6.379 | 4.975 | 0.15 | +/-100 % |
| `spl` | hoch | 0.0119 | 0.0062 | 0.10 | +/-100 % |
| `ms/Step` | niedrig | 66.8 | 69.1 | 0.10 | +/-100 % |
| Episoden | hoch | 44 | 41 | 0.05 | +/-100 % |
| (`sr`) | hoch | 0.0227 | 0.0244 | nur Bericht | - |

- **Score > 0** heisst besser und kommt so ins Ledger.
- **Score >= +0.10** ist die PR-Schwelle, als **Mittelwert beider Seeds**.
- Der Arm mit dem hoechsten Score wird auf **Seed 43** bestaetigt. Bei
  Gleichstand bekommen beide einen Bestaetigungslauf.
- Ein Job, der nicht durchkommt, ist `gescheitert`: kein Score, zaehlt nicht
  gegen die Variante.

**`ms/Step` hat in diesem Duell eine andere Rolle als in Duell 2.** Dort waren
die Trainingsknobs frei und Geschwindigkeit war der staerkste Hebel. Hier sind
sie eingefroren, Cs Tempo-Sockel ist praktisch ausgereizt - von 66.8 ms sind
laut Jianyuans Kostenzerlegung aus r2 rund 57 ms fixer VGGT-Sockel. Nach oben
ist wenig zu holen. `ms/Step` und Episoden wirken deshalb vor allem als
**Strafklausel gegen teure Encoder**: genau deswegen bleiben sie in der Matrix,
obwohl der Encoder freies Spielfeld ist.

## Ableseprotokoll

- **Treffer** = Anzahl der Zeilen mit `episode/success == 1` in der
  `metrics.csv`. Nicht aus `sr * Episoden` gerechnet.
- Alle uebrigen Werte = **letzter geloggter Wert innerhalb des Slots**
  (`metrics/spl`, `metrics/softspl`, `metrics/dtg`, `episode/count`,
  `perf/ms_per_step_interval`).
- Die `metrics.csv` ist im Langformat `step,metric,value` - eine Zeile pro
  Messpunkt, nicht eine Spalte pro Metrik.
- Fehlt `perf/ms_per_step_interval` in einem Lauf, gilt ersatzweise
  `Elapsed_s / N * 1000` mit Vermerk im Ledger.

## Was sich gegenueber Duell 2 geaendert hat, und warum

**Die Latte ist jetzt Cs eigene Config, nicht mehr 6057641.** Gegen die alte
Referenz gewinnt schon die reine Knob-Kombi aus Duell 2 (+0.09) ohne jeden
Adapter-Beitrag. Der Score wuerde dann nicht messen, was dieses Duell fragt.
Gegen C misst er den Adapter.

**Die Trainings-Hyperparameter sind eingefroren.** In Duell 2 waren sie frei,
und der Agent hat den Grossteil seines Budgets in Lotterie-Knobs gesteckt statt
in die Integration. Hier ist der Adapter die einzige freie Achse - plus der
Encoder, siehe unten.

**Der Encoder ist freies Spielfeld.** Der Agent darf eigene Encoder-Module
bauen und routen. Das ist eine bewusste Ausweitung: der reine Pooling-Raum
waere eng, und dieses Duell soll die Richtungen finden, die ein spaeteres
Duell dann verengen kann. `ms/Step` und Episoden bleiben deshalb in der Matrix.

**`buffer_capacity` ist frei, aber gedeckelt.** Mit freiem Encoder darf der
Agent die volle 1374-Token-Sequenz routen. Das ist erlaubt, hat aber einen
Preis, den er kennen muss: siehe `RULES.md` Abschnitt 3.

## Hypothese

Die Frame-Haelfte der Aggregator-Tokens traegt frame-lokale Information, die
die globale Haelfte nicht enthaelt, und der Kamera-Token der Frame-Haelfte
traegt die Blickrichtung. Beides sollte `dtg` und `softspl` frueher bewegen
als der rein globale Vektor - bei praktisch unveraenderten Kosten pro Step.

## Ehrlicher Vorbehalt

Die Referenz steht bei **einem** Treffer in 44 Episoden. Zwei Treffer statt
einem sind statistisch kaum von einem Muenzwurf zu unterscheiden. Der Score
wird real ueber `softspl` und `dtg` entschieden - die sind dicht, jede Episode
liefert einen Wert.

Rechne mit: **ohne Zweit-Treffer verlangt +0.10 ungefaehr `softspl` +40 %,
`dtg` -15 % und `spl` +30 % gleichzeitig.** Ein Duell, das ohne PR endet, ist
ein plausibler Ausgang. Es beantwortet trotzdem die Frage, um die es geht,
naemlich P1 gegen C.

Das Ledger von r2 beziffert ausserdem die **reine Ziehungsvarianz identischer
Configs auf ~+/-0.04 Score**. Ein einzelner Score unter ~0.05 Abstand ist kein
belastbares Ranking. Deshalb laeuft in Welle 2 ein Kontrolllauf von C auf
Seed 42 mit.

## Kontext zu Level 3

| | |
|---|---|
| Curriculum | `data/curriculum/level3_10houses_1goal.json`, 10 HM3D-Haeuser, nur `chair`, 74 997 Train-Episoden |
| Observation | ein RGB-Frame, 518x518 (VGGT), keine Goal-Conditioning |
| Episode | max. 500 Steps (`src/environments/habitat.py:49`) |
| Aktionen | 4 diskret, STOP ist ein No-op und beendet die Episode nicht |
| Success | geodaetische Distanz < `GOAL_RADIUS = 0.2` m (`habitat.py:36`) |
| Reward | `geodesic_delta` + `success_bonus 10.0` + `step_penalty -0.01` |

Zur Einordnung des Duell-Fensters: die CNN-Baseline 6056750 erreichte in
30 Minuten 59 322 Steps bei 2 % SR, also unter dem Random-Agenten (3.84 %). In
diesem Fenster lernt kein Arm die Aufgabe. Gemessen wird, welche Integration am
schnellsten anfaengt, sich in die richtige Richtung zu bewegen.
