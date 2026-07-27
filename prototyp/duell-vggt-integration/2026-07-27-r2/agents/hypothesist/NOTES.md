# Hypothesist - Kandidaten fuer die letzte Welle (Duell 2, r2)

Stand: 2026-07-27 ~13:35 UTC. Submit-Deadline letzte Welle: 14:22 UTC.
Max. 4 parallele GPU-Jobs; einer davon ist der Seed-43-Bestaetigungslauf des
Welle-1-Fuehrenden. Es bleiben also realistisch **2-3 Slots** fuer Neues.

## Vorab: eine Zahl, die die Rangfolge dreht

Ich habe Duell-1 #4 (6057871, kv200k + Lotterie-Knobs) unter der **r2-Matrix**
nachgerechnet (Quelle: `2026-07-27/runs/6057871-aggpool-lottery/metrics.csv`,
letzte geloggte Werte):

| Metrik | Ref 6057641 | 6057871 | rel | Gewicht | Beitrag |
|---|---|---|---|---|---|
| Treffer | 1 | 1 | 0 | 0.45 | 0 |
| softspl | 0.0605 | 0.0658 | +0.087 | 0.15 | +0.013 |
| dtg | 5.193 | 6.255 | -0.204 | 0.15 | -0.031 |
| spl | 0.0201 | 0.0034 | -0.831 | 0.10 | -0.083 |
| ms/Step | 134.1 | 71.4 | +0.467 | 0.10 | +0.047 |
| Episoden | 18 | 40 | +1.0 (Kappung) | 0.05 | +0.050 |
| **Score** | | | | | **~-0.004** |

**Erkenntnis:** Pure Speed ist unter der r2-Matrix fast score-neutral, solange
die Trefferzahl bei 1 bleibt. `spl` und `sr` sind Rolling-Means ueber
Episoden: mehr Episoden bei gleicher Trefferzahl **verduennen spl mechanisch**
(1/40 statt 1/18 -> spl 0.0034 statt 0.0201, Beitrag -0.083). Speed zahlt sich
nur ueber die Lotterie aus: bei geschaetzt p~1/29 Treffer/Episode (2 Treffer
in 58 pooled-Episoden, Duell 1) hat ein 40-Episoden-Lauf E[Treffer]~1.4 und
P(>=2)~0.40; ein zweiter Treffer allein ist +0.45. Speed-Arme sind also
**High-Variance-Tickets**, keine sicheren Punkte. Positiv: softspl ging mit
kv200k rauf (0.0658), das 6-Frame-Fenster hat die Annaeherungsqualitaet nicht
verschlechtert; dtg-Verschlechterung (6.25) kann Episoden-Mix-Rauschen sein.

Konsequenz fuers Ranking: (1) sichere kleine ms/Episoden-Beitraege mitnehmen,
(2) mindestens ein Arm, der dtg/softspl (zusammen 0.30) adressiert,
(3) Varianz durch parallele s42+s43-Submission des besten neuen Arms nutzbar
machen (Regeln verbieten das nicht; Abschnitt 7 verlangt nur dieselbe Config
mit `--env SEED=43`).

## Kandidaten (gerankt nach erwartetem Score-Gewinn pro Risiko)

### K1: `aggregator_pooled_b100k` + Lotterie-Knobs - Rang 1

- **(a) Aenderung:** Subklasse `AggregatorPooledBudget100kAdapter(AggregatorPooledAdapter)`
  in `src/adapters/global_tokens.py` (Muster = `AggregatorPooledBudget200kAdapter`,
  Zeile 148ff: nur `EXTRACTOR_KWARGS = {"compute_heads": False, "total_budget": 100_000}`).
  Registrieren in `src/adapters/__init__.py` (`ADAPTERS`), Run-Preset in
  `scripts/r2dreamer/_run_configs.py` (Muster Zeile 271ff), YAML
  `scripts/slurm/configs/duell2_l3_aggpool_b100k_lottery.yaml` (prefill 1024,
  train_ratio 256, act_entropy 0.1, `SEED: "42"` literal wegen verify.sh:95).
- **(b) Effekt:** Per-Block-Cache 100k/24=4166 statt 8333 Slots
  (`feature_extractor.py:368`): top_k-Eviction ueber ~halb so viele Kandidaten
  plus kleinere KV-Reads in der Global-Attention. Schaetzung 71.5 -> ~62-66
  ms/Step (ms-Beitrag +0.051..+0.055 statt +0.047), N ~21-23k, ~44-46 Episoden
  (Kappung, +0.05), entsprechend mehr Lotterie-Tickets. Fenster ~3 Frames statt
  ~6; softspl blieb schon beim Sprung 1.2M -> 200k stabil (0.0658 vs 0.0605),
  der naechste Halbierungsschritt ist die logische Sonde. Floor-Check:
  4166 > 1374 Anchor-Tokens, Guard `feature_extractor.py:375-379` passiert.
- **(c) Aufwand:** 10-15 min (Subklasse + 1 Registry-Zeile + Preset + YAML).
- **(d) Risiko:** Niedrig. Identischer Codepfad, nur eine Konstante.
  Restrisiko: dtg/softspl kippen beim 3-Frame-Fenster - genau das misst der Arm.
- **Empfehlung:** Als einziger neuer Arm **parallel s42 + s43** submitten
  (2 Slots), damit er im Zeitbudget bestaetigbar ist. Ein Welle-2-Arm ohne
  Seed 43 ist laut GOAL.md "ein Kandidat, kein Ergebnis".

### K2: Aggregator depth=12 (halber Tower) + Budget 100k - Rang 2

- **(a) Aenderung:** Neuer Kwarg `agg_depth: int = 24` in
  `JAXVGGTFeatureExtractor.__init__` (`src/vggt/jax/feature_extractor.py:289-300`),
  durchreichen nach `_init_modules` (Zeile 357-363) als
  `Aggregator(depth=agg_depth)`. Alles Nachgelagerte skaliert selbst mit:
  `_agg_depth` kommt aus `self._aggregator.depth`, Cache-Listen und
  `total_budget // depth` (Zeile 368) haengen daran. Tokens kommen aus
  `out_list[-1]` (Zeile 844), bei depth=12 also Layer 12. Adapter-Subklasse
  `AggregatorPooledDepth12Adapter` mit
  `EXTRACTOR_KWARGS = {"compute_heads": False, "total_budget": 100_000, "agg_depth": 12}`
  (100k haelt das Per-Block-Fenster bei 12 Bloecken auf dem 200k/24-Niveau).
- **(b) Effekt:** Groesster verbleibender ms-Hebel. VGGT-Anteil am
  71.5-ms-Step ist ~55-60 ms (134.1 gesamt, davon VGGT ~120 bei Default-Budget;
  kv200k nahm ~63 ms weg). Der Aggregator-Tower ist davon der Loewenanteil;
  halbe Tiefe spart geschaetzt 25-30 ms -> **~45-50 ms/Step**, N ~26-29k,
  ~55-60 Episoden. ms-Beitrag ~+0.065, Episoden +0.05 (Kappung), maximale
  Ticketzahl (E[Treffer]~1.9 bei p~1/29, P(>=2)~0.56).
- **(c) Aufwand:** 30-40 min inkl. Smoke (Extractor-Kwarg + Subklasse +
  Registry + Preset + YAML + `--smoke`-Startcheck).
- **(d) Risiko:** Mittel-hoch. (1) Layer-12-Features eines auf 24 Layer
  vortrainierten Towers sind kein trainiertes Readout - dtg/softspl (0.30
  Gewicht) koennen deutlich kippen, Downside bis -0.30. (2) Flax-`apply` mit
  ungenutzten `frame_blocks_12..23`-Params muss der Smoke bestaetigen.
  (3) Heads brauchen `intermediate_layer_idx` 4/11/17/23 (Docstring
  feature_extractor.py:77) - depth=12 ist **nur** fuer Token-Arme mit
  `compute_heads=False` gueltig; `write_point_cloud_ply` faellt aus (nur Dumps).
- **Einordnung:** Hoechste Decke, aber als unbestaetigter Welle-2-Arm eher
  Erkenntnis-Wert (offener Faden aus Duell 1 wird beantwortet) als PR-Weg.

### K3: `aggregator_pooled_b50k` - Boden-Sonde - Rang 3

- **(a) Aenderung:** Wie K1, `total_budget=50_000`. Per-Block 2083 Slots,
  Guard haelt (2083 > 1374), Fenster ~1.5 Frames - faktisch gedaechtnisloses
  Streaming.
- **(b) Effekt:** Weitere ~3-8 ms unter K1 (Schaetzung, Eviction-Sort und
  KV-Reads nochmal halbiert). Wissenschaftlich der wertvollste Punkt: haelt
  softspl auch bei ~1.5 Frames, ist der Streaming-Cache auf L3 in diesem
  Budget-Regime schlicht irrelevant - das wuerde die Architekturdiskussion der
  Thesis direkt informieren. Kippen dtg/softspl, ist der Boden zwischen 50k
  und 100k lokalisiert.
- **(c) Aufwand:** 5-10 min huckepack auf K1 (gleiche Dateien, zweite Subklasse).
- **(d) Risiko:** Mittel fuer den Score (Feature-Qualitaet), niedrig fuer die
  Implementierung. Nur starten, wenn nach Bestaetigung+K1(s42+s43) noch ein
  Slot frei ist.

### K4: `rgb_aggregator_pooled_b200k` - der dtg/softspl-Arm - Rang 4

- **(a) Aenderung:** Subklasse von `AggregatorPooledBudget200kAdapter` mit
  `WITH_RGB = True` (der Pfad existiert schon: `GlobalTokensAdapter.__call__`,
  `src/adapters/global_tokens.py:87-107`, haengt das Conv-Feld mit
  `decoder_target=True` an; `replay_image` downsampelt auf 64x64). Registry-Name
  `rgb_aggregator_pooled_b200k` (rgb_-Praefix-Konvention,
  `src/adapters/__init__.py:6-18`), Preset + YAML wie oben.
- **(b) Effekt:** Der pooled Arm hat **kein Decoder-Target und damit keinen
  Rekonstruktionsgradienten** - das dichteste Lernsignal fehlt ihm komplett.
  Ein Conv-Branch + Decoder kostet ~15-25 ms (CNN-only laeuft 30.3 ms/Step
  total, Ledger Duell 1) -> ~90-100 ms/Step, immer noch unter der Referenz.
  Zielt auf softspl/dtg/spl (0.40 Gewicht zusammen), wo Speed-Arme nichts
  liefern oder verduennen.
- **(c) Aufwand:** 15-20 min.
- **(d) Risiko:** Mittel. Gegen-Evidenz: beide Duell-1-Hybrid-Laeufe (RGB+VGGT)
  hatten 0 Treffer in 35 Episoden (Ledger, Erkenntnis 4; n=2, nicht
  signifikant). Dafuer-Evidenz: nur eine Rangfolge-Beobachtung, kein Mechanismus.
  In 30 min Training ist unklar, ob der Recon-Gradient schnell genug wirkt.

### K5: `pointmap_pose` + kv200k - Rang 5, kontingent auf Slot D

- **(a) Aenderung:** Subklasse von `PointMapPoseAdapter`
  (`src/adapters/pointmap_pose.py`) mit `total_budget=200_000` in den
  `EXTRACTOR_KWARGS` (Heads bleiben an, das Budget deckelt nur den
  Aggregator-Cache). Registry + Preset + YAML.
- **(b) Effekt:** Derselbe -60-ms-Mechanismus wie beim pooled Arm,
  uebertragen auf den Geometrie-Arm - der offene Faden "kv200k auf andere
  Arme" aus Duell 1. **Nur sinnvoll, wenn Welle-1 Slot D (6060405) bei
  dtg/softspl besser aussieht als die pooled-Arme**; sonst beschleunigt man
  einen unterlegenen Arm.
- **(c) Aufwand:** 10-15 min.
- **(d) Risiko:** Mechanisch niedrig, Wert komplett kontingent auf D-Ergebnis
  (~14:00 lesbar - knapp vor der Deadline).

### K6: prefill 512 - verworfen

512 gesparte Prefill-Steps sind bei 71.5 ms/Step ~37 s von 30 min (~2%
mehr Trainingsfenster). Kein eigener Slot wert; und in einen K1-Slot gemischt
zerstoert es die Vergleichbarkeit mit Welle 1 (A/C laufen mit prefill 1024).
Gerade Werte beibehalten (Logging-Bug-Paritaet, Duell-1-Erkenntnis 5).

## Empfohlene Slot-Belegung der letzten Welle (4 Slots)

1. Seed-43-Bestaetigung des Welle-1-Fuehrenden (Pflicht, Regeln Abschnitt 1/7).
2. K1 `b100k_lottery` mit SEED=42.
3. K1 `b100k_lottery` mit `--env SEED=43` **parallel** - einziger Weg, wie ein
   Welle-2-Arm noch die PR-Schwelle (Mittel beider Seeds >= +0.10) erreichen kann.
4. K2 depth=12 (Erkenntnis-Slot, beantwortet den Duell-1-Faden) - **oder** K5,
   falls Slot D bei dtg/softspl positiv ueberrascht, **oder** K4, falls der
   Orchestrator die 0.30-dtg/softspl-Wette hoeher gewichtet als die ms-Decke.

Begruendung der Reihenfolge: K1 ist der einzige Kandidat mit niedrigem Risiko
und realistischem Weg zur bestaetigten PR-Schwelle im Restzeitbudget. K2 hat
die hoechste Decke, kann aber nicht mehr bestaetigt werden und traegt das
Feature-Qualitaets-Risiko. K3/K4/K5 sind Erkenntnis- bzw. Kontingenz-Optionen.

## Quellen

- Referenz: `2026-07-27/runs/6057641-aggpool-p2048/metrics.csv` (letzte Werte:
  softspl 0.0605 @8904, dtg 5.193, spl 0.0201, ms 134.1 @9001, 18 Ep., 1 Treffer)
- kv200k-Lauf: `2026-07-27/runs/6057871-aggpool-lottery/metrics.csv` (softspl
  0.0658, dtg 6.255, spl 0.0034, ms 71.4 @19503, 40 Ep., 1 Treffer)
- Budget-Mechanik: `src/vggt/jax/feature_extractor.py:365-399` (uniform =
  total_budget//24, Anchor 1374, Guard 375-379)
- Token-Quelle: `src/vggt/jax/feature_extractor.py:842-845` (out_list[-1])
- Aggregator-Tiefe: `src/vggt/jax/aggregator.py:316` (depth=24, Flax-Attribut)
- Subklassen-Muster: `src/adapters/global_tokens.py:148-175`,
  `src/r2dreamer/AGENTS.md:44-56`
- Duell-1-Erkenntnisse/Faeden: `2026-07-27/LEDGER.md`
