# jianyuan-wang - Arbeitsnotizen Runde 2

Fortsetzung von `../../../2026-07-27/agents/jianyuan-wang/NOTES.md`.
Gleiche Vorbemerkung: Dossier liegt auf diesem Filesystem nicht (macOS-Pfad),
alles hier aus dem Code im Worktree. Das ist StreamVGGT, nicht VGGT - ich bin
nicht Autor davon, die Cache-Mechanik lese ich aus eurem JAX-Port ab.
Ich habe auf diesem Cluster nichts gemessen. "Schaetzung" steht dran, wo es eine ist.

Ein Zugestehen vorab: meine Runde-1-Schaetzung fuer `total_budget=200_000` war
-20 bis -30 ms. Gemessen wurden -62.6 ms (134.1 -> 71.5). Ich habe den
Cache-Term um Faktor ~2.5 unterschaetzt. Das aendert die Rechnung unten, und
zwar gegen weitere Budget-Kuerzungen - siehe Frage 1.

---

## Die Budget-Arithmetik, exakt

`feature_extractor.py:365-372`:

```
anchor_tokens = 5 + (518 // 14) ** 2          # 1374
uniform       = max(total_budget // 24, 1374)
cache_max     = uniform + 1374                # padded MAX pro Block
max_budget    = uniform                       # = budgets_static default
```

Eviction (`attention.py:94`, `attention.py:131`): `n_keep = cache_budget - 1374`
aus einem Kandidatenbereich von `MAX - 1374 = uniform` Slots.
Kontextfenster in Frames = `uniform / 1374` (plus Frame 0 als Anchor).

| total_budget | uniform | MAX | n_keep | Kontext (Frames) |
|---|---|---|---|---|
| 1 200 000 (default) | 50 000 | 51 374 | 48 626 | 36.4 |
| **200 000 (r1)** | 8 333 | 9 707 | 6 959 | **6.1** |
| 100 000 | 4 166 | 5 540 | 2 792 | 3.0 |
| 50 000 | 2 083 | 3 457 | **709** | 1.5 |
| 33 000 | 1 375 | 2 749 | **1** | 0.001 |

**Harter Boden: `total_budget > 32 976`.** Bei `<= 32 976` wird `uniform` auf
1374 geklemmt, `max_budget <= anchor_tokens`, und `_finalize_static_budget_override`
(`feature_extractor.py:376-379`) wirft im Konstruktor. Kein stiller Fallback.

**Weicher Boden: 100k.** Bei 50k ist `n_keep = 709 < 1374`, d.h. der Cache kann
nicht mehr *einen* vollstaendigen Vorframe halten - nach der Eviction bleiben
Frame 0 (Anchor) plus ein knapp halber Frame Residuum. Das ist nicht mehr "kurzes
Fenster", das ist "kein Fenster". Bei 33k bleibt genau 1 Kandidaten-Slot.

---

## Frage 1: bringt 100k/50k noch was? -> Nein, der Hebel ist bei 200k fast ausgereizt

Der Cache-Pfad ist **gepolstert**: `top_k`, die L2-Norm und der Score-Mean laufen
immer ueber den vollen Kandidatenbereich `MAX - 1374`, unabhaengig von `valid_len`
(`attention.py:105-133`). Nur cuDNN ueberspringt ungueltige Slots
(`attention.py:321-332`). Kosten also ~linear in `MAX`, und `MAX` ~linear in
`total_budget`.

Lineares Modell aus dem einen gemessenen Paar (1.2M -> 134.1, 200k -> 71.5):

```
a = 62.6 ms / (51 374 - 9 707) Slots = 1.50e-3 ms/Slot
Cache-Term bei 200k = 9 707 * 1.50e-3 = 14.6 ms
Fixer Sockel        = 71.5 - 14.6     = 56.9 ms   (DINOv2 + frame-Bloecke + Replay + Env)
```

Extrapolation:

| total_budget | Cache-Term | ms/Step (Schaetzung) | Gewinn ggue. 200k | Kontext |
|---|---|---|---|---|
| 200 000 | 14.6 | 71.5 (gemessen) | - | 6.1 Frames |
| 100 000 | 8.3 | **~65** | **-6** | 3.0 Frames |
| 50 000 | 5.2 | **~62** | **-9** | 1.5 Frames (degeneriert) |
| 33 000 | 4.1 | ~61 | -10 | tot |

Das sind **Obergrenzen**: `top_k` ist super-linear, der reale Gewinn liegt
darunter. Und der Sockel von ~57 ms ist die Asymptote - unter die kommt kein
Budget der Welt.

**Befund:** 200k hat schon 83 % des theoretisch verfuegbaren Cache-Gewinns
eingesammelt (62.6 von max ~76 ms). Die restlichen 17 % kosten das Kontextfenster
vollstaendig. **Ich wuerde keinen Wave-Slot auf 100k oder 50k setzen.** Bei einer
Wertung, in der Speed 0.10 wiegt, sind 6 ms von 71.5 (8 %) kein Hebel mehr,
und 50k riskiert genau die Annaeherungsmetriken (dtg/softspl, 0.30), die von
Kontext leben.

Falls du trotzdem ein billiges Los willst: **100k, nicht 50k.** Eine Zeile,
und "Frame-0-Anchor plus 3 Frames" ist noch eine erzaehlbare Konfiguration.
50k ist keine.

---

## Frage 2: `agg_depth=12` - der einzige verbleibende echte Hebel

### Muss `total_budget` mitskalieren? Ja, sonst hebt es sich auf.

`uniform = total_budget // self._agg_depth` (`feature_extractor.py:368`).
Bei `depth=12` und `total_budget=200_000` wird `uniform = 16 666`, `MAX = 18 040`.
Pro Block doppelt so teuer, halb so viele Bloecke -> **Cache-Term unveraendert**
(14.6 ms), Kontextfenster verdoppelt sich auf 12 Frames. Also nicht falsch, aber
der Cache-Gewinn ist Null.

Willst du beides, setze `total_budget=100_000` bei `depth=12`:
`uniform = 8 333`, `MAX = 9 707` - **identische Cache-Geometrie zum heutigen
200k-Arm, gleiches ~6-Frame-Fenster, halb so viele Bloecke.** Das ist die Variante,
die ich bauen wuerde: der Kontext-Trade ist gegenueber r1 unveraendert, nur die
Tiefe aendert sich. Eine Variable pro Lauf, sauber attribuierbar.

Schaetzung: Cache-Term 14.6 -> 7.3 ms. Dazu fallen 12 frame-Bloecke aus dem
Sockel (24 DINOv2 + 24 frame-Bloecke -> 24 + 12, bei gleicher Dim 1024/16 Heads
und 1374 Tokens). Wenn der Transformer-Anteil des Sockels ~35 der 56.9 ms ist
(Schaetzung), sind das weitere ~9 ms. **~55-60 ms/Step, also -12 bis -16 ms
gegenueber 71.5.** Rund doppelt bis dreifach so viel wie alles, was das Budget
noch hergibt.

### Wie riskant fuer die pooled-Token-Qualitaet?

Weniger riskant als in r1 notiert, und der Grund ist der Konsument:

- Bei `compute_heads=False` liest der Extractor `out_list[-1]`
  (`feature_extractor.py:842-845`), bei `depth=12` also Block 11.
- Block 11 ist einer der vier DPT-Readout-Punkte (4/11/17/23,
  `feature_extractor.py:77`) - die Architektur selbst liest dort aus. Kein
  willkuerlicher Abbruch.
- Der Abnehmer ist ein **frisch initialisierter MLP-Branch**, kein vortrainierter
  Head. Er hat keine Erwartung an "letzte Schicht". OOD ist hier eine Frage der
  Informativitaet, nicht der Kalibrierung. Bei einem vortrainierten Point-Head
  waere ich vorsichtig; hier bin ich es weniger.

Was ich **nicht** weiss: ob die geometrische Information in Block 11 fuer diese
Aufgabe reicht. Nie ablatiert, von niemandem. Zwei-Minuten-Vorpruefung, entkoppelt
vom Reward: `write_point_cloud_ply` (`feature_extractor.py:991`) und die Wolke
anschauen. Wenn sie zerfaellt, weisst du es vor dem Slot statt danach. (Achtung:
DPT liest `out_list` an vier Indizes - bei depth=12 muss der Head-Pfad dafuer
angepasst werden. Fuer den *gewerteten* Lauf ist der Head aus, das betrifft nur
den Vorcheck.)

### Exakte Codestellen (3 Edits, ~6 Zeilen)

Alle Listen-Laengen haengen an `self._agg_depth = self._aggregator.depth`
(`feature_extractor.py:362`) - `_warmup`, die Cache-Entry-Schleifen, die
`budgets_static`-Validierung und die Snapshot-Pfade folgen automatisch. Es ist
genau *eine* Stelle, an der die Tiefe entsteht.

1. `feature_extractor.py:294` ff., `__init__`-Signatur - neues Kwarg hinter
   `compute_heads`:
   ```python
   agg_depth: int | None = None,
   ```
2. `feature_extractor.py:349-354`, `_configure_runtime_options` - Parameter
   durchreichen und speichern (wird vor `_init_modules()` aufgerufen, Reihenfolge
   passt):
   ```python
   self._agg_depth_override = agg_depth
   ```
3. `feature_extractor.py:358`, `_init_modules`:
   ```python
   self._aggregator = (
       Aggregator() if self._agg_depth_override is None
       else Aggregator(depth=self._agg_depth_override)
   )
   ```

Adapter-Seite, neue Subklasse in `src/adapters/global_tokens.py` neben
`AggregatorPooledBudget200kAdapter:160`:

```python
EXTRACTOR_KWARGS: dict[str, object] = {
    "compute_heads": False,
    "total_budget": 100_000,   # 100k / 12 = 8333 = identisch zum 200k/24-Arm
    "agg_depth": 12,
}
```

**Das eine Risiko, das du in den ersten 2 Minuten pruefen musst:** die Bloecke
sind inline benannt (`aggregator.py:442,455`: `frame_blocks_{b}` /
`global_blocks_{b}`). Bei `depth=12` instanziiert Flax nur 0-11, der geladene
Parameterbaum enthaelt aber 0-23. Flax `apply` sollte ungenutzte Eintraege
tolerieren (Lookup per Name), aber **ich bin da nicht sicher** und wuerde es nicht
im SLURM-Job herausfinden. Sofort-Test:
`VGGTFeatureExtractor(compute_heads=False, total_budget=100_000, agg_depth=12)`
konstruieren - `_warmup()` laeuft im Konstruktor, ein Shape-/Key-Fehler
schlaegt dort zu. Falls es doch wirft: `self._agg_params` vor dem ersten `apply`
auf `frame_blocks_0..11` / `global_blocks_0..11` filtern, das ist eine
Dict-Comprehension.

---

## Frage 3: kombinieren? -> Nein. Ein Hebel pro Lauf, und der Hebel heisst depth.

Die beiden Hebel sind **nicht orthogonal** - sie greifen ueber
`uniform = total_budget // agg_depth` an derselben Zahl an. "100k + depth 12"
ist nicht Budget-Kuerzung *plus* Tiefen-Kuerzung, es ist *nur* Tiefen-Kuerzung
bei konstantem Kontextfenster. Genau deshalb ist es die richtige Variante: als
Kombination gelesen waere sie nicht attribuierbar, als Tiefenaenderung bei
konstanter Cache-Geometrie ist sie es.

Echtes Kombinieren (z.B. 50k + depth 12 -> uniform 4166, 3 Frames) holt die
~4 ms von Frage 1 zusaetzlich und macht den Lauf uninterpretierbar. Bei 0.10
Gewicht auf ms/Step nicht den Preis wert.

---

## Frage 4: dtg/softspl - was billig in den pooled Vektor kann

Aktueller Vektor, `global_tokens.py:147-157`: `[camera_token, patch_mean, patch_max]`
der global-Haelfte, 3 x 1024 = 3072 fp32. Also: **Pose ist implizit drin** - Token 0
ist der Kamera-Token, das ist der Pose-Traeger des Aggregators. Was fehlt, ist
*metrische* Translation.

**Der teure Weg, den ich nicht empfehle:** `compute_heads=True` und
`features.camera_pose` anhaengen (so macht es `pointmap_pose.py:59`). Aber
`compute_heads` schaltet Kamera- **und** Point-Head gemeinsam
(`feature_extractor.py:857-862`, `_run_heads` ruft beide), und der DPT-Head ist
der teure. Das Flag zu splitten ist machbar, beruehrt aber `_run_heads`,
`HeadOutputs`, `_build_extract_output`, den Kamera-Cache und `_warmup`. Nicht in
40 Minuten vor einer Deadline.

**Der billige Weg (Schaetzung, nie ablatiert - Ableitung aus Prinzipien, keine
Messung):** der Kamera-Token relativ zum Episodenstart, rein adapterseitig,
null VGGT-Kosten.

`ObservationFrame.is_first` existiert (`observation.py:20`), der Adapter kann den
Frame-0-Kamera-Token also selbst halten: bei `is_first` `self._cam0 = tokens[0]`
setzen, sonst `tokens[0] - self._cam0` als vierten 1024er-Block anhaengen.
3072 -> 4096.

Achtung auf die Struktur: `_tokens(features)` sieht den Frame nicht. Entweder
`__call__` ueberschreiben (`global_tokens.py:81-102`) oder `_cam0` in `__call__`
vor dem `_tokens`-Aufruf setzen. Ich wuerde `__call__` ueberschreiben, das ist
ehrlicher als ein Seiteneffekt.

Begruendung: das Frame-0-Anchor-Verhalten ist hier ein Geschenk. Frame 0 liegt
permanent im Cache (`n_anchor = 1374`), der Weltrahmen ist ueber die ganze
Episode der Episodenstart. `cam_t - cam_0` ist damit **Verschiebung seit
Episodenstart in einem stabilen Rahmen**, linear verfuegbar statt als Differenz
zweier absoluter Codes, die der MLP erst lernen muesste. dtg und softspl sind
genau Verschiebungsmetriken. +4 KB/Step Replay, 0 ms VGGT.

Der Vorbehalt, den ich lieber selbst sage: der Kamera-Token ist ein *latenter*
Pose-Traeger, nicht metrisch, und ob eine Differenz im Latentraum sich wie eine
Translation verhaelt, weiss ich nicht. Bei einem 1024-dim Token, das von
Blockschicht zu Blockschicht neu geschrieben wird, ist das eine Vermutung. Wenn
du nur einen Slot fuer eine Feature-Aenderung hast, ist das ein Los, kein Plan.

---

## Empfehlung, geordnet

| Prio | Variante | Parameter | ms/Step (Schaetzung) | Aufwand | Risiko |
|---|---|---|---|---|---|
| **1** | `aggregator_pooled_d12b100k` | `total_budget=100_000`, `agg_depth=12`, `compute_heads=False` | **~55-60** (-12 bis -16) | 3 Edits Extractor + 1 Subklasse, ~15 Zeilen | Flax-Paramtree-Check (2 min), Block-11-Features nie ablatiert |
| **2** | `aggregator_pooled_relpose` (auf 200k-Basis) | wie r1 + 4. Block `cam_t - cam_0` | 71.5 (unveraendert) | ~15 Zeilen, nur Adapter | reine Ableitung, kein Rueckhalt in Messungen |
| - | 100k allein | `total_budget=100_000` | ~65 (-6) | 1 Zeile | Kontext 6 -> 3 Frames |
| - | 50k / 33k | - | ~62 / ~61 | 1 Zeile | **degeneriert, nicht bauen** |

Variante 1 ist der Speed-Lauf, Variante 2 der Annaeherungs-Lauf. Sie beruehren
disjunkte Codepfade (Extractor-Tiefe vs. Adapter-Readout), blockieren sich also
nicht - aber ich wuerde sie **nicht im selben Lauf** kombinieren, sonst ist ein
Treffer nicht zuzuordnen.

Ein Vorbehalt, den ich wie in r1 lieber vorwegnehme: wenn Variante 1 gewinnt,
hat sie mit **mehr Steps** gewonnen, nicht mit besserer 3D-Integration. Eine
Zeile bei gleicher Step-Zahl N gegen den 200k/24-Arm ist der einzige Vergleich,
der die Tiefenfrage tatsaechlich beantwortet. Wuerdest du die mitlaufen lassen?

Und noch eine Bitte, die auch in r1 offen blieb: `benchmark_streaming.py` hat
`--jax-static-budgets`. Ein srun ueber 60 Frames mit `depth=24/budget=200k` gegen
`depth=12/budget=100k` gibt die ms/Frame-Kurve in ~5 Minuten und ersetzt meine
Sockel-Schaetzung von 56.9 ms durch eine Zahl. Kannst du das vor dem Slot laufen
lassen? Meine Extrapolation hat in r1 um Faktor 2.5 danebengelegen.
