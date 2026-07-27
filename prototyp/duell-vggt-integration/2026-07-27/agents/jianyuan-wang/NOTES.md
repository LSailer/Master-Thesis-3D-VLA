# jianyuan-wang - Arbeitsnotizen

Alles, was dieser Agent probiert, findet und verwirft. Roh und fortlaufend.
Kein Politur-Anspruch, aber jede Zahl mit Quelle (SLURM-Job, Run-Dir, Datei:Zeile).

---

## 2026-07-27 - Code-Lesung: wo ms/Step steckt

Vorbemerkung zur Rolle: mein Dossier liegt auf diesem Filesystem nicht vor
(`/Users/lucamac/...` ist ein anderer Rechner). Alles hier stammt aus dem Code
im Worktree plus den Memory-Notizen des Repos. Wo ich schaetze, steht
"Schaetzung" dran. Ich habe auf diesem Cluster nichts gemessen.

Zweite Vorbemerkung: das hier ist StreamVGGT, nicht VGGT. Ich bin nicht Autor
von StreamVGGT. Die Aussagen unten zur KV-Cache-Mechanik lese ich aus eurem
JAX-Port ab, nicht aus eigener Erfahrung mit dem Modell.

### Ausgangsfrage neu gestellt

Nicht "welcher Arm ist der beste 3D-Prior", sondern: **was kann weg, ohne dass
die 3D-Information verschwindet?** Bei 30 Minuten Walltime ist jede ms/Step
direkt in Episoden konvertierbar. Also erst streichen, dann bauen.

Der grosse Block ist der VGGT-Forward mit ~120-132 ms. Der zerfaellt in drei
Teile, und nur einer davon waechst mit der Episodenlaenge:

1. DINOv2-Backbone, 24 Bloecke, 1374 Tokens (`backbone.py:156-170`)
2. Aggregator, 24 x (frame-Block + global-Block) (`aggregator.py:431-510`)
3. KV-Cache-Attention im global-Block plus Eviction
   (`attention.py:249-357`, `attention.py:79-151`)

Punkt 3 waechst. Erster Kandidat.

---

### Befund 1 (wichtigster): der Cache-Budget steht auf 1.2M und wurde nie variiert

`feature_extractor.py:55`

```
_DEFAULT_TOTAL_BUDGET = 1_200_000
```

`feature_extractor.py:365-370` rechnet daraus:

```
self._anchor_tokens = 5 + (518 // 14) ** 2           # 1374
uniform = max(total_budget // self._agg_depth, ...)  # 1_200_000 // 24 = 50_000
self._cache_max = uniform + self._anchor_tokens      # 51_374
```

Konkret heisst das:

- Pro global-Block ein gepolsterter KV-Cache mit **51 374 Slots**. Ueber 24
  Bloecke, k und v, 16 Heads, 64 dim, bf16: `24*2*16*51374*64*2 B` = **~5.05 GB**
  nur Cache (Schaetzung aus den Shapes in `feature_extractor.py:440-452`).
- Jeder Frame haengt 1374 Tokens an. Der Cache ist nach `50000/1374 = 36.4`
  Frames voll. Eine L3-Episode laeuft bis 500 Steps (`habitat.py:49`).
  **Rund 93 % aller Steps einer Episode laufen im gesaettigten Regime.**
- Im gesaettigten Regime laeuft pro Frame und Block eine Eviction:
  `attention.py:131`, `jax.lax.top_k(-scores_masked, n_keep)` mit
  `n_keep = 50000 - 1374 = 48 626` aus ~50 000 Kandidaten, pro Head, 16 Heads,
  24 Bloecke. Ein top_k mit k nahe n ist auf der GPU praktisch ein voller Sort.
  Jeden Env-Step, 24 mal.
- Dazu die Attention: 1374 Queries gegen bis zu 50 000 Keys, 16 Heads, 24
  Bloecke. cuDNN bekommt `key_value_seq_lengths` (`attention.py:321-332`),
  ueberspringt also ungueltige Slots - die gueltigen sind aber eben 50 000.

**Der Hebel:** `total_budget` ist Konstruktor-Argument, und `EXTRACTOR_KWARGS`
geht 1:1 in den Konstruktor (`src/main.py:205`). Eine Zeile in einer
Adapter-Klasse, kein Eingriff in den Modellcode:

```python
# src/adapters/global_tokens.py:54
EXTRACTOR_KWARGS: dict[str, object] = {"compute_heads": False, "total_budget": 200_000}
```

200 000 ist nicht meine Erfindung - euer eigenes Benchmark-Skript nimmt genau
den Wert als Default (`benchmark_streaming.py:178`). Ergibt 8333 Tokens pro
Block, also ca. **6 Frames Kontext**, Cache-Allokation ~0.85 GB statt 5.05 GB.

Geschaetzter Effekt: Attention-Term 6x kleiner, Eviction-Sort 6x kleiner. Wenn
beide zusammen 30-45 ms der 132 ms ausmachen (Schaetzung), landet man bei
**95-110 ms/Step**, also **-20 bis -30 ms/Step**. In 30 Minuten etwa
1500-3000 zusaetzliche Steps.

**Was es kostet:** genau das, was der global-tokens-Arm messen soll - den
akkumulierten Hauskontext. 6 Frames statt 36. Das ist kein Nebeneffekt, das ist
der Trade-off, und er gehoert ins Ledger. Bei 3000-5000 echten
Trainingsschritten halte ich 36 Frames Kontext ohnehin fuer nicht auszahlbar,
aber das ist eine Meinung, keine Messung.

**Wie man es isoliert misst, ohne einen 30-Minuten-Slot zu verbrennen:**
`src/vggt/jax/benchmark_streaming.py` existiert und hat `--jax-static-budgets`.
Ein srun mit zwei Budgets ueber 60 Frames gibt die ms/Frame-Kurve direkt.
Wuerdest du das laufen lassen, bevor ein Budget in einen gewerteten Lauf geht?
Die Zahl ist mehr wert als meine Schaetzung.

**Nebenbefund:** `budgets_static` ist ebenfalls Konstruktor-Argument
(`feature_extractor.py:297`). Ein Tupel mit kleinen Budgets unten und grossen
oben ist damit ohne Codeaenderung testbar. Ich habe das nie ablatiert, sage ich
dazu.

---

### Befund 2: flacherer Aggregator - groesserer Gewinn, echter Codeeingriff

Bei `compute_heads=False` liest der Extractor **nur** die letzte Schicht:

`feature_extractor.py:842-845`

```
def _aggregator_full_tokens(self, out_list):
    final_tokens = out_list[-1]
```

Ein "intermediate layer output" bringt also **null ms**, solange Schicht 23
gerechnet wird. Gewinn gibt es nur, wenn der Turm wirklich frueher aufhoert.

Machbar: `Aggregator.depth` ist Modulfeld (`aggregator.py:316`), die Schleife
ist `for b in range(self.depth)` (`aggregator.py:431`), die Cache-Tiefe wird
davon abgeleitet (`feature_extractor.py:362`). `Aggregator(depth=12)` in
`feature_extractor.py:358` waere die Aenderung. Flax zieht aus dem
Parameterbaum nur, was es instanziiert; ungenutzte Blockparameter stoeren nicht.

Warum 12: der DPT-Head liest im Original die Schichten **4/11/17/23**
(dokumentiert in `feature_extractor.py:77`). Schicht 11 ist ein Readout-Punkt,
den die Architektur selbst benutzt - kein willkuerlicher Abbruch. Das ist das
beste verfuegbare Argument dafuer, dass die Features dort noch geometrisch
tragen.

Geschaetzter Effekt: von 72 Transformer-Bloecken (24 DINOv2 + 24 frame + 24
global) fallen 24 weg, inklusive der 12 teuersten Cache-Bloecke. **Schaetzung
-30 bis -40 % des VGGT-Forwards**, also ~132 -> ~85-92 ms.

Fallstricke aus dem Lesen:

- `_configure_aggregator_cache` teilt `total_budget // self._agg_depth`
  (`feature_extractor.py:368`). Bei depth=12 verdoppelt sich das Budget pro
  Block still. `total_budget` muss mitskaliert werden, sonst frisst Befund 2
  den Gewinn aus Befund 1 wieder auf.
- `_warmup` und die Snapshot-Pfade (`save_cache`/`load_cache`) haengen an der
  Listenlaenge. Sollte durchlaufen, ist aber ungetestet.
- Die Gewichte sind fuer 24 Schichten trainiert. Schicht 11 ohne den Rest des
  Turms auszulesen ist **out of distribution**. Wie stark, kann ich nicht sagen.

**Der Test, den ich vorher machen wuerde und der hier fehlt:** die Punktwolke
anschauen. `write_point_cloud_ply` (`feature_extractor.py:991`) ist da. Wenn die
Wolke bei depth=12 zerfaellt, ist die Antwort in zwei Minuten da statt nach 30.
Das ist eine Repraesentationspruefung entkoppelt vom Reward - die hat der Aufbau
sonst nirgends.

Fuer das Duell: **Befund 2 ist der groessere Gewinn, Befund 1 die einzige
Aenderung ohne Codeeingriff.** Reihenfolge entsprechend.

---

### Befund 3: Aufloesung senken ist im Duell-Budget tot

`backbone.py:96-103` wirft explizit:

```
f"DinoV2Backbone is fixed at {self.img_size}x{self.img_size}; "
f"got {H}x{W}. Reintroducing bicubic-AA pos_embed interpolation "
"is a follow-up."
```

`backbone.py:145-147` addiert `pos_embed` ungefiltert, weil 518 den Schnellpfad
trifft. 322/14 = 23 und 238/14 = 17 gehen glatt auf, aber die
Positions-Embeddings muessten bikubisch interpoliert werden, und daran haengt
mehr als eine Zeile:

- `_IMG_SIZE`, `_PATCH_GRID` (`feature_extractor.py:47-48`) und die daraus
  abgeleiteten `VGGT_IMAGE_SIZE` / `VGGT_PATCH_GRID`, die die Adapter
  konsumieren (`global_tokens.py:16`, `pointmap_pose.py:45`)
- die RoPE-Tabellen (`aggregator.py:413-424`)
- `habitat.py:48 obs_shape = (518, 518, 3)`, also das Rendering
- die Pooling-Faktoren 518 -> 37 (`pointmap.py:27`)
- Replay-Shapes, also inkompatible Checkpoints

Der Gewinn waere gross - 322 gaebe 529 statt 1369 Patches, dense Kosten ~2.6x
runter, Attention quadratisch - aber das ist ein halber Tag plus eine
Qualitaetsfrage, die ich nicht beantworten kann, weil das Modell auf 518
trainiert wurde. **Im Duell nicht anfassen.** Als Faden danach: groesster
verfuegbarer Speed-Hebel ueberhaupt.

---

### Befund 4: `habitat-l3-aggregator-pooled` existiert bereits

Die Frage "ist Aggregator-MLP auf L3 portierbar mit nur RUN_CONFIGS-Eintrag +
YAML?" hat eine kuerzere Antwort als erwartet: der Eintrag ist schon da.

`scripts/r2dreamer/_run_configs.py:259-269`, `adapter="aggregator_pooled"`,
`curriculum="L3"`. Nichts zu portieren.

Warum der Arm schnell ist, aus `src/adapters/global_tokens.py:127-157`:

| | global-tokens | aggregator-pooled |
|---|---|---|
| Replay-Zeile | 1374 x 1024 fp16 = 2.8 MB | 3072 fp32 = 12 KB |
| Branches | Conv (Bild) + Transformer | nur MLP |
| Decoder-Target | ja (Bild) | keins (`WITH_RGB = False`) |
| VGGT-Heads | aus | aus |

Der Gewinn kommt **nicht** aus VGGT - der Forward ist bei beiden identisch,
beide setzen `compute_heads=False` (`global_tokens.py:54`). Er kommt aus dem
Replay (Faktor ~230 pro Zeile; `ReplayBuffer.sample` ist laut Memory-Notiz mit
~59 ms amortisiert der zweitgroesste Posten) und aus dem entfallenen
Transformer-Branch plus Decoder.

Deshalb im Duell attraktiv **und** riskant, und das Risiko ist nicht meines: der
Arm hat kein Rekonstruktionsziel. Ob ein DreamerV3-World-Model bei ~3000-5000
echten Updates ohne Decoder-Target eine brauchbare Repraesentation aufbaut, ist
eine RSSM-Frage - Danijar, nicht ich. Ich kann nur sagen: die 3072 Zahlen sind
`[camera token, patch mean, patch max]` der global-Haelfte, und ein Mean-Pool
ueber 1369 Patch-Tokens ist eine sehr grobe Zusammenfassung. Was davon noch
geometrisch ist statt semantisch, weiss ich nicht.

---

### Geordnete Empfehlung

| # | Aenderung | Codestelle | ms/Step (Schaetzung) | Aufwand | Risiko |
|---|---|---|---|---|---|
| 1 | `total_budget=200_000` in `EXTRACTOR_KWARGS` | `src/adapters/global_tokens.py:54` | -20 bis -30 | 1 Zeile | Kontext 36 -> 6 Frames |
| 2 | Budget isoliert messen statt schaetzen | `src/vggt/jax/benchmark_streaming.py` (existiert) | 0 | 1 srun, ~5 min | keins |
| 3 | `habitat-l3-aggregator-pooled` gewertet starten | `_run_configs.py:259` | Basis ~94 ms | 0 | kein Decoder-Target |
| 4 | `Aggregator(depth=12)`, Budget mitskalieren | `feature_extractor.py:358` + `:368` | -30 bis -40 % | ~20 Zeilen | OOD-Readout, vorher PLY pruefen |
| 5 | Aufloesung 518 -> 322 | `backbone.py:96` u.v.m. | -50 % oder mehr | halber Tag | nicht im Duell |

**Top-Empfehlung fuer den naechsten gewerteten Lauf:**
`habitat-l3-aggregator-pooled` mit `total_budget=200_000`. Zwei Zeilen, greifen
an unabhaengigen Stellen (Replay bzw. Cache-Attention), Schaetzung 95-110
ms/Step statt 171-219. Das verdoppelt bis verdreifacht die Step-Zahl in 30
Minuten.

Ein Vorbehalt, den ich lieber selbst ausspreche als spaeter zu hoeren: wenn der
Arm gewinnt, hat er mit **mehr Steps** gewonnen, nicht mit besserer
3D-Integration. Der Vergleich bei gleicher Step-Zahl N (`GOAL.md:41`) faengt das
gegen die CNN-Baseline ab, **nicht** gegen die anderen 3D-Arme. Wenn das Duell
am Ende sagen soll "diese Integration ist besser", braucht es zusaetzlich eine
Zeile bei gleicher Step-Zahl gegen einen anderen 3D-Arm.

---

### Was ich nicht beantworten kann

- Frage 3, "welcher Arm traegt am meisten lernbare Information pro Sekunde":
  keine Messung, und die vorhandenen Zahlen
  (`docs/06-notes/wandb_2m_runtime_analysis.html`) sind Durchsatz, nicht
  Informationsgehalt. Ich kann nach Kosten ordnen, nicht nach Nutzen. Das
  muesste man probieren.
- Ob 6 Frames Cache-Kontext auf L3 reichen. Haengt daran, wie schnell sich der
  Agent dreht - eine Habitat-Frage.
- Wie stark Features aus Schicht 11 degradieren. Punktwolke anschauen.

