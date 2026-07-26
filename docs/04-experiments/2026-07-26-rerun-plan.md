# Rerun-Plan nach der Adapter-Routing-Migration (2026-07-26)

Stand: HEAD `9429032` (Merge PR #212, 14:16). Erstellt aus einem Review durch die
drei Betreuer-Personas (Braun, Ropinski, Wurzberger); jede tragende Behauptung
unten ist am Code oder an W&B nachgeprüft und mit Fundstelle belegt.

**Kurzfassung:** Der bf16-Change ist *nicht* der Grund für Reruns — er trifft
genau zwei Arme und lässt sich offline abklären. Die eigentlichen Befunde sind,
dass (1) kein einziger Lauf vom aktuellen Code stammt, (2) L2/L3 der CNN-Baseline
nie gelaufen sind, (3) die 3D-Arme an der Walltime sterben statt zu crashen, und
(4) drei Dinge vor dem nächsten GPU-Tag entschieden werden müssen, weil sie sonst
eine ganze Seed-Runde entwerten.

---

## 1. Verifizierte Faktenlage

### Code-Änderungen von heute

| Commit | Was | Trainingspfad betroffen? |
|---|---|---|
| PR #212 (`9429032`) | Observation-Pfad auf Adapter → Routed Fields → Encoder Branches umgebaut, 198 Dateien, +6.5k/−18.6k | **Ja**, alle Arme |
| `31b8d35` | `RoutedCompositeEncoder.compute_dtype` Default `jnp.float32` → `None` | **Ja**, aber nur PointNet-Arme |
| `fd9a386` | `compute_dtype`/`full_bf16` in `_ARCH_FIELDS`, `src/r2dreamer/launch/evaluate.py` | **Nein**, reiner Eval-Pfad |

**Zur bf16-Frage konkret.** Mit `compute_dtype=None` behält jeder Branch seinen
Eigen-Default. Nachgeprüft:

- `ConvEncoder.compute_dtype = jnp.float32` → unverändert
- `PointNetCloudEncoder.compute_dtype = jnp.bfloat16` → **vorher auf fp32 gezwungen, jetzt bf16**
- MLP- und GNN-Branch haben keinen dtype-Knopf → immer fp32
- Fusion-Dense fällt über `if None → jnp.float32` auf fp32 zurück

Numerisch geändert hat sich also **nur** bei Armen mit `Encoder.POINTNET`, das
sind `rgb_house_voxels` und `rgb_house_cloud_episodes`. Nicht `rgb`, nicht
`rgb_pointmap_pose`/`pointmap_pose*`, nicht `pointmap_dense` (CONV_POINTS →
ConvEncoder → fp32), nicht die Token-Arme, nicht GNN.

`fd9a386` erfordert **kein** Retraining. Er repariert nur, dass eine in falscher
Präzision rekonstruierte Policy bei der Offline-Eval unbemerkt blieb — dtype
ändert keine Param-Shapes, `_assert_params_match` konnte es nicht fangen.

### W&B-Bestand (`sailer-luca-university-ulm/3d-vla-objectnav`)

Neuester Lauf: 2026-07-07. **Kein Lauf stammt vom aktuellen HEAD.**

| Arm / Sprosse | Stand |
|---|---|
| `rgb` L1 | vollständig, Replay-Kapazitäts-Ablation 10k/100k/500k/1M, seed 42, `finished` |
| `rgb` **L2** | **existiert nicht** |
| `rgb` **L3** | **existiert nicht** |
| `rgb` L4 | einmal `finished` (`ooi0i2di`, 2026-06-01) |
| alle 3D-Arme | nur L1, überwiegend `crashed` |
| VGGT wp-cp L2/L3/L4 | L2 `crashed` @1.51M, L3 nur @124k, L4 `crashed` @1.43M |

Die älteren `r2d-L2-buffix` / `r2d-L3-buffix` / `*-actfix` (April/Mai) tragen kein
`jax`-Tag und stammen von vor dem JAX-Rewrite. Kein Baseline-Ersatz.

### Die „Crashes" sind Walltime-Kills

Prod-Walltime ist 48 h (`_base.yaml`, `time: "48:00:00"`). Abbruchschritt gegen
172800 s gerechnet:

| Lauf | Abbruch | → ms/Schritt |
|---|---|---|
| `tp1bxeea` l1_live_house_points_pose | 991 039 | 174 |
| `fvwuoux3` l1_hybrid_house_points_pose | 925 657 | 187 |
| `u74c95d8` l1_gnn_house_points_pose | 905 880 | 191 |
| `pcskbp0l` l1_live_house_points_pose | 825 750 | 209 |
| `egu6znfs` l1_..._house_global_embedding | 678 702 | 255 |
| `223vddmo` r2d-L4-wpcp | 1 425 306 | 121 |
| `7ty9rj26` r2d-L2-wpcp | 1 514 250 | 114 |

Jeder Wert liegt im gemessenen Throughput-Band (171–219 ms/Schritt gegen ~28 ms
der CNN-Baseline). Das sind keine Modellfehler. **2M Schritte sind für die
VGGT-Arme in einer 48-h-Partition arithmetisch unerreichbar** (~95–120 h).

Konsequenz: Ein „fertiger" 2M-CNN-Lauf gegen einen bei 1.4M abgeschnittenen
3D-Lauf ist kein Vergleich, sondern ein Scheduling-Artefakt.

---

## 2. Blocker — vor dem nächsten GPU-Tag, kostet keine GPU

Diese Punkte entwerten eine laufende Seed-Runde nachträglich. Sie sind der Grund,
warum die Job-Liste unten *nicht* sofort abgeschickt werden sollte.

### B1 — Das Zero-Init-Gate ist im Refactor verschwunden

`rg -i gate src/r2dreamer/encoders/` findet keinen einzigen Gate-Parameter mehr.
Die Fusion ist eine blanke `nn.Dense(1024)`
(`src/r2dreamer/encoders/routed_composite.py:236-241`). `HybridHousePointsCameraEncoder`
steht in `prototyp/prototyp-encoder-split/DELETIONS.md` als gelöscht, mit der
Begründung „reproduced declaratively by CompositeSpec" — das Gate wurde dabei
nicht reproduziert.

Gleichzeitig behaupten `hybrid_v1.yaml:7`,
`hybrid_house_points_pose_l1_live.yaml:6,19` und
`docs/02-architecture/architecture-data-flow.html:226` weiterhin „zero-init-gated".

Ohne Gate startet der additive Arm nicht mehr als No-op auf der CNN-Baseline. Die
gemessene Differenz ist dann nicht mehr sauber der *gelernten* Nutzung der
3D-Information zuzuschreiben. **Entweder Gate zurück, oder die Additiv-Aussage
fällt und die Configs/Docs werden korrigiert.**

### B2 — `seed: ${SLURM_JOB_ID}`

Betroffen: `l2_cnn.yaml:19`, `l3_cnn.yaml:19`, `l4_cnn.yaml:20`, `hybrid_v1.yaml:19`,
`pointmap_pose_l1.yaml:17`, `house_context_l1.yaml:16`, `global_tokens_l1.yaml:18`.

Der Seed ist die Scheduler-Job-ID: vor dem Lauf unbekannt, nicht reproduzierbar,
und **über die Arme nicht gepaart** — 2D und 3D sehen verschiedene
Episodenfolgen. Der Seed geht bis in die Habitat-Env durch
(`src/environments/habitat.py:193`), die Episodenreihenfolge selbst ist also
unkontrolliert. Die L1-Kapazitätskontrollen verwenden dagegen `seed: 42`.

Fix ohne N Configs pro Seed: `seed: ${SEED}` plus `SEED: "1"` im `env:`-Block.
Trägt, weil `--env` neue Keys aufnimmt (`scripts/slurm/launch.py:145-152`), `env`
als `export` vor dem Trainingskommando emittiert wird (Zeile 264-266) und
`$`-haltige Werte in doppelte Anführungszeichen gesetzt werden (Zeile 176-186).

### B3 — Ungleiches Step-Budget und ungleiche Replay-Kapazität

- `_base.yaml` setzt `steps: 2000000` für alle. Für die 3D-Arme unerreichbar (s.o.).
- `agent_config.py:175` sagt `buffer_capacity = 500_000`, `src/shared/configs.py:52`
  sagt `1_000_000`. Live ist der erste (`src/main.py:499`). `l2_cnn`/`l3_cnn`/`l4_cnn`
  setzen `buffer_capacity` **nicht** → sie laufen auf 500k, während die
  L1-Headline-Baseline auf 1M lief.

Beides angleichen: gemeinsames erreichbares Step-Budget (Vorschlag 800k–1M, die
CNN-Baseline plateaut ohnehin ab ~500k) und `buffer_capacity: 1000000` über die
ganze Leiter.

### B4 — Alle berichteten SR/SPL sind Trainings-Metriken

`val_every: 0` steht in **jeder** prod-Config (nachgeprüft über
`scripts/slurm/configs/*.yaml`). `metrics/sr` und `metrics/spl` schreibt der
Trainings-Episoden-Collector (`src/r2dreamer/launch/loops.py:186`) — auf genau den
Episoden, auf denen trainiert wird, unter der stochastischen Actor-Policy. Der
berichtete Vergleich 0.70 vs. 0.34 ist Trainings-Success, nicht Held-out-Success.

Auf L1 ist der Unterschied verschmerzbar. Auf L3/L4 entscheidet er über die
gesamte Generalisierungsaussage. Held-out-Eval muss die abhängige Variable werden.

### B5 — Der Voxelpuffer wird bei der Eval nicht restauriert

`rg -i 'seed_xyzrgb|_VoxelContextState|voxel' src/r2dreamer/launch/evaluate.py
src/r2dreamer/checkpointing.py` findet nichts. Der Puffer ist nicht Teil des
Checkpoints; `evaluate.py` baut Adapter und Puffer neu.

Eine Policy, die mit gesättigter Karte trainiert wurde, wird also auf einer Karte
evaluiert, die bei null anfängt und über die Eval-Episoden wächst. Damit ist die
Eval-Eingabeverteilung nicht die Trainingsverteilung, und Episode *i* hängt von
Episoden 0..*i*−1 ab. **Korrektheits-Fix, kein Experiment** — muss vor jeder
3D-Zahl im Text passieren.

---

## 3. Der schwerwiegendste Einzelbefund (Ropinski)

**Auf L1 ist `house_context` faktisch eine Konstante. Kein Rerun heilt das.**

Drei Mechanismen greifen ineinander, alle am Code bestätigt:

1. `house_context` ist `buffer=False` (`src/adapters/house_voxels.py:219`) →
   `live=not field.buffer` (`routed_composite.py:80`) → das Feld wird **einmal pro
   Batch** encodiert und über `(B,T)` gebroadcastet (`routed_composite.py:228-233`).
   Zeitliche Variation ist per Konstruktion null.
2. Der Voxelpuffer hat keine Eviction und sättigt nach wenigen Tausend Schritten
   (~2k Voxel/Schritt, Kapazität `1 << 23` = 8.39M).
3. Der Snapshot ist ein deterministischer Even-Stride über einen eingefrorenen
   Store. L1 hat genau eine Scene → der Snapshot ist in jedem Schritt bitgleich.

Der PointNet-/GNN-Zweig liefert also über fast den gesamten Lauf denselben
1024-d-Vektor; sein Beitrag zum RSSM ist ein **Bias**. Das einzige
per-Schritt-3D-Signal in dieser Familie ist die 9-dimensionale VGGT-Pose. Diese
Arme testen nicht „hilft eine 3D-Karte", sondern „hilft VGGT-Ego-Pose".

Zusatz: 16384 von 8.39M Punkten sind 0.2 % (GNN strided nochmal auf 4096 =
0.05 %). Bei ~500 m² Grundfläche ist das ~35 cm mittlerer Punktabstand.
`VOXEL_SIZE_M = 0.01` steht unverändert in `src/adapters/house_voxels.py:88` —
die eigene Empfehlung „2–4 cm Voxel" aus dem Review vom 2026-07-06 ist nicht
umgesetzt.

**Konsequenz für die Job-Planung:** Die L1-house-Arme werden *umformuliert statt
neu gerechnet*. Ein Rerun dieser Arme auf L1 ist verschwendete GPU-Zeit, solange
das Feld nicht pose-abhängig wird (egozentrischer Ausschnitt / BEV um die aktuelle
Position) oder der Arm auf einer Mehr-Scene-Sprosse läuft.

---

## 4. Stufe 0 — ohne GPU, zuerst

| # | Job | Ändert welche Schlussfolgerung |
|---|---|---|
| 0a | Konstanz-Check: `std` von `house_context` über Schritte loggen, aus einem Smoke von `gnn_house_points_pose_l1_live --smoke` | Bestätigt oder widerlegt §3. Der wertvollste Job der Liste, ~30 min Smoke |
| 0b | B1–B3 patchen (Gate-Entscheidung, `${SEED}`, Budget + `buffer_capacity`) | Ohne das misst die nächste Runde etwas anderes als beabsichtigt |
| 0c | B5: Voxel-Puffer neben dem Checkpoint persistieren, in `evaluate.py` re-seeden | Macht jede Offline-Eval der 3D-Arme überhaupt erst gültig |
| 0d | dtype-Probe P1 (unten) | Erledigt die bf16-Frage ohne einen einzigen Rerun |
| 0e | Stratifizierte Re-Analyse vorhandener `eval_results.json` (Kategorie/Scene/Start-Ziel-Distanz) + CNN-Kurve beim tatsächlichen 3D-Cutoff (~800k) ablesen | Liefert den schritt-gematchten 2D-vs-3D-Vergleich gratis |
| 0f | SHA einfrieren und taggen | Alles in derselben Tabelle muss vom selben `git_sha` stammen |

**Probe P1 (dtype, offline, Minuten):** gedumpte PLY aus `pointcloud_dumps/` laden
→ `HouseContextPoseBuffer.seed_xyzrgb` → `house_context_array(16384, float16)` →
`PointNetCloudEncoder` mit **identischen Parametern** einmal
`compute_dtype=jnp.float32`, einmal `jnp.bfloat16`. Berichten: Kosinus-Ähnlichkeit
und max. relative Abweichung des 1024-d-Embeddings plus Gradientennorm durch beide
Pfade. Akzeptanz: cos ≥ 0.999, keine NaN. Eine Zeile in den Anhang.

Achtung: `hybrid_hpp_prodshape_probe` vs. `hybrid_hpp_bf16_prodshape_probe` taugen
dafür **nicht** — die variieren `full_bf16` für das ganze Modell (CNN, RSSM, Heads)
und isolieren den PointNet-Zweig gerade nicht.

---

## 5. Stufe 1 — GPU-Jobs

Alle Kommandos setzen voraus, dass B1–B3 gepatcht sind.

### P1 — Smoke-Sweep am HEAD (7 × ~30 min, `gpu_h100_short`)

Kein einziger Lauf stammt vom HEAD. Nach 198 geänderten Dateien nicht optional.

```bash
bash scripts/slurm/launch.sh l1_cnn_cap1m --smoke
bash scripts/slurm/launch.sh l2_cnn --smoke
bash scripts/slurm/launch.sh l3_cnn --smoke
bash scripts/slurm/launch.sh hybrid_v1 --smoke
bash scripts/slurm/launch.sh pointmap_pose_l1 --smoke
bash scripts/slurm/launch.sh house_context_l1 --smoke
bash scripts/slurm/launch.sh gnn_house_points_pose_l1_live --smoke
```

`assert_min_rows: 5` auf `metrics.csv` ist ein schwaches Gate — es prüft, dass
geloggt wurde, nicht dass gelernt wurde. Bei den PointNet-Armen zusätzlich im
Smoke-Log auf endliche `loss/dyn`-Werte schauen (bf16-Overflow in der
PointNet-Normalisierung zeigt sich in 1500 Schritten, nicht erst nach 40 h).

### P2 — L1-Parität + Baseline-Anker am HEAD (3 Jobs, ~8 GPU-h/Seed)

Ein L1-Lauf gegen den alten `run-4194043` prüfen: reproduziert der Refactor die
alte Baselinekurve? Wenn nicht, ist die gesamte Vorgeschichte nur noch Anhang.

```bash
for s in 1 2 3; do
  bash scripts/slurm/launch.sh l1_cnn_cap1m --smoke-then-prod --env SEED=$s
done
```

(Setzt voraus, dass `l1_cnn_cap1m.yaml:19` `seed: 42` ebenfalls auf `${SEED}`
umgestellt wird — sonst laufen drei identische Runs.)

### P3 — Die Löcher in der Baseline-Leiter: L2 + L3 (6 Jobs, ~24 h je)

Billig (~28 ms/Schritt). Jede 3D-Aussage wird gegen die Baseline gelesen; ohne
L2/L3 ist die vergleichende Frage oberhalb L1 unbeantwortbar.

```bash
for s in 1 2 3; do
  bash scripts/slurm/launch.sh l2_cnn --smoke-then-prod --env SEED=$s
  bash scripts/slurm/launch.sh l3_cnn --smoke-then-prod --env SEED=$s
done
```

### P4 — Hauptachse: ein 3D-Arm auf L3, gleiche Seeds, gleiches Budget (3 Jobs)

**Braucht neue `RUN_CONFIGS`-Zeilen** — es gibt heute keinen Run-Id für einen
3D-Arm auf L2/L3/L4. Das ist der Job, der die zentrale These überhaupt erst
testbar macht.

L3 (10 Häuser, 1 Ziel) ist die Achse, weil Geometrie über Häuser hinweg überträgt
und Zielsemantik nicht. Wenn ein 3D-Prior irgendwo wirkt, dann dort.

Neue Zeilen in `scripts/r2dreamer/_run_configs.py`:

```python
"habitat-l3-hybrid": dict(
    env="habitat", adapter="rgb_pointmap_pose", curriculum="L3",
    output_dir="output/runs/r2dreamer-curriculum-l3-hybrid",
    wandb_name="l3_hybrid", wandb_tags=[...],
),
```

Neue Config `scripts/slurm/configs/l3_hybrid.yaml` — Delta bewusst auf drei Zeilen
halten:

```yaml
extends: hybrid_v1
job_name: l3-hybrid
output_dir: output/runs/r2dreamer-curriculum-l3-hybrid
run_id: habitat-l3-hybrid
curriculum_check: data/curriculum/level3_10houses_1goal.json
args:
  output_dir: output/runs/r2dreamer-curriculum-l3-hybrid/run-${SLURM_JOB_ID}
  buffer_capacity: 1000000
  wandb_name: l3_hybrid-s${SEED}-${SLURM_JOB_ID}
  wandb_tags: curriculum,level3,10houses,chair-only,hybrid,wp-cp,jax,3d-encoder
```

Wenn beim Portieren auf L3 zusätzlich Encoder-Breite oder Learning-Rate angefasst
wird, ist der Sprossen-Vergleich hin.

### P5 — Negativkontrolle L2 mit demselben 3D-Arm (2 Jobs, analog `l2_hybrid.yaml`)

Die wertvollste Ergänzung überhaupt: ein 3D-Prior sollte auf der *Ziel*achse
**nicht** helfen, auf der *Szenen*achse schon. Das ist eine Vorzeichenprognose,
die die Rivalerklärung („3D hilft nur, weil es Parameter hinzufügt") bei keiner
Parameterwahl machen kann.

### P6 — Geometrische Evidenz: `gnn_house_points_pose_l1_live_plydump` (1 Job, 48 h)

Vorher zwei Dinge ändern:

- `pointcloud_dump_steps: "500000,1000000,1500000,2000000"` → `"100000,250000,500000,750000"`.
  Der Lauf stirbt bei ~900k, drei der vier Milestones feuern nie.
- Zusätzlich den **16384-Zeilen-Snapshot** dumpen, nicht nur `buffer.points_xyz`.
  Das sind zwei verschiedene Objekte (Faktor 512); nur das zweite ist das, worauf
  das Modell konditioniert ist.

### P7 — Oracle-Geometrie-Arm (optional, 1–2 Jobs)

Habitat-Tiefensensor + wahre Agentenpose statt VGGT-Point-Map und -Pose, sonst
identisch. In `src/environments/habitat.py` ist heute nur `rgb_sensor`
konfiguriert. Trennt „die 3D-Repräsentation ist schlecht" von „das Weltmodell
nutzt sie nicht" — und ist **billiger** als die VGGT-Arme, weil kein
VGGT-Forward bezahlt wird.

---

## 6. Was **nicht** neu laufen soll

- **`l4_cnn` und jeder L4-3D-Arm.** L4 = 10 Häuser × 6 Ziele überlagert Ziel- und
  Szenen-Generalisierung und isoliert nichts, was L2/L3 nicht schon isolieren.
  Der L4-CNN-Lauf von 2026-06-01 bleibt Anekdote im Anhang.
- **`l1_cnn_cap{10k,100k,500k}`.** Die Kapazitäts-Ablation ist in sich konsistent
  (gleicher Commit, ein Faktor variiert) und beantwortet ihre eigene Frage.
  Commit-gepinnt berichten, **nicht** in dieselbe Figure wie HEAD-Läufe.
- **Alle `*_probe`-Configs**: `l1_cnn_cap1m_seed42_{fp32,bf16,fp16}_probe`,
  `hybrid_hpp_prodshape_probe`, `hybrid_hpp_bf16_prodshape_probe`,
  `jax_buffer_stepopt_probe`. Step-Time- und Präzisionsmessungen, Zweck erfüllt.
- **Die vier `profile_*`-Configs.** Nachgeprüft: `scripts/profiling/` enthält nur
  noch `cprofile_run.py`, alle vier Entrypoints fehlen. Die Configs rendern, aber
  laufen nicht → löschen. Ein Launcher, der einen nicht lauffähigen Job rendert,
  ist ein Reproduzierbarkeitsrisiko.
- **`house_context_l1` / `house_context_l1_long_smoke` als Prod-Job.**
  `HouseCloudEpisodesAdapter` emittiert eine variabel lange `(N,6)`-Wolke in einen
  PointNet-Zweig, dessen eigenes TODO sagt, dass wechselndes N in jedem Schritt neu
  kompiliert.
- **Die L1-house-Arme erneut**, solange §3 gilt. Erst Feld pose-abhängig machen
  oder auf eine Mehr-Scene-Sprosse gehen.
- **Kein 3D-Arm mehr mit `steps: 2000000` unter 48 h Wall.** Entweder gemeinsames
  erreichbares Budget oder `--resume_from`-Verkettung — wobei `--resume_from` laut
  `src/configs/trainer_config.py:58-61` nur `params/opt_state/slow_critic/ema`
  restauriert, **nicht den Replay-Buffer**. Eine Verkettung startet mit leerem
  Buffer und neuem Prefill; das ist ein Confound, kein Workaround.

---

## 7. Seeds

Einigkeit über alle drei Reviewer: **3 Seeds auf der Hauptachse**, feste Werte
(42/43/44 oder 1/2/3), **identisch über beide Arme**, im Laufnamen und im Manifest.
Nebenachsen 2 Seeds, Ablationen 1 Seed — dann aber im Text als „Trend, kein Test"
benannt. Ablationen mit *weniger* Seeds als das Headline-Ergebnis sind der Defekt,
den man am häufigsten sieht.

Bei n = 3 keine Signifikanzsprache: Pro-Seed-Kurven zeigen, Streuung angeben,
Aussage auf Nicht-Überlappung der Seed-Spannen stützen. Abgestürzte oder
nicht-lernende Seeds bleiben mit Abbruchgrund in der Analyse.

Seeds sind der Multiplikator, nicht die Varianten: 3 × 2 Arme × 1 Achse = 6 ist
bezahlbar, 3 × 2 × 4 Sprossen = 24 nicht. Deshalb eine Hauptachse (L3) plus eine
Negativkontrolle (L2) statt der vollen Leiter.

---

## 8. Regel für die Vergleichbarkeit

Jeder Lauf schreibt `git_sha` und `git_dirty` ins Manifest (`src/r2dreamer/manifest.py:52`).
Damit ist die Regel überprüfbar statt Ansichtssache:

> **Alles, was in derselben Tabelle oder derselben Kurve steht, muss vom selben
> `git_sha` stammen. Läufe mit `git_dirty: true` stehen in keiner Vergleichstabelle.**

Praktisch: nicht „alles neu", sondern **„die Vergleichsachse komplett neu"**. Alles
Ältere wandert in ein Vorstudien-Kapitel, mit SHA, Datum und einem Satz, warum es
nicht mit den neuen Zahlen verrechnet wird. Offengelegte Abkürzungen werden anders
bewertet als stillschweigende.

Nebenbefund: Re-Evaluation alter Checkpoints geht am HEAD ohnehin nicht. Der
Adapter-Umbau hat die Flax-Modulpfade umbenannt (`conv_{key}`, `pointnet_{key}`),
und `_assert_params_match` (`src/r2dreamer/agent.py:140`) vergleicht
`jax.tree_util.keystr`-Pfade. Jede Auswertung eines alten Checkpoints muss auf dem
`git_sha` aus dessen `MANIFEST.json` laufen.

---

## 9. Wo die drei Reviewer sich uneinig sind

Das sind die Entscheidungen, die niemand außer dir treffen kann.

1. **Welcher 3D-Arm ist der Arm der Thesis?** Wurzberger fragt zurück; Braun sagt
   „ein additiver Arm, egal welcher, aber genau einer"; Ropinski sagt, die
   house-Familie ist mechanisch kaputt (§3) und `pointmap_pose` ist der einzige mit
   echtem Per-Schritt-3D — aber erst nach Normalisierung, Non-Finite-Guard und einer
   Entscheidung zum Reset-Mode.
2. **L4 streichen?** Braun und Wurzberger: ja, L2/L3 isolieren mehr. Ropinski
   äußert sich nicht dazu.
3. **Wie viel Umbau vor dem nächsten Prod-Job?** Wurzberger will schnell einen
   HEAD-Anker (P1/P2) und den Rest danach. Ropinski will erst die Stufe-0-Liste
   komplett, weil sonst dekorativ ausgewertet wird. Braun will beides, aber die
   Gate-Frage zuerst.

Ropinskis offene Fragen zur 3D-Seite, die noch beantwortet werden müssen:
PointNet nutzt nur XYZ (`points[sample_idx, :3]`, `encoders/pointnet.py:138`),
während der GNN-Zweig `rgb` konkateniert — der Vergleich PointNet vs. GNN ist
damit kein Architekturvergleich, sondern Geometrie-only vs. Geometrie+Farbe. Und
`pointmap_pose` läuft mit `ResetMode.FULL`, VGGT re-ankert also am ersten Frame
jeder Episode; `wp_cp` ist metrisches XYZ in einem episodenabhängigen Frame.
