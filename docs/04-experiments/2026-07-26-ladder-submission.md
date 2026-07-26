# Ladder-Submission 2026-07-26

Umsetzung von P3/P4/P5 aus [dem Rerun-Plan](2026-07-26-rerun-plan.md).
Fünf Arme über je vier Sprossen, alle vom selben SHA, ein Seed, ein Step-Budget.

**SHA:** `4738326` (`experiments(slurm): complete the 5-arm x 4-level curriculum ladder at HEAD`)
**Seed:** `SEED=1` über alle 20 Jobs (`--env SEED=1`)
**Budget:** `steps: 1500000` über alle 20 Jobs
**Modus:** `--smoke-then-prod`, Prod hängt an `afterok` des Smokes
**Ausgeschlossen:** `uc3n089` (bricht Habitat-GL-Sensor-Reads mit Exit 134 ab)

## Job-IDs

| Arm | Adapter | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|
| CNN-Baseline | `rgb` | 6045919/20 | 6045922/23 | 6045924/25 | 6045926/27 |
| WPCP 37,37 | `pointmap_pose` | 6045928/29 | 6045930/31 | 6045932/33 | 6045934/35 |
| Token Pooled Aggregator | `aggregator_pooled` | 6045936/37 | 6045938/39 | 6045940/41 | 6045942/43 |
| Hybrid CNN + Points | `rgb_pointmap_pose` | 6045944/45 | 6045946/47 | 6045948/49 | 6045950/51 |
| Hybrid CNN + Tokens | `rgb_global_tokens` | 6045952/53 | 6045954/55 | 6045956/57 | 6045958/59 |

Jeweils `smoke/prod`.

## Erwartete Reichweite - offengelegt, nicht stillschweigend

ms/Schritt aus W&B; das 48-h-Partitionsmaximum ist die bindende Schranke,
nicht das Step-Budget.

| Arm | ms/Schritt | Referenzlauf | Güte | erreicht von 1.5M | Wall |
|---|---|---|---|---|---|
| CNN-Baseline | 29-37 | `5str1p17` | gemessen, gleicher Arm | **vollständig** (~15 h) | 24 h |
| Token Pooled Aggregator | 94 | `5959vo44` | gemessen, gleicher Arm | **vollständig** (~39 h) | 48 h |
| WPCP 37,37 | 106-121 | `rx29922r`, `7ty9rj26` | gemessen, gleicher Arm | ~1.28M (85 %) | 48 h |
| Hybrid CNN + Points | 164 | `boi0cntv` | gemessen, gleicher Arm | ~0.94M (63 %) | 48 h |
| Hybrid CNN + Tokens | >=254 | `egu6znfs` | **Proxy, Untergrenze** | **<=0.61M (<=41 %)** | 48 h |

Jede betroffene Config trägt diese Zahl plus den Referenzlauf als Kommentar.
Beim Auswerten sind die Kurven an der kürzesten gemeinsamen Reichweite zu
lesen, nicht an den Endpunkten.

### Zur Methode

Abgeleitet als `_runtime / _step` aus der W&B-Summary. Wo der Zähler
`perf/ms_per_step_interval` mitgeschrieben wurde, stimmt die Ableitung auf
1-2 % mit dem geloggten Wert überein - der Prefill amortisiert sich über
Läufe dieser Länge weg:

| Lauf | abgeleitet | geloggt p50 | geloggt letztes Viertel |
|---|---|---|---|
| `egu6znfs` | 254.2 | 251.6 | 250.6 |
| `5str1p17` | 28.9 | 27.5 | 27.4 |
| `fvwuoux3` | 186.5 | 184.7 | 185.1 |

Die älteren wpcp-/hybrid-/agg-Läufe stammen von vor diesem Zähler; dort ist
die Ableitung die einzige verfügbare Quelle.

### Warnung zum Token-Arm

Für `rgb_global_tokens` existiert **kein einziger Prod-Lauf** im W&B-Projekt.
Die einzigen Token-Transformer-Läufe dort sind vier `full_tokens`-Stummel, die
innerhalb von 15 Minuten ohne Schrittzahl endeten. `egu6znfs` ist ein
*house-global-embedding*-Arm, kein Token-Transformer-Arm.

Warum die Smokes nach dem Refactor nicht als Quelle taugen - beide Gründe
geprüft, nicht angenommen:

1. *Sie tragen die Zahl nicht.* `output/smoke/rgb-global-tokens-20260725-152035`
   ist der richtige Arm, endet aber bei Schritt 499 (noch im Prefill, vor dem
   ersten Train-Step), schreibt nur Episoden-Metriken und meldet im Manifest
   `git_dirty: true` auf `add-encoder-routing-adapters`, also vor dem Merge.
   Das W&B-Smoke-Projekt enthält genau einen Lauf, vom 2026-06-01. Die
   `output/diag/pipe-*`-Läufe sind GPU-Speicher-Diagnosen, keine Zeitmessung.
2. *Die Smoke-Form misst die Produktionskosten nicht.* Nach
   `loops.py:453` ist `train_credit += train_ratio / (batch_size * seq_len)`:

   | | `train_ratio` | `batch x seq` | Train-Step feuert | Arbeit je Train-Step |
   |---|---|---|---|---|
   | prod | 512 | 16x64 = 1024 | jeder 2. Schritt | 1024 Elemente |
   | smoke | 16 | 4x16 = 64 | jeder 4. Schritt | 64 Elemente |

   Ein Smoke-Schritt zahlt den Trainingspfad halb so oft bei 16-fach kleinerer
   Arbeit, zusammen etwa 1/32 der Produktionskosten. `train_step` und
   `buffer_sample` sind rund 26 % des Produktionsschritts, ein Smoke
   unterschätzt die Gesamtzeit also systematisch.

   Der VGGT-Forward läuft dagegen einmal je Env-Schritt bei Batch 1, unabhängig
   von der Form. Ein Smoke misst diesen dominierenden Anteil korrekt - nur eben
   nicht die Summe.

Die Richtung des Fehlers ist bekannt: der Proxy verarbeitet ein gepooltes
Embedding, `rgb_global_tokens` dagegen die volle 1374x1024-Sequenz pro
Schritt bei 2.8 MB Replay-Zeile. Der echte Arm sollte **langsamer** sein.
254 ms ist damit eine Untergrenze, und die 0.61M sind eine Obergrenze.

**Zu tun:** sobald die vier `*_global_tokens`-Prod-Jobs (6045953/55/57/59)
eine Stunde laufen, `perf/ms_per_step_interval` ablesen und diese Tabelle
gegen den echten Wert korrigieren. Liegt er deutlich über 254 ms, ist für
diesen Arm zu entscheiden, ob 1.5M sinnvoll bleibt oder ob er mit einem
eigenen, offengelegten Budget geführt wird.

## Warum die 3D-Arme teuer sind

Aus `output/profiles/hybrid_3d60_profile.json` (Produktionsform: Batch 16,
Seq 64, `train_ratio` 512), Steady State. Amortisiert = feuert auf jedem
zweiten Schritt.

| Phase | ms/Schritt | Anteil |
|---|---|---|
| `vggt_forward_internal` | 61.0 | 65 % |
| `train_step` (36.2 x 0.5) | 18.1 | 19 % |
| `buffer_sample` (12.4 x 0.5) | 6.2 | 7 % |
| `resize_rgb` (64 -> 518) | 3.6 | 4 % |
| `act` | 1.5 | 2 % |
| `env_step` | 1.5 | 2 % |
| `vggt_wrapper_internal` + `adapter_post` | 1.3 | 1 % |
| `buffer_add` | 0.02 | ~0 % |
| **`total_step`** | **93.6** | |

Der VGGT-Forward ist allein zwei Drittel des Schritts. Encoder und RSSM sind
Nebenposten. Der Profillauf landet bei 94 ms (p50 112) gegen die 164 ms der
48-h-Läufe; die Differenz ist echtes Habitat-GL-Stepping, der volle
500k-Puffer statt der synthetischen 8192 Zeilen und das Wachstum des
VGGT-KV-Caches über eine Episode. Die Rangfolge ändert sich dadurch nicht.

Was der Hybrid gegenüber dem reinen Geometrie-Arm zusätzlich zahlt
(`output/profiles/replay_train_profile_results.json`, gleiche Form, ohne VGGT):

| Arm | `sample_convert` | `train_step` | zusammen |
|---|---|---|---|
| `cnn` | 11.8 | 33.8 | 45.0 |
| `wpcp` | 3.9 | 24.8 | 31.1 |
| `hybrid` | 15.4 | 35.2 | **49.0** |

Vierfache Replay-Bandbreite plus Conv-Encoder *und* -Decoder; `wpcp` hat
überhaupt kein Rekonstruktionsziel.

## Was mit dieser Runde gefixt wurde

- **B2** - `seed: ${SLURM_JOB_ID}` ist raus. `seed: ${SEED}` gegen einen
  `SEED`-Default in `_base.yaml`, pro Submission mit `--env SEED=n`
  überschreibbar. Auch `house_context_l1` und `pointmap_dense_l1` mitgezogen,
  die denselben Defekt trugen.
- **B3** - `l2/l3/l4_cnn` liefen still auf dem 500k-Default, während die
  L1-Headline-Baseline auf 1M lief. Jede Ladder-Config pinnt ihre Kapazität
  jetzt explizit. Über die Arme hinweg ist sie *nicht* einheitlich und kann es
  nicht sein: eine Global-Token-Zeile ist 2.8 MB, dieser Arm bleibt bei 20000.
  Innerhalb jedes Arms ist sie einheitlich, und darauf kommt es beim
  Sprossen-Vergleich an.
- `l1_cnn.yaml` ist neu und bewusst nicht `l1_cnn_cap1m`: jene Config gehört
  zur Replay-Kapazitäts-Ablation, ist auf Seed 42 gepinnt und wird
  commit-gepinnt als eigene Studie berichtet.
- `hybrid_v1` bekam den `link_external.sh`-Hook, als einziger VGGT-Arm ohne ihn.

## Was diese Runde **nicht** löst

- **B1 (Zero-Init-Gate)** bleibt offen. Die Fusion ist weiterhin eine blanke
  `nn.Dense`; die Additiv-Aussage der Hybrid-Arme steht damit noch nicht.
- **B4 (`val_every: 0`)** bleibt offen. Alle SR/SPL dieser Runde sind
  Trainings-Metriken auf den Trainingsepisoden unter der stochastischen Policy.
  Held-out-Eval muss offline aus den Checkpoints kommen.
- **B5 (Voxelpuffer ohne Restore-Pfad)** ist für diese fünf Arme nicht bindend -
  keiner von ihnen nutzt `HouseContextPoseBuffer`. Für die house-Familie bleibt
  es blockierend.
- **§3** (`house_context` ist auf L1 faktisch konstant) betrifft die
  house-Arme, die hier bewusst nicht laufen.
- Nur **ein Seed**. Der Plan verlangt drei auf der Hauptachse. Seeds 2 und 3
  sind mit `--env SEED=2` bzw. `3` auf demselben SHA nachzuschieben, sonst gilt
  fürs Ergebnis "Trend, kein Test".

## Nachziehen

```bash
# Seeds 2 und 3 auf demselben SHA (4738326)
for s in 2 3; do
  for v in l1_cnn l2_cnn l3_cnn l4_cnn \
           pointmap_pose_l1 l2_pointmap_pose l3_pointmap_pose l4_pointmap_pose \
           aggregator_pooled_l1 l2_aggregator_pooled l3_aggregator_pooled l4_aggregator_pooled \
           hybrid_v1 l2_hybrid l3_hybrid l4_hybrid \
           global_tokens_l1 l2_global_tokens l3_global_tokens l4_global_tokens; do
    bash scripts/slurm/launch.sh "$v" --smoke-then-prod --env SEED=$s --exclude uc3n089
  done
done
```

Läufe nach `MANIFEST.json` beurteilen, nicht nach dem Exit-Code: der
GL-Teardown von Habitat vergiftet den SLURM-Exit-Code.
