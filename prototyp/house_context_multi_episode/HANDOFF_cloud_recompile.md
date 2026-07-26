# HANDOFF: wachsende Cloud rekompiliert jeden Step

Aufgeschoben aus der Adapter-Routing-Migration (2026-07-25, bewusste
Entscheidung: nicht Teil der Migration). Siehe
`docs/notes/adapter-routing-migration.md`.

## Problem

`HouseCloudEpisodesAdapter` (`src/adapters/house_cloud_episodes.py`) hängt
jeden Frame die neuen World-Points an eine wachsende Cloud an und emittiert sie
als Live-Feld mit Shape `(N, 6)`. `N` wächst mit jedem Step, also sieht
`jax.jit` bei jedem Step eine neue Shape und rekompiliert

- `PointNetCloudEncoder` (`src/r2dreamer/encoders/routed_composite.py`),
- und damit `train_step` und `act`.

Bei 518x518 sind das ~268k neue Punkte pro Step. Die Kompilierzeit dominiert
alles andere; der Adapter ist in dieser Form nicht für lange Runs geeignet.

## Gemessen (2026-07-25)

Der Smoke `house_context_l1` (Job 6038442, 20 min Wandtakt, 200 Prefill + 800
Trainingsschritte) lief in **TIMEOUT** und schrieb **null** Metrikzeilen - er kam
in ~17 Minuten nicht durch 200 Prefill-Schritte plus die ersten 50
Trainingsschritte, also über 4 s/Step.

Kontrollvergleich am selben Commit, derselbe Knoten (uc3n104), dasselbe VGGT,
derselbe PointNet-Branch, dieselbe Szene: `hybrid_house_points_pose_l1_live`
(Job 6038443, der Voxel-Arm mit **fester** Cloud) lief COMPLETED durch, Smoke
PASS, 125 Metrikzeilen. Der einzige Unterschied zwischen den beiden ist feste
gegen wachsende Cloud-Shape. Damit ist die Ursache nicht mehr Vermutung.

Der Voxel-Arm hat das Problem nicht: `HouseVoxelsAdapter` emittiert eine
feste `(HOUSE_POINTS, 6)`-Cloud (`_fixed_cloud`, Modulo-Gather über die gültigen
Zeilen), also bleibt die Shape statisch.

## Lösungsrichtungen

1. **Feste Größe wie beim Voxel-Arm.** Nach dem Anhängen auf `N_fix` Zeilen
   bringen - Even-Stride oder FPS. Frühere Messung: eine feste `(501, 6)`-Cloud
   per FPS war der Fix, der den Prefill von 585 auf 131 ms/step brachte
   (`docs/notes/`-Notiz zum Prefill-Bottleneck). Einfachste Variante,
   konsistent mit dem anderen Arm.
2. **Voxel-Dedup beim Anhängen statt am Episodenende.** Dann wächst die Cloud
   viel langsamer, aber die Shape bleibt trotzdem dynamisch - löst das
   Rekompilieren nicht allein.
3. **Bucketing.** Cloud auf die nächste Zweierpotenz padden, damit nur
   log-viele Shapes auftreten. Braucht wieder ein Valid-Count-Feld und
   maskiertes Pooling, also den Contract-Umbau, den die Migration bewusst
   vermieden hat.

Empfehlung: (1).

## Wo anfangen

- `src/adapters/house_cloud_episodes.py`: `__call__` gibt das Live-Feld aus.
- `src/adapters/house_voxels.py::_fixed_cloud`: das Muster, das schon
  funktioniert.
- `tests/adapters/test_routed_pipeline.py` deckt die Verdrahtung ab; ein Test
  auf konstante Shape über mehrere Steps fehlt noch und wäre der erste Schritt.
