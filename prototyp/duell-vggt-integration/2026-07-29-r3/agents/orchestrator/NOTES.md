# Orchestrator-Notizen - Duell 3

## Zeitachse

- 06:10 UTC T+0:00 Start (erster Tool-Call). verify.sh PASS.
- 06:13-06:22 Welle-1-Arme gebaut (P1/P2/P3/P5), tests/adapters 94 passed,
  verify.sh PASS mit den vier neuen YAMLs.
- 06:23-06:27 Welle 1 submittet, 75 s gestaffelt (uv-sync-Race, r2 Slot B):
  P1=6087059, P2=6087060, P3=6087061, P5=6087064, alle gpu_h100_short,
  --exclude uc3n089, SEED=42, 30 min --prod.
- 06:28-06:35 Welle-2-Kandidaten gebaut und committet (P6 quad, P7 frame-only),
  8 gezielte Pipeline-Tests passed. Commits 54d0d9b, danach P6/P7-Commit.

## Entscheidungen

1. Welle 1 = P1 (Pflicht) + P2 + P3 + P5. Alle vier ohne Encoder-Aenderung
   (nur Adapter-Subklassen), damit Welle 1 frueh und risikoarm steht.
   P4 (gelerntes Attention-Pooling) NICHT gebaut: Lernbares Pooling muss im
   Encoder liegen (Adapter ist eingefroren/parameterlos), also muesste die
   volle Sequenz durch den Replay - 1374x2048 fp16 = 5.6 MB/Zeile, ein
   Trainingsbatch (16x64=1024 Zeilen) waere 5.7 GB pro Update. l3_global_tokens
   lief mit 254 ms/Step (halbe Zeilengroesse), r2-Arm H scorte -0.58 ueber
   Tempo-Malus. In einer Matrix mit ms/Step- und Episodengewicht chancenlos ->
   Sackgasse, nicht gefahren.
2. Welle-2-Kandidaten vorbereitet, waehrend die GPUs rechnen:
   P6 aggregator_pooled_full_quad (P1 + 2x2-Quadranten-Means, 14336, 56 KB,
   28 GB < 32-GB-Deckel), P7 aggregator_pooled_frame (frame-only Triple, 3072).
   Finale Auswahl nach Welle-1-Scores.
3. Speicherdeckel-Rechnung (RULES 3.2): P1 12 GB, P2 8 GB, P3 16 GB,
   P5 12 GB, P6 28 GB, P7 6 GB - alle bei buffer_capacity 500 000 unter 32 GB,
   keine Kapazitaetsaenderung noetig.

## Queue-Lage (06:30-06:45)

- Erst-Submit 06:23-06:27 (6087059-64) um 06:33 gecancelt: Worktree hatte kein
  uv.lock und launch.py prod renderte `uv run python` (mit Sync). Bei
  Simultanstart aller vier pending Jobs waere das das r2-Race gewesen.
  Fix: uv.lock aus Main kopiert, launch.py rendert immer --no-sync
  (Branch-Commit). Resubmit 06:33-06:35: 6087073/75/77/78.
- gpu_h100_short ist leer von eigenen Jobs, aber alle vier stehen pending
  Reason=Priority, geschaetzter Start 2026-07-30 14:00 (pessimistisches
  Estimate). r2-Jobs starteten in 10-70 s - am 27.07. um 15:21 lokal; heute
  ist Dienstagmorgen, der Cluster ist voll (PrivateData verbirgt fremde Jobs;
  squeue -t R zeigt clusterweit 0, was nur Sichtbarkeit ist).
- Anfrage-Parameter identisch mit r2 (8 CPU, 64G, 1 GPU, 30 min, Prio 10758,
  QOS normal) - kein Fehler auf unserer Seite, reine Kontention. RULES 4
  erlaubt ausschliesslich gpu_h100_short, also warten + Backfill hoffen.
- Rendered sbatch-Skripte nach runs/<jobid>-<config>/rendered.sbatch gesichert.

## Welle-2-Plan (Entscheidung spaetestens 07:40)

- Szenario A (Welle 1 gelaufen): E = Seed-43 des Fuehrenden, F = Kontrolllauf
  C Seed 42 (duell2_l3_aggpool_b200k_tr128, unveraendert), G/H nach Scores
  aus {P6 quad, P7 frame-only, P8 deep}.
- Szenario B (Welle 1 bei 07:40 noch pending): blind absetzen bis 07:55 -
  E = P1 Seed 43 (Headline-Arm als mutmasslicher Fuehrender), F = Kontrolle C,
  G = P6, H = P8. Jeweils per `sbatch --dependency=afterany:<welle1-job>` 1:1
  gekettet, damit nie mehr als 4 GPU-Jobs parallel laufen (RULES 4).
  Rendered Skripte sind self-contained (export SEED gebacken), also
  --dry-run + manuelles sbatch moeglich.

## Meldungen an Luca

- Nach Welle-1-Submit (06:35): 4 Arme in der Queue (nach Resubmit wegen
  uv-sync-Gefahr), Scores stehen aus. P6/P7/P8 fuer Welle 2 fertig.
