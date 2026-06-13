function n(e){switch(e){case"index":return`@startuml
title "Current Code Architecture"
top to bottom direction

hide stereotype
skinparam ranksep 60
skinparam nodesep 30
skinparam {
  arrowFontSize 10
  defaultTextAlignment center
  wrapWidth 200
  maxMessageSize 100
  shadowing false
}

skinparam person<<Researcher>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<Codebase>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
person "==Researcher / operator\\n<size:10>[CLI, SLURM, W&B]</size>\\n\\nStarts training, evaluation, profiling, and analysis runs.\\nThe current code is organized around script-level run selection, launcher registries, and a JAX/Flax R2Dreamer agent." <<Researcher>> as Researcher
rectangle "==Master-Thesis-3D-VLA current code\\n<size:10>[Python, JAX/Flax, Habitat, VGGT]</size>\\n\\nCurrent main-branch architecture for ObjectNav experiments.\\nThe source contracts are the root and scoped AGENTS.md files plus the current src/ and scripts/ modules." <<Codebase>> as Codebase

Researcher .[#8D8D8D,thickness=2].> Codebase : "<color:#8D8D8D>[...]<color:#8D8D8D>"
@enduml
`;case"view_1dcnbvb":return`@startuml
title "Current Code Overview"
top to bottom direction

hide stereotype
skinparam ranksep 60
skinparam nodesep 30
skinparam {
  arrowFontSize 10
  defaultTextAlignment center
  wrapWidth 200
  maxMessageSize 100
  shadowing false
}

skinparam person<<Researcher>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseOrchestration>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseVggt_boundary>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseEnvironment>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseEncoder_boundary>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseTraining_loop>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseAgent_boundary>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseEvaluation>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
person "==Researcher / operator\\n<size:10>[CLI, SLURM, W&B]</size>\\n\\nStarts training, evaluation, profiling, and analysis runs.\\nThe current code is organized around script-level run selection, launcher registries, and a JAX/Flax R2Dreamer agent." <<Researcher>> as Researcher
rectangle "Master-Thesis-3D-VLA current code" <<Codebase>> as Codebase {
  skinparam RectangleBorderColor<<Codebase>> #3b82f6
  skinparam RectangleFontColor<<Codebase>> #3b82f6
  skinparam RectangleBorderStyle<<Codebase>> dashed

  rectangle "==Experiment orchestration\\n<size:10>[scripts/r2dreamer, scripts/slurm]</size>" <<CodebaseOrchestration>> as CodebaseOrchestration
  rectangle "==VGGT production encoder\\n<size:10>[src/vggt/jax]</size>" <<CodebaseVggt_boundary>> as CodebaseVggt_boundary
  rectangle "==Environment layer\\n<size:10>[src/environments, Habitat/Crafter]</size>" <<CodebaseEnvironment>> as CodebaseEnvironment
  rectangle "==Encoder and adapter layer\\n<size:10>[src/r2dreamer/encoders and adapters]</size>" <<CodebaseEncoder_boundary>> as CodebaseEncoder_boundary
  rectangle "==Trainer and replay loop\\n<size:10>[src/r2dreamer/trainer.py, src/buffer/replay_buffer.py]</size>" <<CodebaseTraining_loop>> as CodebaseTraining_loop
  rectangle "==R2Dreamer agent\\n<size:10>[src/r2dreamer/agent.py]</size>" <<CodebaseAgent_boundary>> as CodebaseAgent_boundary
  rectangle "==Evaluation and parity workflows\\n<size:10>[src/r2dreamer/launch/evaluate.py, launch/parity]</size>" <<CodebaseEvaluation>> as CodebaseEvaluation
}

Researcher .[#8D8D8D,thickness=2].> CodebaseOrchestration : "<color:#8D8D8D>starts run<color:#8D8D8D>"
Researcher .[#8D8D8D,thickness=2].> CodebaseEvaluation : "<color:#8D8D8D>runs evaluation<color:#8D8D8D>"
CodebaseOrchestration .[#8D8D8D,thickness=2].> CodebaseEnvironment : "<color:#8D8D8D>[...]<color:#8D8D8D>"
CodebaseOrchestration .[#8D8D8D,thickness=2].> CodebaseEncoder_boundary : "<color:#8D8D8D>resolve encoder<color:#8D8D8D>"
CodebaseEnvironment .[#8D8D8D,thickness=2].> CodebaseEncoder_boundary : "<color:#8D8D8D>[...]<color:#8D8D8D>"
CodebaseEncoder_boundary .[#8D8D8D,thickness=2].> CodebaseTraining_loop : "<color:#8D8D8D>obs adapter<color:#8D8D8D>"
CodebaseEncoder_boundary .[#8D8D8D,thickness=2].> CodebaseAgent_boundary : "<color:#8D8D8D>agent overrides and module_cls<color:#8D8D8D>"
CodebaseVggt_boundary .[#8D8D8D,thickness=2].> CodebaseEncoder_boundary : "<color:#8D8D8D>[...]<color:#8D8D8D>"
CodebaseTraining_loop .[#8D8D8D,thickness=2].> CodebaseAgent_boundary : "<color:#8D8D8D>[...]<color:#8D8D8D>"
CodebaseAgent_boundary .[#8D8D8D,thickness=2].> CodebaseTraining_loop : "<color:#8D8D8D>action and metrics<color:#8D8D8D>"
CodebaseAgent_boundary .[#8D8D8D,thickness=2].> CodebaseEvaluation : "<color:#8D8D8D>save<color:#8D8D8D>"
CodebaseEvaluation .[#8D8D8D,thickness=2].> CodebaseAgent_boundary : "<color:#8D8D8D>compare behavior<color:#8D8D8D>"
@enduml
`;case"view_1omwq3l":return`@startuml
title "Experiment Orchestration"
top to bottom direction

hide stereotype
skinparam ranksep 60
skinparam nodesep 30
skinparam {
  arrowFontSize 10
  defaultTextAlignment center
  wrapWidth 200
  maxMessageSize 100
  shadowing false
}

skinparam person<<Researcher>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseOrchestrationSlurm_configs>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseOrchestrationRun_dispatcher>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseOrchestrationRun_configs>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseOrchestrationPublic_entrypoint>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseOrchestrationTrain_entry>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseOrchestrationRegistries>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam database<<CodebaseOrchestrationCurriculum>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseEnvironment>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseEncoder_boundary>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
person "==Researcher / operator\\n<size:10>[CLI, SLURM, W&B]</size>\\n\\nStarts training, evaluation, profiling, and analysis runs.\\nThe current code is organized around script-level run selection, launcher registries, and a JAX/Flax R2Dreamer agent." <<Researcher>> as Researcher
rectangle "Experiment orchestration" <<CodebaseOrchestration>> as CodebaseOrchestration {
  skinparam RectangleBorderColor<<CodebaseOrchestration>> #3b82f6
  skinparam RectangleFontColor<<CodebaseOrchestration>> #3b82f6
  skinparam RectangleBorderStyle<<CodebaseOrchestration>> dashed

  rectangle "==SLURM configs\\n<size:10>[scripts/slurm/configs/*.yaml and legacy *.sbatch]</size>\\n\\nRender or call run.py with a run_id and train flags. GPU work is launched through the cluster scheduler." <<CodebaseOrchestrationSlurm_configs>> as CodebaseOrchestrationSlurm_configs
  rectangle "==scripts/r2dreamer/run.py\\n<size:10>[Single run-id dispatcher]</size>\\n\\nAccepts run.py <run-id> [train flags] and forwards to _run_configs.launch_run." <<CodebaseOrchestrationRun_dispatcher>> as CodebaseOrchestrationRun_dispatcher
  rectangle "==RUN_CONFIGS\\n<size:10>[scripts/r2dreamer/_run_configs.py]</size>\\n\\nSingle source of truth for env, encoder, curriculum, output_dir, wandb_name, and wandb_tags.\\nKnown run ids cover Habitat L1-L4 CNN/VGGT variants and Crafter CNN." <<CodebaseOrchestrationRun_configs>> as CodebaseOrchestrationRun_configs
  rectangle "==src/main.py\\n<size:10>[train, evaluate, parity commands]</size>\\n\\nPublic CLI dispatcher for train/evaluate and parity workflows." <<CodebaseOrchestrationPublic_entrypoint>> as CodebaseOrchestrationPublic_entrypoint
  rectangle "==src.r2dreamer.launch.train.train\\n<size:10>[Launcher composition root]</size>\\n\\nResolves curriculum, encoder, env, agent config, trainer config, agent, and Trainer.\\nRuns Trainer.run() and returns the Trainer for programmatic callers." <<CodebaseOrchestrationTrain_entry>> as CodebaseOrchestrationTrain_entry
  rectangle "==launch registries\\n<size:10>[src/r2dreamer/launch/registries.py]</size>\\n\\nMaps encoder strings to Encoder classes and env strings to env factories." <<CodebaseOrchestrationRegistries>> as CodebaseOrchestrationRegistries
  database "==Curriculum JSON\\n<size:10>[data/curriculum/*.json]</size>\\n\\nHabitat L1-L4 curriculum files resolved by launch/curricula.py and _helpers.py." <<CodebaseOrchestrationCurriculum>> as CodebaseOrchestrationCurriculum
}
rectangle "==Environment layer\\n<size:10>[src/environments, Habitat/Crafter]</size>" <<CodebaseEnvironment>> as CodebaseEnvironment
rectangle "==Encoder and adapter layer\\n<size:10>[src/r2dreamer/encoders and adapters]</size>" <<CodebaseEncoder_boundary>> as CodebaseEncoder_boundary

Researcher .[#8D8D8D,thickness=2].> CodebaseOrchestrationRun_dispatcher : "<color:#8D8D8D>starts run<color:#8D8D8D>"
CodebaseOrchestrationRun_dispatcher .[#8D8D8D,thickness=2].> CodebaseOrchestrationRun_configs : "<color:#8D8D8D>select run id<color:#8D8D8D>"
CodebaseOrchestrationSlurm_configs .[#8D8D8D,thickness=2].> CodebaseOrchestrationRun_dispatcher : "<color:#8D8D8D>render or call<color:#8D8D8D>"
CodebaseOrchestrationRun_configs .[#8D8D8D,thickness=2].> CodebaseOrchestrationPublic_entrypoint : "<color:#8D8D8D>call src.main.train<color:#8D8D8D>"
CodebaseOrchestrationPublic_entrypoint .[#8D8D8D,thickness=2].> CodebaseOrchestrationTrain_entry : "<color:#8D8D8D>dispatch train<color:#8D8D8D>"
CodebaseOrchestrationTrain_entry .[#8D8D8D,thickness=2].> CodebaseOrchestrationRegistries : "<color:#8D8D8D>resolve env and encoder<color:#8D8D8D>"
CodebaseOrchestrationTrain_entry .[#8D8D8D,thickness=2].> CodebaseOrchestrationCurriculum : "<color:#8D8D8D>resolve Habitat curriculum<color:#8D8D8D>"
CodebaseOrchestrationTrain_entry .[#8D8D8D,thickness=2].> CodebaseEnvironment : "<color:#8D8D8D>[...]<color:#8D8D8D>"
CodebaseOrchestrationRegistries .[#8D8D8D,thickness=2].> CodebaseEncoder_boundary : "<color:#8D8D8D>resolve encoder<color:#8D8D8D>"
@enduml
`;case"view_spshjc":return`@startuml
title "Encoder Specs And Adapters"
top to bottom direction

hide stereotype
skinparam ranksep 60
skinparam nodesep 30
skinparam {
  arrowFontSize 10
  defaultTextAlignment center
  wrapWidth 200
  maxMessageSize 100
  shadowing false
}

skinparam rectangle<<CodebaseOrchestration>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseEnvironment>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseVggt_boundary>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseEncoder_boundaryEncoder_spec>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseEncoder_boundaryObs_adapter>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseEncoder_boundaryVggt_adapter>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseEncoder_boundaryHybrid_adapter>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseAgent_boundary>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseEncoder_boundaryCnn_spec>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseEncoder_boundaryWp_cp_spec>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseEncoder_boundaryAggregator_spec>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseEncoder_boundaryDense_wp_spec>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseEncoder_boundaryWp_cp>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseEncoder_boundaryPooled_agg>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseEncoder_boundaryDense_wp>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseEncoder_boundaryHybrid_spec>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseEncoder_boundaryHybrid_obs>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseTraining_loop>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
rectangle "==Experiment orchestration\\n<size:10>[scripts/r2dreamer, scripts/slurm]</size>" <<CodebaseOrchestration>> as CodebaseOrchestration
rectangle "==Environment layer\\n<size:10>[src/environments, Habitat/Crafter]</size>" <<CodebaseEnvironment>> as CodebaseEnvironment
rectangle "==VGGT production encoder\\n<size:10>[src/vggt/jax]</size>" <<CodebaseVggt_boundary>> as CodebaseVggt_boundary
rectangle "Encoder and adapter layer" <<CodebaseEncoder_boundary>> as CodebaseEncoder_boundary {
  skinparam RectangleBorderColor<<CodebaseEncoder_boundary>> #3b82f6
  skinparam RectangleFontColor<<CodebaseEncoder_boundary>> #3b82f6
  skinparam RectangleBorderStyle<<CodebaseEncoder_boundary>> dashed

  rectangle "==EncoderSpec\\n<size:10>[obs_shape, env_render_resolution, encoder_type, module_cls]</size>\\n\\nLauncher-side contract that keeps adapter output, environment render size, and the Flax encoder module aligned.\\ntrain.py copies the spec into R2DreamerConfig." <<CodebaseEncoder_boundaryEncoder_spec>> as CodebaseEncoder_boundaryEncoder_spec
  rectangle "==ObsAdapter\\n<size:10>[src/r2dreamer/adapters/obs_adapter.py]</size>\\n\\nDefault RGB passthrough adapter with buffer shape (3,64,64), dtype uint8, and normalize_on_sample=True." <<CodebaseEncoder_boundaryObs_adapter>> as CodebaseEncoder_boundaryObs_adapter
  rectangle "==VGGTObsAdapter\\n<size:10>[src/r2dreamer/adapters/vggt_adapter.py]</size>\\n\\nRuns JAXVGGTFeatureExtractor per frame and returns replay features plus one-step agent features.\\nFeature kinds: wp_cp, aggregator, wp_dense, agg_raw." <<CodebaseEncoder_boundaryVggt_adapter>> as CodebaseEncoder_boundaryVggt_adapter
  rectangle "==HybridObsAdapter\\n<size:10>[src/r2dreamer/adapters/hybrid_adapter.py]</size>\\n\\nRuns VGGT for WP/CP, resizes the same 518x518 frame to 64x64 RGB, and stores explicit replay fields image and wp_cp." <<CodebaseEncoder_boundaryHybrid_adapter>> as CodebaseEncoder_boundaryHybrid_adapter
  rectangle "==cnn\\n<size:10>[ConvEncoder, obs_shape (3,64,64)]</size>" <<CodebaseEncoder_boundaryCnn_spec>> as CodebaseEncoder_boundaryCnn_spec
  rectangle "==vggt / vggt_wp_cp_64\\n<size:10>[VGGTEncoder MLP]</size>\\n\\nWP/CP vector is K*K*3 world points plus 9 camera-pose values. K=37 gives 4116; K=64 gives 12297." <<CodebaseEncoder_boundaryWp_cp_spec>> as CodebaseEncoder_boundaryWp_cp_spec
  rectangle "==vggt_aggregator_mlp\\n<size:10>[VGGTAggregatorMLPEncoder]</size>\\n\\nPools VGGT global aggregator tokens into [camera | mean_patches | max_patches] = 3072 float32 values." <<CodebaseEncoder_boundaryAggregator_spec>> as CodebaseEncoder_boundaryAggregator_spec
  rectangle "==vggt_wp_dense_cnn\\n<size:10>[WPConvEncoder]</size>\\n\\nStores dense world points as (3,518,518) float16 and encodes them with a symlog conv stack." <<CodebaseEncoder_boundaryDense_wp_spec>> as CodebaseEncoder_boundaryDense_wp_spec
  rectangle "==WP/CP features\\n<size:10>[4116 or 12297 float32]</size>" <<CodebaseEncoder_boundaryWp_cp>> as CodebaseEncoder_boundaryWp_cp
  rectangle "==Pooled aggregator features\\n<size:10>[3072 float32]</size>" <<CodebaseEncoder_boundaryPooled_agg>> as CodebaseEncoder_boundaryPooled_agg
  rectangle "==Dense world-point map\\n<size:10>[3 x 518 x 518 float16]</size>" <<CodebaseEncoder_boundaryDense_wp>> as CodebaseEncoder_boundaryDense_wp
  rectangle "==hybrid\\n<size:10>[HybridEncoder]</size>\\n\\nStores image uint8 (3,64,64) and WP/CP float32 separately, then packs to 16404 features at the JAX boundary." <<CodebaseEncoder_boundaryHybrid_spec>> as CodebaseEncoder_boundaryHybrid_spec
  rectangle "==Hybrid replay observation\\n<size:10>[image uint8 + wp_cp float32]</size>\\n\\nReplay keeps modalities inspectable under explicit keys; obs_batch packs them into the legacy flat tensor." <<CodebaseEncoder_boundaryHybrid_obs>> as CodebaseEncoder_boundaryHybrid_obs
}
rectangle "==R2Dreamer agent\\n<size:10>[src/r2dreamer/agent.py]</size>" <<CodebaseAgent_boundary>> as CodebaseAgent_boundary
rectangle "==Trainer and replay loop\\n<size:10>[src/r2dreamer/trainer.py, src/buffer/replay_buffer.py]</size>" <<CodebaseTraining_loop>> as CodebaseTraining_loop

CodebaseOrchestration .[#8D8D8D,thickness=2].> CodebaseEncoder_boundaryEncoder_spec : "<color:#8D8D8D>resolve encoder<color:#8D8D8D>"
CodebaseEnvironment .[#8D8D8D,thickness=2].> CodebaseEncoder_boundaryObs_adapter : "<color:#8D8D8D>CNN path<color:#8D8D8D>"
CodebaseEnvironment .[#8D8D8D,thickness=2].> CodebaseEncoder_boundaryVggt_adapter : "<color:#8D8D8D>VGGT standalone path<color:#8D8D8D>"
CodebaseEnvironment .[#8D8D8D,thickness=2].> CodebaseEncoder_boundaryHybrid_adapter : "<color:#8D8D8D>hybrid path<color:#8D8D8D>"
CodebaseVggt_boundary .[#8D8D8D,thickness=2].> CodebaseEncoder_boundaryVggt_adapter : "<color:#8D8D8D>extract features<color:#8D8D8D>"
CodebaseVggt_boundary .[#8D8D8D,thickness=2].> CodebaseEncoder_boundaryHybrid_adapter : "<color:#8D8D8D>extract WP/CP<color:#8D8D8D>"
CodebaseEncoder_boundaryEncoder_spec .[#8D8D8D,thickness=2].> CodebaseEncoder_boundaryObs_adapter : "<color:#8D8D8D>default RGB adapter<color:#8D8D8D>"
CodebaseEncoder_boundaryEncoder_spec .[#8D8D8D,thickness=2].> CodebaseEncoder_boundaryVggt_adapter : "<color:#8D8D8D>VGGT standalone variants<color:#8D8D8D>"
CodebaseEncoder_boundaryEncoder_spec .[#8D8D8D,thickness=2].> CodebaseEncoder_boundaryHybrid_adapter : "<color:#8D8D8D>hybrid variant<color:#8D8D8D>"
CodebaseEncoder_boundaryObs_adapter .[#8D8D8D,thickness=2].> CodebaseEncoder_boundaryCnn_spec : "<color:#8D8D8D>declares<color:#8D8D8D>"
CodebaseEncoder_boundaryVggt_adapter .[#8D8D8D,thickness=2].> CodebaseEncoder_boundaryWp_cp_spec : "<color:#8D8D8D>wp_cp readout<color:#8D8D8D>"
CodebaseEncoder_boundaryVggt_adapter .[#8D8D8D,thickness=2].> CodebaseEncoder_boundaryAggregator_spec : "<color:#8D8D8D>aggregator readout<color:#8D8D8D>"
CodebaseEncoder_boundaryVggt_adapter .[#8D8D8D,thickness=2].> CodebaseEncoder_boundaryDense_wp_spec : "<color:#8D8D8D>dense WP readout<color:#8D8D8D>"
CodebaseEncoder_boundaryVggt_adapter .[#8D8D8D,thickness=2].> CodebaseEncoder_boundaryWp_cp : "<color:#8D8D8D>extract world_points + camera_pose<color:#8D8D8D>"
CodebaseEncoder_boundaryVggt_adapter .[#8D8D8D,thickness=2].> CodebaseEncoder_boundaryPooled_agg : "<color:#8D8D8D>pool final global tokens<color:#8D8D8D>"
CodebaseEncoder_boundaryVggt_adapter .[#8D8D8D,thickness=2].> CodebaseEncoder_boundaryDense_wp : "<color:#8D8D8D>request return_dense<color:#8D8D8D>"
CodebaseEncoder_boundaryHybrid_adapter .[#8D8D8D,thickness=2].> CodebaseEncoder_boundaryHybrid_spec : "<color:#8D8D8D>hybrid readout<color:#8D8D8D>"
CodebaseEncoder_boundaryHybrid_adapter .[#8D8D8D,thickness=2].> CodebaseEncoder_boundaryHybrid_obs : "<color:#8D8D8D>store two fields<color:#8D8D8D>"
CodebaseEncoder_boundaryEncoder_spec .[#8D8D8D,thickness=2].> CodebaseAgent_boundary : "<color:#8D8D8D>agent overrides and module_cls<color:#8D8D8D>"
CodebaseEncoder_boundaryObs_adapter .[#8D8D8D,thickness=2].> CodebaseTraining_loop : "<color:#8D8D8D>obs adapter<color:#8D8D8D>"
CodebaseEncoder_boundaryVggt_adapter .[#8D8D8D,thickness=2].> CodebaseTraining_loop : "<color:#8D8D8D>obs adapter<color:#8D8D8D>"
CodebaseEncoder_boundaryHybrid_adapter .[#8D8D8D,thickness=2].> CodebaseTraining_loop : "<color:#8D8D8D>obs adapter<color:#8D8D8D>"
@enduml
`;case"view_12nwcpr":return`@startuml
title "Trainer Replay Loop"
top to bottom direction

hide stereotype
skinparam ranksep 60
skinparam nodesep 30
skinparam {
  arrowFontSize 10
  defaultTextAlignment center
  wrapWidth 200
  maxMessageSize 100
  shadowing false
}

skinparam rectangle<<CodebaseEncoder_boundary>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseAgent_boundary>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseTraining_loopTrainer>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam database<<CodebaseTraining_loopReplay>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam database<<CodebaseTraining_loopMetrics>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseTraining_loopSampled_batch>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseTraining_loopConvert_batch>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseTraining_loopObs_batch>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
rectangle "==Encoder and adapter layer\\n<size:10>[src/r2dreamer/encoders and adapters]</size>" <<CodebaseEncoder_boundary>> as CodebaseEncoder_boundary
rectangle "==R2Dreamer agent\\n<size:10>[src/r2dreamer/agent.py]</size>" <<CodebaseAgent_boundary>> as CodebaseAgent_boundary
rectangle "Trainer and replay loop" <<CodebaseTraining_loop>> as CodebaseTraining_loop {
  skinparam RectangleBorderColor<<CodebaseTraining_loop>> #3b82f6
  skinparam RectangleFontColor<<CodebaseTraining_loop>> #3b82f6
  skinparam RectangleBorderStyle<<CodebaseTraining_loop>> dashed

  rectangle "==Trainer\\n<size:10>[Prefill, collect, train_ratio, val, log, checkpoint]</size>\\n\\nOwns environment stepping, replay insertion, batch sampling, validation env loop, metrics, W&B logging, videos, checkpoints, and MANIFEST writes." <<CodebaseTraining_loopTrainer>> as CodebaseTraining_loopTrainer
  database "==ReplayBuffer\\n<size:10>[NumPy ring buffer]</size>\\n\\nStores observations as one array or a mapping of named fields.\\nSamples fixed length (B,T) windows and returns JAX arrays with is_first, actions, rewards, dones, and terminals." <<CodebaseTraining_loopReplay>> as CodebaseTraining_loopReplay
  database "==Run artifacts\\n<size:10>[output/runs, metrics.csv, checkpoints, MANIFEST.json, W&B]</size>\\n\\nTrainer writes local checkpoints and metrics, manifest provenance, optional videos, and W&B summaries." <<CodebaseTraining_loopMetrics>> as CodebaseTraining_loopMetrics
  rectangle "==Sampled sequence batch\\n<size:10>[obs + actions + rewards + dones + terminals + is_first]</size>\\n\\nReplay sample window consumed by convert_batch and agent.train_step." <<CodebaseTraining_loopSampled_batch>> as CodebaseTraining_loopSampled_batch
  rectangle "==convert_batch\\n<size:10>[trainer.py]</size>\\n\\nOne-hot encodes actions to (B,T,A), maps dones to is_last, terminals to is_terminal, and preserves is_first." <<CodebaseTraining_loopConvert_batch>> as CodebaseTraining_loopConvert_batch
  rectangle "==obs_batch bridge\\n<size:10>[src/r2dreamer/obs_batch.py]</size>\\n\\nNormalizes CNN RGB, casts VGGT features to float32, packs hybrid dict observations, and reshapes observations to B*T before the Flax encoder." <<CodebaseTraining_loopObs_batch>> as CodebaseTraining_loopObs_batch
}

CodebaseEncoder_boundary .[#8D8D8D,thickness=2].> CodebaseTraining_loopTrainer : "<color:#8D8D8D>obs adapter<color:#8D8D8D>"
CodebaseAgent_boundary .[#8D8D8D,thickness=2].> CodebaseTraining_loopTrainer : "<color:#8D8D8D>action and metrics<color:#8D8D8D>"
CodebaseTraining_loopTrainer .[#8D8D8D,thickness=2].> CodebaseTraining_loopReplay : "<color:#8D8D8D>add transition<color:#8D8D8D>"
CodebaseTraining_loopTrainer .[#8D8D8D,thickness=2].> CodebaseTraining_loopMetrics : "<color:#8D8D8D>log and save<color:#8D8D8D>"
CodebaseTraining_loopReplay .[#8D8D8D,thickness=2].> CodebaseTraining_loopSampled_batch : "<color:#8D8D8D>sample (B,T)<color:#8D8D8D>"
CodebaseTraining_loopSampled_batch .[#8D8D8D,thickness=2].> CodebaseTraining_loopConvert_batch : "<color:#8D8D8D>format batch<color:#8D8D8D>"
CodebaseTraining_loopConvert_batch .[#8D8D8D,thickness=2].> CodebaseTraining_loopObs_batch : "<color:#8D8D8D>agent boundary<color:#8D8D8D>"
CodebaseTraining_loopTrainer .[#8D8D8D,thickness=2].> CodebaseAgent_boundary : "<color:#8D8D8D>act and train_step<color:#8D8D8D>"
CodebaseTraining_loopObs_batch .[#8D8D8D,thickness=2].> CodebaseAgent_boundary : "<color:#8D8D8D>B*T observations<color:#8D8D8D>"
@enduml
`;case"view_iera9v":return`@startuml
title "R2Dreamer Agent Internals"
top to bottom direction

hide stereotype
skinparam ranksep 60
skinparam nodesep 30
skinparam {
  arrowFontSize 10
  defaultTextAlignment center
  wrapWidth 200
  maxMessageSize 100
  shadowing false
}

skinparam rectangle<<CodebaseEncoder_boundary>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseTraining_loop>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseEvaluation>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseAgent_boundaryConfig>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseAgent_boundaryEncoder_mod>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseAgent_boundaryEmbed>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseAgent_boundaryRssm>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseAgent_boundaryRssm_feat>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseAgent_boundaryHeads>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseAgent_boundaryLosses>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseAgent_boundaryAgent>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
rectangle "==Encoder and adapter layer\\n<size:10>[src/r2dreamer/encoders and adapters]</size>" <<CodebaseEncoder_boundary>> as CodebaseEncoder_boundary
rectangle "==Trainer and replay loop\\n<size:10>[src/r2dreamer/trainer.py, src/buffer/replay_buffer.py]</size>" <<CodebaseTraining_loop>> as CodebaseTraining_loop
rectangle "==Evaluation and parity workflows\\n<size:10>[src/r2dreamer/launch/evaluate.py, launch/parity]</size>" <<CodebaseEvaluation>> as CodebaseEvaluation
rectangle "R2Dreamer agent" <<CodebaseAgent_boundary>> as CodebaseAgent_boundary {
  skinparam RectangleBorderColor<<CodebaseAgent_boundary>> #3b82f6
  skinparam RectangleFontColor<<CodebaseAgent_boundary>> #3b82f6
  skinparam RectangleBorderStyle<<CodebaseAgent_boundary>> dashed

  rectangle "==R2DreamerConfig\\n<size:10>[src/r2dreamer/config.py]</size>\\n\\nConfig-first source of truth for RSSM sizes, encoder choice, loss scales, batch/sequence settings, train_ratio, replay capacity, optimizer, and run defaults." <<CodebaseAgent_boundaryConfig>> as CodebaseAgent_boundaryConfig
  rectangle "==Flax observation encoder\\n<size:10>[ConvEncoder, VGGTEncoder, VGGTAggregatorMLPEncoder, WPConvEncoder, HybridEncoder]</size>\\n\\nChosen by EncoderSpec.module_cls and instantiated inside the agent, on the JAX side of the boundary." <<CodebaseAgent_boundaryEncoder_mod>> as CodebaseAgent_boundaryEncoder_mod
  rectangle "==Observation embed\\n<size:10>[usually 1024 or hybrid 2048]</size>\\n\\nEncoder output that conditions the RSSM posterior." <<CodebaseAgent_boundaryEmbed>> as CodebaseAgent_boundaryEmbed
  rectangle "==R2RSSM\\n<size:10>[src/r2dreamer/world_model/rssm.py]</size>\\n\\nBlock-GRU latent dynamics with observe, img_step, and get_feat." <<CodebaseAgent_boundaryRssm>> as CodebaseAgent_boundaryRssm
  rectangle "==RSSM feature\\n<size:10>[deter_size + stoch_classes*stoch_discrete]</size>\\n\\nDefault feature size is 2048 deterministic + 512 stochastic = 2560." <<CodebaseAgent_boundaryRssm_feat>> as CodebaseAgent_boundaryRssm_feat
  rectangle "==Prediction and control heads\\n<size:10>[src/r2dreamer/world_model/heads.py]</size>" <<CodebaseAgent_boundaryHeads>> as CodebaseAgent_boundaryHeads
  rectangle "==Loss composition\\n<size:10>[world_model, behavior, representation]</size>" <<CodebaseAgent_boundaryLosses>> as CodebaseAgent_boundaryLosses
  rectangle "==R2DreamerAgent\\n<size:10>[JAX/Flax composition root]</size>\\n\\nOwns params, optimizer state, slow critic EMA, acting state, JIT-compiled train_step and act.\\nA single shared forward pass feeds world-model, behavior, and representation losses under one jax.grad." <<CodebaseAgent_boundaryAgent>> as CodebaseAgent_boundaryAgent
}

CodebaseEncoder_boundary .[#8D8D8D,thickness=2].> CodebaseAgent_boundaryConfig : "<color:#8D8D8D>agent overrides and module_cls<color:#8D8D8D>"
CodebaseTraining_loop .[#8D8D8D,thickness=2].> CodebaseAgent_boundaryAgent : "<color:#8D8D8D>act and train_step<color:#8D8D8D>"
CodebaseTraining_loop .[#8D8D8D,thickness=2].> CodebaseAgent_boundaryEncoder_mod : "<color:#8D8D8D>B*T observations<color:#8D8D8D>"
CodebaseEvaluation .[#8D8D8D,thickness=2].> CodebaseAgent_boundaryAgent : "<color:#8D8D8D>compare behavior<color:#8D8D8D>"
CodebaseAgent_boundaryConfig .[#8D8D8D,thickness=2].> CodebaseAgent_boundaryAgent : "<color:#8D8D8D>initialize<color:#8D8D8D>"
CodebaseAgent_boundaryLosses .[#8D8D8D,thickness=2].> CodebaseAgent_boundaryAgent : "<color:#8D8D8D>update params and optimizer state<color:#8D8D8D>"
CodebaseAgent_boundaryEncoder_mod .[#8D8D8D,thickness=2].> CodebaseAgent_boundaryEmbed : "<color:#8D8D8D>encode obs<color:#8D8D8D>"
CodebaseAgent_boundaryEmbed .[#8D8D8D,thickness=2].> CodebaseAgent_boundaryRssm : "<color:#8D8D8D>posterior observe<color:#8D8D8D>"
CodebaseAgent_boundaryRssm .[#8D8D8D,thickness=2].> CodebaseAgent_boundaryRssm_feat : "<color:#8D8D8D>post states and feat<color:#8D8D8D>"
CodebaseAgent_boundaryRssm_feat .[#8D8D8D,thickness=2].> CodebaseAgent_boundaryHeads : "<color:#8D8D8D>[...]<color:#8D8D8D>"
CodebaseAgent_boundaryRssm_feat .[#8D8D8D,thickness=2].> CodebaseAgent_boundaryLosses : "<color:#8D8D8D>[...]<color:#8D8D8D>"
CodebaseAgent_boundaryAgent .[#8D8D8D,thickness=2].> CodebaseTraining_loop : "<color:#8D8D8D>action and metrics<color:#8D8D8D>"
CodebaseAgent_boundaryAgent .[#8D8D8D,thickness=2].> CodebaseEvaluation : "<color:#8D8D8D>save<color:#8D8D8D>"
@enduml
`;case"view_12tzet7":return`@startuml
title "Loss Composition"
top to bottom direction

hide stereotype
skinparam ranksep 60
skinparam nodesep 30
skinparam {
  arrowFontSize 10
  defaultTextAlignment center
  wrapWidth 200
  maxMessageSize 100
  shadowing false
}

skinparam rectangle<<CodebaseAgent_boundaryRssm_feat>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseAgent_boundaryLossesWm_loss>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseAgent_boundaryLossesBehavior_loss>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseAgent_boundaryLossesRep_loss>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseAgent_boundaryLossesOptimizer>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseAgent_boundaryAgent>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
rectangle "==RSSM feature\\n<size:10>[deter_size + stoch_classes*stoch_discrete]</size>\\n\\nDefault feature size is 2048 deterministic + 512 stochastic = 2560." <<CodebaseAgent_boundaryRssm_feat>> as CodebaseAgent_boundaryRssm_feat
rectangle "Loss composition" <<CodebaseAgent_boundaryLosses>> as CodebaseAgent_boundaryLosses {
  skinparam RectangleBorderColor<<CodebaseAgent_boundaryLosses>> #3b82f6
  skinparam RectangleFontColor<<CodebaseAgent_boundaryLosses>> #3b82f6
  skinparam RectangleBorderStyle<<CodebaseAgent_boundaryLosses>> dashed

  rectangle "==world_model_loss\\n<size:10>[KL dyn/rep + reward + continue + optional decoder]</size>" <<CodebaseAgent_boundaryLossesWm_loss>> as CodebaseAgent_boundaryLossesWm_loss
  rectangle "==behavior_loss\\n<size:10>[Detached imagination, lambda-return, actor and critic losses]</size>" <<CodebaseAgent_boundaryLossesBehavior_loss>> as CodebaseAgent_boundaryLossesBehavior_loss
  rectangle "==representation_loss\\n<size:10>[Barlow Twins + replay-value]</size>" <<CodebaseAgent_boundaryLossesRep_loss>> as CodebaseAgent_boundaryLossesRep_loss
  rectangle "==LaProp + AGC update\\n<size:10>[src/shared/optim.py]</size>\\n\\nWeighted total loss is differentiated once and updates the single params pytree." <<CodebaseAgent_boundaryLossesOptimizer>> as CodebaseAgent_boundaryLossesOptimizer
}
rectangle "==R2DreamerAgent\\n<size:10>[JAX/Flax composition root]</size>\\n\\nOwns params, optimizer state, slow critic EMA, acting state, JIT-compiled train_step and act.\\nA single shared forward pass feeds world-model, behavior, and representation losses under one jax.grad." <<CodebaseAgent_boundaryAgent>> as CodebaseAgent_boundaryAgent

CodebaseAgent_boundaryRssm_feat .[#8D8D8D,thickness=2].> CodebaseAgent_boundaryLossesWm_loss : "<color:#8D8D8D>world-model terms<color:#8D8D8D>"
CodebaseAgent_boundaryRssm_feat .[#8D8D8D,thickness=2].> CodebaseAgent_boundaryLossesBehavior_loss : "<color:#8D8D8D>imagination starts<color:#8D8D8D>"
CodebaseAgent_boundaryRssm_feat .[#8D8D8D,thickness=2].> CodebaseAgent_boundaryLossesRep_loss : "<color:#8D8D8D>representation terms<color:#8D8D8D>"
CodebaseAgent_boundaryLossesWm_loss .[#8D8D8D,thickness=2].> CodebaseAgent_boundaryLossesOptimizer : "<color:#8D8D8D>weighted sum<color:#8D8D8D>"
CodebaseAgent_boundaryLossesBehavior_loss .[#8D8D8D,thickness=2].> CodebaseAgent_boundaryLossesOptimizer : "<color:#8D8D8D>weighted sum<color:#8D8D8D>"
CodebaseAgent_boundaryLossesRep_loss .[#8D8D8D,thickness=2].> CodebaseAgent_boundaryLossesOptimizer : "<color:#8D8D8D>weighted sum<color:#8D8D8D>"
CodebaseAgent_boundaryLossesOptimizer .[#8D8D8D,thickness=2].> CodebaseAgent_boundaryAgent : "<color:#8D8D8D>update params and optimizer state<color:#8D8D8D>"
@enduml
`;case"view_1byfr7e":return`@startuml
title "VGGT Streaming Encoder"
top to bottom direction

hide stereotype
skinparam ranksep 60
skinparam nodesep 30
skinparam {
  arrowFontSize 10
  defaultTextAlignment center
  wrapWidth 200
  maxMessageSize 100
  shadowing false
}

skinparam rectangle<<CodebaseVggt_boundaryExtractor>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseVggt_boundaryAggregator>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseEncoder_boundary>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam database<<CodebaseVggt_boundaryAgg_cache>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseVggt_boundaryCamera_head>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseVggt_boundaryPoint_head>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseVggt_boundaryAggregator_tokens>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam database<<CodebaseVggt_boundaryCamera_cache>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseVggt_boundaryCamera_pose>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseVggt_boundaryDense_world_points>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseVggt_boundaryWorld_points>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
rectangle "VGGT production encoder" <<CodebaseVggt_boundary>> as CodebaseVggt_boundary {
  skinparam RectangleBorderColor<<CodebaseVggt_boundary>> #3b82f6
  skinparam RectangleFontColor<<CodebaseVggt_boundary>> #3b82f6
  skinparam RectangleBorderStyle<<CodebaseVggt_boundary>> dashed

  rectangle "==JAXVGGTFeatureExtractor\\n<size:10>[src/vggt/jax/feature_extractor.py]</size>\\n\\nDrop-in JAX backend for StreamVGGT.\\nLoads HuggingFace StreamVGGT weights, keeps streaming caches as instance state, resets at episode boundaries, and exposes extract(rgb)." <<CodebaseVggt_boundaryExtractor>> as CodebaseVggt_boundaryExtractor
  rectangle "==Aggregator\\n<size:10>[24 alternating attention blocks]</size>\\n\\nConsumes fixed 518x518 RGB, emits camera + register + patch tokens, and supports streaming cache paths." <<CodebaseVggt_boundaryAggregator>> as CodebaseVggt_boundaryAggregator
  database "==Aggregator padded KV cache\\n<size:10>[per-block (k_pad, v_pad, valid_len)]</size>\\n\\nFixed-shape padded cache keeps JIT stable. Per-block budgets are Python static args; eviction uses budgeted cache control." <<CodebaseVggt_boundaryAgg_cache>> as CodebaseVggt_boundaryAgg_cache
  rectangle "==CameraHead\\n<size:10>[pose output]</size>" <<CodebaseVggt_boundaryCamera_head>> as CodebaseVggt_boundaryCamera_head
  rectangle "==DPTHead\\n<size:10>[dense 518 x 518 x 3 points]</size>" <<CodebaseVggt_boundaryPoint_head>> as CodebaseVggt_boundaryPoint_head
  rectangle "==Aggregator features\\n<size:10>[1374 x 1024 global stream]</size>\\n\\nFinal global-stream tokens: 1 camera + 4 registers + 37x37 patches, 1024 dims. Pooled variants drop registers when flattening raw or pooling patches." <<CodebaseVggt_boundaryAggregator_tokens>> as CodebaseVggt_boundaryAggregator_tokens
  database "==Camera-head padded KV cache\\n<size:10>[max_camera_frames guard]</size>\\n\\nCamera head cache fails loudly on overflow instead of silently clamping dynamic updates." <<CodebaseVggt_boundaryCamera_cache>> as CodebaseVggt_boundaryCamera_cache
  rectangle "==camera_pose\\n<size:10>[9 float32 values]</size>" <<CodebaseVggt_boundaryCamera_pose>> as CodebaseVggt_boundaryCamera_pose
  rectangle "==dense_world_points\\n<size:10>[518 x 518 x 3 float32]</size>" <<CodebaseVggt_boundaryDense_world_points>> as CodebaseVggt_boundaryDense_world_points
  rectangle "==world_points\\n<size:10>[K x K x 3]</size>\\n\\nDense point map pooled to K=37 by default or K=64 for vggt_wp_cp_64." <<CodebaseVggt_boundaryWorld_points>> as CodebaseVggt_boundaryWorld_points
}
rectangle "==Encoder and adapter layer\\n<size:10>[src/r2dreamer/encoders and adapters]</size>" <<CodebaseEncoder_boundary>> as CodebaseEncoder_boundary

CodebaseVggt_boundaryExtractor .[#8D8D8D,thickness=2].> CodebaseVggt_boundaryAggregator : "<color:#8D8D8D>run one frame<color:#8D8D8D>"
CodebaseVggt_boundaryAggregator .[#8D8D8D,thickness=2].> CodebaseVggt_boundaryAgg_cache : "<color:#8D8D8D>read and update<color:#8D8D8D>"
CodebaseVggt_boundaryAggregator .[#8D8D8D,thickness=2].> CodebaseVggt_boundaryCamera_head : "<color:#8D8D8D>if compute_heads=True<color:#8D8D8D>"
CodebaseVggt_boundaryAggregator .[#8D8D8D,thickness=2].> CodebaseVggt_boundaryPoint_head : "<color:#8D8D8D>if compute_heads=True<color:#8D8D8D>"
CodebaseVggt_boundaryAggregator .[#8D8D8D,thickness=2].> CodebaseVggt_boundaryAggregator_tokens : "<color:#8D8D8D>expose final global stream<color:#8D8D8D>"
CodebaseVggt_boundaryCamera_head .[#8D8D8D,thickness=2].> CodebaseVggt_boundaryCamera_cache : "<color:#8D8D8D>read and update<color:#8D8D8D>"
CodebaseVggt_boundaryCamera_head .[#8D8D8D,thickness=2].> CodebaseVggt_boundaryCamera_pose : "<color:#8D8D8D>pose<color:#8D8D8D>"
CodebaseVggt_boundaryPoint_head .[#8D8D8D,thickness=2].> CodebaseVggt_boundaryDense_world_points : "<color:#8D8D8D>dense points<color:#8D8D8D>"
CodebaseVggt_boundaryDense_world_points .[#8D8D8D,thickness=2].> CodebaseVggt_boundaryWorld_points : "<color:#8D8D8D>pool to K x K<color:#8D8D8D>"
CodebaseVggt_boundaryExtractor .[#8D8D8D,thickness=2].> CodebaseEncoder_boundary : "<color:#8D8D8D>[...]<color:#8D8D8D>"
@enduml
`;case"view_118k4sm":return`@startuml
title "Evaluation And Parity"
top to bottom direction

hide stereotype
skinparam ranksep 60
skinparam nodesep 30
skinparam {
  arrowFontSize 10
  defaultTextAlignment center
  wrapWidth 200
  maxMessageSize 100
  shadowing false
}

skinparam person<<Researcher>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseEvaluationParity>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseAgent_boundary>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam rectangle<<CodebaseEvaluationEvaluate>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
skinparam database<<CodebaseEvaluationCheckpoints>>{
  BackgroundColor #3b82f6
  FontColor #eff6ff
  BorderColor #2563eb
}
person "==Researcher / operator\\n<size:10>[CLI, SLURM, W&B]</size>\\n\\nStarts training, evaluation, profiling, and analysis runs.\\nThe current code is organized around script-level run selection, launcher registries, and a JAX/Flax R2Dreamer agent." <<Researcher>> as Researcher
rectangle "Evaluation and parity workflows" <<CodebaseEvaluation>> as CodebaseEvaluation {
  skinparam RectangleBorderColor<<CodebaseEvaluation>> #3b82f6
  skinparam RectangleFontColor<<CodebaseEvaluation>> #3b82f6
  skinparam RectangleBorderStyle<<CodebaseEvaluation>> dashed

  rectangle "==parity workflows\\n<size:10>[train_parity.py, benchmark.py]</size>\\n\\nJAX/PyTorch parity training and benchmark commands for debugging numerical drift." <<CodebaseEvaluationParity>> as CodebaseEvaluationParity
  rectangle "==evaluate()\\n<size:10>[checkpoint evaluation]</size>\\n\\nLoads a policy checkpoint, constructs the matching env and encoder, runs episodes, and logs metrics." <<CodebaseEvaluationEvaluate>> as CodebaseEvaluationEvaluate
  database "==Policy checkpoints\\n<size:10>[pickle step_*.pkl]</size>\\n\\nContain params, opt_state, slow_critic_params, ema_state, and step." <<CodebaseEvaluationCheckpoints>> as CodebaseEvaluationCheckpoints
}
rectangle "==R2Dreamer agent\\n<size:10>[src/r2dreamer/agent.py]</size>" <<CodebaseAgent_boundary>> as CodebaseAgent_boundary

Researcher .[#8D8D8D,thickness=2].> CodebaseEvaluationEvaluate : "<color:#8D8D8D>runs evaluation<color:#8D8D8D>"
CodebaseAgent_boundary .[#8D8D8D,thickness=2].> CodebaseEvaluationCheckpoints : "<color:#8D8D8D>save<color:#8D8D8D>"
CodebaseEvaluationEvaluate .[#8D8D8D,thickness=2].> CodebaseEvaluationCheckpoints : "<color:#8D8D8D>load<color:#8D8D8D>"
CodebaseEvaluationCheckpoints .[#8D8D8D,thickness=2].> CodebaseEvaluationEvaluate : "<color:#8D8D8D>restore<color:#8D8D8D>"
CodebaseEvaluationParity .[#8D8D8D,thickness=2].> CodebaseAgent_boundary : "<color:#8D8D8D>compare behavior<color:#8D8D8D>"
@enduml
`;default:throw new Error("Unknown viewId: "+e)}}export{n as pumlSource};
