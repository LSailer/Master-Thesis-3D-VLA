function a(e){switch(e){case"index":return`---
title: "Current Code Architecture"
---
graph TB
  Researcher[fa:fa-user Researcher / operator]
  Codebase[Master-Thesis-3D-VLA current code]
  Researcher -. "[...]" .-> Codebase
`;case"view_1dcnbvb":return`---
title: "Current Code Overview"
---
graph TB
  Researcher[fa:fa-user Researcher / operator]
  subgraph Codebase["Master-Thesis-3D-VLA current code"]
    Codebase.Orchestration[Experiment orchestration]
    Codebase.Vggt_boundary[VGGT production encoder]
    Codebase.Environment[Environment layer]
    Codebase.Encoder_boundary[Encoder and adapter layer]
    Codebase.Training_loop[Trainer and replay loop]
    Codebase.Agent_boundary[R2Dreamer agent]
    Codebase.Evaluation[Evaluation and parity workflows]
  end
  Researcher -. "starts run" .-> Codebase.Orchestration
  Researcher -. "runs evaluation" .-> Codebase.Evaluation
  Codebase.Orchestration -. "[...]" .-> Codebase.Environment
  Codebase.Orchestration -. "resolve encoder" .-> Codebase.Encoder_boundary
  Codebase.Environment -. "[...]" .-> Codebase.Encoder_boundary
  Codebase.Encoder_boundary -. "obs adapter" .-> Codebase.Training_loop
  Codebase.Encoder_boundary -. "agent overrides and module_cls" .-> Codebase.Agent_boundary
  Codebase.Vggt_boundary -. "[...]" .-> Codebase.Encoder_boundary
  Codebase.Training_loop -. "[...]" .-> Codebase.Agent_boundary
  Codebase.Agent_boundary -. "action and metrics" .-> Codebase.Training_loop
  Codebase.Agent_boundary -. "save" .-> Codebase.Evaluation
  Codebase.Evaluation -. "compare behavior" .-> Codebase.Agent_boundary
`;case"view_1omwq3l":return`---
title: "Experiment Orchestration"
---
graph TB
  Researcher[fa:fa-user Researcher / operator]
  subgraph CodebaseOrchestration["Experiment orchestration"]
    CodebaseOrchestration.Slurm_configs[SLURM configs]
    CodebaseOrchestration.Run_dispatcher[scripts/r2dreamer/run.py]
    CodebaseOrchestration.Run_configs[RUN_CONFIGS]
    CodebaseOrchestration.Public_entrypoint[src/main.py]
    CodebaseOrchestration.Train_entry[src.r2dreamer.launch.train.train]
    CodebaseOrchestration.Registries[launch registries]
    CodebaseOrchestration.Curriculum([Curriculum JSON])
  end
  CodebaseEnvironment[Environment layer]
  CodebaseEncoder_boundary[Encoder and adapter layer]
  Researcher -. "starts run" .-> CodebaseOrchestration.Run_dispatcher
  CodebaseOrchestration.Run_dispatcher -. "select run id" .-> CodebaseOrchestration.Run_configs
  CodebaseOrchestration.Slurm_configs -. "render or call" .-> CodebaseOrchestration.Run_dispatcher
  CodebaseOrchestration.Run_configs -. "call src.main.train" .-> CodebaseOrchestration.Public_entrypoint
  CodebaseOrchestration.Public_entrypoint -. "dispatch train" .-> CodebaseOrchestration.Train_entry
  CodebaseOrchestration.Train_entry -. "resolve env and encoder" .-> CodebaseOrchestration.Registries
  CodebaseOrchestration.Train_entry -. "resolve Habitat curriculum" .-> CodebaseOrchestration.Curriculum
  CodebaseOrchestration.Train_entry -. "[...]" .-> CodebaseEnvironment
  CodebaseOrchestration.Registries -. "resolve encoder" .-> CodebaseEncoder_boundary
`;case"view_spshjc":return`---
title: "Encoder Specs And Adapters"
---
graph TB
  CodebaseOrchestration[Experiment orchestration]
  CodebaseEnvironment[Environment layer]
  CodebaseVggt_boundary[VGGT production encoder]
  subgraph CodebaseEncoder_boundary["Encoder and adapter layer"]
    CodebaseEncoder_boundary.Encoder_spec[EncoderSpec]
    CodebaseEncoder_boundary.Obs_adapter[ObsAdapter]
    CodebaseEncoder_boundary.Vggt_adapter[VGGTObsAdapter]
    CodebaseEncoder_boundary.Hybrid_adapter[HybridObsAdapter]
    CodebaseEncoder_boundary.Cnn_spec[cnn]
    CodebaseEncoder_boundary.Wp_cp_spec[vggt / vggt_wp_cp_64]
    CodebaseEncoder_boundary.Aggregator_spec[vggt_aggregator_mlp]
    CodebaseEncoder_boundary.Dense_wp_spec[vggt_wp_dense_cnn]
    CodebaseEncoder_boundary.Wp_cp[WP/CP features]
    CodebaseEncoder_boundary.Pooled_agg[Pooled aggregator features]
    CodebaseEncoder_boundary.Dense_wp[Dense world-point map]
    CodebaseEncoder_boundary.Hybrid_spec[hybrid]
    CodebaseEncoder_boundary.Hybrid_obs[Hybrid replay observation]
  end
  CodebaseAgent_boundary[R2Dreamer agent]
  CodebaseTraining_loop[Trainer and replay loop]
  CodebaseOrchestration -. "resolve encoder" .-> CodebaseEncoder_boundary.Encoder_spec
  CodebaseEnvironment -. "CNN path" .-> CodebaseEncoder_boundary.Obs_adapter
  CodebaseEnvironment -. "VGGT standalone path" .-> CodebaseEncoder_boundary.Vggt_adapter
  CodebaseEnvironment -. "hybrid path" .-> CodebaseEncoder_boundary.Hybrid_adapter
  CodebaseVggt_boundary -. "extract features" .-> CodebaseEncoder_boundary.Vggt_adapter
  CodebaseVggt_boundary -. "extract WP/CP" .-> CodebaseEncoder_boundary.Hybrid_adapter
  CodebaseEncoder_boundary.Encoder_spec -. "default RGB adapter" .-> CodebaseEncoder_boundary.Obs_adapter
  CodebaseEncoder_boundary.Encoder_spec -. "VGGT standalone variants" .-> CodebaseEncoder_boundary.Vggt_adapter
  CodebaseEncoder_boundary.Encoder_spec -. "hybrid variant" .-> CodebaseEncoder_boundary.Hybrid_adapter
  CodebaseEncoder_boundary.Obs_adapter -. "declares" .-> CodebaseEncoder_boundary.Cnn_spec
  CodebaseEncoder_boundary.Vggt_adapter -. "wp_cp readout" .-> CodebaseEncoder_boundary.Wp_cp_spec
  CodebaseEncoder_boundary.Vggt_adapter -. "aggregator readout" .-> CodebaseEncoder_boundary.Aggregator_spec
  CodebaseEncoder_boundary.Vggt_adapter -. "dense WP readout" .-> CodebaseEncoder_boundary.Dense_wp_spec
  CodebaseEncoder_boundary.Vggt_adapter -. "extract world_points + camera_pose" .-> CodebaseEncoder_boundary.Wp_cp
  CodebaseEncoder_boundary.Vggt_adapter -. "pool final global tokens" .-> CodebaseEncoder_boundary.Pooled_agg
  CodebaseEncoder_boundary.Vggt_adapter -. "request return_dense" .-> CodebaseEncoder_boundary.Dense_wp
  CodebaseEncoder_boundary.Hybrid_adapter -. "hybrid readout" .-> CodebaseEncoder_boundary.Hybrid_spec
  CodebaseEncoder_boundary.Hybrid_adapter -. "store two fields" .-> CodebaseEncoder_boundary.Hybrid_obs
  CodebaseEncoder_boundary.Encoder_spec -. "agent overrides and module_cls" .-> CodebaseAgent_boundary
  CodebaseEncoder_boundary.Obs_adapter -. "obs adapter" .-> CodebaseTraining_loop
  CodebaseEncoder_boundary.Vggt_adapter -. "obs adapter" .-> CodebaseTraining_loop
  CodebaseEncoder_boundary.Hybrid_adapter -. "obs adapter" .-> CodebaseTraining_loop
`;case"view_12nwcpr":return`---
title: "Trainer Replay Loop"
---
graph TB
  CodebaseEncoder_boundary[Encoder and adapter layer]
  CodebaseAgent_boundary[R2Dreamer agent]
  subgraph CodebaseTraining_loop["Trainer and replay loop"]
    CodebaseTraining_loop.Trainer[Trainer]
    CodebaseTraining_loop.Replay([ReplayBuffer])
    CodebaseTraining_loop.Metrics([Run artifacts])
    CodebaseTraining_loop.Sampled_batch[Sampled sequence batch]
    CodebaseTraining_loop.Convert_batch[convert_batch]
    CodebaseTraining_loop.Obs_batch[obs_batch bridge]
  end
  CodebaseEncoder_boundary -. "obs adapter" .-> CodebaseTraining_loop.Trainer
  CodebaseAgent_boundary -. "action and metrics" .-> CodebaseTraining_loop.Trainer
  CodebaseTraining_loop.Trainer -. "add transition" .-> CodebaseTraining_loop.Replay
  CodebaseTraining_loop.Trainer -. "log and save" .-> CodebaseTraining_loop.Metrics
  CodebaseTraining_loop.Replay -. "sample (B,T)" .-> CodebaseTraining_loop.Sampled_batch
  CodebaseTraining_loop.Sampled_batch -. "format batch" .-> CodebaseTraining_loop.Convert_batch
  CodebaseTraining_loop.Convert_batch -. "agent boundary" .-> CodebaseTraining_loop.Obs_batch
  CodebaseTraining_loop.Trainer -. "act and train_step" .-> CodebaseAgent_boundary
  CodebaseTraining_loop.Obs_batch -. "B*T observations" .-> CodebaseAgent_boundary
`;case"view_iera9v":return`---
title: "R2Dreamer Agent Internals"
---
graph TB
  CodebaseEncoder_boundary[Encoder and adapter layer]
  CodebaseTraining_loop[Trainer and replay loop]
  CodebaseEvaluation[Evaluation and parity workflows]
  subgraph CodebaseAgent_boundary["R2Dreamer agent"]
    CodebaseAgent_boundary.Config[R2DreamerConfig]
    CodebaseAgent_boundary.Encoder_mod[Flax observation encoder]
    CodebaseAgent_boundary.Embed[Observation embed]
    CodebaseAgent_boundary.Rssm[R2RSSM]
    CodebaseAgent_boundary.Rssm_feat[RSSM feature]
    CodebaseAgent_boundary.Heads[Prediction and control heads]
    CodebaseAgent_boundary.Losses[Loss composition]
    CodebaseAgent_boundary.Agent[R2DreamerAgent]
  end
  CodebaseEncoder_boundary -. "agent overrides and module_cls" .-> CodebaseAgent_boundary.Config
  CodebaseTraining_loop -. "act and train_step" .-> CodebaseAgent_boundary.Agent
  CodebaseTraining_loop -. "B*T observations" .-> CodebaseAgent_boundary.Encoder_mod
  CodebaseEvaluation -. "compare behavior" .-> CodebaseAgent_boundary.Agent
  CodebaseAgent_boundary.Config -. "initialize" .-> CodebaseAgent_boundary.Agent
  CodebaseAgent_boundary.Losses -. "update params and optimizer state" .-> CodebaseAgent_boundary.Agent
  CodebaseAgent_boundary.Encoder_mod -. "encode obs" .-> CodebaseAgent_boundary.Embed
  CodebaseAgent_boundary.Embed -. "posterior observe" .-> CodebaseAgent_boundary.Rssm
  CodebaseAgent_boundary.Rssm -. "post states and feat" .-> CodebaseAgent_boundary.Rssm_feat
  CodebaseAgent_boundary.Rssm_feat -. "[...]" .-> CodebaseAgent_boundary.Heads
  CodebaseAgent_boundary.Rssm_feat -. "[...]" .-> CodebaseAgent_boundary.Losses
  CodebaseAgent_boundary.Agent -. "action and metrics" .-> CodebaseTraining_loop
  CodebaseAgent_boundary.Agent -. "save" .-> CodebaseEvaluation
`;case"view_12tzet7":return`---
title: "Loss Composition"
---
graph TB
  CodebaseAgent_boundaryRssm_feat[RSSM feature]
  subgraph CodebaseAgent_boundaryLosses["Loss composition"]
    CodebaseAgent_boundaryLosses.Wm_loss[world_model_loss]
    CodebaseAgent_boundaryLosses.Behavior_loss[behavior_loss]
    CodebaseAgent_boundaryLosses.Rep_loss[representation_loss]
    CodebaseAgent_boundaryLosses.Optimizer[LaProp + AGC update]
  end
  CodebaseAgent_boundaryAgent[R2DreamerAgent]
  CodebaseAgent_boundaryRssm_feat -. "world-model terms" .-> CodebaseAgent_boundaryLosses.Wm_loss
  CodebaseAgent_boundaryRssm_feat -. "imagination starts" .-> CodebaseAgent_boundaryLosses.Behavior_loss
  CodebaseAgent_boundaryRssm_feat -. "representation terms" .-> CodebaseAgent_boundaryLosses.Rep_loss
  CodebaseAgent_boundaryLosses.Wm_loss -. "weighted sum" .-> CodebaseAgent_boundaryLosses.Optimizer
  CodebaseAgent_boundaryLosses.Behavior_loss -. "weighted sum" .-> CodebaseAgent_boundaryLosses.Optimizer
  CodebaseAgent_boundaryLosses.Rep_loss -. "weighted sum" .-> CodebaseAgent_boundaryLosses.Optimizer
  CodebaseAgent_boundaryLosses.Optimizer -. "update params and optimizer state" .-> CodebaseAgent_boundaryAgent
`;case"view_1byfr7e":return`---
title: "VGGT Streaming Encoder"
---
graph TB
  subgraph CodebaseVggt_boundary["VGGT production encoder"]
    CodebaseVggt_boundary.Extractor[JAXVGGTFeatureExtractor]
    CodebaseVggt_boundary.Aggregator[Aggregator]
    CodebaseVggt_boundary.Agg_cache([Aggregator padded KV cache])
    CodebaseVggt_boundary.Camera_head[CameraHead]
    CodebaseVggt_boundary.Point_head[DPTHead]
    CodebaseVggt_boundary.Aggregator_tokens[Aggregator features]
    CodebaseVggt_boundary.Camera_cache([Camera-head padded KV cache])
    CodebaseVggt_boundary.Camera_pose[camera_pose]
    CodebaseVggt_boundary.Dense_world_points[dense_world_points]
    CodebaseVggt_boundary.World_points[world_points]
  end
  CodebaseEncoder_boundary[Encoder and adapter layer]
  CodebaseVggt_boundary.Extractor -. "run one frame" .-> CodebaseVggt_boundary.Aggregator
  CodebaseVggt_boundary.Aggregator -. "read and update" .-> CodebaseVggt_boundary.Agg_cache
  CodebaseVggt_boundary.Aggregator -. "if compute_heads=True" .-> CodebaseVggt_boundary.Camera_head
  CodebaseVggt_boundary.Aggregator -. "if compute_heads=True" .-> CodebaseVggt_boundary.Point_head
  CodebaseVggt_boundary.Aggregator -. "expose final global stream" .-> CodebaseVggt_boundary.Aggregator_tokens
  CodebaseVggt_boundary.Camera_head -. "read and update" .-> CodebaseVggt_boundary.Camera_cache
  CodebaseVggt_boundary.Camera_head -. "pose" .-> CodebaseVggt_boundary.Camera_pose
  CodebaseVggt_boundary.Point_head -. "dense points" .-> CodebaseVggt_boundary.Dense_world_points
  CodebaseVggt_boundary.Dense_world_points -. "pool to K x K" .-> CodebaseVggt_boundary.World_points
  CodebaseVggt_boundary.Extractor -. "[...]" .-> CodebaseEncoder_boundary
`;case"view_118k4sm":return`---
title: "Evaluation And Parity"
---
graph TB
  Researcher[fa:fa-user Researcher / operator]
  subgraph CodebaseEvaluation["Evaluation and parity workflows"]
    CodebaseEvaluation.Parity[parity workflows]
    CodebaseEvaluation.Evaluate[evaluate()]
    CodebaseEvaluation.Checkpoints([Policy checkpoints])
  end
  CodebaseAgent_boundary[R2Dreamer agent]
  Researcher -. "runs evaluation" .-> CodebaseEvaluation.Evaluate
  CodebaseAgent_boundary -. "save" .-> CodebaseEvaluation.Checkpoints
  CodebaseEvaluation.Evaluate -. "load" .-> CodebaseEvaluation.Checkpoints
  CodebaseEvaluation.Checkpoints -. "restore" .-> CodebaseEvaluation.Evaluate
  CodebaseEvaluation.Parity -. "compare behavior" .-> CodebaseAgent_boundary
`;default:throw new Error("Unknown viewId: "+e)}}export{a as mmdSource};
