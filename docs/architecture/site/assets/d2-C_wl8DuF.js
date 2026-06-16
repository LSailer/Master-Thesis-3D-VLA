function n(e){switch(e){case"index":return`direction: down

Researcher: {
  label: "Researcher / operator"
  shape: person
}
Codebase: {
  label: "Master-Thesis-3D-VLA current code"
}

Researcher -> Codebase: "[...]"
`;case"view_1dcnbvb":return`direction: down

Researcher: {
  label: "Researcher / operator"
  shape: person
}
Codebase: {
  label: "Master-Thesis-3D-VLA current code"

  Orchestration: {
    label: "Experiment orchestration"
  }
  Vggt_boundary: {
    label: "VGGT production encoder"
  }
  Environment: {
    label: "Environment layer"
  }
  Encoder_boundary: {
    label: "Encoder and adapter layer"
  }
  Training_loop: {
    label: "Trainer and replay loop"
  }
  Agent_boundary: {
    label: "R2Dreamer agent"
  }
  Evaluation: {
    label: "Evaluation and parity workflows"
  }
}

Researcher -> Codebase.Orchestration: "starts run"
Researcher -> Codebase.Evaluation: "runs evaluation"
Codebase.Orchestration -> Codebase.Environment: "[...]"
Codebase.Orchestration -> Codebase.Encoder_boundary: "resolve encoder"
Codebase.Environment -> Codebase.Encoder_boundary: "[...]"
Codebase.Encoder_boundary -> Codebase.Training_loop: "obs adapter"
Codebase.Encoder_boundary -> Codebase.Agent_boundary: "agent overrides and module_cls"
Codebase.Vggt_boundary -> Codebase.Encoder_boundary: "[...]"
Codebase.Training_loop -> Codebase.Agent_boundary: "[...]"
Codebase.Agent_boundary -> Codebase.Training_loop: "action and metrics"
Codebase.Agent_boundary -> Codebase.Evaluation: "save"
Codebase.Evaluation -> Codebase.Agent_boundary: "compare behavior"
`;case"view_1omwq3l":return`direction: down

Researcher: {
  label: "Researcher / operator"
  shape: person
}
CodebaseOrchestration: {
  label: "Experiment orchestration"

  Slurm_configs: {
    label: "SLURM configs"
  }
  Run_dispatcher: {
    label: "scripts/r2dreamer/run.py"
  }
  Run_configs: {
    label: "RUN_CONFIGS"
  }
  Public_entrypoint: {
    label: "src/main.py"
  }
  Train_entry: {
    label: "src.r2dreamer.launch.train.train"
  }
  Registries: {
    label: "launch registries"
  }
  Curriculum: {
    label: "Curriculum JSON"
    shape: stored_data
  }
}
CodebaseEnvironment: {
  label: "Environment layer"
}
CodebaseEncoder_boundary: {
  label: "Encoder and adapter layer"
}

Researcher -> CodebaseOrchestration.Run_dispatcher: "starts run"
CodebaseOrchestration.Run_dispatcher -> CodebaseOrchestration.Run_configs: "select run id"
CodebaseOrchestration.Slurm_configs -> CodebaseOrchestration.Run_dispatcher: "render or call"
CodebaseOrchestration.Run_configs -> CodebaseOrchestration.Public_entrypoint: "call src.main.train"
CodebaseOrchestration.Public_entrypoint -> CodebaseOrchestration.Train_entry: "dispatch train"
CodebaseOrchestration.Train_entry -> CodebaseOrchestration.Registries: "resolve env and encoder"
CodebaseOrchestration.Train_entry -> CodebaseOrchestration.Curriculum: "resolve Habitat curriculum"
CodebaseOrchestration.Train_entry -> CodebaseEnvironment: "[...]"
CodebaseOrchestration.Registries -> CodebaseEncoder_boundary: "resolve encoder"
`;case"view_spshjc":return`direction: down

CodebaseOrchestration: {
  label: "Experiment orchestration"
}
CodebaseEnvironment: {
  label: "Environment layer"
}
CodebaseVggt_boundary: {
  label: "VGGT production encoder"
}
CodebaseEncoder_boundary: {
  label: "Encoder and adapter layer"

  Encoder_spec: {
    label: "EncoderSpec"
  }
  Obs_adapter: {
    label: "ObsAdapter"
  }
  Vggt_adapter: {
    label: "VGGTObsAdapter"
  }
  Hybrid_adapter: {
    label: "HybridObsAdapter"
  }
  Cnn_spec: {
    label: "cnn"
  }
  Wp_cp_spec: {
    label: "vggt / vggt_wp_cp_64"
  }
  Aggregator_spec: {
    label: "vggt_aggregator_mlp"
  }
  Dense_wp_spec: {
    label: "vggt_wp_dense_cnn"
  }
  Wp_cp: {
    label: "WP/CP features"
  }
  Pooled_agg: {
    label: "Pooled aggregator features"
  }
  Dense_wp: {
    label: "Dense world-point map"
  }
  Hybrid_spec: {
    label: "hybrid"
  }
  Hybrid_obs: {
    label: "Hybrid replay observation"
  }
}
CodebaseAgent_boundary: {
  label: "R2Dreamer agent"
}
CodebaseTraining_loop: {
  label: "Trainer and replay loop"
}

CodebaseOrchestration -> CodebaseEncoder_boundary.Encoder_spec: "resolve encoder"
CodebaseEnvironment -> CodebaseEncoder_boundary.Obs_adapter: "CNN path"
CodebaseEnvironment -> CodebaseEncoder_boundary.Vggt_adapter: "VGGT standalone path"
CodebaseEnvironment -> CodebaseEncoder_boundary.Hybrid_adapter: "hybrid path"
CodebaseVggt_boundary -> CodebaseEncoder_boundary.Vggt_adapter: "extract features"
CodebaseVggt_boundary -> CodebaseEncoder_boundary.Hybrid_adapter: "extract WP/CP"
CodebaseEncoder_boundary.Encoder_spec -> CodebaseEncoder_boundary.Obs_adapter: "default RGB adapter"
CodebaseEncoder_boundary.Encoder_spec -> CodebaseEncoder_boundary.Vggt_adapter: "VGGT standalone variants"
CodebaseEncoder_boundary.Encoder_spec -> CodebaseEncoder_boundary.Hybrid_adapter: "hybrid variant"
CodebaseEncoder_boundary.Obs_adapter -> CodebaseEncoder_boundary.Cnn_spec: "declares"
CodebaseEncoder_boundary.Vggt_adapter -> CodebaseEncoder_boundary.Wp_cp_spec: "wp_cp readout"
CodebaseEncoder_boundary.Vggt_adapter -> CodebaseEncoder_boundary.Aggregator_spec: "aggregator readout"
CodebaseEncoder_boundary.Vggt_adapter -> CodebaseEncoder_boundary.Dense_wp_spec: "dense WP readout"
CodebaseEncoder_boundary.Vggt_adapter -> CodebaseEncoder_boundary.Wp_cp: "extract world_points + camera_pose"
CodebaseEncoder_boundary.Vggt_adapter -> CodebaseEncoder_boundary.Pooled_agg: "pool final global tokens"
CodebaseEncoder_boundary.Vggt_adapter -> CodebaseEncoder_boundary.Dense_wp: "request return_dense"
CodebaseEncoder_boundary.Hybrid_adapter -> CodebaseEncoder_boundary.Hybrid_spec: "hybrid readout"
CodebaseEncoder_boundary.Hybrid_adapter -> CodebaseEncoder_boundary.Hybrid_obs: "store two fields"
CodebaseEncoder_boundary.Encoder_spec -> CodebaseAgent_boundary: "agent overrides and module_cls"
CodebaseEncoder_boundary.Obs_adapter -> CodebaseTraining_loop: "obs adapter"
CodebaseEncoder_boundary.Vggt_adapter -> CodebaseTraining_loop: "obs adapter"
CodebaseEncoder_boundary.Hybrid_adapter -> CodebaseTraining_loop: "obs adapter"
`;case"view_12nwcpr":return`direction: down

CodebaseEncoder_boundary: {
  label: "Encoder and adapter layer"
}
CodebaseAgent_boundary: {
  label: "R2Dreamer agent"
}
CodebaseTraining_loop: {
  label: "Trainer and replay loop"

  Trainer: {
    label: "Trainer"
  }
  Replay: {
    label: "ReplayBuffer"
    shape: stored_data
  }
  Metrics: {
    label: "Run artifacts"
    shape: stored_data
  }
  Sampled_batch: {
    label: "Sampled sequence batch"
  }
  Convert_batch: {
    label: "convert_batch"
  }
  Obs_batch: {
    label: "obs_batch bridge"
  }
}

CodebaseEncoder_boundary -> CodebaseTraining_loop.Trainer: "obs adapter"
CodebaseAgent_boundary -> CodebaseTraining_loop.Trainer: "action and metrics"
CodebaseTraining_loop.Trainer -> CodebaseTraining_loop.Replay: "add transition"
CodebaseTraining_loop.Trainer -> CodebaseTraining_loop.Metrics: "log and save"
CodebaseTraining_loop.Replay -> CodebaseTraining_loop.Sampled_batch: "sample (B,T)"
CodebaseTraining_loop.Sampled_batch -> CodebaseTraining_loop.Convert_batch: "format batch"
CodebaseTraining_loop.Convert_batch -> CodebaseTraining_loop.Obs_batch: "agent boundary"
CodebaseTraining_loop.Trainer -> CodebaseAgent_boundary: "act and train_step"
CodebaseTraining_loop.Obs_batch -> CodebaseAgent_boundary: "B*T observations"
`;case"view_iera9v":return`direction: down

CodebaseEncoder_boundary: {
  label: "Encoder and adapter layer"
}
CodebaseTraining_loop: {
  label: "Trainer and replay loop"
}
CodebaseEvaluation: {
  label: "Evaluation and parity workflows"
}
CodebaseAgent_boundary: {
  label: "R2Dreamer agent"

  Config: {
    label: "R2DreamerConfig"
  }
  Encoder_mod: {
    label: "Flax observation encoder"
  }
  Embed: {
    label: "Observation embed"
  }
  Rssm: {
    label: "R2RSSM"
  }
  Rssm_feat: {
    label: "RSSM feature"
  }
  Heads: {
    label: "Prediction and control heads"
  }
  Losses: {
    label: "Loss composition"
  }
  Agent: {
    label: "R2DreamerAgent"
  }
}

CodebaseEncoder_boundary -> CodebaseAgent_boundary.Config: "agent overrides and module_cls"
CodebaseTraining_loop -> CodebaseAgent_boundary.Agent: "act and train_step"
CodebaseTraining_loop -> CodebaseAgent_boundary.Encoder_mod: "B*T observations"
CodebaseEvaluation -> CodebaseAgent_boundary.Agent: "compare behavior"
CodebaseAgent_boundary.Config -> CodebaseAgent_boundary.Agent: "initialize"
CodebaseAgent_boundary.Losses -> CodebaseAgent_boundary.Agent: "update params and optimizer state"
CodebaseAgent_boundary.Encoder_mod -> CodebaseAgent_boundary.Embed: "encode obs"
CodebaseAgent_boundary.Embed -> CodebaseAgent_boundary.Rssm: "posterior observe"
CodebaseAgent_boundary.Rssm -> CodebaseAgent_boundary.Rssm_feat: "post states and feat"
CodebaseAgent_boundary.Rssm_feat -> CodebaseAgent_boundary.Heads: "[...]"
CodebaseAgent_boundary.Rssm_feat -> CodebaseAgent_boundary.Losses: "[...]"
CodebaseAgent_boundary.Agent -> CodebaseTraining_loop: "action and metrics"
CodebaseAgent_boundary.Agent -> CodebaseEvaluation: "save"
`;case"view_12tzet7":return`direction: down

CodebaseAgent_boundaryRssm_feat: {
  label: "RSSM feature"
}
CodebaseAgent_boundaryLosses: {
  label: "Loss composition"

  Wm_loss: {
    label: "world_model_loss"
  }
  Behavior_loss: {
    label: "behavior_loss"
  }
  Rep_loss: {
    label: "representation_loss"
  }
  Optimizer: {
    label: "LaProp + AGC update"
  }
}
CodebaseAgent_boundaryAgent: {
  label: "R2DreamerAgent"
}

CodebaseAgent_boundaryRssm_feat -> CodebaseAgent_boundaryLosses.Wm_loss: "world-model terms"
CodebaseAgent_boundaryRssm_feat -> CodebaseAgent_boundaryLosses.Behavior_loss: "imagination starts"
CodebaseAgent_boundaryRssm_feat -> CodebaseAgent_boundaryLosses.Rep_loss: "representation terms"
CodebaseAgent_boundaryLosses.Wm_loss -> CodebaseAgent_boundaryLosses.Optimizer: "weighted sum"
CodebaseAgent_boundaryLosses.Behavior_loss -> CodebaseAgent_boundaryLosses.Optimizer: "weighted sum"
CodebaseAgent_boundaryLosses.Rep_loss -> CodebaseAgent_boundaryLosses.Optimizer: "weighted sum"
CodebaseAgent_boundaryLosses.Optimizer -> CodebaseAgent_boundaryAgent: "update params and optimizer state"
`;case"view_1byfr7e":return`direction: down

CodebaseVggt_boundary: {
  label: "VGGT production encoder"

  Extractor: {
    label: "JAXVGGTFeatureExtractor"
  }
  Aggregator: {
    label: "Aggregator"
  }
  Agg_cache: {
    label: "Aggregator padded KV cache"
    shape: stored_data
  }
  Camera_head: {
    label: "CameraHead"
  }
  Point_head: {
    label: "DPTHead"
  }
  Aggregator_tokens: {
    label: "Aggregator features"
  }
  Camera_cache: {
    label: "Camera-head padded KV cache"
    shape: stored_data
  }
  Camera_pose: {
    label: "camera_pose"
  }
  Dense_world_points: {
    label: "dense_world_points"
  }
  World_points: {
    label: "world_points"
  }
}
CodebaseEncoder_boundary: {
  label: "Encoder and adapter layer"
}

CodebaseVggt_boundary.Extractor -> CodebaseVggt_boundary.Aggregator: "run one frame"
CodebaseVggt_boundary.Aggregator -> CodebaseVggt_boundary.Agg_cache: "read and update"
CodebaseVggt_boundary.Aggregator -> CodebaseVggt_boundary.Camera_head: "if compute_heads=True"
CodebaseVggt_boundary.Aggregator -> CodebaseVggt_boundary.Point_head: "if compute_heads=True"
CodebaseVggt_boundary.Aggregator -> CodebaseVggt_boundary.Aggregator_tokens: "expose final global stream"
CodebaseVggt_boundary.Camera_head -> CodebaseVggt_boundary.Camera_cache: "read and update"
CodebaseVggt_boundary.Camera_head -> CodebaseVggt_boundary.Camera_pose: "pose"
CodebaseVggt_boundary.Point_head -> CodebaseVggt_boundary.Dense_world_points: "dense points"
CodebaseVggt_boundary.Dense_world_points -> CodebaseVggt_boundary.World_points: "pool to K x K"
CodebaseVggt_boundary.Extractor -> CodebaseEncoder_boundary: "[...]"
`;case"view_118k4sm":return`direction: down

Researcher: {
  label: "Researcher / operator"
  shape: person
}
CodebaseEvaluation: {
  label: "Evaluation and parity workflows"

  Parity: {
    label: "parity workflows"
  }
  Evaluate: {
    label: "evaluate()"
  }
  Checkpoints: {
    label: "Policy checkpoints"
    shape: stored_data
  }
}
CodebaseAgent_boundary: {
  label: "R2Dreamer agent"
}

Researcher -> CodebaseEvaluation.Evaluate: "runs evaluation"
CodebaseAgent_boundary -> CodebaseEvaluation.Checkpoints: "save"
CodebaseEvaluation.Evaluate -> CodebaseEvaluation.Checkpoints: "load"
CodebaseEvaluation.Checkpoints -> CodebaseEvaluation.Evaluate: "restore"
CodebaseEvaluation.Parity -> CodebaseAgent_boundary: "compare behavior"
`;default:throw new Error("Unknown viewId: "+e)}}export{n as d2Source};
