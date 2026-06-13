function t(e){switch(e){case"index":return`digraph {
    graph [TBbalance=min,
        bgcolor=transparent,
        compound=true,
        fontname=Arial,
        fontsize=20,
        labeljust=l,
        labelloc=t,
        layout=dot,
        likec4_viewId=index,
        nodesep=1.528,
        outputorder=nodesfirst,
        pad=0.209,
        rankdir=TB,
        ranksep=1.667,
        splines=spline
    ];
    node [color="#2563eb",
        fillcolor="#3b82f6",
        fontcolor="#eff6ff",
        fontname=Arial,
        label="\\N",
        penwidth=0,
        shape=rect,
        style=filled
    ];
    edge [arrowsize=0.75,
        color="#8D8D8D",
        fontcolor="#C9C9C9",
        fontname=Arial,
        fontsize=14,
        penwidth=2
    ];
    researcher [height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Researcher / operator</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">CLI, SLURM, W&amp;B</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Starts training, evaluation, profiling, and<BR/>analysis runs.<BR/>The current code is organized around<BR/>script-level run selection, launcher<BR/>registries, and a JAX/Flax R2Dreamer agent.</FONT></TD></TR></TABLE>>,
        likec4_id=researcher,
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    codebase [height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Master-Thesis-3D-VLA current code</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">Python, JAX/Flax, Habitat, VGGT</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Current main-branch architecture for<BR/>ObjectNav experiments.<BR/>The source contracts are the root and scoped<BR/>AGENTS.md files plus the current src/ and<BR/>scripts/ modules.</FONT></TD></TR></TABLE>>,
        likec4_id=codebase,
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    researcher -> codebase [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14"><B>[...]</B></FONT></TD></TR></TABLE>>,
        likec4_id=yfmm58,
        style=dashed];
}
`;case"view_1dcnbvb":return`digraph {
    graph [TBbalance=min,
        bgcolor=transparent,
        compound=true,
        fontname=Arial,
        fontsize=20,
        labeljust=l,
        labelloc=t,
        layout=dot,
        likec4_viewId=view_1dcnbvb,
        nodesep=1.528,
        outputorder=nodesfirst,
        pad=0.209,
        rankdir=TB,
        ranksep=1.667,
        splines=spline
    ];
    node [color="#2563eb",
        fillcolor="#3b82f6",
        fontcolor="#eff6ff",
        fontname=Arial,
        label="\\N",
        penwidth=0,
        shape=rect,
        style=filled
    ];
    edge [arrowsize=0.75,
        color="#8D8D8D",
        fontcolor="#C9C9C9",
        fontname=Arial,
        fontsize=14,
        penwidth=2
    ];
    subgraph cluster_codebase {
        graph [color="#1b3d88",
            fillcolor="#194b9e",
            label=<<FONT POINT-SIZE="11" COLOR="#bfdbfeb3"><B>MASTER-THESIS-3D-VLA CURRENT CODE</B></FONT>>,
            likec4_depth=1,
            likec4_id=codebase,
            likec4_level=0,
            margin=40,
            style=filled
        ];
        orchestration [height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Experiment orchestration</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">scripts/r2dreamer, scripts/slurm</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.orchestration",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        vggt_boundary [height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">VGGT production encoder</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">src/vggt/jax</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.vggt_boundary",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        environment [height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Environment layer</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">src/environments, Habitat/Crafter</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.environment",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        encoder_boundary [height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Encoder and adapter layer</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">src/r2dreamer/encoders and adapters</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.encoder_boundary",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        training_loop [height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Trainer and replay loop</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">src/r2dreamer/trainer.py,</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.training_loop",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        agent_boundary [height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">R2Dreamer agent</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">src/r2dreamer/agent.py</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.agent_boundary",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        evaluation [height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Evaluation and parity workflows</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">src/r2dreamer/launch/evaluate.py,</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.evaluation",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
    }
    researcher [height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Researcher / operator</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">CLI, SLURM, W&amp;B</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Starts training, evaluation, profiling, and<BR/>analysis runs.<BR/>The current code is organized around<BR/>script-level run selection, launcher<BR/>registries, and a JAX/Flax R2Dreamer agent.</FONT></TD></TR></TABLE>>,
        likec4_id=researcher,
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    researcher -> orchestration [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">starts run</FONT></TD></TR></TABLE>>,
        likec4_id=vlgbk9,
        style=dashed];
    researcher -> evaluation [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">runs evaluation</FONT></TD></TR></TABLE>>,
        likec4_id="1av0kpw",
        style=dashed];
    orchestration -> environment [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14"><B>[...]</B></FONT></TD></TR></TABLE>>,
        likec4_id=qozk1w,
        style=dashed,
        weight=2];
    orchestration -> encoder_boundary [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">resolve encoder</FONT></TD></TR></TABLE>>,
        likec4_id="1p7t2ua",
        style=dashed,
        weight=2];
    vggt_boundary -> encoder_boundary [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14"><B>[...]</B></FONT></TD></TR></TABLE>>,
        likec4_id=q7lwyk,
        minlen=1,
        style=dashed];
    environment -> encoder_boundary [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14"><B>[...]</B></FONT></TD></TR></TABLE>>,
        likec4_id="1botwss",
        style=dashed];
    encoder_boundary -> training_loop [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">obs adapter</FONT></TD></TR></TABLE>>,
        likec4_id="18z3qii",
        style=dashed];
    encoder_boundary -> agent_boundary [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">agent overrides and module_cls</FONT></TD></TR></TABLE>>,
        likec4_id=f57l53,
        style=dashed];
    training_loop -> agent_boundary [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14"><B>[...]</B></FONT></TD></TR></TABLE>>,
        likec4_id="1qi6qfr",
        style=dashed];
    agent_boundary -> training_loop [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">action and metrics</FONT></TD></TR></TABLE>>,
        likec4_id="1dh6kev",
        style=dashed];
    agent_boundary -> evaluation [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">save</FONT></TD></TR></TABLE>>,
        likec4_id="1a20vr6",
        style=dashed];
    evaluation -> agent_boundary [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">compare behavior</FONT></TD></TR></TABLE>>,
        likec4_id=dd0zeq,
        style=dashed];
}
`;case"view_1omwq3l":return`digraph {
    graph [TBbalance=min,
        bgcolor=transparent,
        compound=true,
        fontname=Arial,
        fontsize=20,
        labeljust=l,
        labelloc=t,
        layout=dot,
        likec4_viewId=view_1omwq3l,
        nodesep=1.528,
        outputorder=nodesfirst,
        pad=0.209,
        rankdir=TB,
        ranksep=1.667,
        splines=spline
    ];
    node [color="#2563eb",
        fillcolor="#3b82f6",
        fontcolor="#eff6ff",
        fontname=Arial,
        penwidth=0,
        shape=rect,
        style=filled
    ];
    edge [arrowsize=0.75,
        color="#8D8D8D",
        fontcolor="#C9C9C9",
        fontname=Arial,
        fontsize=14,
        penwidth=2
    ];
    subgraph cluster_orchestration {
        graph [color="#1b3d88",
            fillcolor="#194b9e",
            label=<<FONT POINT-SIZE="11" COLOR="#bfdbfeb3"><B>EXPERIMENT ORCHESTRATION</B></FONT>>,
            likec4_depth=1,
            likec4_id="codebase.orchestration",
            likec4_level=0,
            margin=40,
            style=filled
        ];
        slurm_configs [group="codebase.orchestration",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">SLURM configs</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">scripts/slurm/configs/*.yaml and legacy</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Render or call run.py with a run_id and train<BR/>flags. GPU work is launched through the<BR/>cluster scheduler.</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.orchestration.slurm_configs",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        run_dispatcher [group="codebase.orchestration",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">scripts/r2dreamer/run.py</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">Single run-id dispatcher</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Accepts run.py &lt;run-id&gt; [train flags] and<BR/>forwards to _run_configs.launch_run.</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.orchestration.run_dispatcher",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        run_configs [group="codebase.orchestration",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">RUN_CONFIGS</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">scripts/r2dreamer/_run_configs.py</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Single source of truth for env, encoder,<BR/>curriculum, output_dir, wandb_name, and<BR/>wandb_tags.<BR/>Known run ids cover Habitat L1-L4 CNN/VGGT<BR/>variants and Crafter CNN.</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.orchestration.run_configs",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        public_entrypoint [group="codebase.orchestration",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">src/main.py</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">train, evaluate, parity commands</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Public CLI dispatcher for train/evaluate and<BR/>parity workflows.</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.orchestration.public_entrypoint",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        train_entry [group="codebase.orchestration",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">src.r2dreamer.launch.train.train</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">Launcher composition root</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Resolves curriculum, encoder, env, agent<BR/>config, trainer config, agent, and Trainer.<BR/>Runs Trainer.run() and returns the Trainer<BR/>for programmatic callers.</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.orchestration.train_entry",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        registries [group="codebase.orchestration",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">launch registries</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">src/r2dreamer/launch/registries.py</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Maps encoder strings to Encoder classes and<BR/>env strings to env factories.</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.orchestration.registries",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        curriculum [group="codebase.orchestration",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Curriculum JSON</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">data/curriculum/*.json</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Habitat L1-L4 curriculum files resolved by<BR/>launch/curricula.py and _helpers.py.</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.orchestration.curriculum",
            likec4_level=1,
            margin="0.223,0",
            penwidth=2,
            shape=cylinder,
            width=4.445];
    }
    researcher [height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Researcher / operator</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">CLI, SLURM, W&amp;B</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Starts training, evaluation, profiling, and<BR/>analysis runs.<BR/>The current code is organized around<BR/>script-level run selection, launcher<BR/>registries, and a JAX/Flax R2Dreamer agent.</FONT></TD></TR></TABLE>>,
        likec4_id=researcher,
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    researcher -> run_dispatcher [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">starts run</FONT></TD></TR></TABLE>>,
        likec4_id=b7ia0m,
        minlen=1,
        style=dashed];
    slurm_configs -> run_dispatcher [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">render or call</FONT></TD></TR></TABLE>>,
        likec4_id=e0u5ew,
        minlen=1,
        style=dashed,
        weight=3];
    run_dispatcher -> run_configs [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">select run id</FONT></TD></TR></TABLE>>,
        likec4_id=chbd1w,
        style=dashed,
        weight=3];
    run_configs -> public_entrypoint [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">call src.main.train</FONT></TD></TR></TABLE>>,
        likec4_id="1j6x3dv",
        style=dashed];
    public_entrypoint -> train_entry [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">dispatch train</FONT></TD></TR></TABLE>>,
        likec4_id=w4qgcn,
        style=dashed,
        weight=2];
    train_entry -> registries [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">resolve env and encoder</FONT></TD></TR></TABLE>>,
        likec4_id=a9em1e,
        style=dashed,
        weight=2];
    train_entry -> curriculum [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">resolve Habitat curriculum</FONT></TD></TR></TABLE>>,
        likec4_id="1dfr3y4",
        minlen=1,
        style=dashed,
        weight=2];
    environment [height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Environment layer</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">src/environments, Habitat/Crafter</FONT></TD></TR></TABLE>>,
        likec4_id="codebase.environment",
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    train_entry -> environment [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14"><B>[...]</B></FONT></TD></TR></TABLE>>,
        likec4_id="2jouo1",
        minlen=1,
        style=dashed];
    encoder_boundary [height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Encoder and adapter layer</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">src/r2dreamer/encoders and adapters</FONT></TD></TR></TABLE>>,
        likec4_id="codebase.encoder_boundary",
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    registries -> encoder_boundary [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">resolve encoder</FONT></TD></TR></TABLE>>,
        likec4_id="13ngo4f",
        minlen=1,
        style=dashed];
}
`;case"view_spshjc":return`digraph {
    graph [TBbalance=min,
        bgcolor=transparent,
        compound=true,
        fontname=Arial,
        fontsize=20,
        labeljust=l,
        labelloc=t,
        layout=dot,
        likec4_viewId=view_spshjc,
        nodesep=1.528,
        outputorder=nodesfirst,
        pad=0.209,
        rankdir=TB,
        ranksep=1.667,
        splines=spline
    ];
    node [color="#2563eb",
        fillcolor="#3b82f6",
        fontcolor="#eff6ff",
        fontname=Arial,
        penwidth=0,
        shape=rect,
        style=filled
    ];
    edge [arrowsize=0.75,
        color="#8D8D8D",
        fontcolor="#C9C9C9",
        fontname=Arial,
        fontsize=14,
        penwidth=2
    ];
    subgraph cluster_encoder_boundary {
        graph [color="#1b3d88",
            fillcolor="#194b9e",
            label=<<FONT POINT-SIZE="11" COLOR="#bfdbfeb3"><B>ENCODER AND ADAPTER LAYER</B></FONT>>,
            likec4_depth=1,
            likec4_id="codebase.encoder_boundary",
            likec4_level=0,
            margin=40,
            style=filled
        ];
        encoder_spec [height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">EncoderSpec</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">obs_shape, env_render_resolution,</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Launcher-side contract that keeps adapter<BR/>output, environment render size, and the Flax<BR/>encoder module aligned.<BR/>train.py copies the spec into<BR/>R2DreamerConfig.</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.encoder_boundary.encoder_spec",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        obs_adapter [height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">ObsAdapter</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">src/r2dreamer/adapters/obs_adapter.py</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Default RGB passthrough adapter with buffer<BR/>shape (3,64,64), dtype uint8, and<BR/>normalize_on_sample=True.</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.encoder_boundary.obs_adapter",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        vggt_adapter [height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">VGGTObsAdapter</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">src/r2dreamer/adapters/vggt_adapter.py</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Runs JAXVGGTFeatureExtractor per frame and<BR/>returns replay features plus one-step agent<BR/>features.<BR/>Feature kinds: wp_cp, aggregator, wp_dense,<BR/>agg_raw.</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.encoder_boundary.vggt_adapter",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        hybrid_adapter [height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">HybridObsAdapter</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">src/r2dreamer/adapters/hybrid_adapter.py</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Runs VGGT for WP/CP, resizes the same 518x518<BR/>frame to 64x64 RGB, and stores explicit<BR/>replay fields image and wp_cp.</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.encoder_boundary.hybrid_adapter",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        cnn_spec [height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">cnn</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">ConvEncoder, obs_shape (3,64,64)</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.encoder_boundary.cnn_spec",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        wp_cp_spec [height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">vggt / vggt_wp_cp_64</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">VGGTEncoder MLP</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">WP/CP vector is K*K*3 world points plus 9<BR/>camera-pose values. K=37 gives 4116; K=64<BR/>gives 12297.</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.encoder_boundary.wp_cp_spec",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        aggregator_spec [height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">vggt_aggregator_mlp</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">VGGTAggregatorMLPEncoder</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Pools VGGT global aggregator tokens into<BR/>[camera | mean_patches | max_patches] = 3072<BR/>float32 values.</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.encoder_boundary.aggregator_spec",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        dense_wp_spec [height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">vggt_wp_dense_cnn</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">WPConvEncoder</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Stores dense world points as (3,518,518)<BR/>float16 and encodes them with a symlog conv<BR/>stack.</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.encoder_boundary.dense_wp_spec",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        wp_cp [height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">WP/CP features</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">4116 or 12297 float32</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.encoder_boundary.wp_cp",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        pooled_agg [height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Pooled aggregator features</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">3072 float32</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.encoder_boundary.pooled_agg",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        dense_wp [height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Dense world-point map</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">3 x 518 x 518 float16</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.encoder_boundary.dense_wp",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        hybrid_spec [height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">hybrid</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">HybridEncoder</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Stores image uint8 (3,64,64) and WP/CP<BR/>float32 separately, then packs to 16404<BR/>features at the JAX boundary.</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.encoder_boundary.hybrid_spec",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        hybrid_obs [height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Hybrid replay observation</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">image uint8 + wp_cp float32</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Replay keeps modalities inspectable under<BR/>explicit keys; obs_batch packs them into the<BR/>legacy flat tensor.</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.encoder_boundary.hybrid_obs",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
    }
    orchestration [height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Experiment orchestration</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">scripts/r2dreamer, scripts/slurm</FONT></TD></TR></TABLE>>,
        likec4_id="codebase.orchestration",
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    orchestration -> encoder_spec [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">resolve encoder</FONT></TD></TR></TABLE>>,
        likec4_id="1vz35eq",
        minlen=1,
        style=dashed];
    environment [height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Environment layer</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">src/environments, Habitat/Crafter</FONT></TD></TR></TABLE>>,
        likec4_id="codebase.environment",
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    environment -> obs_adapter [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">CNN path</FONT></TD></TR></TABLE>>,
        likec4_id="1jsp3sk",
        style=dashed];
    environment -> vggt_adapter [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">VGGT standalone path</FONT></TD></TR></TABLE>>,
        likec4_id=ebfx7s,
        style=dashed];
    environment -> hybrid_adapter [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">hybrid path</FONT></TD></TR></TABLE>>,
        likec4_id=pwk012,
        style=dashed];
    vggt_boundary [height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">VGGT production encoder</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">src/vggt/jax</FONT></TD></TR></TABLE>>,
        likec4_id="codebase.vggt_boundary",
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    vggt_boundary -> vggt_adapter [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">extract features</FONT></TD></TR></TABLE>>,
        likec4_id="5zx4mw",
        style=dashed];
    vggt_boundary -> hybrid_adapter [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">extract WP/CP</FONT></TD></TR></TABLE>>,
        likec4_id="17c0kja",
        style=dashed];
    encoder_spec -> obs_adapter [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">default RGB adapter</FONT></TD></TR></TABLE>>,
        likec4_id=voehn6,
        style=dashed,
        weight=2];
    encoder_spec -> vggt_adapter [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">VGGT standalone variants</FONT></TD></TR></TABLE>>,
        likec4_id="10ye6vi",
        style=dashed,
        weight=2];
    encoder_spec -> hybrid_adapter [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">hybrid variant</FONT></TD></TR></TABLE>>,
        likec4_id=pi9zi8,
        style=dashed,
        weight=2];
    agent_boundary [height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">R2Dreamer agent</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">src/r2dreamer/agent.py</FONT></TD></TR></TABLE>>,
        likec4_id="codebase.agent_boundary",
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    encoder_spec -> agent_boundary [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">agent overrides and module_cls</FONT></TD></TR></TABLE>>,
        likec4_id=pxbpt3,
        minlen=1,
        style=dashed];
    obs_adapter -> cnn_spec [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">declares</FONT></TD></TR></TABLE>>,
        likec4_id=n56wed,
        minlen=1,
        style=dashed,
        weight=2];
    training_loop [height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Trainer and replay loop</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">src/r2dreamer/trainer.py,</FONT></TD></TR></TABLE>>,
        likec4_id="codebase.training_loop",
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    obs_adapter -> training_loop [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">obs adapter</FONT></TD></TR></TABLE>>,
        likec4_id=ekl6ma,
        style=dashed];
    vggt_adapter -> wp_cp_spec [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">wp_cp readout</FONT></TD></TR></TABLE>>,
        likec4_id="1ya9n01",
        minlen=1,
        style=dashed,
        weight=2];
    vggt_adapter -> aggregator_spec [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">aggregator readout</FONT></TD></TR></TABLE>>,
        likec4_id="191vaeb",
        minlen=1,
        style=dashed,
        weight=2];
    vggt_adapter -> dense_wp_spec [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">dense WP readout</FONT></TD></TR></TABLE>>,
        likec4_id="1zd7uj",
        minlen=1,
        style=dashed,
        weight=2];
    vggt_adapter -> wp_cp [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">extract world_points + camera_pose</FONT></TD></TR></TABLE>>,
        likec4_id="10k3rgr",
        minlen=1,
        style=dashed,
        weight=2];
    vggt_adapter -> pooled_agg [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">pool final global tokens</FONT></TD></TR></TABLE>>,
        likec4_id="13zyl8j",
        minlen=1,
        style=dashed,
        weight=2];
    vggt_adapter -> dense_wp [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">request return_dense</FONT></TD></TR></TABLE>>,
        likec4_id="1euu2wx",
        minlen=1,
        style=dashed,
        weight=2];
    vggt_adapter -> training_loop [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">obs adapter</FONT></TD></TR></TABLE>>,
        likec4_id=c84i7y,
        style=dashed];
    hybrid_adapter -> hybrid_spec [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">hybrid readout</FONT></TD></TR></TABLE>>,
        likec4_id="1qabfrs",
        minlen=1,
        style=dashed,
        weight=2];
    hybrid_adapter -> hybrid_obs [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">store two fields</FONT></TD></TR></TABLE>>,
        likec4_id="1sb3zr7",
        minlen=1,
        style=dashed,
        weight=2];
    hybrid_adapter -> training_loop [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">obs adapter</FONT></TD></TR></TABLE>>,
        likec4_id="1dw02eo",
        style=dashed];
}
`;case"view_12nwcpr":return`digraph {
    graph [TBbalance=min,
        bgcolor=transparent,
        compound=true,
        fontname=Arial,
        fontsize=20,
        labeljust=l,
        labelloc=t,
        layout=dot,
        likec4_viewId=view_12nwcpr,
        nodesep=1.528,
        outputorder=nodesfirst,
        pad=0.209,
        rankdir=TB,
        ranksep=1.667,
        splines=spline
    ];
    node [color="#2563eb",
        fillcolor="#3b82f6",
        fontcolor="#eff6ff",
        fontname=Arial,
        penwidth=0,
        shape=rect,
        style=filled
    ];
    edge [arrowsize=0.75,
        color="#8D8D8D",
        fontcolor="#C9C9C9",
        fontname=Arial,
        fontsize=14,
        penwidth=2
    ];
    subgraph cluster_training_loop {
        graph [color="#1b3d88",
            fillcolor="#194b9e",
            label=<<FONT POINT-SIZE="11" COLOR="#bfdbfeb3"><B>TRAINER AND REPLAY LOOP</B></FONT>>,
            likec4_depth=1,
            likec4_id="codebase.training_loop",
            likec4_level=0,
            margin=40,
            style=filled
        ];
        trainer [group="codebase.training_loop",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Trainer</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">Prefill, collect, train_ratio, val, log,</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Owns environment stepping, replay insertion,<BR/>batch sampling, validation env loop, metrics,<BR/>W&amp;B logging, videos, checkpoints, and<BR/>MANIFEST writes.</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.training_loop.trainer",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        replay [group="codebase.training_loop",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">ReplayBuffer</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">NumPy ring buffer</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Stores observations as one array or a mapping<BR/>of named fields.<BR/>Samples fixed length (B,T) windows and<BR/>returns JAX arrays with is_first, actions,<BR/>rewards, dones, and terminals.</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.training_loop.replay",
            likec4_level=1,
            margin="0.223,0",
            penwidth=2,
            shape=cylinder,
            width=4.445];
        metrics [group="codebase.training_loop",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Run artifacts</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">output/runs, metrics.csv, checkpoints,</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Trainer writes local checkpoints and metrics,<BR/>manifest provenance, optional videos, and W&amp;B<BR/>summaries.</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.training_loop.metrics",
            likec4_level=1,
            margin="0.223,0",
            penwidth=2,
            shape=cylinder,
            width=4.445];
        sampled_batch [group="codebase.training_loop",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Sampled sequence batch</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">obs + actions + rewards + dones + terminals +</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Replay sample window consumed by<BR/>convert_batch and agent.train_step.</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.training_loop.sampled_batch",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        convert_batch [group="codebase.training_loop",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">convert_batch</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">trainer.py</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">One-hot encodes actions to (B,T,A), maps<BR/>dones to is_last, terminals to is_terminal,<BR/>and preserves is_first.</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.training_loop.convert_batch",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        obs_batch [group="codebase.training_loop",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">obs_batch bridge</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">src/r2dreamer/obs_batch.py</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Normalizes CNN RGB, casts VGGT features to<BR/>float32, packs hybrid dict observations, and<BR/>reshapes observations to B*T before the Flax<BR/>encoder.</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.training_loop.obs_batch",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
    }
    encoder_boundary [height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Encoder and adapter layer</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">src/r2dreamer/encoders and adapters</FONT></TD></TR></TABLE>>,
        likec4_id="codebase.encoder_boundary",
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    encoder_boundary -> trainer [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">obs adapter</FONT></TD></TR></TABLE>>,
        likec4_id="4ep1k3",
        minlen=1,
        style=dashed];
    agent_boundary [height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">R2Dreamer agent</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">src/r2dreamer/agent.py</FONT></TD></TR></TABLE>>,
        likec4_id="codebase.agent_boundary",
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    agent_boundary -> trainer [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">action and metrics</FONT></TD></TR></TABLE>>,
        likec4_id="1cts8b2",
        style=dashed];
    trainer -> agent_boundary [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">act and train_step</FONT></TD></TR></TABLE>>,
        likec4_id="1kqfgpa",
        style=dashed];
    trainer -> replay [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">add transition</FONT></TD></TR></TABLE>>,
        likec4_id="1nk2ezy",
        style=dashed,
        weight=2];
    trainer -> metrics [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">log and save</FONT></TD></TR></TABLE>>,
        likec4_id="1nj17ii",
        minlen=1,
        style=dashed,
        weight=2];
    replay -> sampled_batch [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">sample (B,T)</FONT></TD></TR></TABLE>>,
        likec4_id="1f3j08o",
        style=dashed];
    sampled_batch -> convert_batch [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">format batch</FONT></TD></TR></TABLE>>,
        likec4_id="1pqkr7j",
        style=dashed];
    convert_batch -> obs_batch [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">agent boundary</FONT></TD></TR></TABLE>>,
        likec4_id="1k8yabn",
        style=dashed,
        weight=2];
    obs_batch -> agent_boundary [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">B*T observations</FONT></TD></TR></TABLE>>,
        likec4_id=etxi9w,
        style=dashed];
}
`;case"view_iera9v":return`digraph {
    graph [TBbalance=min,
        bgcolor=transparent,
        compound=true,
        fontname=Arial,
        fontsize=20,
        labeljust=l,
        labelloc=t,
        layout=dot,
        likec4_viewId=view_iera9v,
        nodesep=1.528,
        outputorder=nodesfirst,
        pad=0.209,
        rankdir=TB,
        ranksep=1.667,
        splines=spline
    ];
    node [color="#2563eb",
        fillcolor="#3b82f6",
        fontcolor="#eff6ff",
        fontname=Arial,
        penwidth=0,
        shape=rect,
        style=filled
    ];
    edge [arrowsize=0.75,
        color="#8D8D8D",
        fontcolor="#C9C9C9",
        fontname=Arial,
        fontsize=14,
        penwidth=2
    ];
    subgraph cluster_agent_boundary {
        graph [color="#1b3d88",
            fillcolor="#194b9e",
            label=<<FONT POINT-SIZE="11" COLOR="#bfdbfeb3"><B>R2DREAMER AGENT</B></FONT>>,
            likec4_depth=1,
            likec4_id="codebase.agent_boundary",
            likec4_level=0,
            margin=40,
            style=filled
        ];
        config [group="codebase.agent_boundary",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">R2DreamerConfig</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">src/r2dreamer/config.py</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Config-first source of truth for RSSM sizes,<BR/>encoder choice, loss scales, batch/sequence<BR/>settings, train_ratio, replay capacity,<BR/>optimizer, and run defaults.</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.agent_boundary.config",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        encoder_mod [group="codebase.agent_boundary",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Flax observation encoder</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">ConvEncoder, VGGTEncoder,</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Chosen by EncoderSpec.module_cls and<BR/>instantiated inside the agent, on the JAX<BR/>side of the boundary.</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.agent_boundary.encoder_mod",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        embed [group="codebase.agent_boundary",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Observation embed</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">usually 1024 or hybrid 2048</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Encoder output that conditions the RSSM<BR/>posterior.</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.agent_boundary.embed",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        rssm [group="codebase.agent_boundary",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">R2RSSM</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">src/r2dreamer/world_model/rssm.py</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Block-GRU latent dynamics with observe,<BR/>img_step, and get_feat.</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.agent_boundary.rssm",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        rssm_feat [group="codebase.agent_boundary",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">RSSM feature</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">deter_size + stoch_classes*stoch_discrete</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Default feature size is 2048 deterministic +<BR/>512 stochastic = 2560.</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.agent_boundary.rssm_feat",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        heads [group="codebase.agent_boundary",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Prediction and control heads</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">src/r2dreamer/world_model/heads.py</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.agent_boundary.heads",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        losses [group="codebase.agent_boundary",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Loss composition</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">world_model, behavior, representation</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.agent_boundary.losses",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        agent [group="codebase.agent_boundary",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">R2DreamerAgent</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">JAX/Flax composition root</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Owns params, optimizer state, slow critic<BR/>EMA, acting state, JIT-compiled train_step<BR/>and act.<BR/>A single shared forward pass feeds<BR/>world-model, behavior, and representation</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.agent_boundary.agent",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
    }
    encoder_boundary [height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Encoder and adapter layer</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">src/r2dreamer/encoders and adapters</FONT></TD></TR></TABLE>>,
        likec4_id="codebase.encoder_boundary",
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    encoder_boundary -> config [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">agent overrides and module_cls</FONT></TD></TR></TABLE>>,
        likec4_id=oxmair,
        minlen=1,
        style=dashed];
    training_loop [height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Trainer and replay loop</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">src/r2dreamer/trainer.py,</FONT></TD></TR></TABLE>>,
        likec4_id="codebase.training_loop",
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    training_loop -> encoder_mod [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">B*T observations</FONT></TD></TR></TABLE>>,
        likec4_id=fihl6c,
        style=dashed];
    training_loop -> agent [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">act and train_step</FONT></TD></TR></TABLE>>,
        likec4_id="7ffx28",
        style=dashed];
    evaluation [height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Evaluation and parity workflows</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">src/r2dreamer/launch/evaluate.py,</FONT></TD></TR></TABLE>>,
        likec4_id="codebase.evaluation",
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    evaluation -> agent [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">compare behavior</FONT></TD></TR></TABLE>>,
        likec4_id="1h0uyr9",
        style=dashed];
    config -> agent [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">initialize</FONT></TD></TR></TABLE>>,
        likec4_id="1n7y4t5",
        style=dashed,
        weight=2];
    encoder_mod -> embed [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">encode obs</FONT></TD></TR></TABLE>>,
        likec4_id=vyn3ak,
        style=dashed,
        weight=2];
    embed -> rssm [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">posterior observe</FONT></TD></TR></TABLE>>,
        likec4_id="14kazr2",
        style=dashed];
    rssm -> rssm_feat [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">post states and feat</FONT></TD></TR></TABLE>>,
        likec4_id="53v4cj",
        style=dashed];
    rssm_feat -> heads [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14"><B>[...]</B></FONT></TD></TR></TABLE>>,
        likec4_id="1bnvxt3",
        minlen=1,
        style=dashed];
    rssm_feat -> losses [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14"><B>[...]</B></FONT></TD></TR></TABLE>>,
        likec4_id="7w9xex",
        style=dashed];
    losses -> agent [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">update params and optimizer state</FONT></TD></TR></TABLE>>,
        likec4_id=phsy6e,
        style=dashed,
        weight=2];
    agent -> training_loop [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">action and metrics</FONT></TD></TR></TABLE>>,
        likec4_id="10gl5ts",
        style=dashed];
    agent -> evaluation [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">save</FONT></TD></TR></TABLE>>,
        likec4_id="18pm31x",
        style=dashed];
}
`;case"view_12tzet7":return`digraph {
    graph [TBbalance=min,
        bgcolor=transparent,
        compound=true,
        fontname=Arial,
        fontsize=20,
        labeljust=l,
        labelloc=t,
        layout=dot,
        likec4_viewId=view_12tzet7,
        nodesep=1.528,
        outputorder=nodesfirst,
        pad=0.209,
        rankdir=TB,
        ranksep=1.667,
        splines=spline
    ];
    node [color="#2563eb",
        fillcolor="#3b82f6",
        fontcolor="#eff6ff",
        fontname=Arial,
        penwidth=0,
        shape=rect,
        style=filled
    ];
    edge [arrowsize=0.75,
        color="#8D8D8D",
        fontcolor="#C9C9C9",
        fontname=Arial,
        fontsize=14,
        penwidth=2
    ];
    subgraph cluster_losses {
        graph [color="#1b3d88",
            fillcolor="#194b9e",
            label=<<FONT POINT-SIZE="11" COLOR="#bfdbfeb3"><B>LOSS COMPOSITION</B></FONT>>,
            likec4_depth=1,
            likec4_id="codebase.agent_boundary.losses",
            likec4_level=0,
            margin=40,
            style=filled
        ];
        wm_loss [group="codebase.agent_boundary.losses",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">world_model_loss</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">KL dyn/rep + reward + continue + optional</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.agent_boundary.losses.wm_loss",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        behavior_loss [group="codebase.agent_boundary.losses",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">behavior_loss</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">Detached imagination, lambda-return, actor</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.agent_boundary.losses.behavior_loss",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        rep_loss [group="codebase.agent_boundary.losses",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">representation_loss</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">Barlow Twins + replay-value</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.agent_boundary.losses.rep_loss",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        optimizer [group="codebase.agent_boundary.losses",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">LaProp + AGC update</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">src/shared/optim.py</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Weighted total loss is differentiated once<BR/>and updates the single params pytree.</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.agent_boundary.losses.optimizer",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
    }
    rssm_feat [height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">RSSM feature</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">deter_size + stoch_classes*stoch_discrete</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Default feature size is 2048 deterministic +<BR/>512 stochastic = 2560.</FONT></TD></TR></TABLE>>,
        likec4_id="codebase.agent_boundary.rssm_feat",
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    rssm_feat -> wm_loss [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">world-model terms</FONT></TD></TR></TABLE>>,
        likec4_id=a2sbep,
        style=dashed];
    rssm_feat -> behavior_loss [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">imagination starts</FONT></TD></TR></TABLE>>,
        likec4_id="1hzvk9j",
        style=dashed];
    rssm_feat -> rep_loss [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">representation terms</FONT></TD></TR></TABLE>>,
        likec4_id=jddo8s,
        style=dashed];
    wm_loss -> optimizer [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">weighted sum</FONT></TD></TR></TABLE>>,
        likec4_id="17vvm5j",
        style=dashed,
        weight=2];
    behavior_loss -> optimizer [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">weighted sum</FONT></TD></TR></TABLE>>,
        likec4_id="1t0bju9",
        style=dashed,
        weight=2];
    rep_loss -> optimizer [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">weighted sum</FONT></TD></TR></TABLE>>,
        likec4_id="2ue6ey",
        style=dashed,
        weight=2];
    agent [height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">R2DreamerAgent</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">JAX/Flax composition root</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Owns params, optimizer state, slow critic<BR/>EMA, acting state, JIT-compiled train_step<BR/>and act.<BR/>A single shared forward pass feeds<BR/>world-model, behavior, and representation</FONT></TD></TR></TABLE>>,
        likec4_id="codebase.agent_boundary.agent",
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    optimizer -> agent [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">update params and optimizer state</FONT></TD></TR></TABLE>>,
        likec4_id=kz00wz,
        minlen=1,
        style=dashed];
}
`;case"view_1byfr7e":return`digraph {
    graph [TBbalance=min,
        bgcolor=transparent,
        compound=true,
        fontname=Arial,
        fontsize=20,
        labeljust=l,
        labelloc=t,
        layout=dot,
        likec4_viewId=view_1byfr7e,
        nodesep=1.528,
        outputorder=nodesfirst,
        pad=0.209,
        rankdir=TB,
        ranksep=1.667,
        splines=spline
    ];
    node [color="#2563eb",
        fillcolor="#3b82f6",
        fontcolor="#eff6ff",
        fontname=Arial,
        penwidth=0,
        shape=rect,
        style=filled
    ];
    edge [arrowsize=0.75,
        color="#8D8D8D",
        fontcolor="#C9C9C9",
        fontname=Arial,
        fontsize=14,
        penwidth=2
    ];
    subgraph cluster_vggt_boundary {
        graph [color="#1b3d88",
            fillcolor="#194b9e",
            label=<<FONT POINT-SIZE="11" COLOR="#bfdbfeb3"><B>VGGT PRODUCTION ENCODER</B></FONT>>,
            likec4_depth=1,
            likec4_id="codebase.vggt_boundary",
            likec4_level=0,
            margin=40,
            style=filled
        ];
        extractor [height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">JAXVGGTFeatureExtractor</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">src/vggt/jax/feature_extractor.py</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Drop-in JAX backend for StreamVGGT.<BR/>Loads HuggingFace StreamVGGT weights, keeps<BR/>streaming caches as instance state, resets at<BR/>episode boundaries, and exposes extract(rgb).</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.vggt_boundary.extractor",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        aggregator [height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Aggregator</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">24 alternating attention blocks</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Consumes fixed 518x518 RGB, emits camera +<BR/>register + patch tokens, and supports<BR/>streaming cache paths.</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.vggt_boundary.aggregator",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        agg_cache [height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Aggregator padded KV cache</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">per-block (k_pad, v_pad, valid_len)</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Fixed-shape padded cache keeps JIT stable.<BR/>Per-block budgets are Python static args;<BR/>eviction uses budgeted cache control.</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.vggt_boundary.agg_cache",
            likec4_level=1,
            margin="0.223,0",
            penwidth=2,
            shape=cylinder,
            width=4.445];
        camera_head [height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">CameraHead</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">pose output</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.vggt_boundary.camera_head",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        point_head [height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">DPTHead</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">dense 518 x 518 x 3 points</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.vggt_boundary.point_head",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        aggregator_tokens [height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Aggregator features</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">1374 x 1024 global stream</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Final global-stream tokens: 1 camera + 4<BR/>registers + 37x37 patches, 1024 dims. Pooled<BR/>variants drop registers when flattening raw<BR/>or pooling patches.</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.vggt_boundary.aggregator_tokens",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        camera_cache [height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Camera-head padded KV cache</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">max_camera_frames guard</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Camera head cache fails loudly on overflow<BR/>instead of silently clamping dynamic updates.</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.vggt_boundary.camera_cache",
            likec4_level=1,
            margin="0.223,0",
            penwidth=2,
            shape=cylinder,
            width=4.445];
        camera_pose [height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">camera_pose</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">9 float32 values</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.vggt_boundary.camera_pose",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        dense_world_points [height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">dense_world_points</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">518 x 518 x 3 float32</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.vggt_boundary.dense_world_points",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        world_points [height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">world_points</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">K x K x 3</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Dense point map pooled to K=37 by default or<BR/>K=64 for vggt_wp_cp_64.</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.vggt_boundary.world_points",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
    }
    extractor -> aggregator [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">run one frame</FONT></TD></TR></TABLE>>,
        likec4_id="1gbumeb",
        style=dashed,
        weight=2];
    encoder_boundary [height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Encoder and adapter layer</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">src/r2dreamer/encoders and adapters</FONT></TD></TR></TABLE>>,
        likec4_id="codebase.encoder_boundary",
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    extractor -> encoder_boundary [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14"><B>[...]</B></FONT></TD></TR></TABLE>>,
        likec4_id="1bry8g2",
        minlen=1,
        style=dashed];
    aggregator -> agg_cache [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">read and update</FONT></TD></TR></TABLE>>,
        likec4_id=dfly75,
        minlen=1,
        style=dashed];
    aggregator -> camera_head [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">if compute_heads=True</FONT></TD></TR></TABLE>>,
        likec4_id="14nz7a5",
        style=dashed];
    aggregator -> point_head [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">if compute_heads=True</FONT></TD></TR></TABLE>>,
        likec4_id=mqa07c,
        style=dashed];
    aggregator -> aggregator_tokens [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">expose final global stream</FONT></TD></TR></TABLE>>,
        likec4_id=o1wgal,
        minlen=1,
        style=dashed];
    camera_head -> camera_cache [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">read and update</FONT></TD></TR></TABLE>>,
        likec4_id="1f0bx5a",
        minlen=1,
        style=dashed];
    camera_head -> camera_pose [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">pose</FONT></TD></TR></TABLE>>,
        likec4_id="1yflm7f",
        minlen=1,
        style=dashed];
    point_head -> dense_world_points [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">dense points</FONT></TD></TR></TABLE>>,
        likec4_id="1iqgfmd",
        style=dashed];
    dense_world_points -> world_points [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">pool to K x K</FONT></TD></TR></TABLE>>,
        likec4_id="1kgsu4s",
        minlen=1,
        style=dashed];
}
`;case"view_118k4sm":return`digraph {
    graph [TBbalance=min,
        bgcolor=transparent,
        compound=true,
        fontname=Arial,
        fontsize=20,
        labeljust=l,
        labelloc=t,
        layout=dot,
        likec4_viewId=view_118k4sm,
        nodesep=1.528,
        outputorder=nodesfirst,
        pad=0.209,
        rankdir=TB,
        ranksep=1.667,
        splines=spline
    ];
    node [color="#2563eb",
        fillcolor="#3b82f6",
        fontcolor="#eff6ff",
        fontname=Arial,
        penwidth=0,
        shape=rect,
        style=filled
    ];
    edge [arrowsize=0.75,
        color="#8D8D8D",
        fontcolor="#C9C9C9",
        fontname=Arial,
        fontsize=14,
        penwidth=2
    ];
    subgraph cluster_evaluation {
        graph [color="#1b3d88",
            fillcolor="#194b9e",
            label=<<FONT POINT-SIZE="11" COLOR="#bfdbfeb3"><B>EVALUATION AND PARITY WORKFLOWS</B></FONT>>,
            likec4_depth=1,
            likec4_id="codebase.evaluation",
            likec4_level=0,
            margin=40,
            style=filled
        ];
        parity [height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">parity workflows</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">train_parity.py, benchmark.py</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">JAX/PyTorch parity training and benchmark<BR/>commands for debugging numerical drift.</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.evaluation.parity",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        evaluate [group="codebase.evaluation",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">evaluate()</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">checkpoint evaluation</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Loads a policy checkpoint, constructs the<BR/>matching env and encoder, runs episodes, and<BR/>logs metrics.</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.evaluation.evaluate",
            likec4_level=1,
            margin="0.223,0.223",
            width=4.445];
        checkpoints [group="codebase.evaluation",
            height=2.5,
            label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Policy checkpoints</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">pickle step_*.pkl</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Contain params, opt_state,<BR/>slow_critic_params, ema_state, and step.</FONT></TD></TR></TABLE>>,
            likec4_id="codebase.evaluation.checkpoints",
            likec4_level=1,
            margin="0.223,0",
            penwidth=2,
            shape=cylinder,
            width=4.445];
    }
    researcher [height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">Researcher / operator</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">CLI, SLURM, W&amp;B</FONT></TD></TR><TR><TD><FONT POINT-SIZE="15" COLOR="#bfdbfe">Starts training, evaluation, profiling, and<BR/>analysis runs.<BR/>The current code is organized around<BR/>script-level run selection, launcher<BR/>registries, and a JAX/Flax R2Dreamer agent.</FONT></TD></TR></TABLE>>,
        likec4_id=researcher,
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    researcher -> evaluate [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">runs evaluation</FONT></TD></TR></TABLE>>,
        likec4_id="1s763ld",
        minlen=1,
        style=dashed];
    agent_boundary [height=2.5,
        label=<<TABLE BORDER="0" CELLPADDING="0" CELLSPACING="4"><TR><TD><FONT POINT-SIZE="20">R2Dreamer agent</FONT></TD></TR><TR><TD><FONT POINT-SIZE="13" COLOR="#bfdbfe">src/r2dreamer/agent.py</FONT></TD></TR></TABLE>>,
        likec4_id="codebase.agent_boundary",
        likec4_level=0,
        margin="0.223,0.223",
        width=4.445];
    parity -> agent_boundary [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">compare behavior</FONT></TD></TR></TABLE>>,
        likec4_id=rlj8yz,
        minlen=1,
        style=dashed];
    agent_boundary -> checkpoints [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">save</FONT></TD></TR></TABLE>>,
        likec4_id="1g8cbrp",
        style=dashed];
    evaluate -> checkpoints [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">load</FONT></TD></TR></TABLE>>,
        likec4_id="1avyt3c",
        minlen=0,
        style=dashed];
    checkpoints -> evaluate [arrowhead=normal,
        label=<<TABLE BORDER="0" CELLPADDING="3" CELLSPACING="0" BGCOLOR="#18191BA0"><TR><TD ALIGN="TEXT" BALIGN="LEFT"><FONT POINT-SIZE="14">restore</FONT></TD></TR></TABLE>>,
        likec4_id=jjjch4,
        minlen=0,
        style=dashed];
}
`;default:throw new Error("Unknown viewId: "+e)}}function n(e){switch(e){case"index":return`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
 "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<!-- Generated by graphviz version 14.0.4 (0)
 -->
<!-- Pages: 1 -->
<svg width="390pt" height="533pt"
 viewBox="0.00 0.00 390.00 533.00" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<g id="graph0" class="graph" transform="scale(1 1) rotate(0) translate(15.05 517.85)">
<!-- researcher -->
<g id="node1" class="node">
<title>researcher</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="349.4,-502.8 10.82,-502.8 10.82,-322.8 349.4,-322.8 349.4,-502.8"/>
<text xml:space="preserve" text-anchor="start" x="82.84" y="-463.6" font-family="Arial" font-size="20.00" fill="#eff6ff">Researcher / operator</text>
<text xml:space="preserve" text-anchor="start" x="125.21" y="-441.9" font-family="Arial" font-size="13.00" fill="#bfdbfe">CLI, SLURM, W&amp;B</text>
<text xml:space="preserve" text-anchor="start" x="48.37" y="-420.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">Starts training, evaluation, profiling, and</text>
<text xml:space="preserve" text-anchor="start" x="134.26" y="-402.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">analysis runs.</text>
<text xml:space="preserve" text-anchor="start" x="55.45" y="-384.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">The current code is organized around</text>
<text xml:space="preserve" text-anchor="start" x="66.72" y="-366.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">script&#45;level run selection, launcher</text>
<text xml:space="preserve" text-anchor="start" x="30.88" y="-348.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">registries, and a JAX/Flax R2Dreamer agent.</text>
</g>
<!-- codebase -->
<g id="node2" class="node">
<title>codebase</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="360.22,-180 0,-180 0,0 360.22,0 360.22,-180"/>
<text xml:space="preserve" text-anchor="start" x="20.06" y="-140.8" font-family="Arial" font-size="20.00" fill="#eff6ff">Master&#45;Thesis&#45;3D&#45;VLA current code</text>
<text xml:space="preserve" text-anchor="start" x="84.02" y="-119.1" font-family="Arial" font-size="13.00" fill="#bfdbfe">Python, JAX/Flax, Habitat, VGGT</text>
<text xml:space="preserve" text-anchor="start" x="59.23" y="-97.7" font-family="Arial" font-size="15.00" fill="#bfdbfe">Current main&#45;branch architecture for</text>
<text xml:space="preserve" text-anchor="start" x="100.08" y="-79.7" font-family="Arial" font-size="15.00" fill="#bfdbfe">ObjectNav experiments.</text>
<text xml:space="preserve" text-anchor="start" x="28.77" y="-61.7" font-family="Arial" font-size="15.00" fill="#bfdbfe">The source contracts are the root and scoped</text>
<text xml:space="preserve" text-anchor="start" x="38.81" y="-43.7" font-family="Arial" font-size="15.00" fill="#bfdbfe">AGENTS.md files plus the current src/ and</text>
<text xml:space="preserve" text-anchor="start" x="123.84" y="-25.7" font-family="Arial" font-size="15.00" fill="#bfdbfe">scripts/ modules.</text>
</g>
<!-- researcher&#45;&gt;codebase -->
<g id="edge1" class="edge">
<title>researcher&#45;&gt;codebase</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M180.11,-322.87C180.11,-281.67 180.11,-232.56 180.11,-190.17"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="182.73,-190.36 180.11,-182.86 177.48,-190.36 182.73,-190.36"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="180.11,-240 180.11,-262.8 207.1,-262.8 207.1,-240 180.11,-240"/>
<text xml:space="preserve" text-anchor="start" x="183.11" y="-248.2" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">[...]</text>
</g>
</g>
</svg>
`;case"view_1dcnbvb":return`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
 "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<!-- Generated by graphviz version 14.0.4 (0)
 -->
<!-- Pages: 1 -->
<svg width="1238pt" height="2204pt"
 viewBox="0.00 0.00 1238.00 2204.00" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<g id="graph0" class="graph" transform="scale(1 1) rotate(0) translate(15.05 2189.05)">
<g id="clust1" class="cluster">
<title>cluster_codebase</title>
<polygon fill="#194b9e" stroke="#1b3d88" points="8,-8 8,-1903.2 981,-1903.2 981,-8 8,-8"/>
<text xml:space="preserve" text-anchor="start" x="16" y="-1890.3" font-family="Arial" font-weight="bold" font-size="11.00" fill="#bfdbfe" fill-opacity="0.701961">MASTER&#45;THESIS&#45;3D&#45;VLA CURRENT CODE</text>
</g>
<!-- orchestration -->
<g id="node1" class="node">
<title>orchestration</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="511.02,-1842 190.98,-1842 190.98,-1662 511.02,-1662 511.02,-1842"/>
<text xml:space="preserve" text-anchor="start" x="239.84" y="-1755.8" font-family="Arial" font-size="20.00" fill="#eff6ff">Experiment orchestration</text>
<text xml:space="preserve" text-anchor="start" x="260.35" y="-1734.1" font-family="Arial" font-size="13.00" fill="#bfdbfe">scripts/r2dreamer, scripts/slurm</text>
</g>
<!-- vggt_boundary -->
<g id="node2" class="node">
<title>vggt_boundary</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="941.02,-1842 620.98,-1842 620.98,-1662 941.02,-1662 941.02,-1842"/>
<text xml:space="preserve" text-anchor="start" x="664.27" y="-1755.8" font-family="Arial" font-size="20.00" fill="#eff6ff">VGGT production encoder</text>
<text xml:space="preserve" text-anchor="start" x="748.13" y="-1734.1" font-family="Arial" font-size="13.00" fill="#bfdbfe">src/vggt/jax</text>
</g>
<!-- environment -->
<g id="node3" class="node">
<title>environment</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="368.02,-1519.2 47.98,-1519.2 47.98,-1339.2 368.02,-1339.2 368.02,-1519.2"/>
<text xml:space="preserve" text-anchor="start" x="127.41" y="-1433" font-family="Arial" font-size="20.00" fill="#eff6ff">Environment layer</text>
<text xml:space="preserve" text-anchor="start" x="112.63" y="-1411.3" font-family="Arial" font-size="13.00" fill="#bfdbfe">src/environments, Habitat/Crafter</text>
</g>
<!-- encoder_boundary -->
<g id="node4" class="node">
<title>encoder_boundary</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="748.02,-1196.4 427.98,-1196.4 427.98,-1016.4 748.02,-1016.4 748.02,-1196.4"/>
<text xml:space="preserve" text-anchor="start" x="470.14" y="-1110.2" font-family="Arial" font-size="20.00" fill="#eff6ff">Encoder and adapter layer</text>
<text xml:space="preserve" text-anchor="start" x="479.25" y="-1088.5" font-family="Arial" font-size="13.00" fill="#bfdbfe">src/r2dreamer/encoders and adapters</text>
</g>
<!-- training_loop -->
<g id="node5" class="node">
<title>training_loop</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="517.02,-873.6 196.98,-873.6 196.98,-693.6 517.02,-693.6 517.02,-873.6"/>
<text xml:space="preserve" text-anchor="start" x="254.16" y="-787.4" font-family="Arial" font-size="20.00" fill="#eff6ff">Trainer and replay loop</text>
<text xml:space="preserve" text-anchor="start" x="285.84" y="-765.7" font-family="Arial" font-size="13.00" fill="#bfdbfe">src/r2dreamer/trainer.py,</text>
</g>
<!-- agent_boundary -->
<g id="node6" class="node">
<title>agent_boundary</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="752.02,-550.8 431.98,-550.8 431.98,-370.8 752.02,-370.8 752.02,-550.8"/>
<text xml:space="preserve" text-anchor="start" x="512.52" y="-464.6" font-family="Arial" font-size="20.00" fill="#eff6ff">R2Dreamer agent</text>
<text xml:space="preserve" text-anchor="start" x="524.8" y="-442.9" font-family="Arial" font-size="13.00" fill="#bfdbfe">src/r2dreamer/agent.py</text>
</g>
<!-- evaluation -->
<g id="node7" class="node">
<title>evaluation</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="903.12,-228 582.88,-228 582.88,-48 903.12,-48 903.12,-228"/>
<text xml:space="preserve" text-anchor="start" x="602.93" y="-141.8" font-family="Arial" font-size="20.00" fill="#eff6ff">Evaluation and parity workflows</text>
<text xml:space="preserve" text-anchor="start" x="644.73" y="-120.1" font-family="Arial" font-size="13.00" fill="#bfdbfe">src/r2dreamer/launch/evaluate.py,</text>
</g>
<!-- researcher -->
<g id="node8" class="node">
<title>researcher</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="898.29,-2174 559.71,-2174 559.71,-1994 898.29,-1994 898.29,-2174"/>
<text xml:space="preserve" text-anchor="start" x="631.73" y="-2134.8" font-family="Arial" font-size="20.00" fill="#eff6ff">Researcher / operator</text>
<text xml:space="preserve" text-anchor="start" x="674.1" y="-2113.1" font-family="Arial" font-size="13.00" fill="#bfdbfe">CLI, SLURM, W&amp;B</text>
<text xml:space="preserve" text-anchor="start" x="597.26" y="-2091.7" font-family="Arial" font-size="15.00" fill="#bfdbfe">Starts training, evaluation, profiling, and</text>
<text xml:space="preserve" text-anchor="start" x="683.15" y="-2073.7" font-family="Arial" font-size="15.00" fill="#bfdbfe">analysis runs.</text>
<text xml:space="preserve" text-anchor="start" x="604.34" y="-2055.7" font-family="Arial" font-size="15.00" fill="#bfdbfe">The current code is organized around</text>
<text xml:space="preserve" text-anchor="start" x="615.61" y="-2037.7" font-family="Arial" font-size="15.00" fill="#bfdbfe">script&#45;level run selection, launcher</text>
<text xml:space="preserve" text-anchor="start" x="579.77" y="-2019.7" font-family="Arial" font-size="15.00" fill="#bfdbfe">registries, and a JAX/Flax R2Dreamer agent.</text>
</g>
<!-- orchestration&#45;&gt;environment -->
<g id="edge3" class="edge">
<title>orchestration&#45;&gt;environment</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M311.36,-1662.07C292.88,-1620.61 270.82,-1571.14 251.85,-1528.56"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="254.28,-1527.58 248.83,-1521.79 249.49,-1529.71 254.28,-1527.58"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="283.79,-1579.2 283.79,-1602 310.78,-1602 310.78,-1579.2 283.79,-1579.2"/>
<text xml:space="preserve" text-anchor="start" x="286.79" y="-1587.4" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">[...]</text>
</g>
<!-- orchestration&#45;&gt;encoder_boundary -->
<g id="edge4" class="edge">
<title>orchestration&#45;&gt;encoder_boundary</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M383.71,-1662.19C427.76,-1542.54 505.93,-1330.28 551.74,-1205.88"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="554.08,-1207.12 554.2,-1199.18 549.15,-1205.31 554.08,-1207.12"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="500.77,-1417.8 500.77,-1440.6 606.38,-1440.6 606.38,-1417.8 500.77,-1417.8"/>
<text xml:space="preserve" text-anchor="start" x="503.77" y="-1425" font-family="Arial" font-size="14.00" fill="#c9c9c9">resolve encoder</text>
</g>
<!-- vggt_boundary&#45;&gt;encoder_boundary -->
<g id="edge5" class="edge">
<title>vggt_boundary&#45;&gt;encoder_boundary</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M777.67,-1662.38C772.27,-1577.42 758.11,-1446.58 721,-1339.2 704.97,-1292.83 679.49,-1245.21 655.1,-1205.29"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="657.35,-1203.95 651.18,-1198.95 652.89,-1206.71 657.35,-1203.95"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="762.51,-1417.8 762.51,-1440.6 789.5,-1440.6 789.5,-1417.8 762.51,-1417.8"/>
<text xml:space="preserve" text-anchor="start" x="765.51" y="-1426" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">[...]</text>
</g>
<!-- environment&#45;&gt;encoder_boundary -->
<g id="edge6" class="edge">
<title>environment&#45;&gt;encoder_boundary</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M313.35,-1339.27C363.59,-1296.85 423.77,-1246.05 474.97,-1202.82"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="476.43,-1205.03 480.46,-1198.18 473.04,-1201.01 476.43,-1205.03"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="409.39,-1256.4 409.39,-1279.2 436.39,-1279.2 436.39,-1256.4 409.39,-1256.4"/>
<text xml:space="preserve" text-anchor="start" x="412.39" y="-1264.6" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">[...]</text>
</g>
<!-- encoder_boundary&#45;&gt;training_loop -->
<g id="edge7" class="edge">
<title>encoder_boundary&#45;&gt;training_loop</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M523.96,-1016.47C493.86,-974.66 457.89,-924.71 427.06,-881.89"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="429.25,-880.44 422.73,-875.89 424.99,-883.51 429.25,-880.44"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="479.43,-933.6 479.43,-956.4 559.37,-956.4 559.37,-933.6 479.43,-933.6"/>
<text xml:space="preserve" text-anchor="start" x="482.43" y="-940.8" font-family="Arial" font-size="14.00" fill="#c9c9c9">obs adapter</text>
</g>
<!-- encoder_boundary&#45;&gt;agent_boundary -->
<g id="edge8" class="edge">
<title>encoder_boundary&#45;&gt;agent_boundary</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M605.78,-1016.75C623.2,-917.61 644.86,-752.38 628,-610.8 626.06,-594.5 622.96,-577.4 619.38,-560.83"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="621.95,-560.3 617.75,-553.55 616.83,-561.45 621.95,-560.3"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="634.09,-772.2 634.09,-795 838.55,-795 838.55,-772.2 634.09,-772.2"/>
<text xml:space="preserve" text-anchor="start" x="637.09" y="-779.4" font-family="Arial" font-size="14.00" fill="#c9c9c9">agent overrides and module_cls</text>
</g>
<!-- training_loop&#45;&gt;agent_boundary -->
<g id="edge9" class="edge">
<title>training_loop&#45;&gt;agent_boundary</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M362.03,-693.62C366.81,-665.48 375.48,-635.34 391.01,-610.8 403.35,-591.29 419.24,-573.46 436.66,-557.45"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="438.11,-559.67 441.96,-552.71 434.61,-555.75 438.11,-559.67"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="391.01,-610.8 391.01,-633.6 418,-633.6 418,-610.8 391.01,-610.8"/>
<text xml:space="preserve" text-anchor="start" x="394.01" y="-619" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">[...]</text>
</g>
<!-- agent_boundary&#45;&gt;training_loop -->
<g id="edge10" class="edge">
<title>agent_boundary&#45;&gt;training_loop</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M526.93,-550.63C496.31,-592.42 459.72,-642.37 428.35,-685.2"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="426.25,-683.62 423.94,-691.22 430.49,-686.72 426.25,-683.62"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="481.55,-610.8 481.55,-633.6 601.15,-633.6 601.15,-610.8 481.55,-610.8"/>
<text xml:space="preserve" text-anchor="start" x="484.55" y="-618" font-family="Arial" font-size="14.00" fill="#c9c9c9">action and metrics</text>
</g>
<!-- agent_boundary&#45;&gt;evaluation -->
<g id="edge11" class="edge">
<title>agent_boundary&#45;&gt;evaluation</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M562.82,-370.9C558.45,-343.36 558.73,-313.56 570.43,-288 579.29,-268.65 592.03,-250.85 606.65,-234.82"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="608.17,-237.03 611.41,-229.77 604.35,-233.42 608.17,-237.03"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="570.43,-288 570.43,-310.8 606,-310.8 606,-288 570.43,-288"/>
<text xml:space="preserve" text-anchor="start" x="573.43" y="-295.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">save</text>
</g>
<!-- evaluation&#45;&gt;agent_boundary -->
<g id="edge12" class="edge">
<title>evaluation&#45;&gt;agent_boundary</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M701.19,-227.83C681.64,-269.36 658.3,-318.95 638.22,-361.6"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="635.97,-360.22 635.15,-368.13 640.72,-362.46 635.97,-360.22"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="672.03,-288 672.03,-310.8 790.09,-310.8 790.09,-288 672.03,-288"/>
<text xml:space="preserve" text-anchor="start" x="675.03" y="-295.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">compare behavior</text>
</g>
<!-- researcher&#45;&gt;orchestration -->
<g id="edge1" class="edge">
<title>researcher&#45;&gt;orchestration</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M627.21,-1994.13C575.62,-1949.1 512.94,-1894.38 460.33,-1848.45"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="462.3,-1846.69 454.93,-1843.73 458.85,-1850.64 462.3,-1846.69"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="551.33,-1911.2 551.33,-1934 615.69,-1934 615.69,-1911.2 551.33,-1911.2"/>
<text xml:space="preserve" text-anchor="start" x="554.33" y="-1918.4" font-family="Arial" font-size="14.00" fill="#c9c9c9">starts run</text>
</g>
<!-- researcher&#45;&gt;evaluation -->
<g id="edge2" class="edge">
<title>researcher&#45;&gt;evaluation</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M898.1,-2012.74C998.74,-1958.47 1107,-1872.35 1107,-1753 1107,-1753 1107,-1753 1107,-459.8 1107,-346.88 1007.19,-264.26 912.1,-211.16"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="913.41,-208.89 905.57,-207.58 910.88,-213.49 913.41,-208.89"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1107,-1095 1107,-1117.8 1207.95,-1117.8 1207.95,-1095 1107,-1095"/>
<text xml:space="preserve" text-anchor="start" x="1110" y="-1102.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">runs evaluation</text>
</g>
</g>
</svg>
`;case"view_1omwq3l":return`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
 "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<!-- Generated by graphviz version 14.0.4 (0)
 -->
<!-- Pages: 1 -->
<svg width="1289pt" height="2216pt"
 viewBox="0.00 0.00 1289.00 2216.00" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<g id="graph0" class="graph" transform="scale(1 1) rotate(0) translate(15.05 2201.05)">
<g id="clust1" class="cluster">
<title>cluster_orchestration</title>
<polygon fill="#194b9e" stroke="#1b3d88" points="8,-282.8 8,-2178 862,-2178 862,-282.8 8,-282.8"/>
<text xml:space="preserve" text-anchor="start" x="16" y="-2165.1" font-family="Arial" font-weight="bold" font-size="11.00" fill="#bfdbfe" fill-opacity="0.701961">EXPERIMENT ORCHESTRATION</text>
</g>
<!-- slurm_configs -->
<g id="node1" class="node">
<title>slurm_configs</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="810.55,-2116.8 479.45,-2116.8 479.45,-1936.8 810.55,-1936.8 810.55,-2116.8"/>
<text xml:space="preserve" text-anchor="start" x="575.53" y="-2059.6" font-family="Arial" font-size="20.00" fill="#eff6ff">SLURM configs</text>
<text xml:space="preserve" text-anchor="start" x="533.01" y="-2037.9" font-family="Arial" font-size="13.00" fill="#bfdbfe">scripts/slurm/configs/*.yaml and legacy</text>
<text xml:space="preserve" text-anchor="start" x="499.5" y="-2016.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">Render or call run.py with a run_id and train</text>
<text xml:space="preserve" text-anchor="start" x="510.76" y="-1998.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">flags. GPU work is launched through the</text>
<text xml:space="preserve" text-anchor="start" x="586.22" y="-1980.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">cluster scheduler.</text>
</g>
<!-- run_dispatcher -->
<g id="node2" class="node">
<title>run_dispatcher</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="805.02,-1794 484.98,-1794 484.98,-1614 805.02,-1614 805.02,-1794"/>
<text xml:space="preserve" text-anchor="start" x="536.63" y="-1727.8" font-family="Arial" font-size="20.00" fill="#eff6ff">scripts/r2dreamer/run.py</text>
<text xml:space="preserve" text-anchor="start" x="576.71" y="-1706.1" font-family="Arial" font-size="13.00" fill="#bfdbfe">Single run&#45;id dispatcher</text>
<text xml:space="preserve" text-anchor="start" x="512.01" y="-1684.7" font-family="Arial" font-size="15.00" fill="#bfdbfe">Accepts run.py &lt;run&#45;id&gt; [train flags] and</text>
<text xml:space="preserve" text-anchor="start" x="521.59" y="-1666.7" font-family="Arial" font-size="15.00" fill="#bfdbfe">forwards to _run_configs.launch_run.</text>
</g>
<!-- run_configs -->
<g id="node3" class="node">
<title>run_configs</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="821.79,-1471.2 468.21,-1471.2 468.21,-1291.2 821.79,-1291.2 821.79,-1471.2"/>
<text xml:space="preserve" text-anchor="start" x="572.22" y="-1432" font-family="Arial" font-size="20.00" fill="#eff6ff">RUN_CONFIGS</text>
<text xml:space="preserve" text-anchor="start" x="546.74" y="-1410.3" font-family="Arial" font-size="13.00" fill="#bfdbfe">scripts/r2dreamer/_run_configs.py</text>
<text xml:space="preserve" text-anchor="start" x="515.76" y="-1388.9" font-family="Arial" font-size="15.00" fill="#bfdbfe">Single source of truth for env, encoder,</text>
<text xml:space="preserve" text-anchor="start" x="507" y="-1370.9" font-family="Arial" font-size="15.00" fill="#bfdbfe">curriculum, output_dir, wandb_name, and</text>
<text xml:space="preserve" text-anchor="start" x="602.47" y="-1352.9" font-family="Arial" font-size="15.00" fill="#bfdbfe">wandb_tags.</text>
<text xml:space="preserve" text-anchor="start" x="488.27" y="-1334.9" font-family="Arial" font-size="15.00" fill="#bfdbfe">Known run ids cover Habitat L1&#45;L4 CNN/VGGT</text>
<text xml:space="preserve" text-anchor="start" x="558.72" y="-1316.9" font-family="Arial" font-size="15.00" fill="#bfdbfe">variants and Crafter CNN.</text>
</g>
<!-- public_entrypoint -->
<g id="node4" class="node">
<title>public_entrypoint</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="808.47,-1148.4 481.53,-1148.4 481.53,-968.4 808.47,-968.4 808.47,-1148.4"/>
<text xml:space="preserve" text-anchor="start" x="593.88" y="-1082.2" font-family="Arial" font-size="20.00" fill="#eff6ff">src/main.py</text>
<text xml:space="preserve" text-anchor="start" x="551.07" y="-1060.5" font-family="Arial" font-size="13.00" fill="#bfdbfe">train, evaluate, parity commands</text>
<text xml:space="preserve" text-anchor="start" x="501.58" y="-1039.1" font-family="Arial" font-size="15.00" fill="#bfdbfe">Public CLI dispatcher for train/evaluate and</text>
<text xml:space="preserve" text-anchor="start" x="589.57" y="-1021.1" font-family="Arial" font-size="15.00" fill="#bfdbfe">parity workflows.</text>
</g>
<!-- train_entry -->
<g id="node5" class="node">
<title>train_entry</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="805.02,-825.6 484.98,-825.6 484.98,-645.6 805.02,-645.6 805.02,-825.6"/>
<text xml:space="preserve" text-anchor="start" x="506.05" y="-777.4" font-family="Arial" font-size="20.00" fill="#eff6ff">src.r2dreamer.launch.train.train</text>
<text xml:space="preserve" text-anchor="start" x="568.4" y="-755.7" font-family="Arial" font-size="13.00" fill="#bfdbfe">Launcher composition root</text>
<text xml:space="preserve" text-anchor="start" x="507.02" y="-734.3" font-family="Arial" font-size="15.00" fill="#bfdbfe">Resolves curriculum, encoder, env, agent</text>
<text xml:space="preserve" text-anchor="start" x="509.92" y="-716.3" font-family="Arial" font-size="15.00" fill="#bfdbfe">config, trainer config, agent, and Trainer.</text>
<text xml:space="preserve" text-anchor="start" x="505.36" y="-698.3" font-family="Arial" font-size="15.00" fill="#bfdbfe">Runs Trainer.run() and returns the Trainer</text>
<text xml:space="preserve" text-anchor="start" x="562.48" y="-680.3" font-family="Arial" font-size="15.00" fill="#bfdbfe">for programmatic callers.</text>
</g>
<!-- registries -->
<g id="node6" class="node">
<title>registries</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="391.8,-502.8 48.2,-502.8 48.2,-322.8 391.8,-322.8 391.8,-502.8"/>
<text xml:space="preserve" text-anchor="start" x="147.19" y="-436.6" font-family="Arial" font-size="20.00" fill="#eff6ff">launch registries</text>
<text xml:space="preserve" text-anchor="start" x="121.74" y="-414.9" font-family="Arial" font-size="13.00" fill="#bfdbfe">src/r2dreamer/launch/registries.py</text>
<text xml:space="preserve" text-anchor="start" x="68.25" y="-393.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">Maps encoder strings to Encoder classes and</text>
<text xml:space="preserve" text-anchor="start" x="128.71" y="-375.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">env strings to env factories.</text>
</g>
<!-- curriculum -->
<g id="node7" class="node">
<title>curriculum</title>
<path fill="#3b82f6" stroke="#2563eb" stroke-width="2" d="M822.02,-486.44C822.02,-495.47 750.3,-502.8 662,-502.8 573.7,-502.8 501.98,-495.47 501.98,-486.44 501.98,-486.44 501.98,-339.16 501.98,-339.16 501.98,-330.13 573.7,-322.8 662,-322.8 750.3,-322.8 822.02,-330.13 822.02,-339.16 822.02,-339.16 822.02,-486.44 822.02,-486.44"/>
<path fill="none" stroke="#2563eb" stroke-width="2" d="M822.02,-486.44C822.02,-477.41 750.3,-470.07 662,-470.07 573.7,-470.07 501.98,-477.41 501.98,-486.44"/>
<text xml:space="preserve" text-anchor="start" x="584.21" y="-436.6" font-family="Arial" font-size="20.00" fill="#eff6ff">Curriculum JSON</text>
<text xml:space="preserve" text-anchor="start" x="599.5" y="-414.9" font-family="Arial" font-size="13.00" fill="#bfdbfe">data/curriculum/*.json</text>
<text xml:space="preserve" text-anchor="start" x="524.44" y="-393.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">Habitat L1&#45;L4 curriculum files resolved by</text>
<text xml:space="preserve" text-anchor="start" x="541.93" y="-375.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">launch/curricula.py and _helpers.py.</text>
</g>
<!-- researcher -->
<g id="node8" class="node">
<title>researcher</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="1259.29,-2116.8 920.71,-2116.8 920.71,-1936.8 1259.29,-1936.8 1259.29,-2116.8"/>
<text xml:space="preserve" text-anchor="start" x="992.73" y="-2077.6" font-family="Arial" font-size="20.00" fill="#eff6ff">Researcher / operator</text>
<text xml:space="preserve" text-anchor="start" x="1035.1" y="-2055.9" font-family="Arial" font-size="13.00" fill="#bfdbfe">CLI, SLURM, W&amp;B</text>
<text xml:space="preserve" text-anchor="start" x="958.26" y="-2034.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">Starts training, evaluation, profiling, and</text>
<text xml:space="preserve" text-anchor="start" x="1044.15" y="-2016.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">analysis runs.</text>
<text xml:space="preserve" text-anchor="start" x="965.34" y="-1998.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">The current code is organized around</text>
<text xml:space="preserve" text-anchor="start" x="976.61" y="-1980.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">script&#45;level run selection, launcher</text>
<text xml:space="preserve" text-anchor="start" x="940.77" y="-1962.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">registries, and a JAX/Flax R2Dreamer agent.</text>
</g>
<!-- environment -->
<g id="node9" class="node">
<title>environment</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="1252.02,-502.8 931.98,-502.8 931.98,-322.8 1252.02,-322.8 1252.02,-502.8"/>
<text xml:space="preserve" text-anchor="start" x="1011.41" y="-416.6" font-family="Arial" font-size="20.00" fill="#eff6ff">Environment layer</text>
<text xml:space="preserve" text-anchor="start" x="996.63" y="-394.9" font-family="Arial" font-size="13.00" fill="#bfdbfe">src/environments, Habitat/Crafter</text>
</g>
<!-- encoder_boundary -->
<g id="node10" class="node">
<title>encoder_boundary</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="380.02,-180 59.98,-180 59.98,0 380.02,0 380.02,-180"/>
<text xml:space="preserve" text-anchor="start" x="102.14" y="-93.8" font-family="Arial" font-size="20.00" fill="#eff6ff">Encoder and adapter layer</text>
<text xml:space="preserve" text-anchor="start" x="111.25" y="-72.1" font-family="Arial" font-size="13.00" fill="#bfdbfe">src/r2dreamer/encoders and adapters</text>
</g>
<!-- slurm_configs&#45;&gt;run_dispatcher -->
<g id="edge2" class="edge">
<title>slurm_configs&#45;&gt;run_dispatcher</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M645,-1936.87C645,-1895.67 645,-1846.56 645,-1804.17"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="647.63,-1804.36 645,-1796.86 642.38,-1804.36 647.63,-1804.36"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="645,-1854 645,-1876.8 732.7,-1876.8 732.7,-1854 645,-1854"/>
<text xml:space="preserve" text-anchor="start" x="648" y="-1861.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">render or call</text>
</g>
<!-- run_dispatcher&#45;&gt;run_configs -->
<g id="edge3" class="edge">
<title>run_dispatcher&#45;&gt;run_configs</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M645,-1614.07C645,-1572.87 645,-1523.76 645,-1481.37"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="647.63,-1481.56 645,-1474.06 642.38,-1481.56 647.63,-1481.56"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="645,-1531.2 645,-1554 726.48,-1554 726.48,-1531.2 645,-1531.2"/>
<text xml:space="preserve" text-anchor="start" x="648" y="-1538.4" font-family="Arial" font-size="14.00" fill="#c9c9c9">select run id</text>
</g>
<!-- run_configs&#45;&gt;public_entrypoint -->
<g id="edge4" class="edge">
<title>run_configs&#45;&gt;public_entrypoint</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M645,-1291.27C645,-1250.07 645,-1200.96 645,-1158.57"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="647.63,-1158.76 645,-1151.26 642.38,-1158.76 647.63,-1158.76"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="645,-1208.4 645,-1231.2 759.92,-1231.2 759.92,-1208.4 645,-1208.4"/>
<text xml:space="preserve" text-anchor="start" x="648" y="-1215.6" font-family="Arial" font-size="14.00" fill="#c9c9c9">call src.main.train</text>
</g>
<!-- public_entrypoint&#45;&gt;train_entry -->
<g id="edge5" class="edge">
<title>public_entrypoint&#45;&gt;train_entry</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M645,-968.47C645,-927.27 645,-878.16 645,-835.77"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="647.63,-835.96 645,-828.46 642.38,-835.96 647.63,-835.96"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="645,-885.6 645,-908.4 734.27,-908.4 734.27,-885.6 645,-885.6"/>
<text xml:space="preserve" text-anchor="start" x="648" y="-892.8" font-family="Arial" font-size="14.00" fill="#c9c9c9">dispatch train</text>
</g>
<!-- train_entry&#45;&gt;registries -->
<g id="edge6" class="edge">
<title>train_entry&#45;&gt;registries</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M527.18,-645.67C470.87,-603.16 403.41,-552.24 346.07,-508.96"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="347.67,-506.88 340.1,-504.45 344.5,-511.07 347.67,-506.88"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="445.24,-562.8 445.24,-585.6 604.57,-585.6 604.57,-562.8 445.24,-562.8"/>
<text xml:space="preserve" text-anchor="start" x="448.24" y="-570" font-family="Arial" font-size="14.00" fill="#c9c9c9">resolve env and encoder</text>
</g>
<!-- train_entry&#45;&gt;curriculum -->
<g id="edge7" class="edge">
<title>train_entry&#45;&gt;curriculum</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M649.71,-645.67C651.88,-604.81 654.46,-556.18 656.69,-514.03"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="659.3,-514.39 657.07,-506.77 654.06,-514.12 659.3,-514.39"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="654.01,-562.8 654.01,-585.6 821.84,-585.6 821.84,-562.8 654.01,-562.8"/>
<text xml:space="preserve" text-anchor="start" x="657.01" y="-570" font-family="Arial" font-size="14.00" fill="#c9c9c9">resolve Habitat curriculum</text>
</g>
<!-- train_entry&#45;&gt;environment -->
<g id="edge8" class="edge">
<title>train_entry&#45;&gt;environment</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M768.92,-645.67C828.27,-603.07 899.39,-552.03 959.78,-508.69"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="961.2,-510.9 965.77,-504.39 958.14,-506.63 961.2,-510.9"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="881.9,-562.8 881.9,-585.6 908.9,-585.6 908.9,-562.8 881.9,-562.8"/>
<text xml:space="preserve" text-anchor="start" x="884.9" y="-571" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">[...]</text>
</g>
<!-- registries&#45;&gt;encoder_boundary -->
<g id="edge9" class="edge">
<title>registries&#45;&gt;encoder_boundary</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M220,-322.87C220,-281.67 220,-232.56 220,-190.17"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="222.63,-190.36 220,-182.86 217.38,-190.36 222.63,-190.36"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="220,-240 220,-262.8 325.61,-262.8 325.61,-240 220,-240"/>
<text xml:space="preserve" text-anchor="start" x="223" y="-247.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">resolve encoder</text>
</g>
<!-- researcher&#45;&gt;run_dispatcher -->
<g id="edge1" class="edge">
<title>researcher&#45;&gt;run_dispatcher</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M966.63,-1936.87C907.55,-1894.27 836.75,-1843.23 776.63,-1799.89"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="778.3,-1797.86 770.68,-1795.6 775.22,-1802.11 778.3,-1797.86"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="880.84,-1854 880.84,-1876.8 945.19,-1876.8 945.19,-1854 880.84,-1854"/>
<text xml:space="preserve" text-anchor="start" x="883.84" y="-1861.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">starts run</text>
</g>
</g>
</svg>
`;case"view_spshjc":return`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
 "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<!-- Generated by graphviz version 14.0.4 (0)
 -->
<!-- Pages: 1 -->
<svg width="5116pt" height="1236pt"
 viewBox="0.00 0.00 5116.00 1236.00" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<g id="graph0" class="graph" transform="scale(1 1) rotate(0) translate(15.05 1220.65)">
<g id="clust1" class="cluster">
<title>cluster_encoder_boundary</title>
<polygon fill="#194b9e" stroke="#1b3d88" points="360.02,-8 360.02,-934.8 4296.02,-934.8 4296.02,-8 360.02,-8"/>
<text xml:space="preserve" text-anchor="start" x="368.02" y="-921.9" font-family="Arial" font-weight="bold" font-size="11.00" fill="#bfdbfe" fill-opacity="0.701961">ENCODER AND ADAPTER LAYER</text>
</g>
<!-- encoder_spec -->
<g id="node1" class="node">
<title>encoder_spec</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="3395,-873.6 3053.04,-873.6 3053.04,-693.6 3395,-693.6 3395,-873.6"/>
<text xml:space="preserve" text-anchor="start" x="3163.98" y="-834.4" font-family="Arial" font-size="20.00" fill="#eff6ff">EncoderSpec</text>
<text xml:space="preserve" text-anchor="start" x="3122.11" y="-812.7" font-family="Arial" font-size="13.00" fill="#bfdbfe">obs_shape, env_render_resolution,</text>
<text xml:space="preserve" text-anchor="start" x="3083.52" y="-791.3" font-family="Arial" font-size="15.00" fill="#bfdbfe">Launcher&#45;side contract that keeps adapter</text>
<text xml:space="preserve" text-anchor="start" x="3073.1" y="-773.3" font-family="Arial" font-size="15.00" fill="#bfdbfe">output, environment render size, and the Flax</text>
<text xml:space="preserve" text-anchor="start" x="3141.88" y="-755.3" font-family="Arial" font-size="15.00" fill="#bfdbfe">encoder module aligned.</text>
<text xml:space="preserve" text-anchor="start" x="3131.05" y="-737.3" font-family="Arial" font-size="15.00" fill="#bfdbfe">train.py copies the spec into</text>
<text xml:space="preserve" text-anchor="start" x="3161.5" y="-719.3" font-family="Arial" font-size="15.00" fill="#bfdbfe">R2DreamerConfig.</text>
</g>
<!-- obs_adapter -->
<g id="node2" class="node">
<title>obs_adapter</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="3393.74,-550.8 3054.3,-550.8 3054.3,-370.8 3393.74,-370.8 3393.74,-550.8"/>
<text xml:space="preserve" text-anchor="start" x="3170.66" y="-493.6" font-family="Arial" font-size="20.00" fill="#eff6ff">ObsAdapter</text>
<text xml:space="preserve" text-anchor="start" x="3109.85" y="-471.9" font-family="Arial" font-size="13.00" fill="#bfdbfe">src/r2dreamer/adapters/obs_adapter.py</text>
<text xml:space="preserve" text-anchor="start" x="3074.35" y="-450.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">Default RGB passthrough adapter with buffer</text>
<text xml:space="preserve" text-anchor="start" x="3113.94" y="-432.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">shape (3,64,64), dtype uint8, and</text>
<text xml:space="preserve" text-anchor="start" x="3128.76" y="-414.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">normalize_on_sample=True.</text>
</g>
<!-- vggt_adapter -->
<g id="node3" class="node">
<title>vggt_adapter</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="2543.3,-550.8 2184.74,-550.8 2184.74,-370.8 2543.3,-370.8 2543.3,-550.8"/>
<text xml:space="preserve" text-anchor="start" x="2282.32" y="-511.6" font-family="Arial" font-size="20.00" fill="#eff6ff">VGGTObsAdapter</text>
<text xml:space="preserve" text-anchor="start" x="2248.04" y="-489.9" font-family="Arial" font-size="13.00" fill="#bfdbfe">src/r2dreamer/adapters/vggt_adapter.py</text>
<text xml:space="preserve" text-anchor="start" x="2204.79" y="-468.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">Runs JAXVGGTFeatureExtractor per frame and</text>
<text xml:space="preserve" text-anchor="start" x="2221.02" y="-450.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">returns replay features plus one&#45;step agent</text>
<text xml:space="preserve" text-anchor="start" x="2334.84" y="-432.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">features.</text>
<text xml:space="preserve" text-anchor="start" x="2211.84" y="-414.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">Feature kinds: wp_cp, aggregator, wp_dense,</text>
<text xml:space="preserve" text-anchor="start" x="2333.17" y="-396.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">agg_raw.</text>
</g>
<!-- hybrid_adapter -->
<g id="node4" class="node">
<title>hybrid_adapter</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="4112.14,-550.8 3731.9,-550.8 3731.9,-370.8 4112.14,-370.8 4112.14,-550.8"/>
<text xml:space="preserve" text-anchor="start" x="3839.76" y="-493.6" font-family="Arial" font-size="20.00" fill="#eff6ff">HybridObsAdapter</text>
<text xml:space="preserve" text-anchor="start" x="3800.62" y="-471.9" font-family="Arial" font-size="13.00" fill="#bfdbfe">src/r2dreamer/adapters/hybrid_adapter.py</text>
<text xml:space="preserve" text-anchor="start" x="3751.96" y="-450.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">Runs VGGT for WP/CP, resizes the same 518x518</text>
<text xml:space="preserve" text-anchor="start" x="3789.46" y="-432.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">frame to 64x64 RGB, and stores explicit</text>
<text xml:space="preserve" text-anchor="start" x="3819.05" y="-414.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">replay fields image and wp_cp.</text>
</g>
<!-- cnn_spec -->
<g id="node5" class="node">
<title>cnn_spec</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="3384.04,-228 3064,-228 3064,-48 3384.04,-48 3384.04,-228"/>
<text xml:space="preserve" text-anchor="start" x="3207.9" y="-141.8" font-family="Arial" font-size="20.00" fill="#eff6ff">cnn</text>
<text xml:space="preserve" text-anchor="start" x="3121.4" y="-120.1" font-family="Arial" font-size="13.00" fill="#bfdbfe">ConvEncoder, obs_shape (3,64,64)</text>
</g>
<!-- wp_cp_spec -->
<g id="node6" class="node">
<title>wp_cp_spec</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="739.75,-228 400.29,-228 400.29,-48 739.75,-48 739.75,-228"/>
<text xml:space="preserve" text-anchor="start" x="472.73" y="-170.8" font-family="Arial" font-size="20.00" fill="#eff6ff">vggt / vggt_wp_cp_64</text>
<text xml:space="preserve" text-anchor="start" x="512.22" y="-149.1" font-family="Arial" font-size="13.00" fill="#bfdbfe">VGGTEncoder MLP</text>
<text xml:space="preserve" text-anchor="start" x="429.13" y="-127.7" font-family="Arial" font-size="15.00" fill="#bfdbfe">WP/CP vector is K*K*3 world points plus 9</text>
<text xml:space="preserve" text-anchor="start" x="420.34" y="-109.7" font-family="Arial" font-size="15.00" fill="#bfdbfe">camera&#45;pose values. K=37 gives 4116; K=64</text>
<text xml:space="preserve" text-anchor="start" x="527.49" y="-91.7" font-family="Arial" font-size="15.00" fill="#bfdbfe">gives 12297.</text>
</g>
<!-- aggregator_spec -->
<g id="node7" class="node">
<title>aggregator_spec</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="1210.52,-228 849.52,-228 849.52,-48 1210.52,-48 1210.52,-228"/>
<text xml:space="preserve" text-anchor="start" x="935.51" y="-170.8" font-family="Arial" font-size="20.00" fill="#eff6ff">vggt_aggregator_mlp</text>
<text xml:space="preserve" text-anchor="start" x="941.87" y="-149.1" font-family="Arial" font-size="13.00" fill="#bfdbfe">VGGTAggregatorMLPEncoder</text>
<text xml:space="preserve" text-anchor="start" x="889.11" y="-127.7" font-family="Arial" font-size="15.00" fill="#bfdbfe">Pools VGGT global aggregator tokens into</text>
<text xml:space="preserve" text-anchor="start" x="869.57" y="-109.7" font-family="Arial" font-size="15.00" fill="#bfdbfe">[camera | mean_patches | max_patches] = 3072</text>
<text xml:space="preserve" text-anchor="start" x="981.65" y="-91.7" font-family="Arial" font-size="15.00" fill="#bfdbfe">float32 values.</text>
</g>
<!-- dense_wp_spec -->
<g id="node8" class="node">
<title>dense_wp_spec</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="1663.83,-228 1320.21,-228 1320.21,-48 1663.83,-48 1663.83,-228"/>
<text xml:space="preserve" text-anchor="start" x="1400.28" y="-170.8" font-family="Arial" font-size="20.00" fill="#eff6ff">vggt_wp_dense_cnn</text>
<text xml:space="preserve" text-anchor="start" x="1442.17" y="-149.1" font-family="Arial" font-size="13.00" fill="#bfdbfe">WPConvEncoder</text>
<text xml:space="preserve" text-anchor="start" x="1355.27" y="-127.7" font-family="Arial" font-size="15.00" fill="#bfdbfe">Stores dense world points as (3,518,518)</text>
<text xml:space="preserve" text-anchor="start" x="1340.27" y="-109.7" font-family="Arial" font-size="15.00" fill="#bfdbfe">float16 and encodes them with a symlog conv</text>
<text xml:space="preserve" text-anchor="start" x="1472.43" y="-91.7" font-family="Arial" font-size="15.00" fill="#bfdbfe">stack.</text>
</g>
<!-- wp_cp -->
<g id="node9" class="node">
<title>wp_cp</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="2094.04,-228 1774,-228 1774,-48 2094.04,-48 2094.04,-228"/>
<text xml:space="preserve" text-anchor="start" x="1862.33" y="-141.8" font-family="Arial" font-size="20.00" fill="#eff6ff">WP/CP features</text>
<text xml:space="preserve" text-anchor="start" x="1870.77" y="-120.1" font-family="Arial" font-size="13.00" fill="#bfdbfe">4116 or 12297 float32</text>
</g>
<!-- pooled_agg -->
<g id="node10" class="node">
<title>pooled_agg</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="2524.04,-228 2204,-228 2204,-48 2524.04,-48 2524.04,-228"/>
<text xml:space="preserve" text-anchor="start" x="2242.82" y="-141.8" font-family="Arial" font-size="20.00" fill="#eff6ff">Pooled aggregator features</text>
<text xml:space="preserve" text-anchor="start" x="2328.24" y="-120.1" font-family="Arial" font-size="13.00" fill="#bfdbfe">3072 float32</text>
</g>
<!-- dense_wp -->
<g id="node11" class="node">
<title>dense_wp</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="2954.04,-228 2634,-228 2634,-48 2954.04,-48 2954.04,-228"/>
<text xml:space="preserve" text-anchor="start" x="2691.19" y="-141.8" font-family="Arial" font-size="20.00" fill="#eff6ff">Dense world&#45;point map</text>
<text xml:space="preserve" text-anchor="start" x="2733.67" y="-120.1" font-family="Arial" font-size="13.00" fill="#bfdbfe">3 x 518 x 518 float16</text>
</g>
<!-- hybrid_spec -->
<g id="node12" class="node">
<title>hybrid_spec</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="3814.04,-228 3494,-228 3494,-48 3814.04,-48 3814.04,-228"/>
<text xml:space="preserve" text-anchor="start" x="3626.78" y="-170.8" font-family="Arial" font-size="20.00" fill="#eff6ff">hybrid</text>
<text xml:space="preserve" text-anchor="start" x="3611.03" y="-149.1" font-family="Arial" font-size="13.00" fill="#bfdbfe">HybridEncoder</text>
<text xml:space="preserve" text-anchor="start" x="3518.12" y="-127.7" font-family="Arial" font-size="15.00" fill="#bfdbfe">Stores image uint8 (3,64,64) and WP/CP</text>
<text xml:space="preserve" text-anchor="start" x="3523.1" y="-109.7" font-family="Arial" font-size="15.00" fill="#bfdbfe">float32 separately, then packs to 16404</text>
<text xml:space="preserve" text-anchor="start" x="3554.79" y="-91.7" font-family="Arial" font-size="15.00" fill="#bfdbfe">features at the JAX boundary.</text>
</g>
<!-- hybrid_obs -->
<g id="node13" class="node">
<title>hybrid_obs</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="4256.4,-228 3923.64,-228 3923.64,-48 4256.4,-48 4256.4,-228"/>
<text xml:space="preserve" text-anchor="start" x="3976.63" y="-170.8" font-family="Arial" font-size="20.00" fill="#eff6ff">Hybrid replay observation</text>
<text xml:space="preserve" text-anchor="start" x="4008.9" y="-149.1" font-family="Arial" font-size="13.00" fill="#bfdbfe">image uint8 + wp_cp float32</text>
<text xml:space="preserve" text-anchor="start" x="3947.44" y="-127.7" font-family="Arial" font-size="15.00" fill="#bfdbfe">Replay keeps modalities inspectable under</text>
<text xml:space="preserve" text-anchor="start" x="3943.69" y="-109.7" font-family="Arial" font-size="15.00" fill="#bfdbfe">explicit keys; obs_batch packs them into the</text>
<text xml:space="preserve" text-anchor="start" x="4031.24" y="-91.7" font-family="Arial" font-size="15.00" fill="#bfdbfe">legacy flat tensor.</text>
</g>
<!-- orchestration -->
<g id="node14" class="node">
<title>orchestration</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="3384.04,-1205.6 3064,-1205.6 3064,-1025.6 3384.04,-1025.6 3384.04,-1205.6"/>
<text xml:space="preserve" text-anchor="start" x="3112.86" y="-1119.4" font-family="Arial" font-size="20.00" fill="#eff6ff">Experiment orchestration</text>
<text xml:space="preserve" text-anchor="start" x="3133.37" y="-1097.7" font-family="Arial" font-size="13.00" fill="#bfdbfe">scripts/r2dreamer, scripts/slurm</text>
</g>
<!-- environment -->
<g id="node15" class="node">
<title>environment</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="5086.04,-873.6 4766,-873.6 4766,-693.6 5086.04,-693.6 5086.04,-873.6"/>
<text xml:space="preserve" text-anchor="start" x="4845.43" y="-787.4" font-family="Arial" font-size="20.00" fill="#eff6ff">Environment layer</text>
<text xml:space="preserve" text-anchor="start" x="4830.65" y="-765.7" font-family="Arial" font-size="13.00" fill="#bfdbfe">src/environments, Habitat/Crafter</text>
</g>
<!-- vggt_boundary -->
<g id="node16" class="node">
<title>vggt_boundary</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="4656.04,-873.6 4336,-873.6 4336,-693.6 4656.04,-693.6 4656.04,-873.6"/>
<text xml:space="preserve" text-anchor="start" x="4379.29" y="-787.4" font-family="Arial" font-size="20.00" fill="#eff6ff">VGGT production encoder</text>
<text xml:space="preserve" text-anchor="start" x="4463.15" y="-765.7" font-family="Arial" font-size="13.00" fill="#bfdbfe">src/vggt/jax</text>
</g>
<!-- agent_boundary -->
<g id="node17" class="node">
<title>agent_boundary</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="320.04,-550.8 0,-550.8 0,-370.8 320.04,-370.8 320.04,-550.8"/>
<text xml:space="preserve" text-anchor="start" x="80.54" y="-464.6" font-family="Arial" font-size="20.00" fill="#eff6ff">R2Dreamer agent</text>
<text xml:space="preserve" text-anchor="start" x="92.82" y="-442.9" font-family="Arial" font-size="13.00" fill="#bfdbfe">src/r2dreamer/agent.py</text>
</g>
<!-- training_loop -->
<g id="node18" class="node">
<title>training_loop</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="4686.04,-228 4366,-228 4366,-48 4686.04,-48 4686.04,-228"/>
<text xml:space="preserve" text-anchor="start" x="4423.18" y="-141.8" font-family="Arial" font-size="20.00" fill="#eff6ff">Trainer and replay loop</text>
<text xml:space="preserve" text-anchor="start" x="4454.86" y="-120.1" font-family="Arial" font-size="13.00" fill="#bfdbfe">src/r2dreamer/trainer.py,</text>
</g>
<!-- encoder_spec&#45;&gt;obs_adapter -->
<g id="edge7" class="edge">
<title>encoder_spec&#45;&gt;obs_adapter</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M3224.02,-693.67C3224.02,-652.47 3224.02,-603.36 3224.02,-560.97"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="3226.65,-561.16 3224.02,-553.66 3221.4,-561.16 3226.65,-561.16"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="3224.02,-610.8 3224.02,-633.6 3357.65,-633.6 3357.65,-610.8 3224.02,-610.8"/>
<text xml:space="preserve" text-anchor="start" x="3227.02" y="-618" font-family="Arial" font-size="14.00" fill="#c9c9c9">default RGB adapter</text>
</g>
<!-- encoder_spec&#45;&gt;vggt_adapter -->
<g id="edge8" class="edge">
<title>encoder_spec&#45;&gt;vggt_adapter</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M3053.08,-718.84C2908.69,-664.97 2702.15,-587.93 2552.89,-532.25"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="2553.93,-529.84 2545.98,-529.68 2552.09,-534.76 2553.93,-529.84"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="2819.81,-610.8 2819.81,-633.6 2990.78,-633.6 2990.78,-610.8 2819.81,-610.8"/>
<text xml:space="preserve" text-anchor="start" x="2822.81" y="-618" font-family="Arial" font-size="14.00" fill="#c9c9c9">VGGT standalone variants</text>
</g>
<!-- encoder_spec&#45;&gt;hybrid_adapter -->
<g id="edge9" class="edge">
<title>encoder_spec&#45;&gt;hybrid_adapter</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M3394.76,-694.25C3449.41,-666.73 3510.4,-636.81 3566.98,-610.8 3617.01,-587.8 3671.6,-564.18 3722.64,-542.71"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="3723.44,-545.22 3729.34,-539.9 3721.41,-540.38 3723.44,-545.22"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="3566.98,-610.8 3566.98,-633.6 3657.02,-633.6 3657.02,-610.8 3566.98,-610.8"/>
<text xml:space="preserve" text-anchor="start" x="3569.98" y="-618" font-family="Arial" font-size="14.00" fill="#c9c9c9">hybrid variant</text>
</g>
<!-- encoder_spec&#45;&gt;agent_boundary -->
<g id="edge10" class="edge">
<title>encoder_spec&#45;&gt;agent_boundary</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M3053.23,-777.7C2513.49,-760.86 849.65,-698 333.02,-550.8 331.92,-550.49 330.83,-550.17 329.73,-549.84"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="330.65,-547.38 322.71,-547.63 329.07,-552.39 330.65,-547.38"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="741.3,-610.8 741.3,-633.6 945.75,-633.6 945.75,-610.8 741.3,-610.8"/>
<text xml:space="preserve" text-anchor="start" x="744.3" y="-618" font-family="Arial" font-size="14.00" fill="#c9c9c9">agent overrides and module_cls</text>
</g>
<!-- obs_adapter&#45;&gt;cnn_spec -->
<g id="edge11" class="edge">
<title>obs_adapter&#45;&gt;cnn_spec</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M3224.02,-370.87C3224.02,-329.67 3224.02,-280.56 3224.02,-238.17"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="3226.65,-238.36 3224.02,-230.86 3221.4,-238.36 3226.65,-238.36"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="3224.02,-288 3224.02,-310.8 3282.94,-310.8 3282.94,-288 3224.02,-288"/>
<text xml:space="preserve" text-anchor="start" x="3227.02" y="-295.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">declares</text>
</g>
<!-- obs_adapter&#45;&gt;training_loop -->
<g id="edge12" class="edge">
<title>obs_adapter&#45;&gt;training_loop</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M3393.64,-422.99C3478.45,-405.48 3582.85,-385.26 3677.02,-370.8 3915.7,-334.16 3986.98,-384.24 4217.02,-310.8 4274.66,-292.4 4333.73,-262.69 4384.58,-233.12"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="4385.9,-235.39 4391.04,-229.33 4383.24,-230.86 4385.9,-235.39"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="4271.03,-288 4271.03,-310.8 4350.98,-310.8 4350.98,-288 4271.03,-288"/>
<text xml:space="preserve" text-anchor="start" x="4274.03" y="-295.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">obs adapter</text>
</g>
<!-- vggt_adapter&#45;&gt;wp_cp_spec -->
<g id="edge13" class="edge">
<title>vggt_adapter&#45;&gt;wp_cp_spec</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M2185,-446.93C1890.64,-422.77 1287.79,-360.62 794.02,-228 779.35,-224.06 764.34,-219.45 749.37,-214.43"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="750.36,-212 742.42,-212.06 748.67,-216.97 750.36,-212"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1132.45,-288 1132.45,-310.8 1230.29,-310.8 1230.29,-288 1132.45,-288"/>
<text xml:space="preserve" text-anchor="start" x="1135.45" y="-295.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">wp_cp readout</text>
</g>
<!-- vggt_adapter&#45;&gt;aggregator_spec -->
<g id="edge14" class="edge">
<title>vggt_adapter&#45;&gt;aggregator_spec</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M2184.91,-430.78C1965.63,-393.25 1584.08,-321.46 1265.02,-228 1250.34,-223.7 1235.27,-218.91 1220.19,-213.84"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1221.1,-211.37 1213.16,-211.44 1219.41,-216.34 1221.1,-211.37"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1563.28,-288 1563.28,-310.8 1688.36,-310.8 1688.36,-288 1563.28,-288"/>
<text xml:space="preserve" text-anchor="start" x="1566.28" y="-295.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">aggregator readout</text>
</g>
<!-- vggt_adapter&#45;&gt;dense_wp_spec -->
<g id="edge15" class="edge">
<title>vggt_adapter&#45;&gt;dense_wp_spec</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M2184.92,-399.99C2106.71,-373.41 2014.08,-341.28 1931.06,-310.8 1836.09,-275.93 1813.24,-264.86 1719.02,-228 1704.17,-222.19 1688.77,-216.16 1673.31,-210.1"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1674.43,-207.73 1666.49,-207.43 1672.52,-212.61 1674.43,-207.73"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1931.06,-288 1931.06,-310.8 2053.02,-310.8 2053.02,-288 1931.06,-288"/>
<text xml:space="preserve" text-anchor="start" x="1934.06" y="-295.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">dense WP readout</text>
</g>
<!-- vggt_adapter&#45;&gt;wp_cp -->
<g id="edge16" class="edge">
<title>vggt_adapter&#45;&gt;wp_cp</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M2194.89,-370.81C2164.29,-352.33 2133.36,-331.99 2105.73,-310.8 2076.38,-288.27 2046.97,-261.01 2021,-234.89"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="2023.03,-233.21 2015.89,-229.72 2019.29,-236.9 2023.03,-233.21"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="2105.73,-288 2105.73,-310.8 2337.02,-310.8 2337.02,-288 2105.73,-288"/>
<text xml:space="preserve" text-anchor="start" x="2108.73" y="-295.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">extract world_points + camera_pose</text>
</g>
<!-- vggt_adapter&#45;&gt;pooled_agg -->
<g id="edge17" class="edge">
<title>vggt_adapter&#45;&gt;pooled_agg</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M2364.02,-370.87C2364.02,-329.67 2364.02,-280.56 2364.02,-238.17"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="2366.65,-238.36 2364.02,-230.86 2361.4,-238.36 2366.65,-238.36"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="2364.02,-288 2364.02,-310.8 2512.45,-310.8 2512.45,-288 2364.02,-288"/>
<text xml:space="preserve" text-anchor="start" x="2367.02" y="-295.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">pool final global tokens</text>
</g>
<!-- vggt_adapter&#45;&gt;dense_wp -->
<g id="edge18" class="edge">
<title>vggt_adapter&#45;&gt;dense_wp</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M2483.23,-370.87C2540.32,-328.27 2608.74,-277.23 2666.83,-233.89"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="2668.09,-236.23 2672.53,-229.64 2664.95,-232.02 2668.09,-236.23"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="2591.91,-288 2591.91,-310.8 2731,-310.8 2731,-288 2591.91,-288"/>
<text xml:space="preserve" text-anchor="start" x="2594.91" y="-295.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">request return_dense</text>
</g>
<!-- vggt_adapter&#45;&gt;training_loop -->
<g id="edge19" class="edge">
<title>vggt_adapter&#45;&gt;training_loop</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M2543.18,-430.76C2670.13,-411.14 2844.63,-386.12 2999.02,-370.8 3452.24,-325.82 3572.65,-384.94 4022.02,-310.8 4039.13,-307.98 4306.67,-233.8 4323.02,-228 4334.07,-224.08 4345.37,-219.83 4356.69,-215.39"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="4357.39,-217.94 4363.39,-212.74 4355.45,-213.06 4357.39,-217.94"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="4106.61,-288 4106.61,-310.8 4186.55,-310.8 4186.55,-288 4106.61,-288"/>
<text xml:space="preserve" text-anchor="start" x="4109.61" y="-295.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">obs adapter</text>
</g>
<!-- hybrid_adapter&#45;&gt;hybrid_spec -->
<g id="edge20" class="edge">
<title>hybrid_adapter&#45;&gt;hybrid_spec</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M3824.95,-370.87C3805.83,-351.74 3786.45,-331.14 3769.52,-310.8 3750.2,-287.6 3731.12,-261.13 3714.21,-236.03"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="3716.59,-234.87 3710.24,-230.09 3712.22,-237.79 3716.59,-234.87"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="3769.52,-288 3769.52,-310.8 3865.02,-310.8 3865.02,-288 3769.52,-288"/>
<text xml:space="preserve" text-anchor="start" x="3772.52" y="-295.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">hybrid readout</text>
</g>
<!-- hybrid_adapter&#45;&gt;hybrid_obs -->
<g id="edge21" class="edge">
<title>hybrid_adapter&#45;&gt;hybrid_obs</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M3891.61,-371.01C3886.99,-343.33 3887.26,-313.42 3899.65,-288 3909.13,-268.53 3922.53,-250.86 3937.87,-235.05"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="3939.43,-237.2 3942.9,-230.05 3935.73,-233.47 3939.43,-237.2"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="3899.65,-288 3899.65,-310.8 3999.02,-310.8 3999.02,-288 3899.65,-288"/>
<text xml:space="preserve" text-anchor="start" x="3902.65" y="-295.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">store two fields</text>
</g>
<!-- hybrid_adapter&#45;&gt;training_loop -->
<g id="edge22" class="edge">
<title>hybrid_adapter&#45;&gt;training_loop</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M4112.12,-433.45C4209.18,-412.62 4324.84,-375.72 4412.02,-310.8 4438.19,-291.32 4460.17,-263.93 4477.63,-236.74"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="4479.79,-238.25 4481.54,-230.5 4475.34,-235.46 4479.79,-238.25"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="4437.04,-288 4437.04,-310.8 4516.99,-310.8 4516.99,-288 4437.04,-288"/>
<text xml:space="preserve" text-anchor="start" x="4440.04" y="-295.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">obs adapter</text>
</g>
<!-- orchestration&#45;&gt;encoder_spec -->
<g id="edge1" class="edge">
<title>orchestration&#45;&gt;encoder_spec</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M3224.02,-1025.73C3224.02,-981.9 3224.02,-928.88 3224.02,-883.74"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="3226.65,-883.87 3224.02,-876.37 3221.4,-883.87 3226.65,-883.87"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="3224.02,-942.8 3224.02,-965.6 3329.63,-965.6 3329.63,-942.8 3224.02,-942.8"/>
<text xml:space="preserve" text-anchor="start" x="3227.02" y="-950" font-family="Arial" font-size="14.00" fill="#c9c9c9">resolve encoder</text>
</g>
<!-- environment&#45;&gt;obs_adapter -->
<g id="edge2" class="edge">
<title>environment&#45;&gt;obs_adapter</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M4766.14,-711.26C4747.78,-704.61 4729.15,-698.53 4711.02,-693.6 4525.48,-643.09 4471.37,-669.98 4282.55,-633.6 4240.96,-625.59 4231.7,-618.34 4190.02,-610.8 3964.13,-569.93 3903.81,-586.29 3677.02,-550.8 3586.66,-536.66 3486.84,-517.44 3403.96,-500.48"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="3404.52,-497.92 3396.64,-498.98 3403.46,-503.06 3404.52,-497.92"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="4282.55,-610.8 4282.55,-633.6 4350.02,-633.6 4350.02,-610.8 4282.55,-610.8"/>
<text xml:space="preserve" text-anchor="start" x="4285.55" y="-618" font-family="Arial" font-size="14.00" fill="#c9c9c9">CNN path</text>
</g>
<!-- environment&#45;&gt;vggt_adapter -->
<g id="edge3" class="edge">
<title>environment&#45;&gt;vggt_adapter</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M4766.09,-710.18C4747.78,-703.76 4729.17,-698.03 4711.02,-693.6 4408.86,-619.92 4320.99,-682.36 4013.82,-633.6 3969.74,-626.6 3960.17,-617.35 3916.02,-610.8 3512.01,-550.9 3405.31,-592.51 2999.02,-550.8 2848.75,-535.37 2679.36,-511.29 2553.35,-492.07"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="2554.01,-489.52 2546.2,-490.98 2553.22,-494.71 2554.01,-489.52"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="4013.82,-610.8 4013.82,-633.6 4163.02,-633.6 4163.02,-610.8 4013.82,-610.8"/>
<text xml:space="preserve" text-anchor="start" x="4016.82" y="-618" font-family="Arial" font-size="14.00" fill="#c9c9c9">VGGT standalone path</text>
</g>
<!-- environment&#45;&gt;hybrid_adapter -->
<g id="edge4" class="edge">
<title>environment&#45;&gt;hybrid_adapter</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M4766.28,-706.64C4694.25,-674.26 4607.45,-637.81 4527.02,-610.8 4394.24,-566.2 4241,-528.6 4121.97,-502.33"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="4122.86,-499.84 4114.97,-500.79 4121.73,-504.96 4122.86,-499.84"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="4582.67,-610.8 4582.67,-633.6 4657.94,-633.6 4657.94,-610.8 4582.67,-610.8"/>
<text xml:space="preserve" text-anchor="start" x="4585.67" y="-618" font-family="Arial" font-size="14.00" fill="#c9c9c9">hybrid path</text>
</g>
<!-- vggt_boundary&#45;&gt;vggt_adapter -->
<g id="edge5" class="edge">
<title>vggt_boundary&#45;&gt;vggt_adapter</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M4336.16,-697.87C4331.77,-696.35 4327.38,-694.92 4323.02,-693.6 4095.06,-624.66 4025.73,-671.03 3790.53,-633.6 3741,-625.72 3729.64,-618.08 3680.02,-610.8 3379.4,-566.7 3300.9,-585.23 2999.02,-550.8 2848.8,-533.67 2679.23,-509.66 2553.13,-490.9"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="2553.79,-488.34 2545.98,-489.83 2553.01,-493.53 2553.79,-488.34"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="3790.53,-610.8 3790.53,-633.6 3893.02,-633.6 3893.02,-610.8 3790.53,-610.8"/>
<text xml:space="preserve" text-anchor="start" x="3793.53" y="-618" font-family="Arial" font-size="14.00" fill="#c9c9c9">extract features</text>
</g>
<!-- vggt_boundary&#45;&gt;hybrid_adapter -->
<g id="edge6" class="edge">
<title>vggt_boundary&#45;&gt;hybrid_adapter</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M4450.28,-694.02C4431.12,-664.07 4406.4,-632.71 4377.02,-610.8 4302.67,-555.36 4206.69,-519.94 4122.06,-497.62"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="4122.92,-495.13 4115,-495.79 4121.6,-500.21 4122.92,-495.13"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="4402.51,-610.8 4402.51,-633.6 4500.3,-633.6 4500.3,-610.8 4402.51,-610.8"/>
<text xml:space="preserve" text-anchor="start" x="4405.51" y="-618" font-family="Arial" font-size="14.00" fill="#c9c9c9">extract WP/CP</text>
</g>
</g>
</svg>
`;case"view_12nwcpr":return`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
 "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<!-- Generated by graphviz version 14.0.4 (0)
 -->
<!-- Pages: 1 -->
<svg width="1330pt" height="1901pt"
 viewBox="0.00 0.00 1330.00 1901.00" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<g id="graph0" class="graph" transform="scale(1 1) rotate(0) translate(15.05 1886.45)">
<g id="clust1" class="cluster">
<title>cluster_training_loop</title>
<polygon fill="#194b9e" stroke="#1b3d88" points="8,-8 8,-1600.6 910,-1600.6 910,-8 8,-8"/>
<text xml:space="preserve" text-anchor="start" x="16" y="-1587.7" font-family="Arial" font-weight="bold" font-size="11.00" fill="#bfdbfe" fill-opacity="0.701961">TRAINER AND REPLAY LOOP</text>
</g>
<!-- trainer -->
<g id="node1" class="node">
<title>trainer</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="860.13,-1539.4 519.87,-1539.4 519.87,-1359.4 860.13,-1359.4 860.13,-1539.4"/>
<text xml:space="preserve" text-anchor="start" x="658.33" y="-1491.2" font-family="Arial" font-size="20.00" fill="#eff6ff">Trainer</text>
<text xml:space="preserve" text-anchor="start" x="593.18" y="-1469.5" font-family="Arial" font-size="13.00" fill="#bfdbfe">Prefill, collect, train_ratio, val, log,</text>
<text xml:space="preserve" text-anchor="start" x="539.92" y="-1448.1" font-family="Arial" font-size="15.00" fill="#bfdbfe">Owns environment stepping, replay insertion,</text>
<text xml:space="preserve" text-anchor="start" x="542.84" y="-1430.1" font-family="Arial" font-size="15.00" fill="#bfdbfe">batch sampling, validation env loop, metrics,</text>
<text xml:space="preserve" text-anchor="start" x="560.34" y="-1412.1" font-family="Arial" font-size="15.00" fill="#bfdbfe">W&amp;B logging, videos, checkpoints, and</text>
<text xml:space="preserve" text-anchor="start" x="628.33" y="-1394.1" font-family="Arial" font-size="15.00" fill="#bfdbfe">MANIFEST writes.</text>
</g>
<!-- replay -->
<g id="node2" class="node">
<title>replay</title>
<path fill="#3b82f6" stroke="#2563eb" stroke-width="2" d="M399.97,-1198.4C399.97,-1208.44 321.1,-1216.6 224,-1216.6 126.9,-1216.6 48.03,-1208.44 48.03,-1198.4 48.03,-1198.4 48.03,-1034.6 48.03,-1034.6 48.03,-1024.56 126.9,-1016.4 224,-1016.4 321.1,-1016.4 399.97,-1024.56 399.97,-1034.6 399.97,-1034.6 399.97,-1198.4 399.97,-1198.4"/>
<path fill="none" stroke="#2563eb" stroke-width="2" d="M399.97,-1198.4C399.97,-1188.36 321.1,-1180.2 224,-1180.2 126.9,-1180.2 48.03,-1188.36 48.03,-1198.4"/>
<text xml:space="preserve" text-anchor="start" x="166.19" y="-1167.3" font-family="Arial" font-size="20.00" fill="#eff6ff">ReplayBuffer</text>
<text xml:space="preserve" text-anchor="start" x="171.62" y="-1145.6" font-family="Arial" font-size="13.00" fill="#bfdbfe">NumPy ring buffer</text>
<text xml:space="preserve" text-anchor="start" x="68.08" y="-1124.2" font-family="Arial" font-size="15.00" fill="#bfdbfe">Stores observations as one array or a mapping</text>
<text xml:space="preserve" text-anchor="start" x="171.05" y="-1106.2" font-family="Arial" font-size="15.00" fill="#bfdbfe">of named fields.</text>
<text xml:space="preserve" text-anchor="start" x="90.19" y="-1088.2" font-family="Arial" font-size="15.00" fill="#bfdbfe">Samples fixed length (B,T) windows and</text>
<text xml:space="preserve" text-anchor="start" x="92.71" y="-1070.2" font-family="Arial" font-size="15.00" fill="#bfdbfe">returns JAX arrays with is_first, actions,</text>
<text xml:space="preserve" text-anchor="start" x="121.45" y="-1052.2" font-family="Arial" font-size="15.00" fill="#bfdbfe">rewards, dones, and terminals.</text>
</g>
<!-- metrics -->
<g id="node3" class="node">
<title>metrics</title>
<path fill="#3b82f6" stroke="#2563eb" stroke-width="2" d="M869.73,-1190.14C869.73,-1199.17 789.17,-1206.5 690,-1206.5 590.83,-1206.5 510.27,-1199.17 510.27,-1190.14 510.27,-1190.14 510.27,-1042.86 510.27,-1042.86 510.27,-1033.83 590.83,-1026.5 690,-1026.5 789.17,-1026.5 869.73,-1033.83 869.73,-1042.86 869.73,-1042.86 869.73,-1190.14 869.73,-1190.14"/>
<path fill="none" stroke="#2563eb" stroke-width="2" d="M869.73,-1190.14C869.73,-1181.11 789.17,-1173.77 690,-1173.77 590.83,-1173.77 510.27,-1181.11 510.27,-1190.14"/>
<text xml:space="preserve" text-anchor="start" x="633.87" y="-1149.3" font-family="Arial" font-size="20.00" fill="#eff6ff">Run artifacts</text>
<text xml:space="preserve" text-anchor="start" x="581.62" y="-1127.6" font-family="Arial" font-size="13.00" fill="#bfdbfe">output/runs, metrics.csv, checkpoints,</text>
<text xml:space="preserve" text-anchor="start" x="542.45" y="-1106.2" font-family="Arial" font-size="15.00" fill="#bfdbfe">Trainer writes local checkpoints and metrics,</text>
<text xml:space="preserve" text-anchor="start" x="530.32" y="-1088.2" font-family="Arial" font-size="15.00" fill="#bfdbfe">manifest provenance, optional videos, and W&amp;B</text>
<text xml:space="preserve" text-anchor="start" x="651.24" y="-1070.2" font-family="Arial" font-size="15.00" fill="#bfdbfe">summaries.</text>
</g>
<!-- sampled_batch -->
<g id="node4" class="node">
<title>sampled_batch</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="385.02,-873.6 64.98,-873.6 64.98,-693.6 385.02,-693.6 385.02,-873.6"/>
<text xml:space="preserve" text-anchor="start" x="112.14" y="-807.4" font-family="Arial" font-size="20.00" fill="#eff6ff">Sampled sequence batch</text>
<text xml:space="preserve" text-anchor="start" x="91.5" y="-785.7" font-family="Arial" font-size="13.00" fill="#bfdbfe">obs + actions + rewards + dones + terminals +</text>
<text xml:space="preserve" text-anchor="start" x="101.61" y="-764.3" font-family="Arial" font-size="15.00" fill="#bfdbfe">Replay sample window consumed by</text>
<text xml:space="preserve" text-anchor="start" x="105.33" y="-746.3" font-family="Arial" font-size="15.00" fill="#bfdbfe">convert_batch and agent.train_step.</text>
</g>
<!-- convert_batch -->
<g id="node5" class="node">
<title>convert_batch</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="385.12,-550.8 64.88,-550.8 64.88,-370.8 385.12,-370.8 385.12,-550.8"/>
<text xml:space="preserve" text-anchor="start" x="162.18" y="-493.6" font-family="Arial" font-size="20.00" fill="#eff6ff">convert_batch</text>
<text xml:space="preserve" text-anchor="start" x="197.91" y="-471.9" font-family="Arial" font-size="13.00" fill="#bfdbfe">trainer.py</text>
<text xml:space="preserve" text-anchor="start" x="84.94" y="-450.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">One&#45;hot encodes actions to (B,T,A), maps</text>
<text xml:space="preserve" text-anchor="start" x="89.52" y="-432.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">dones to is_last, terminals to is_terminal,</text>
<text xml:space="preserve" text-anchor="start" x="151.64" y="-414.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">and preserves is_first.</text>
</g>
<!-- obs_batch -->
<g id="node6" class="node">
<title>obs_batch</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="401.76,-228 48.24,-228 48.24,-48 401.76,-48 401.76,-228"/>
<text xml:space="preserve" text-anchor="start" x="148.28" y="-179.8" font-family="Arial" font-size="20.00" fill="#eff6ff">obs_batch bridge</text>
<text xml:space="preserve" text-anchor="start" x="144.07" y="-158.1" font-family="Arial" font-size="13.00" fill="#bfdbfe">src/r2dreamer/obs_batch.py</text>
<text xml:space="preserve" text-anchor="start" x="68.29" y="-136.7" font-family="Arial" font-size="15.00" fill="#bfdbfe">Normalizes CNN RGB, casts VGGT features to</text>
<text xml:space="preserve" text-anchor="start" x="81.17" y="-118.7" font-family="Arial" font-size="15.00" fill="#bfdbfe">float32, packs hybrid dict observations, and</text>
<text xml:space="preserve" text-anchor="start" x="74.5" y="-100.7" font-family="Arial" font-size="15.00" fill="#bfdbfe">reshapes observations to B*T before the Flax</text>
<text xml:space="preserve" text-anchor="start" x="195.81" y="-82.7" font-family="Arial" font-size="15.00" fill="#bfdbfe">encoder.</text>
</g>
<!-- encoder_boundary -->
<g id="node7" class="node">
<title>encoder_boundary</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="850.02,-1871.4 529.98,-1871.4 529.98,-1691.4 850.02,-1691.4 850.02,-1871.4"/>
<text xml:space="preserve" text-anchor="start" x="572.14" y="-1785.2" font-family="Arial" font-size="20.00" fill="#eff6ff">Encoder and adapter layer</text>
<text xml:space="preserve" text-anchor="start" x="581.25" y="-1763.5" font-family="Arial" font-size="13.00" fill="#bfdbfe">src/r2dreamer/encoders and adapters</text>
</g>
<!-- agent_boundary -->
<g id="node8" class="node">
<title>agent_boundary</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="1300.02,-1206.5 979.98,-1206.5 979.98,-1026.5 1300.02,-1026.5 1300.02,-1206.5"/>
<text xml:space="preserve" text-anchor="start" x="1060.52" y="-1120.3" font-family="Arial" font-size="20.00" fill="#eff6ff">R2Dreamer agent</text>
<text xml:space="preserve" text-anchor="start" x="1072.8" y="-1098.6" font-family="Arial" font-size="13.00" fill="#bfdbfe">src/r2dreamer/agent.py</text>
</g>
<!-- trainer&#45;&gt;replay -->
<g id="edge4" class="edge">
<title>trainer&#45;&gt;replay</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M564.81,-1359.51C504.05,-1316.36 430.75,-1264.31 367.71,-1219.55"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="369.39,-1217.52 361.75,-1215.32 366.35,-1221.8 369.39,-1217.52"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="470.97,-1276.6 470.97,-1299.4 561.03,-1299.4 561.03,-1276.6 470.97,-1276.6"/>
<text xml:space="preserve" text-anchor="start" x="473.97" y="-1283.8" font-family="Arial" font-size="14.00" fill="#c9c9c9">add transition</text>
</g>
<!-- trainer&#45;&gt;metrics -->
<g id="edge5" class="edge">
<title>trainer&#45;&gt;metrics</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M680.69,-1359.46C678.99,-1339.71 677.48,-1318.86 676.61,-1299.4 675.42,-1272.87 676.44,-1244.2 678.36,-1217.64"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="680.98,-1217.88 678.94,-1210.2 675.74,-1217.48 680.98,-1217.88"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="676.61,-1276.6 676.61,-1299.4 762,-1299.4 762,-1276.6 676.61,-1276.6"/>
<text xml:space="preserve" text-anchor="start" x="679.61" y="-1283.8" font-family="Arial" font-size="14.00" fill="#c9c9c9">log and save</text>
</g>
<!-- trainer&#45;&gt;agent_boundary -->
<g id="edge3" class="edge">
<title>trainer&#45;&gt;agent_boundary</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M778.94,-1359.82C808.89,-1331.85 843.23,-1301.71 876.7,-1276.6 906.7,-1254.09 940.06,-1231.96 972.67,-1211.69"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="973.81,-1214.07 978.81,-1207.89 971.05,-1209.6 973.81,-1214.07"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="876.7,-1276.6 876.7,-1299.4 994,-1299.4 994,-1276.6 876.7,-1276.6"/>
<text xml:space="preserve" text-anchor="start" x="879.7" y="-1283.8" font-family="Arial" font-size="14.00" fill="#c9c9c9">act and train_step</text>
</g>
<!-- replay&#45;&gt;sampled_batch -->
<g id="edge6" class="edge">
<title>replay&#45;&gt;sampled_batch</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M224.3,-1015.77C224.43,-974.02 224.57,-925.6 224.7,-883.89"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="227.32,-883.98 224.72,-876.47 222.07,-883.96 227.32,-883.98"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="224.53,-933.6 224.53,-956.4 310.65,-956.4 310.65,-933.6 224.53,-933.6"/>
<text xml:space="preserve" text-anchor="start" x="227.53" y="-940.8" font-family="Arial" font-size="14.00" fill="#c9c9c9">sample (B,T)</text>
</g>
<!-- sampled_batch&#45;&gt;convert_batch -->
<g id="edge7" class="edge">
<title>sampled_batch&#45;&gt;convert_batch</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M225,-693.67C225,-652.47 225,-603.36 225,-560.97"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="227.63,-561.16 225,-553.66 222.38,-561.16 227.63,-561.16"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="225,-610.8 225,-633.6 308.81,-633.6 308.81,-610.8 225,-610.8"/>
<text xml:space="preserve" text-anchor="start" x="228" y="-618" font-family="Arial" font-size="14.00" fill="#c9c9c9">format batch</text>
</g>
<!-- convert_batch&#45;&gt;obs_batch -->
<g id="edge8" class="edge">
<title>convert_batch&#45;&gt;obs_batch</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M225,-370.87C225,-329.67 225,-280.56 225,-238.17"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="227.63,-238.36 225,-230.86 222.38,-238.36 227.63,-238.36"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="225,-288 225,-310.8 328.3,-310.8 328.3,-288 225,-288"/>
<text xml:space="preserve" text-anchor="start" x="228" y="-295.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">agent boundary</text>
</g>
<!-- obs_batch&#45;&gt;agent_boundary -->
<g id="edge9" class="edge">
<title>obs_batch&#45;&gt;agent_boundary</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M308.19,-227.78C479.89,-411.02 872.24,-829.75 1049.78,-1019.22"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1047.67,-1020.8 1054.71,-1024.48 1051.5,-1017.21 1047.67,-1020.8"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="682.5,-610.8 682.5,-633.6 795.11,-633.6 795.11,-610.8 682.5,-610.8"/>
<text xml:space="preserve" text-anchor="start" x="685.5" y="-618" font-family="Arial" font-size="14.00" fill="#c9c9c9">B*T observations</text>
</g>
<!-- encoder_boundary&#45;&gt;trainer -->
<g id="edge1" class="edge">
<title>encoder_boundary&#45;&gt;trainer</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M690,-1691.53C690,-1647.7 690,-1594.68 690,-1549.54"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="692.63,-1549.67 690,-1542.17 687.38,-1549.67 692.63,-1549.67"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="690,-1608.6 690,-1631.4 769.94,-1631.4 769.94,-1608.6 690,-1608.6"/>
<text xml:space="preserve" text-anchor="start" x="693" y="-1615.8" font-family="Arial" font-size="14.00" fill="#c9c9c9">obs adapter</text>
</g>
<!-- agent_boundary&#45;&gt;trainer -->
<g id="edge2" class="edge">
<title>agent_boundary&#45;&gt;trainer</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1095.8,-1206.29C1076.31,-1238.76 1050.98,-1273.78 1021,-1299.4 976.94,-1337.06 921.81,-1366.89 869.26,-1389.66"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="868.47,-1387.14 862.6,-1392.5 870.53,-1391.97 868.47,-1387.14"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1042.18,-1276.6 1042.18,-1299.4 1161.78,-1299.4 1161.78,-1276.6 1042.18,-1276.6"/>
<text xml:space="preserve" text-anchor="start" x="1045.18" y="-1283.8" font-family="Arial" font-size="14.00" fill="#c9c9c9">action and metrics</text>
</g>
</g>
</svg>
`;case"view_iera9v":return`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
 "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<!-- Generated by graphviz version 14.0.4 (0)
 -->
<!-- Pages: 1 -->
<svg width="1538pt" height="2479pt"
 viewBox="0.00 0.00 1538.00 2479.00" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<g id="graph0" class="graph" transform="scale(1 1) rotate(0) translate(15.05 2463.85)">
<g id="clust1" class="cluster">
<title>cluster_agent_boundary</title>
<polygon fill="#194b9e" stroke="#1b3d88" points="8,-282.8 8,-2178 1118,-2178 1118,-282.8 8,-282.8"/>
<text xml:space="preserve" text-anchor="start" x="16" y="-2165.1" font-family="Arial" font-weight="bold" font-size="11.00" fill="#bfdbfe" fill-opacity="0.701961">R2DREAMER AGENT</text>
</g>
<!-- config -->
<g id="node1" class="node">
<title>config</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="385.9,-2116.8 48.1,-2116.8 48.1,-1936.8 385.9,-1936.8 385.9,-2116.8"/>
<text xml:space="preserve" text-anchor="start" x="136.41" y="-2068.6" font-family="Arial" font-size="20.00" fill="#eff6ff">R2DreamerConfig</text>
<text xml:space="preserve" text-anchor="start" x="148.72" y="-2046.9" font-family="Arial" font-size="13.00" fill="#bfdbfe">src/r2dreamer/config.py</text>
<text xml:space="preserve" text-anchor="start" x="74.87" y="-2025.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">Config&#45;first source of truth for RSSM sizes,</text>
<text xml:space="preserve" text-anchor="start" x="68.16" y="-2007.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">encoder choice, loss scales, batch/sequence</text>
<text xml:space="preserve" text-anchor="start" x="97.36" y="-1989.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">settings, train_ratio, replay capacity,</text>
<text xml:space="preserve" text-anchor="start" x="126.53" y="-1971.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">optimizer, and run defaults.</text>
</g>
<!-- encoder_mod -->
<g id="node2" class="node">
<title>encoder_mod</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="1076.02,-2116.8 755.98,-2116.8 755.98,-1936.8 1076.02,-1936.8 1076.02,-2116.8"/>
<text xml:space="preserve" text-anchor="start" x="803.71" y="-2059.6" font-family="Arial" font-size="20.00" fill="#eff6ff">Flax observation encoder</text>
<text xml:space="preserve" text-anchor="start" x="828.57" y="-2037.9" font-family="Arial" font-size="13.00" fill="#bfdbfe">ConvEncoder, VGGTEncoder,</text>
<text xml:space="preserve" text-anchor="start" x="778.41" y="-2016.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">Chosen by EncoderSpec.module_cls and</text>
<text xml:space="preserve" text-anchor="start" x="781.74" y="-1998.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">instantiated inside the agent, on the JAX</text>
<text xml:space="preserve" text-anchor="start" x="845.95" y="-1980.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">side of the boundary.</text>
</g>
<!-- embed -->
<g id="node3" class="node">
<title>embed</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="1076.02,-1794 755.98,-1794 755.98,-1614 1076.02,-1614 1076.02,-1794"/>
<text xml:space="preserve" text-anchor="start" x="828.73" y="-1727.8" font-family="Arial" font-size="20.00" fill="#eff6ff">Observation embed</text>
<text xml:space="preserve" text-anchor="start" x="836.14" y="-1706.1" font-family="Arial" font-size="13.00" fill="#bfdbfe">usually 1024 or hybrid 2048</text>
<text xml:space="preserve" text-anchor="start" x="778.42" y="-1684.7" font-family="Arial" font-size="15.00" fill="#bfdbfe">Encoder output that conditions the RSSM</text>
<text xml:space="preserve" text-anchor="start" x="884.74" y="-1666.7" font-family="Arial" font-size="15.00" fill="#bfdbfe">posterior.</text>
</g>
<!-- rssm -->
<g id="node4" class="node">
<title>rssm</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="1076.02,-1471.2 755.98,-1471.2 755.98,-1291.2 1076.02,-1291.2 1076.02,-1471.2"/>
<text xml:space="preserve" text-anchor="start" x="874.33" y="-1405" font-family="Arial" font-size="20.00" fill="#eff6ff">R2RSSM</text>
<text xml:space="preserve" text-anchor="start" x="812.33" y="-1383.3" font-family="Arial" font-size="13.00" fill="#bfdbfe">src/r2dreamer/world_model/rssm.py</text>
<text xml:space="preserve" text-anchor="start" x="778.03" y="-1361.9" font-family="Arial" font-size="15.00" fill="#bfdbfe">Block&#45;GRU latent dynamics with observe,</text>
<text xml:space="preserve" text-anchor="start" x="837.61" y="-1343.9" font-family="Arial" font-size="15.00" fill="#bfdbfe">img_step, and get_feat.</text>
</g>
<!-- rssm_feat -->
<g id="node5" class="node">
<title>rssm_feat</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="1077.59,-1148.4 754.41,-1148.4 754.41,-968.4 1077.59,-968.4 1077.59,-1148.4"/>
<text xml:space="preserve" text-anchor="start" x="853.2" y="-1082.2" font-family="Arial" font-size="20.00" fill="#eff6ff">RSSM feature</text>
<text xml:space="preserve" text-anchor="start" x="793.34" y="-1060.5" font-family="Arial" font-size="13.00" fill="#bfdbfe">deter_size + stoch_classes*stoch_discrete</text>
<text xml:space="preserve" text-anchor="start" x="774.47" y="-1039.1" font-family="Arial" font-size="15.00" fill="#bfdbfe">Default feature size is 2048 deterministic +</text>
<text xml:space="preserve" text-anchor="start" x="840.74" y="-1021.1" font-family="Arial" font-size="15.00" fill="#bfdbfe">512 stochastic = 2560.</text>
</g>
<!-- heads -->
<g id="node6" class="node">
<title>heads</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="648.02,-825.6 327.98,-825.6 327.98,-645.6 648.02,-645.6 648.02,-825.6"/>
<text xml:space="preserve" text-anchor="start" x="361.25" y="-739.4" font-family="Arial" font-size="20.00" fill="#eff6ff">Prediction and control heads</text>
<text xml:space="preserve" text-anchor="start" x="380.7" y="-717.7" font-family="Arial" font-size="13.00" fill="#bfdbfe">src/r2dreamer/world_model/heads.py</text>
</g>
<!-- losses -->
<g id="node7" class="node">
<title>losses</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="1078.02,-825.6 757.98,-825.6 757.98,-645.6 1078.02,-645.6 1078.02,-825.6"/>
<text xml:space="preserve" text-anchor="start" x="840.74" y="-739.4" font-family="Arial" font-size="20.00" fill="#eff6ff">Loss composition</text>
<text xml:space="preserve" text-anchor="start" x="807.44" y="-717.7" font-family="Arial" font-size="13.00" fill="#bfdbfe">world_model, behavior, representation</text>
</g>
<!-- agent -->
<g id="node8" class="node">
<title>agent</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="976.95,-502.8 655.05,-502.8 655.05,-322.8 976.95,-322.8 976.95,-502.8"/>
<text xml:space="preserve" text-anchor="start" x="738.19" y="-463.6" font-family="Arial" font-size="20.00" fill="#eff6ff">R2DreamerAgent</text>
<text xml:space="preserve" text-anchor="start" x="740.5" y="-441.9" font-family="Arial" font-size="13.00" fill="#bfdbfe">JAX/Flax composition root</text>
<text xml:space="preserve" text-anchor="start" x="680.13" y="-420.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">Owns params, optimizer state, slow critic</text>
<text xml:space="preserve" text-anchor="start" x="675.11" y="-402.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">EMA, acting state, JIT&#45;compiled train_step</text>
<text xml:space="preserve" text-anchor="start" x="789.31" y="-384.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">and act.</text>
<text xml:space="preserve" text-anchor="start" x="698.85" y="-366.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">A single shared forward pass feeds</text>
<text xml:space="preserve" text-anchor="start" x="675.5" y="-348.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">world&#45;model, behavior, and representation</text>
</g>
<!-- encoder_boundary -->
<g id="node9" class="node">
<title>encoder_boundary</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="377.02,-2448.8 56.98,-2448.8 56.98,-2268.8 377.02,-2268.8 377.02,-2448.8"/>
<text xml:space="preserve" text-anchor="start" x="99.14" y="-2362.6" font-family="Arial" font-size="20.00" fill="#eff6ff">Encoder and adapter layer</text>
<text xml:space="preserve" text-anchor="start" x="108.25" y="-2340.9" font-family="Arial" font-size="13.00" fill="#bfdbfe">src/r2dreamer/encoders and adapters</text>
</g>
<!-- training_loop -->
<g id="node10" class="node">
<title>training_loop</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="1508.02,-825.6 1187.98,-825.6 1187.98,-645.6 1508.02,-645.6 1508.02,-825.6"/>
<text xml:space="preserve" text-anchor="start" x="1245.16" y="-739.4" font-family="Arial" font-size="20.00" fill="#eff6ff">Trainer and replay loop</text>
<text xml:space="preserve" text-anchor="start" x="1276.84" y="-717.7" font-family="Arial" font-size="13.00" fill="#bfdbfe">src/r2dreamer/trainer.py,</text>
</g>
<!-- evaluation -->
<g id="node11" class="node">
<title>evaluation</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="975.12,-180 654.88,-180 654.88,0 975.12,0 975.12,-180"/>
<text xml:space="preserve" text-anchor="start" x="674.93" y="-93.8" font-family="Arial" font-size="20.00" fill="#eff6ff">Evaluation and parity workflows</text>
<text xml:space="preserve" text-anchor="start" x="716.73" y="-72.1" font-family="Arial" font-size="13.00" fill="#bfdbfe">src/r2dreamer/launch/evaluate.py,</text>
</g>
<!-- config&#45;&gt;agent -->
<g id="edge5" class="edge">
<title>config&#45;&gt;agent</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M217,-1936.95C217,-1872.55 217,-1783.39 217,-1705 217,-1705 217,-1705 217,-734.6 217,-539.18 467.83,-462.34 644.86,-432.44"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="645.13,-435.06 652.1,-431.25 644.27,-429.88 645.13,-435.06"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="217,-1208.4 217,-1231.2 272.8,-1231.2 272.8,-1208.4 217,-1208.4"/>
<text xml:space="preserve" text-anchor="start" x="220" y="-1215.6" font-family="Arial" font-size="14.00" fill="#c9c9c9">initialize</text>
</g>
<!-- encoder_mod&#45;&gt;embed -->
<g id="edge6" class="edge">
<title>encoder_mod&#45;&gt;embed</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M916,-1936.87C916,-1895.67 916,-1846.56 916,-1804.17"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="918.63,-1804.36 916,-1796.86 913.38,-1804.36 918.63,-1804.36"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="916,-1854 916,-1876.8 994.39,-1876.8 994.39,-1854 916,-1854"/>
<text xml:space="preserve" text-anchor="start" x="919" y="-1861.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">encode obs</text>
</g>
<!-- embed&#45;&gt;rssm -->
<g id="edge7" class="edge">
<title>embed&#45;&gt;rssm</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M916,-1614.07C916,-1572.87 916,-1523.76 916,-1481.37"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="918.63,-1481.56 916,-1474.06 913.38,-1481.56 918.63,-1481.56"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="916,-1531.2 916,-1554 1030.17,-1554 1030.17,-1531.2 916,-1531.2"/>
<text xml:space="preserve" text-anchor="start" x="919" y="-1538.4" font-family="Arial" font-size="14.00" fill="#c9c9c9">posterior observe</text>
</g>
<!-- rssm&#45;&gt;rssm_feat -->
<g id="edge8" class="edge">
<title>rssm&#45;&gt;rssm_feat</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M916,-1291.27C916,-1250.07 916,-1200.96 916,-1158.57"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="918.63,-1158.76 916,-1151.26 913.38,-1158.76 918.63,-1158.76"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="916,-1208.4 916,-1231.2 1044.19,-1231.2 1044.19,-1208.4 916,-1208.4"/>
<text xml:space="preserve" text-anchor="start" x="919" y="-1215.6" font-family="Arial" font-size="14.00" fill="#c9c9c9">post states and feat</text>
</g>
<!-- rssm_feat&#45;&gt;heads -->
<g id="edge9" class="edge">
<title>rssm_feat&#45;&gt;heads</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M797.35,-968.47C740.52,-925.87 672.42,-874.83 614.6,-831.49"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="616.51,-829.64 608.94,-827.25 613.36,-833.84 616.51,-829.64"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="714.83,-885.6 714.83,-908.4 741.83,-908.4 741.83,-885.6 714.83,-885.6"/>
<text xml:space="preserve" text-anchor="start" x="717.83" y="-893.8" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">[...]</text>
</g>
<!-- rssm_feat&#45;&gt;losses -->
<g id="edge10" class="edge">
<title>rssm_feat&#45;&gt;losses</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M916.55,-968.47C916.81,-927.27 917.12,-878.16 917.38,-835.77"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="920.01,-835.97 917.43,-828.46 914.76,-835.94 920.01,-835.97"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="917.06,-885.6 917.06,-908.4 944.05,-908.4 944.05,-885.6 917.06,-885.6"/>
<text xml:space="preserve" text-anchor="start" x="920.06" y="-893.8" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">[...]</text>
</g>
<!-- losses&#45;&gt;agent -->
<g id="edge11" class="edge">
<title>losses&#45;&gt;agent</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M828.2,-645.92C814.36,-627.38 802.13,-606.94 794.44,-585.6 786.18,-562.71 785.3,-537.13 787.96,-512.83"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="790.55,-513.29 788.91,-505.51 785.34,-512.61 790.55,-513.29"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="794.44,-562.8 794.44,-585.6 1016,-585.6 1016,-562.8 794.44,-562.8"/>
<text xml:space="preserve" text-anchor="start" x="797.44" y="-570" font-family="Arial" font-size="14.00" fill="#c9c9c9">update params and optimizer state</text>
</g>
<!-- agent&#45;&gt;training_loop -->
<g id="edge12" class="edge">
<title>agent&#45;&gt;training_loop</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M963.31,-502.63C1034.35,-545.47 1119.61,-596.88 1191.78,-640.4"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1190.14,-642.48 1197.92,-644.1 1192.86,-637.98 1190.14,-642.48"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1097.95,-562.8 1097.95,-585.6 1217.56,-585.6 1217.56,-562.8 1097.95,-562.8"/>
<text xml:space="preserve" text-anchor="start" x="1100.95" y="-570" font-family="Arial" font-size="14.00" fill="#c9c9c9">action and metrics</text>
</g>
<!-- agent&#45;&gt;evaluation -->
<g id="edge13" class="edge">
<title>agent&#45;&gt;evaluation</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M771.29,-323.07C763.48,-303.67 756.54,-282.91 752.43,-262.8 747.45,-238.47 751.36,-212.97 759.19,-189.29"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="761.58,-190.41 761.62,-182.46 756.63,-188.64 761.58,-190.41"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="752.43,-240 752.43,-262.8 788,-262.8 788,-240 752.43,-240"/>
<text xml:space="preserve" text-anchor="start" x="755.43" y="-247.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">save</text>
</g>
<!-- encoder_boundary&#45;&gt;config -->
<g id="edge1" class="edge">
<title>encoder_boundary&#45;&gt;config</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M217,-2268.93C217,-2225.1 217,-2172.08 217,-2126.94"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="219.63,-2127.07 217,-2119.57 214.38,-2127.07 219.63,-2127.07"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="217,-2186 217,-2208.8 421.45,-2208.8 421.45,-2186 217,-2186"/>
<text xml:space="preserve" text-anchor="start" x="220" y="-2193.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">agent overrides and module_cls</text>
</g>
<!-- training_loop&#45;&gt;encoder_mod -->
<g id="edge2" class="edge">
<title>training_loop&#45;&gt;encoder_mod</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1336.94,-825.29C1329.86,-889.61 1322,-978.75 1322,-1057.4 1322,-1705 1322,-1705 1322,-1705 1322,-1832.79 1196.64,-1917.61 1085.24,-1967.77"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1084.44,-1965.25 1078.65,-1970.69 1086.57,-1970.05 1084.44,-1965.25"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1322,-1369.8 1322,-1392.6 1434.61,-1392.6 1434.61,-1369.8 1322,-1369.8"/>
<text xml:space="preserve" text-anchor="start" x="1325" y="-1377" font-family="Arial" font-size="14.00" fill="#c9c9c9">B*T observations</text>
</g>
<!-- training_loop&#45;&gt;agent -->
<g id="edge3" class="edge">
<title>training_loop&#45;&gt;agent</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1325.95,-645.78C1314.7,-615.99 1298.12,-584.79 1274,-562.8 1195.31,-491.05 1081.56,-453.81 987.11,-434.51"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="987.7,-431.95 979.83,-433.06 986.68,-437.1 987.7,-431.95"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1294.07,-562.8 1294.07,-585.6 1411.36,-585.6 1411.36,-562.8 1294.07,-562.8"/>
<text xml:space="preserve" text-anchor="start" x="1297.07" y="-570" font-family="Arial" font-size="14.00" fill="#c9c9c9">act and train_step</text>
</g>
<!-- evaluation&#45;&gt;agent -->
<g id="edge4" class="edge">
<title>evaluation&#45;&gt;agent</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M815.28,-179.83C815.41,-221.01 815.56,-270.12 815.69,-312.52"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="813.06,-312.35 815.71,-319.84 818.31,-312.33 813.06,-312.35"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="815.53,-240 815.53,-262.8 933.59,-262.8 933.59,-240 815.53,-240"/>
<text xml:space="preserve" text-anchor="start" x="818.53" y="-247.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">compare behavior</text>
</g>
</g>
</svg>
`;case"view_12tzet7":return`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
 "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<!-- Generated by graphviz version 14.0.4 (0)
 -->
<!-- Pages: 1 -->
<svg width="1306pt" height="1188pt"
 viewBox="0.00 0.00 1306.00 1188.00" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<g id="graph0" class="graph" transform="scale(1 1) rotate(0) translate(15.05 1172.65)">
<g id="clust1" class="cluster">
<title>cluster_losses</title>
<polygon fill="#194b9e" stroke="#1b3d88" points="8,-282.8 8,-886.8 1268,-886.8 1268,-282.8 8,-282.8"/>
<text xml:space="preserve" text-anchor="start" x="16" y="-873.9" font-family="Arial" font-weight="bold" font-size="11.00" fill="#bfdbfe" fill-opacity="0.701961">LOSS COMPOSITION</text>
</g>
<!-- wm_loss -->
<g id="node1" class="node">
<title>wm_loss</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="368.02,-825.6 47.98,-825.6 47.98,-645.6 368.02,-645.6 368.02,-825.6"/>
<text xml:space="preserve" text-anchor="start" x="127.96" y="-739.4" font-family="Arial" font-size="20.00" fill="#eff6ff">world_model_loss</text>
<text xml:space="preserve" text-anchor="start" x="87.13" y="-717.7" font-family="Arial" font-size="13.00" fill="#bfdbfe">KL dyn/rep + reward + continue + optional</text>
</g>
<!-- behavior_loss -->
<g id="node2" class="node">
<title>behavior_loss</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="798.02,-825.6 477.98,-825.6 477.98,-645.6 798.02,-645.6 798.02,-825.6"/>
<text xml:space="preserve" text-anchor="start" x="576.3" y="-739.4" font-family="Arial" font-size="20.00" fill="#eff6ff">behavior_loss</text>
<text xml:space="preserve" text-anchor="start" x="512.99" y="-717.7" font-family="Arial" font-size="13.00" fill="#bfdbfe">Detached imagination, lambda&#45;return, actor</text>
</g>
<!-- rep_loss -->
<g id="node3" class="node">
<title>rep_loss</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="1228.02,-825.6 907.98,-825.6 907.98,-645.6 1228.02,-645.6 1228.02,-825.6"/>
<text xml:space="preserve" text-anchor="start" x="980.72" y="-739.4" font-family="Arial" font-size="20.00" fill="#eff6ff">representation_loss</text>
<text xml:space="preserve" text-anchor="start" x="986.54" y="-717.7" font-family="Arial" font-size="13.00" fill="#bfdbfe">Barlow Twins + replay&#45;value</text>
</g>
<!-- optimizer -->
<g id="node4" class="node">
<title>optimizer</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="798.02,-502.8 477.98,-502.8 477.98,-322.8 798.02,-322.8 798.02,-502.8"/>
<text xml:space="preserve" text-anchor="start" x="539.32" y="-436.6" font-family="Arial" font-size="20.00" fill="#eff6ff">LaProp + AGC update</text>
<text xml:space="preserve" text-anchor="start" x="581.28" y="-414.9" font-family="Arial" font-size="13.00" fill="#bfdbfe">src/shared/optim.py</text>
<text xml:space="preserve" text-anchor="start" x="503.34" y="-393.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">Weighted total loss is differentiated once</text>
<text xml:space="preserve" text-anchor="start" x="510.42" y="-375.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">and updates the single params pytree.</text>
</g>
<!-- rssm_feat -->
<g id="node5" class="node">
<title>rssm_feat</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="799.59,-1157.6 476.41,-1157.6 476.41,-977.6 799.59,-977.6 799.59,-1157.6"/>
<text xml:space="preserve" text-anchor="start" x="575.2" y="-1091.4" font-family="Arial" font-size="20.00" fill="#eff6ff">RSSM feature</text>
<text xml:space="preserve" text-anchor="start" x="515.34" y="-1069.7" font-family="Arial" font-size="13.00" fill="#bfdbfe">deter_size + stoch_classes*stoch_discrete</text>
<text xml:space="preserve" text-anchor="start" x="496.47" y="-1048.3" font-family="Arial" font-size="15.00" fill="#bfdbfe">Default feature size is 2048 deterministic +</text>
<text xml:space="preserve" text-anchor="start" x="562.74" y="-1030.3" font-family="Arial" font-size="15.00" fill="#bfdbfe">512 stochastic = 2560.</text>
</g>
<!-- agent -->
<g id="node6" class="node">
<title>agent</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="798.95,-180 477.05,-180 477.05,0 798.95,0 798.95,-180"/>
<text xml:space="preserve" text-anchor="start" x="560.19" y="-140.8" font-family="Arial" font-size="20.00" fill="#eff6ff">R2DreamerAgent</text>
<text xml:space="preserve" text-anchor="start" x="562.5" y="-119.1" font-family="Arial" font-size="13.00" fill="#bfdbfe">JAX/Flax composition root</text>
<text xml:space="preserve" text-anchor="start" x="502.13" y="-97.7" font-family="Arial" font-size="15.00" fill="#bfdbfe">Owns params, optimizer state, slow critic</text>
<text xml:space="preserve" text-anchor="start" x="497.11" y="-79.7" font-family="Arial" font-size="15.00" fill="#bfdbfe">EMA, acting state, JIT&#45;compiled train_step</text>
<text xml:space="preserve" text-anchor="start" x="611.31" y="-61.7" font-family="Arial" font-size="15.00" fill="#bfdbfe">and act.</text>
<text xml:space="preserve" text-anchor="start" x="520.85" y="-43.7" font-family="Arial" font-size="15.00" fill="#bfdbfe">A single shared forward pass feeds</text>
<text xml:space="preserve" text-anchor="start" x="497.5" y="-25.7" font-family="Arial" font-size="15.00" fill="#bfdbfe">world&#45;model, behavior, and representation</text>
</g>
<!-- wm_loss&#45;&gt;optimizer -->
<g id="edge4" class="edge">
<title>wm_loss&#45;&gt;optimizer</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M327.21,-645.67C384.3,-603.07 452.72,-552.03 510.81,-508.69"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="512.07,-511.03 516.51,-504.44 508.93,-506.82 512.07,-511.03"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="435.89,-562.8 435.89,-585.6 528.27,-585.6 528.27,-562.8 435.89,-562.8"/>
<text xml:space="preserve" text-anchor="start" x="438.89" y="-570" font-family="Arial" font-size="14.00" fill="#c9c9c9">weighted sum</text>
</g>
<!-- behavior_loss&#45;&gt;optimizer -->
<g id="edge5" class="edge">
<title>behavior_loss&#45;&gt;optimizer</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M638,-645.67C638,-604.47 638,-555.36 638,-512.97"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="640.63,-513.16 638,-505.66 635.38,-513.16 640.63,-513.16"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="638,-562.8 638,-585.6 730.38,-585.6 730.38,-562.8 638,-562.8"/>
<text xml:space="preserve" text-anchor="start" x="641" y="-570" font-family="Arial" font-size="14.00" fill="#c9c9c9">weighted sum</text>
</g>
<!-- rep_loss&#45;&gt;optimizer -->
<g id="edge6" class="edge">
<title>rep_loss&#45;&gt;optimizer</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M948.79,-645.67C891.7,-603.07 823.28,-552.03 765.19,-508.69"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="767.07,-506.82 759.49,-504.44 763.93,-511.03 767.07,-506.82"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="865.89,-562.8 865.89,-585.6 958.27,-585.6 958.27,-562.8 865.89,-562.8"/>
<text xml:space="preserve" text-anchor="start" x="868.89" y="-570" font-family="Arial" font-size="14.00" fill="#c9c9c9">weighted sum</text>
</g>
<!-- optimizer&#45;&gt;agent -->
<g id="edge7" class="edge">
<title>optimizer&#45;&gt;agent</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M638,-322.87C638,-281.67 638,-232.56 638,-190.17"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="640.63,-190.36 638,-182.86 635.38,-190.36 640.63,-190.36"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="638,-240 638,-262.8 859.56,-262.8 859.56,-240 638,-240"/>
<text xml:space="preserve" text-anchor="start" x="641" y="-247.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">update params and optimizer state</text>
</g>
<!-- rssm_feat&#45;&gt;wm_loss -->
<g id="edge1" class="edge">
<title>rssm_feat&#45;&gt;wm_loss</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M522.2,-977.73C463.28,-932.51 391.63,-877.53 331.64,-831.49"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="333.58,-829.67 326.03,-827.18 330.38,-833.83 333.58,-829.67"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="435.89,-894.8 435.89,-917.6 557.03,-917.6 557.03,-894.8 435.89,-894.8"/>
<text xml:space="preserve" text-anchor="start" x="438.89" y="-902" font-family="Arial" font-size="14.00" fill="#c9c9c9">world&#45;model terms</text>
</g>
<!-- rssm_feat&#45;&gt;behavior_loss -->
<g id="edge2" class="edge">
<title>rssm_feat&#45;&gt;behavior_loss</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M638,-977.73C638,-933.9 638,-880.88 638,-835.74"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="640.63,-835.87 638,-828.37 635.38,-835.87 640.63,-835.87"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="638,-894.8 638,-917.6 753.72,-917.6 753.72,-894.8 638,-894.8"/>
<text xml:space="preserve" text-anchor="start" x="641" y="-902" font-family="Arial" font-size="14.00" fill="#c9c9c9">imagination starts</text>
</g>
<!-- rssm_feat&#45;&gt;rep_loss -->
<g id="edge3" class="edge">
<title>rssm_feat&#45;&gt;rep_loss</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M753.8,-977.73C812.72,-932.51 884.37,-877.53 944.36,-831.49"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="945.62,-833.83 949.97,-827.18 942.42,-829.67 945.62,-833.83"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="853,-894.8 853,-917.6 987.39,-917.6 987.39,-894.8 853,-894.8"/>
<text xml:space="preserve" text-anchor="start" x="856" y="-902" font-family="Arial" font-size="14.00" fill="#c9c9c9">representation terms</text>
</g>
</g>
</svg>
`;case"view_1byfr7e":return`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
 "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<!-- Generated by graphviz version 14.0.4 (0)
 -->
<!-- Pages: 1 -->
<svg width="2130pt" height="1618pt"
 viewBox="0.00 0.00 2130.00 1618.00" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<g id="graph0" class="graph" transform="scale(1 1) rotate(0) translate(15.05 1603.45)">
<g id="clust1" class="cluster">
<title>cluster_vggt_boundary</title>
<polygon fill="#194b9e" stroke="#1b3d88" points="8,-8 8,-1580.4 1740,-1580.4 1740,-8 8,-8"/>
<text xml:space="preserve" text-anchor="start" x="16" y="-1567.5" font-family="Arial" font-weight="bold" font-size="11.00" fill="#bfdbfe" fill-opacity="0.701961">VGGT PRODUCTION ENCODER</text>
</g>
<!-- extractor -->
<g id="node1" class="node">
<title>extractor</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="1272.39,-1519.2 899.61,-1519.2 899.61,-1339.2 1272.39,-1339.2 1272.39,-1519.2"/>
<text xml:space="preserve" text-anchor="start" x="964.85" y="-1471" font-family="Arial" font-size="20.00" fill="#eff6ff">JAXVGGTFeatureExtractor</text>
<text xml:space="preserve" text-anchor="start" x="993.51" y="-1449.3" font-family="Arial" font-size="13.00" fill="#bfdbfe">src/vggt/jax/feature_extractor.py</text>
<text xml:space="preserve" text-anchor="start" x="954.7" y="-1427.9" font-family="Arial" font-size="15.00" fill="#bfdbfe">Drop&#45;in JAX backend for StreamVGGT.</text>
<text xml:space="preserve" text-anchor="start" x="919.67" y="-1409.9" font-family="Arial" font-size="15.00" fill="#bfdbfe">Loads HuggingFace StreamVGGT weights, keeps</text>
<text xml:space="preserve" text-anchor="start" x="935.93" y="-1391.9" font-family="Arial" font-size="15.00" fill="#bfdbfe">streaming caches as instance state, resets at</text>
<text xml:space="preserve" text-anchor="start" x="931.32" y="-1373.9" font-family="Arial" font-size="15.00" fill="#bfdbfe">episode boundaries, and exposes extract(rgb).</text>
</g>
<!-- aggregator -->
<g id="node2" class="node">
<title>aggregator</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="1264.67,-1196.4 907.33,-1196.4 907.33,-1016.4 1264.67,-1016.4 1264.67,-1196.4"/>
<text xml:space="preserve" text-anchor="start" x="1036.52" y="-1139.2" font-family="Arial" font-size="20.00" fill="#eff6ff">Aggregator</text>
<text xml:space="preserve" text-anchor="start" x="999.64" y="-1117.5" font-family="Arial" font-size="13.00" fill="#bfdbfe">24 alternating attention blocks</text>
<text xml:space="preserve" text-anchor="start" x="927.39" y="-1096.1" font-family="Arial" font-size="15.00" fill="#bfdbfe">Consumes fixed 518x518 RGB, emits camera +</text>
<text xml:space="preserve" text-anchor="start" x="962.39" y="-1078.1" font-family="Arial" font-size="15.00" fill="#bfdbfe">register + patch tokens, and supports</text>
<text xml:space="preserve" text-anchor="start" x="1008.46" y="-1060.1" font-family="Arial" font-size="15.00" fill="#bfdbfe">streaming cache paths.</text>
</g>
<!-- agg_cache -->
<g id="node3" class="node">
<title>agg_cache</title>
<path fill="#3b82f6" stroke="#2563eb" stroke-width="2" d="M386.31,-857.24C386.31,-866.27 310.42,-873.6 217,-873.6 123.58,-873.6 47.69,-866.27 47.69,-857.24 47.69,-857.24 47.69,-709.96 47.69,-709.96 47.69,-700.93 123.58,-693.6 217,-693.6 310.42,-693.6 386.31,-700.93 386.31,-709.96 386.31,-709.96 386.31,-857.24 386.31,-857.24"/>
<path fill="none" stroke="#2563eb" stroke-width="2" d="M386.31,-857.24C386.31,-848.21 310.42,-840.87 217,-840.87 123.58,-840.87 47.69,-848.21 47.69,-857.24"/>
<text xml:space="preserve" text-anchor="start" x="85.79" y="-816.4" font-family="Arial" font-size="20.00" fill="#eff6ff">Aggregator padded KV cache</text>
<text xml:space="preserve" text-anchor="start" x="115.83" y="-794.7" font-family="Arial" font-size="13.00" fill="#bfdbfe">per&#45;block (k_pad, v_pad, valid_len)</text>
<text xml:space="preserve" text-anchor="start" x="67.74" y="-773.3" font-family="Arial" font-size="15.00" fill="#bfdbfe">Fixed&#45;shape padded cache keeps JIT stable.</text>
<text xml:space="preserve" text-anchor="start" x="79.85" y="-755.3" font-family="Arial" font-size="15.00" fill="#bfdbfe">Per&#45;block budgets are Python static args;</text>
<text xml:space="preserve" text-anchor="start" x="91.5" y="-737.3" font-family="Arial" font-size="15.00" fill="#bfdbfe">eviction uses budgeted cache control.</text>
</g>
<!-- camera_head -->
<g id="node4" class="node">
<title>camera_head</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="816.02,-873.6 495.98,-873.6 495.98,-693.6 816.02,-693.6 816.02,-873.6"/>
<text xml:space="preserve" text-anchor="start" x="596.53" y="-787.4" font-family="Arial" font-size="20.00" fill="#eff6ff">CameraHead</text>
<text xml:space="preserve" text-anchor="start" x="622.03" y="-765.7" font-family="Arial" font-size="13.00" fill="#bfdbfe">pose output</text>
</g>
<!-- point_head -->
<g id="node5" class="node">
<title>point_head</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="1246.02,-873.6 925.98,-873.6 925.98,-693.6 1246.02,-693.6 1246.02,-873.6"/>
<text xml:space="preserve" text-anchor="start" x="1042.09" y="-787.4" font-family="Arial" font-size="20.00" fill="#eff6ff">DPTHead</text>
<text xml:space="preserve" text-anchor="start" x="1008.3" y="-765.7" font-family="Arial" font-size="13.00" fill="#bfdbfe">dense 518 x 518 x 3 points</text>
</g>
<!-- aggregator_tokens -->
<g id="node6" class="node">
<title>aggregator_tokens</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="1700.44,-873.6 1355.56,-873.6 1355.56,-693.6 1700.44,-693.6 1700.44,-873.6"/>
<text xml:space="preserve" text-anchor="start" x="1439.61" y="-825.4" font-family="Arial" font-size="20.00" fill="#eff6ff">Aggregator features</text>
<text xml:space="preserve" text-anchor="start" x="1451.39" y="-803.7" font-family="Arial" font-size="13.00" fill="#bfdbfe">1374 x 1024 global stream</text>
<text xml:space="preserve" text-anchor="start" x="1391.89" y="-782.3" font-family="Arial" font-size="15.00" fill="#bfdbfe">Final global&#45;stream tokens: 1 camera + 4</text>
<text xml:space="preserve" text-anchor="start" x="1375.62" y="-764.3" font-family="Arial" font-size="15.00" fill="#bfdbfe">registers + 37x37 patches, 1024 dims. Pooled</text>
<text xml:space="preserve" text-anchor="start" x="1387.1" y="-746.3" font-family="Arial" font-size="15.00" fill="#bfdbfe">variants drop registers when flattening raw</text>
<text xml:space="preserve" text-anchor="start" x="1464.62" y="-728.3" font-family="Arial" font-size="15.00" fill="#bfdbfe">or pooling patches.</text>
</g>
<!-- camera_cache -->
<g id="node7" class="node">
<title>camera_cache</title>
<path fill="#3b82f6" stroke="#2563eb" stroke-width="2" d="M389.55,-534.44C389.55,-543.47 313.11,-550.8 219,-550.8 124.89,-550.8 48.45,-543.47 48.45,-534.44 48.45,-534.44 48.45,-387.16 48.45,-387.16 48.45,-378.13 124.89,-370.8 219,-370.8 313.11,-370.8 389.55,-378.13 389.55,-387.16 389.55,-387.16 389.55,-534.44 389.55,-534.44"/>
<path fill="none" stroke="#2563eb" stroke-width="2" d="M389.55,-534.44C389.55,-525.41 313.11,-518.07 219,-518.07 124.89,-518.07 48.45,-525.41 48.45,-534.44"/>
<text xml:space="preserve" text-anchor="start" x="76.13" y="-484.6" font-family="Arial" font-size="20.00" fill="#eff6ff">Camera&#45;head padded KV cache</text>
<text xml:space="preserve" text-anchor="start" x="139.52" y="-462.9" font-family="Arial" font-size="13.00" fill="#bfdbfe">max_camera_frames guard</text>
<text xml:space="preserve" text-anchor="start" x="73.92" y="-441.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">Camera head cache fails loudly on overflow</text>
<text xml:space="preserve" text-anchor="start" x="68.5" y="-423.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">instead of silently clamping dynamic updates.</text>
</g>
<!-- camera_pose -->
<g id="node8" class="node">
<title>camera_pose</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="820.02,-550.8 499.98,-550.8 499.98,-370.8 820.02,-370.8 820.02,-550.8"/>
<text xml:space="preserve" text-anchor="start" x="599.41" y="-464.6" font-family="Arial" font-size="20.00" fill="#eff6ff">camera_pose</text>
<text xml:space="preserve" text-anchor="start" x="614.47" y="-442.9" font-family="Arial" font-size="13.00" fill="#bfdbfe">9 float32 values</text>
</g>
<!-- dense_world_points -->
<g id="node9" class="node">
<title>dense_world_points</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="1250.02,-550.8 929.98,-550.8 929.98,-370.8 1250.02,-370.8 1250.02,-550.8"/>
<text xml:space="preserve" text-anchor="start" x="1001.05" y="-464.6" font-family="Arial" font-size="20.00" fill="#eff6ff">dense_world_points</text>
<text xml:space="preserve" text-anchor="start" x="1029.65" y="-442.9" font-family="Arial" font-size="13.00" fill="#bfdbfe">518 x 518 x 3 float32</text>
</g>
<!-- world_points -->
<g id="node10" class="node">
<title>world_points</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="1262.87,-228 917.13,-228 917.13,-48 1262.87,-48 1262.87,-228"/>
<text xml:space="preserve" text-anchor="start" x="1033.86" y="-161.8" font-family="Arial" font-size="20.00" fill="#eff6ff">world_points</text>
<text xml:space="preserve" text-anchor="start" x="1063.99" y="-140.1" font-family="Arial" font-size="13.00" fill="#bfdbfe">K x K x 3</text>
<text xml:space="preserve" text-anchor="start" x="937.19" y="-118.7" font-family="Arial" font-size="15.00" fill="#bfdbfe">Dense point map pooled to K=37 by default or</text>
<text xml:space="preserve" text-anchor="start" x="1004.73" y="-100.7" font-family="Arial" font-size="15.00" fill="#bfdbfe">K=64 for vggt_wp_cp_64.</text>
</g>
<!-- encoder_boundary -->
<g id="node11" class="node">
<title>encoder_boundary</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="2100.02,-1196.4 1779.98,-1196.4 1779.98,-1016.4 2100.02,-1016.4 2100.02,-1196.4"/>
<text xml:space="preserve" text-anchor="start" x="1822.14" y="-1110.2" font-family="Arial" font-size="20.00" fill="#eff6ff">Encoder and adapter layer</text>
<text xml:space="preserve" text-anchor="start" x="1831.25" y="-1088.5" font-family="Arial" font-size="13.00" fill="#bfdbfe">src/r2dreamer/encoders and adapters</text>
</g>
<!-- extractor&#45;&gt;aggregator -->
<g id="edge1" class="edge">
<title>extractor&#45;&gt;aggregator</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1086,-1339.27C1086,-1298.07 1086,-1248.96 1086,-1206.57"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1088.63,-1206.76 1086,-1199.26 1083.38,-1206.76 1088.63,-1206.76"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1086,-1256.4 1086,-1279.2 1179.16,-1279.2 1179.16,-1256.4 1086,-1256.4"/>
<text xml:space="preserve" text-anchor="start" x="1089" y="-1263.6" font-family="Arial" font-size="14.00" fill="#c9c9c9">run one frame</text>
</g>
<!-- extractor&#45;&gt;encoder_boundary -->
<g id="edge2" class="edge">
<title>extractor&#45;&gt;encoder_boundary</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1272.29,-1372.93C1410.74,-1329.91 1603.19,-1265.97 1767,-1196.4 1768.22,-1195.88 1769.45,-1195.36 1770.68,-1194.83"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1771.57,-1197.3 1777.39,-1191.89 1769.47,-1192.49 1771.57,-1197.3"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1601.9,-1256.4 1601.9,-1279.2 1628.89,-1279.2 1628.89,-1256.4 1601.9,-1256.4"/>
<text xml:space="preserve" text-anchor="start" x="1604.9" y="-1264.6" font-family="Arial" font-weight="bold" font-size="14.00" fill="#c9c9c9">[...]</text>
</g>
<!-- aggregator&#45;&gt;agg_cache -->
<g id="edge3" class="edge">
<title>aggregator&#45;&gt;agg_cache</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M907.46,-1043.03C777.66,-997.25 597.93,-932.95 441,-873.6 426.63,-868.17 411.78,-862.46 396.88,-856.67"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="397.84,-854.23 389.9,-853.96 395.94,-859.12 397.84,-854.23"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="653.11,-933.6 653.11,-956.4 761.09,-956.4 761.09,-933.6 653.11,-933.6"/>
<text xml:space="preserve" text-anchor="start" x="656.11" y="-940.8" font-family="Arial" font-size="14.00" fill="#c9c9c9">read and update</text>
</g>
<!-- aggregator&#45;&gt;camera_head -->
<g id="edge4" class="edge">
<title>aggregator&#45;&gt;camera_head</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M966.79,-1016.47C909.7,-973.87 841.28,-922.83 783.19,-879.49"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="785.07,-877.62 777.49,-875.24 781.93,-881.83 785.07,-877.62"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="883.89,-933.6 883.89,-956.4 1037.37,-956.4 1037.37,-933.6 883.89,-933.6"/>
<text xml:space="preserve" text-anchor="start" x="886.89" y="-940.8" font-family="Arial" font-size="14.00" fill="#c9c9c9">if compute_heads=True</text>
</g>
<!-- aggregator&#45;&gt;point_head -->
<g id="edge5" class="edge">
<title>aggregator&#45;&gt;point_head</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1086,-1016.47C1086,-975.27 1086,-926.16 1086,-883.77"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1088.63,-883.96 1086,-876.46 1083.38,-883.96 1088.63,-883.96"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1086,-933.6 1086,-956.4 1239.48,-956.4 1239.48,-933.6 1086,-933.6"/>
<text xml:space="preserve" text-anchor="start" x="1089" y="-940.8" font-family="Arial" font-size="14.00" fill="#c9c9c9">if compute_heads=True</text>
</g>
<!-- aggregator&#45;&gt;aggregator_tokens -->
<g id="edge6" class="edge">
<title>aggregator&#45;&gt;aggregator_tokens</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1208.53,-1016.47C1267.22,-973.87 1337.55,-922.83 1397.26,-879.49"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1398.63,-881.74 1403.16,-875.21 1395.55,-877.49 1398.63,-881.74"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1320.25,-933.6 1320.25,-956.4 1488.9,-956.4 1488.9,-933.6 1320.25,-933.6"/>
<text xml:space="preserve" text-anchor="start" x="1323.25" y="-940.8" font-family="Arial" font-size="14.00" fill="#c9c9c9">expose final global stream</text>
</g>
<!-- camera_head&#45;&gt;camera_cache -->
<g id="edge7" class="edge">
<title>camera_head&#45;&gt;camera_cache</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M534.85,-693.67C476.23,-650.63 405.85,-598.97 346.43,-555.34"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="348.15,-553.36 340.56,-551.03 345.05,-557.59 348.15,-553.36"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="450.6,-610.8 450.6,-633.6 558.58,-633.6 558.58,-610.8 450.6,-610.8"/>
<text xml:space="preserve" text-anchor="start" x="453.6" y="-618" font-family="Arial" font-size="14.00" fill="#c9c9c9">read and update</text>
</g>
<!-- camera_head&#45;&gt;camera_pose -->
<g id="edge8" class="edge">
<title>camera_head&#45;&gt;camera_pose</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M657.11,-693.67C657.62,-652.47 658.23,-603.36 658.76,-560.97"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="661.39,-561.19 658.85,-553.66 656.14,-561.12 661.39,-561.19"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="658.12,-610.8 658.12,-633.6 694.48,-633.6 694.48,-610.8 658.12,-610.8"/>
<text xml:space="preserve" text-anchor="start" x="661.12" y="-618" font-family="Arial" font-size="14.00" fill="#c9c9c9">pose</text>
</g>
<!-- point_head&#45;&gt;dense_world_points -->
<g id="edge9" class="edge">
<title>point_head&#45;&gt;dense_world_points</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1087.11,-693.67C1087.62,-652.47 1088.23,-603.36 1088.76,-560.97"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1091.39,-561.19 1088.85,-553.66 1086.14,-561.12 1091.39,-561.19"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1088.12,-610.8 1088.12,-633.6 1173.51,-633.6 1173.51,-610.8 1088.12,-610.8"/>
<text xml:space="preserve" text-anchor="start" x="1091.12" y="-618" font-family="Arial" font-size="14.00" fill="#c9c9c9">dense points</text>
</g>
<!-- dense_world_points&#45;&gt;world_points -->
<g id="edge10" class="edge">
<title>dense_world_points&#45;&gt;world_points</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M1090,-370.87C1090,-329.67 1090,-280.56 1090,-238.17"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1092.63,-238.36 1090,-230.86 1087.38,-238.36 1092.63,-238.36"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="1090,-288 1090,-310.8 1175.38,-310.8 1175.38,-288 1090,-288"/>
<text xml:space="preserve" text-anchor="start" x="1093" y="-295.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">pool to K x K</text>
</g>
</g>
</svg>
`;case"view_118k4sm":return`<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
 "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<!-- Generated by graphviz version 14.0.4 (0)
 -->
<!-- Pages: 1 -->
<svg width="1380pt" height="865pt"
 viewBox="0.00 0.00 1380.00 865.00" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<g id="graph0" class="graph" transform="scale(1 1) rotate(0) translate(15.05 849.85)">
<g id="clust1" class="cluster">
<title>cluster_evaluation</title>
<polygon fill="#194b9e" stroke="#1b3d88" points="8,-282.8 8,-564 1342,-564 1342,-282.8 8,-282.8"/>
<text xml:space="preserve" text-anchor="start" x="16" y="-551.1" font-family="Arial" font-weight="bold" font-size="11.00" fill="#bfdbfe" fill-opacity="0.701961">EVALUATION AND PARITY WORKFLOWS</text>
</g>
<!-- parity -->
<g id="node1" class="node">
<title>parity</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="377.71,-502.8 48.29,-502.8 48.29,-322.8 377.71,-322.8 377.71,-502.8"/>
<text xml:space="preserve" text-anchor="start" x="141.87" y="-436.6" font-family="Arial" font-size="20.00" fill="#eff6ff">parity workflows</text>
<text xml:space="preserve" text-anchor="start" x="127.74" y="-414.9" font-family="Arial" font-size="13.00" fill="#bfdbfe">train_parity.py, benchmark.py</text>
<text xml:space="preserve" text-anchor="start" x="68.35" y="-393.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">JAX/PyTorch parity training and benchmark</text>
<text xml:space="preserve" text-anchor="start" x="77.1" y="-375.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">commands for debugging numerical drift.</text>
</g>
<!-- evaluate -->
<g id="node2" class="node">
<title>evaluate</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="839.99,-502.8 488.01,-502.8 488.01,-322.8 839.99,-322.8 839.99,-502.8"/>
<text xml:space="preserve" text-anchor="start" x="619.53" y="-445.6" font-family="Arial" font-size="20.00" fill="#eff6ff">evaluate()</text>
<text xml:space="preserve" text-anchor="start" x="601.49" y="-423.9" font-family="Arial" font-size="13.00" fill="#bfdbfe">checkpoint evaluation</text>
<text xml:space="preserve" text-anchor="start" x="527.26" y="-402.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">Loads a policy checkpoint, constructs the</text>
<text xml:space="preserve" text-anchor="start" x="508.06" y="-384.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">matching env and encoder, runs episodes, and</text>
<text xml:space="preserve" text-anchor="start" x="621.91" y="-366.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">logs metrics.</text>
</g>
<!-- checkpoints -->
<g id="node3" class="node">
<title>checkpoints</title>
<path fill="#3b82f6" stroke="#2563eb" stroke-width="2" d="M1302.02,-486.44C1302.02,-495.47 1230.3,-502.8 1142,-502.8 1053.7,-502.8 981.98,-495.47 981.98,-486.44 981.98,-486.44 981.98,-339.16 981.98,-339.16 981.98,-330.13 1053.7,-322.8 1142,-322.8 1230.3,-322.8 1302.02,-330.13 1302.02,-339.16 1302.02,-339.16 1302.02,-486.44 1302.02,-486.44"/>
<path fill="none" stroke="#2563eb" stroke-width="2" d="M1302.02,-486.44C1302.02,-477.41 1230.3,-470.07 1142,-470.07 1053.7,-470.07 981.98,-477.41 981.98,-486.44"/>
<text xml:space="preserve" text-anchor="start" x="1059.74" y="-436.6" font-family="Arial" font-size="20.00" fill="#eff6ff">Policy checkpoints</text>
<text xml:space="preserve" text-anchor="start" x="1095.03" y="-414.9" font-family="Arial" font-size="13.00" fill="#bfdbfe">pickle step_*.pkl</text>
<text xml:space="preserve" text-anchor="start" x="1051.95" y="-393.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">Contain params, opt_state,</text>
<text xml:space="preserve" text-anchor="start" x="1004.02" y="-375.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">slow_critic_params, ema_state, and step.</text>
</g>
<!-- researcher -->
<g id="node4" class="node">
<title>researcher</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="833.29,-834.8 494.71,-834.8 494.71,-654.8 833.29,-654.8 833.29,-834.8"/>
<text xml:space="preserve" text-anchor="start" x="566.73" y="-795.6" font-family="Arial" font-size="20.00" fill="#eff6ff">Researcher / operator</text>
<text xml:space="preserve" text-anchor="start" x="609.1" y="-773.9" font-family="Arial" font-size="13.00" fill="#bfdbfe">CLI, SLURM, W&amp;B</text>
<text xml:space="preserve" text-anchor="start" x="532.26" y="-752.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">Starts training, evaluation, profiling, and</text>
<text xml:space="preserve" text-anchor="start" x="618.15" y="-734.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">analysis runs.</text>
<text xml:space="preserve" text-anchor="start" x="539.34" y="-716.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">The current code is organized around</text>
<text xml:space="preserve" text-anchor="start" x="550.61" y="-698.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">script&#45;level run selection, launcher</text>
<text xml:space="preserve" text-anchor="start" x="514.77" y="-680.5" font-family="Arial" font-size="15.00" fill="#bfdbfe">registries, and a JAX/Flax R2Dreamer agent.</text>
</g>
<!-- agent_boundary -->
<g id="node5" class="node">
<title>agent_boundary</title>
<polygon fill="#3b82f6" stroke="#2563eb" stroke-width="0" points="837.02,-180 516.98,-180 516.98,0 837.02,0 837.02,-180"/>
<text xml:space="preserve" text-anchor="start" x="597.52" y="-93.8" font-family="Arial" font-size="20.00" fill="#eff6ff">R2Dreamer agent</text>
<text xml:space="preserve" text-anchor="start" x="609.8" y="-72.1" font-family="Arial" font-size="13.00" fill="#bfdbfe">src/r2dreamer/agent.py</text>
</g>
<!-- parity&#45;&gt;agent_boundary -->
<g id="edge2" class="edge">
<title>parity&#45;&gt;agent_boundary</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M341.63,-322.87C403.36,-280.19 477.37,-229.02 540.13,-185.63"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="541.35,-187.98 546.03,-181.55 538.37,-183.66 541.35,-187.98"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="458.91,-240 458.91,-262.8 576.97,-262.8 576.97,-240 458.91,-240"/>
<text xml:space="preserve" text-anchor="start" x="461.91" y="-247.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">compare behavior</text>
</g>
<!-- evaluate&#45;&gt;checkpoints -->
<g id="edge4" class="edge">
<title>evaluate&#45;&gt;checkpoints</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M839.78,-387.85C868.95,-385.93 898.92,-385.31 927.22,-387 941.46,-387.85 956.2,-389.02 970.99,-390.39"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="970.49,-392.98 978.21,-391.08 970.99,-387.75 970.49,-392.98"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="894.75,-387 894.75,-409.8 927.22,-409.8 927.22,-387 894.75,-387"/>
<text xml:space="preserve" text-anchor="start" x="897.75" y="-394.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">load</text>
</g>
<!-- checkpoints&#45;&gt;evaluate -->
<g id="edge5" class="edge">
<title>checkpoints&#45;&gt;evaluate</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M981.06,-412.8C939.07,-412.8 893.47,-412.8 850.29,-412.8"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="850.48,-410.18 842.98,-412.8 850.48,-415.43 850.48,-410.18"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="886.2,-415.8 886.2,-438.6 935.77,-438.6 935.77,-415.8 886.2,-415.8"/>
<text xml:space="preserve" text-anchor="start" x="889.2" y="-423" font-family="Arial" font-size="14.00" fill="#c9c9c9">restore</text>
</g>
<!-- researcher&#45;&gt;evaluate -->
<g id="edge1" class="edge">
<title>researcher&#45;&gt;evaluate</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M664,-654.93C664,-611.1 664,-558.08 664,-512.94"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="666.63,-513.07 664,-505.57 661.38,-513.07 666.63,-513.07"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="664,-572 664,-594.8 764.95,-594.8 764.95,-572 664,-572"/>
<text xml:space="preserve" text-anchor="start" x="667" y="-579.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">runs evaluation</text>
</g>
<!-- agent_boundary&#45;&gt;checkpoints -->
<g id="edge3" class="edge">
<title>agent_boundary&#45;&gt;checkpoints</title>
<path fill="none" stroke="#8d8d8d" stroke-width="2" stroke-dasharray="5,2" d="M805.85,-179.89C868.96,-223.43 944.87,-275.8 1008.58,-319.75"/>
<polygon fill="#8d8d8d" stroke="#8d8d8d" stroke-width="2" points="1006.93,-321.8 1014.59,-323.9 1009.91,-317.48 1006.93,-321.8"/>
<polygon fill="#18191b" fill-opacity="0.627451" stroke="none" points="923.44,-240 923.44,-262.8 959.01,-262.8 959.01,-240 923.44,-240"/>
<text xml:space="preserve" text-anchor="start" x="926.44" y="-247.2" font-family="Arial" font-size="14.00" fill="#c9c9c9">save</text>
</g>
</g>
</svg>
`;default:throw new Error("Unknown viewId: "+e)}}export{t as dotSource,n as svgSource};
