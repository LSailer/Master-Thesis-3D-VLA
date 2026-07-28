
"""Parser behavior tests for the train/eval CLI flags.

``src.main`` is the composition root now, so the flag-to-config translation
under test is ``_config_from_args`` (a flag named like a config field fills it)
plus ``_renamed_agent_flags`` (the handful whose names or values differ).
"""

from src.configs.config import LATENT_PRESETS, R2DreamerConfig
from src.main import _agent_config, _config_from_args, _renamed_agent_flags
from src.launch.parser import build_parser


def _train_args(*argv: str):
    return build_parser().parse_args(list(argv))


def test_parser_does_not_expose_encoder_era_flags():
    parser = build_parser()
    args = parser.parse_args([])
    help_text = parser.format_help()

    # Dispatch by encoder-type string and its sidecar inputs are gone: a variant
    # is selected by adapter name, and each adapter declares its own pipeline.
    for flag in (
        "encoder_type",
        "wandb_notes_file",
        "static_house_context_path",
        "static_house_points_path",
    ):
        assert not hasattr(args, flag)
        assert f"--{flag}" not in help_text


def test_mlp_layers_help_describes_the_routed_mlp_branch():
    help_text = " ".join(build_parser().format_help().split())

    assert "Depth of the composite encoder's MLP branch" in help_text
    assert "Only affects variants that route a field to the MLP branch" in help_text


def test_parser_defaults_to_scalars_only_no_video():
    args = _train_args()

    assert args.mode == "train"
    assert args.seed == 42
    assert args.curriculum == "L1"
    assert args.max_episode_steps == 500
    assert args.latent_preset == "12m"
    assert args.video_log_every == 0
    assert args.video_log_episodes == 0
    assert args.log_video_episodes == 0


def test_val_flags_are_gone():
    # The val-episode loop was removed with the orchestrator refactor; a YAML
    # still rendering val_every must fail at parse time, not be ignored.
    args = _train_args()
    for flag in ("val_every", "val_episodes", "val_video_episodes"):
        assert not hasattr(args, flag)


def test_buffer_capacity_override_accepts_hyphen_and_underscore_aliases():
    parser = build_parser()

    hyphen = parser.parse_args(["--buffer-capacity", "500000"])
    underscore = parser.parse_args(["--buffer_capacity", "100000"])

    assert hyphen.buffer_capacity == 500_000
    assert underscore.buffer_capacity == 100_000


class TestConfigFromArgs:
    """A flag named like a config field fills it; an unset flag does not."""

    def test_matching_flag_fills_the_config_field(self):
        args = _train_args("--buffer_capacity", "500000")

        config = _config_from_args(R2DreamerConfig, args)

        assert config.buffer_capacity == 500_000

    def test_a_flag_wins_over_a_supplied_default(self):
        # The precedence that lets an explicit CLI override beat a size preset.
        args = _train_args("--buffer-capacity", "500000")

        config = _config_from_args(
            R2DreamerConfig, args, defaults={"buffer_capacity": 1_000_000}
        )

        assert config.buffer_capacity == 500_000

    def test_an_unset_flag_leaves_the_supplied_default(self):
        args = _train_args()

        config = _config_from_args(
            R2DreamerConfig, args, defaults={"buffer_capacity": 1_000_000}
        )

        assert config.buffer_capacity == 1_000_000

    def test_overrides_win_over_everything_read_from_args(self):
        args = _train_args("--buffer_capacity", "500000")

        config = _config_from_args(
            R2DreamerConfig, args, buffer_capacity=7, num_actions=4
        )

        assert config.buffer_capacity == 7


class TestRenamedAgentFlags:
    """Flags whose name or value differs from the config field they set."""

    def test_loss_weight_flags_map_onto_scale_fields(self):
        args = _train_args(
            "--actor_loss_weight", "2.0",
            "--value_loss_weight", "3.0",
            "--repval_loss_weight", "4.0",
        )

        renamed = _renamed_agent_flags(args)

        assert renamed["scale_policy"] == 2.0
        assert renamed["scale_value"] == 3.0
        assert renamed["scale_repval"] == 4.0

    def test_unset_flags_are_absent_rather_than_none(self):
        # A None would overwrite the config field's own default.
        assert _renamed_agent_flags(_train_args()) == {}

    def test_barlow_grad_to_encoder_clears_the_stop_gradient(self):
        renamed = _renamed_agent_flags(_train_args("--barlow_grad_to_encoder"))

        assert renamed["barlow_stop_grad"] is False

    def test_compute_dtype_short_aliases_expand(self):
        def dtype(value: str) -> str:
            return _renamed_agent_flags(_train_args("--compute_dtype", value))[
                "compute_dtype"
            ]

        assert dtype("bf16") == "bfloat16"
        assert dtype("fp16") == "float16"
        assert dtype("bfloat16") == "bfloat16"


class TestAgentConfig:
    """The assembled agent config: size preset, then flags, then renamed flags."""

    def _config(self, *argv: str) -> R2DreamerConfig:
        return _agent_config(
            args=_train_args(*argv),
            num_actions=4,
            output_dir="/tmp/r2dreamer-test",
        )

    def test_adapter_name_is_recorded_as_provenance(self):
        config = self._config()

        assert config.adapter == "rgb"
        assert config.num_actions == 4
        assert config.logdir == "/tmp/r2dreamer-test"

    def test_table_model_size_preset_reaches_the_agent_config(self):
        config = self._config("--latent_preset", "200m")

        expected = LATENT_PRESETS["200m"]
        assert config.deter_size == 8192
        assert config.hidden_size == expected["hidden_size"]
        assert config.stoch_classes == expected["stoch_classes"]
        assert config.stoch_discrete == expected["stoch_discrete"]
        assert config.encoder_depth == expected["encoder_depth"]
        assert config.mlp_units == expected["mlp_units"]

    def test_an_explicit_size_flag_wins_over_the_preset(self):
        config = self._config("--latent_preset", "200m", "--deter_size", "512")

        assert config.deter_size == 512
        assert config.hidden_size == LATENT_PRESETS["200m"]["hidden_size"]

    def test_compute_dtype_override_reaches_the_agent_config(self):
        assert self._config("--compute_dtype", "bfloat16").compute_dtype == "bfloat16"

    def test_full_bf16_flag_reaches_the_agent_config(self):
        # The flag was silently dropped once because the config field was
        # missing; assert the plumbing in both directions.
        assert self._config().full_bf16 is False
        assert self._config("--full_bf16").full_bf16 is True
