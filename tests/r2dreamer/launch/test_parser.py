
"""Parser behavior tests for r2dreamer train launch flags."""

from types import SimpleNamespace

from src.configs.config import LATENT_PRESETS
from src.r2dreamer.launch.parser import _build_parser_eval, _build_parser_train
from src.r2dreamer.launch.train import _agent_overrides_from_args


def test_train_parser_does_not_expose_wandb_notes_file():
    parser = _build_parser_train()
    args = parser.parse_args([])

    assert not hasattr(args, "wandb_notes_file")
    assert "--wandb_notes_file" not in parser.format_help()


def test_mlp_layers_help_matches_conv_encoder_guard():
    help_text = " ".join(_build_parser_train().format_help().split())

    assert "Only valid for VGGT MLP encoders" in help_text
    assert "CNN/dense-WP conv encoders require the default value (1)" in help_text
    assert "Ignored by the CNN/dense-WP conv encoders" not in help_text


def test_train_parser_defaults_to_scalars_only_no_validation_or_video():
    args = _build_parser_train().parse_args([])

    assert args.latent_preset == "12m"
    assert args.val_every == 0
    assert args.video_log_every == 0
    assert args.val_video_episodes == 0
    assert args.video_log_episodes == 0


def test_eval_parser_defaults_to_no_video_logging():
    args = _build_parser_eval().parse_args([])

    assert args.log_video_episodes == 0


def test_buffer_capacity_override_accepts_hyphen_and_underscore_aliases():
    parser = _build_parser_train()

    hyphen = parser.parse_args(["--buffer-capacity", "500000"])
    underscore = parser.parse_args(["--buffer_capacity", "100000"])

    assert hyphen.buffer_capacity == 500_000
    assert underscore.buffer_capacity == 100_000


def test_buffer_capacity_override_wins_over_encoder_default():
    args = _build_parser_train().parse_args(["--buffer-capacity", "500000"])
    encoder_spec = SimpleNamespace(agent_overrides={"buffer_capacity": 1_000_000})

    overrides = _agent_overrides_from_args(args, encoder_spec, latent_presets={})

    assert overrides["buffer_capacity"] == 500_000


def test_compute_dtype_override_reaches_agent_config():
    args = _build_parser_train().parse_args(["--compute_dtype", "bfloat16"])
    encoder_spec = SimpleNamespace(agent_overrides={})

    overrides = _agent_overrides_from_args(args, encoder_spec, latent_presets={})

    assert overrides["compute_dtype"] == "bfloat16"


def test_table_model_size_preset_reaches_agent_config():
    args = _build_parser_train().parse_args(["--latent_preset", "200m"])
    encoder_spec = SimpleNamespace(agent_overrides={})

    overrides = _agent_overrides_from_args(args, encoder_spec, LATENT_PRESETS)

    assert overrides["deter_size"] == 8192
    assert overrides["hidden_size"] == 1024
    assert overrides["stoch_classes"] == 32
    assert overrides["stoch_discrete"] == 64
    assert overrides["encoder_depth"] == 64
    assert overrides["mlp_units"] == 1024


def test_static_house_context_path_accepts_hyphen_and_underscore_aliases():
    parser = _build_parser_train()

    hyphen = parser.parse_args(["--static-house-context-path", "house.ply"])
    underscore = parser.parse_args(["--static_house_context_path", "other.ply"])

    assert hyphen.static_house_context_path == "house.ply"
    assert underscore.static_house_context_path == "other.ply"


def test_static_house_points_path_accepts_hyphen_and_underscore_aliases():
    parser = _build_parser_train()

    hyphen = parser.parse_args(["--static-house-points-path", "house.ply"])
    underscore = parser.parse_args(["--static_house_points_path", "other.ply"])

    assert hyphen.static_house_points_path == "house.ply"
    assert underscore.static_house_points_path == "other.ply"
