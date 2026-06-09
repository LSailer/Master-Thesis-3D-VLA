"""Parser behavior tests for r2dreamer train launch flags."""

from src.r2dreamer.launch.parser import _build_parser_train


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


def test_train_parser_defaults_disable_in_run_val_and_video():
    parser = _build_parser_train()
    args = parser.parse_args([])

    assert args.val_every == 0
    assert args.video_log_every == 0


def test_train_parser_allows_explicit_val_and_video_opt_in():
    parser = _build_parser_train()
    args = parser.parse_args(["--val_every", "50000", "--video_log_every", "25000"])

    assert args.val_every == 50_000
    assert args.video_log_every == 25_000
