"""Parser behavior tests for r2dreamer launch flags."""

from src.r2dreamer.launch.parser import _build_parser_eval, _build_parser_train


def test_train_parser_does_not_expose_wandb_notes_file():
    parser = _build_parser_train()
    args = parser.parse_args([])

    assert not hasattr(args, "wandb_notes_file")
    assert "--wandb_notes_file" not in parser.format_help()


def test_eval_parser_accepts_3d27_aggregator_encoder():
    parser = _build_parser_eval()
    args = parser.parse_args(["--encoder", "vggt_aggregator_mlp"])

    assert args.encoder == "vggt_aggregator_mlp"
