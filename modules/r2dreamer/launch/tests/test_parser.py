"""Parser behavior tests for r2dreamer train launch flags."""

from modules.r2dreamer.launch.parser import _build_parser_train


def test_train_parser_does_not_expose_wandb_notes_file():
    parser = _build_parser_train()
    args = parser.parse_args([])

    assert not hasattr(args, "wandb_notes_file")
    assert "--wandb_notes_file" not in parser.format_help()
