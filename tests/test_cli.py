from typer.testing import CliRunner

from preference_lab.cli import app


def test_evaluate_accepts_config_option(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "paths:\n"
        "  train_data: data/sample_preferences.jsonl\n"
        f"  output_dir: {tmp_path.as_posix()}\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["evaluate", "--config", str(config)])

    assert result.exit_code == 0
    assert (tmp_path / "metrics.json").exists()
