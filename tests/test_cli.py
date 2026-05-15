from typer.testing import CliRunner

from gar.cli import app


def test_cli_lists_commands():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "ingest" in result.output
    assert "reset" in result.output
    assert "doctor" in result.output
    assert "chat" in result.output
    assert "serve" in result.output
