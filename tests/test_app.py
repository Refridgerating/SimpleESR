import sys
from unittest.mock import patch

from esr_lab import app


def test_main_runs(tmp_path, monkeypatch):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("a,b\n1,2\n")

    monkeypatch.setattr(sys, "argv", ["prog", str(csv_file)])

    with patch("esr_lab.app.ESRPlotter.plot") as plot_mock:
        app.main()
        plot_mock.assert_called_once()
