from unittest.mock import patch

from esr_lab import gui


def test_gui_main(monkeypatch, tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("a,b\n1,2\n")

    class DummyTk:
        def withdraw(self):
            pass

    monkeypatch.setattr(gui.tk, "Tk", lambda: DummyTk())
    monkeypatch.setattr(gui.filedialog, "askopenfilename", lambda **kwargs: str(csv_file))

    with patch("esr_lab.gui.ESRPlotter.plot") as plot_mock:
        gui.main()
        plot_mock.assert_called_once()
