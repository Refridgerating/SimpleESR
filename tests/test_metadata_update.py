from esr_lab.gui import SpanPeakSelector
from esr_lab.io import ESRLoader


class DummyVar:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


def make_csv(path, freq):
    path.write_text(
        f"Frequency;{freq}\n\nMeas\nBField [mT];MW_Absorption []\n1;2\n"
    )
    return path


def test_metadata_updates_on_trace_change(tmp_path):
    csv1 = make_csv(tmp_path / "a.csv", 1)
    csv2 = make_csv(tmp_path / "b.csv", 2)
    spec1 = ESRLoader.load_csv(csv1)
    spec2 = ESRLoader.load_csv(csv2)

    selector = SpanPeakSelector([spec1, spec2])
    selector._update_metadata_display()
    assert "Frequency: 1" in selector.metadata_text

    selector.trace_var = DummyVar("Trace 2")
    selector._on_trace_change()
    assert "Frequency: 2" in selector.metadata_text
