from esr_lab.io import ESRLoader


def test_load_csv_comma(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("a,b\n1,2\n")

    spectrum = ESRLoader.load_csv(csv_file)
    assert spectrum.field.tolist() == [1]
    assert spectrum.intensity.tolist() == [2]


def test_load_csv_semicolon(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("a;b\n3;4\n")

    spectrum = ESRLoader.load_csv(csv_file)
    assert spectrum.field.tolist() == [3]
    assert spectrum.intensity.tolist() == [4]
