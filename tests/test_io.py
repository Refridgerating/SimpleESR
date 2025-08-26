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


def test_load_csv_with_metadata(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text(
        "Frequency;9.5\n"
        "Modulation;0.2\n"
        "ModulationFreq;100\n"
        "Bfrom;0\n"
        "Bto;10\n"
        "MicrowavePower;5\n"
        "SweepTime;60\n"
        "Temperature;300\n"
        "\nMeas\nBField [mT];MW_Absorption []\n1;2\n"
    )

    spectrum = ESRLoader.load_csv(csv_file)
    assert spectrum.field.tolist() == [1]
    assert spectrum.intensity.tolist() == [2]
    assert spectrum.metadata == {
        "Frequency": 9.5,
        "Modulation": 0.2,
        "ModulationFreq": 100.0,
        "Bfrom": 0.0,
        "Bto": 10.0,
        "MicrowavePower": 5.0,
        "SweepTime": 60.0,
        "Temperature": 300.0,
    }
