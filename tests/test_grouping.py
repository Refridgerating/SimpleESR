import numpy as np

from esr_lab.services import parse_replicate_label, summarize_replicate_fits


def test_parse_replicate_label_for_lab_style_filename():
    name = "20250102_162256837_CoCal-60C-IP-R3-2JAN2024"
    group, rep = parse_replicate_label(name)
    assert rep == 3
    assert group == "20250102_162256837_CoCal-60C-IP-2JAN2024"


def test_summarize_replicate_fits_aggregates_h_res():
    labels = [
        "20250102_162256837_CoCal-60C-IP-R1-2JAN2024",
        "20250102_162256837_CoCal-60C-IP-R2-2JAN2024",
        "20250102_162256837_CoCal-60C-IP-R3-2JAN2024",
    ]
    payloads = [
        {
            "fits": [
                {
                    "peak": 1,
                    "kind": "derivative",
                    "h_res": 339.0,
                    "delta": 1.00,
                    "g": 2.0023,
                    "chi2": 1.1,
                    "stderr": (0.2, 0.1, 0.1, 0.1),
                }
            ]
        },
        {
            "fits": [
                {
                    "peak": 1,
                    "kind": "derivative",
                    "h_res": 339.2,
                    "delta": 1.05,
                    "g": 2.0018,
                    "chi2": 0.9,
                    "stderr": (0.2, 0.1, 0.1, 0.1),
                }
            ]
        },
        {
            "fits": [
                {
                    "peak": 1,
                    "kind": "derivative",
                    "h_res": 338.9,
                    "delta": 1.10,
                    "g": 2.0020,
                    "chi2": 1.0,
                    "stderr": (0.2, 0.1, 0.1, 0.1),
                }
            ]
        },
    ]

    rows = summarize_replicate_fits(labels, payloads, min_replicates=2, max_chi2=10.0)
    assert len(rows) == 1
    row = rows[0]
    assert row["n_total"] == 3
    assert row["n_used"] == 3
    assert row["kind"] == "derivative"
    assert np.isclose(row["h_res_mean"], np.mean([339.0, 339.2, 338.9]))
    assert np.isfinite(row["h_res_wmean"])
    assert np.isfinite(row["h_res_err_total"])
    assert "R1" in row["included"]
    assert row["rejected"] == ""


def test_summarize_replicate_fits_rejects_bad_chi2():
    labels = [
        "sample-R1",
        "sample-R2",
        "sample-R3",
    ]
    payloads = [
        {"fits": [{"peak": 1, "kind": "derivative", "h_res": 100.0, "chi2": 1.0, "stderr": (0.1, 0.1, 0.1, 0.1)}]},
        {"fits": [{"peak": 1, "kind": "derivative", "h_res": 100.2, "chi2": 1000.0, "stderr": (0.1, 0.1, 0.1, 0.1)}]},
        {"fits": [{"peak": 1, "kind": "derivative", "h_res": 99.9, "chi2": 1.2, "stderr": (0.1, 0.1, 0.1, 0.1)}]},
    ]

    rows = summarize_replicate_fits(labels, payloads, min_replicates=2, max_chi2=25.0)
    assert len(rows) == 1
    row = rows[0]
    assert row["n_total"] == 3
    assert row["n_used"] == 2
    assert "chi2>25.0" in row["rejected"]
