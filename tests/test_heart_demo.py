"""Tests for the lightweight Heart framework demonstration."""

import pandas as pd

from examples import heart_demo


def test_validation_demo_uses_cli_without_loading_model(
    tmp_path,
    monkeypatch,
    capsys,
):
    config_path = tmp_path / "heart.yaml"
    config_path.write_text(
        "dataset_name: heart\n",
        encoding="utf-8",
    )

    received_arguments = []

    def fake_cli_main(arguments):
        received_arguments.append(arguments)
        return 0

    monkeypatch.setattr(
        heart_demo,
        "CONFIG_PATH",
        config_path,
    )
    monkeypatch.setattr(
        heart_demo,
        "cli_main",
        fake_cli_main,
    )

    heart_demo.main([])

    assert received_arguments == [
        [
            "--config",
            str(config_path),
            "--validate-only",
        ]
    ]

    output = capsys.readouterr().out
    assert "Validation demo completed successfully." in output
    assert "serialized model was not loaded" in output


def test_result_summary_includes_global_local_and_lime_outputs(
    tmp_path,
    capsys,
):
    pd.DataFrame(
        {
            "Feature": ["thal"],
            "Importance": [0.48],
            "RawValue": [0.48],
            "Method": ["coefficients"],
        }
    ).to_csv(
        tmp_path / "native_importance.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "Feature": ["ca"],
            "Importance": [0.05],
            "Std": [0.01],
        }
    ).to_csv(
        tmp_path / "permutation_importance.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "Feature": ["thal"],
            "MeanAbsSHAP": [0.09],
            "MeanSHAP": [-0.01],
        }
    ).to_csv(
        tmp_path / "shap_importance.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "Feature": ["thal"],
            "NativeRank": [1.0],
            "SHAPRank": [1.0],
            "PermutationRank": [2.0],
        }
    ).to_csv(
        tmp_path / "rank_compare.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "CaseType": ["TP"],
            "CaseIndex": [11],
            "y_true": [1],
            "y_pred": [1],
            "y_prob": [0.96],
            "distance": [0.46],
        }
    ).to_csv(
        tmp_path / "representative_cases.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "CaseIndex": [11, 11, 11, 11],
            "Prediction": [1, 1, 1, 1],
            "Probability": [0.96, 0.96, 0.96, 0.96],
            "Rule": [
                "ca > 0.34",
                "cp > -0.16",
                "thal > -0.89",
                "age > 0.40",
            ],
            "Weight": [0.13, 0.12, 0.09, 0.01],
        }
    ).to_csv(
        tmp_path / "lime_explanations.csv",
        index=False,
    )

    heart_demo.summarize_results(
        tmp_path,
        top_n=5,
    )

    output = capsys.readouterr().out

    assert "Top 5 native features" in output
    assert "Top 5 permutation features" in output
    assert "Top 5 SHAP features" in output
    assert "Global rank comparison" in output
    assert "Representative cases" in output
    assert "Three strongest LIME rules" in output

    assert "ca > 0.34" in output
    assert "cp > -0.16" in output
    assert "thal > -0.89" in output

    # Only the three strongest rules should be displayed.
    assert "age > 0.40" not in output