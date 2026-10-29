"""Create a tolerant, machine-readable comparison of two XAI result folders."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _ordered(df):
    """Return a deterministically ordered copy for row-wise comparison.

    Args:
        df: Result DataFrame to order.

    Returns:
        DataFrame sorted stably by preferred identifier columns, by available
        text columns when identifiers are absent, or only reindexed when no
        suitable sort key exists.
    """
    preferred = [c for c in ("Feature", "CaseIndex", "CaseType", "Rule", "Rank") if c in df]
    fallback = [c for c in df if not pd.api.types.is_numeric_dtype(df[c])]
    keys = preferred or fallback
    return df.sort_values(keys, kind="stable").reset_index(drop=True) if keys else df.reset_index(drop=True)


def compare_csv(reference, candidate, rtol=1e-5, atol=1e-8):
    """Compare two result CSV files using schema and numeric tolerances.

    Args:
        reference: Path to the canonical CSV file.
        candidate: Path to the rerun CSV file.
        rtol: Relative tolerance passed to :func:`numpy.allclose`.
        atol: Absolute tolerance passed to :func:`numpy.allclose`.

    Returns:
        Machine-readable report with status ``EXACT``, ``CLOSE``,
        ``DIFFERENT``, ``SCHEMA_DIFFERENT``, or ``SHAPE_DIFFERENT`` and
        column-level numeric differences where applicable.

    Raises:
        OSError: If either CSV file cannot be read.
    """
    left, right = pd.read_csv(reference), pd.read_csv(candidate)
    if left.columns.tolist() != right.columns.tolist():
        return {"status": "SCHEMA_DIFFERENT", "reference_columns": left.columns.tolist(),
                "candidate_columns": right.columns.tolist()}
    if left.shape != right.shape:
        return {"status": "SHAPE_DIFFERENT", "reference_shape": list(left.shape),
                "candidate_shape": list(right.shape)}
    left, right = _ordered(left), _ordered(right)
    numeric = [c for c in left if pd.api.types.is_numeric_dtype(left[c])
               and pd.api.types.is_numeric_dtype(right[c])]
    text = [c for c in left if c not in numeric]
    text_equal = all(left[c].fillna("<NA>").astype(str).equals(
        right[c].fillna("<NA>").astype(str)) for c in text)
    details = {}
    close = True
    for column in numeric:
        a, b = left[column].to_numpy(float), right[column].to_numpy(float)
        column_close = np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
        close &= column_close
        finite = np.isfinite(a) & np.isfinite(b)
        details[column] = {
            "close": bool(column_close),
            "max_absolute_difference": float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else None,
        }
    exact = left.equals(right)
    return {"status": "EXACT" if exact else "CLOSE" if text_equal and close else "DIFFERENT",
            "text_equal": text_equal, "numeric": details}


def compare_directories(reference, candidate, rtol=1e-5, atol=1e-8):
    """Compare every top-level CSV present in either result directory.

    Args:
        reference: Directory containing canonical result CSV files.
        candidate: Directory containing candidate rerun CSV files.
        rtol: Relative numeric comparison tolerance.
        atol: Absolute numeric comparison tolerance.

    Returns:
        Report containing resolved directories, tolerances, status counts, and
        one comparison result per discovered filename. Files present in only
        one directory receive ``MISSING`` status.
    """
    reference, candidate = Path(reference), Path(candidate)
    names = sorted({p.name for p in reference.glob("*.csv")} | {p.name for p in candidate.glob("*.csv")})
    files = {}
    for name in names:
        left, right = reference / name, candidate / name
        if not left.exists() or not right.exists():
            files[name] = {"status": "MISSING", "reference": left.exists(), "candidate": right.exists()}
        else:
            files[name] = compare_csv(left, right, rtol, atol)
    counts = pd.Series([value["status"] for value in files.values()]).value_counts().to_dict()
    return {"reference": str(reference.resolve()), "candidate": str(candidate.resolve()),
            "rtol": rtol, "atol": atol, "summary": counts, "files": files}


def main(argv=None):
    """Execute the result-directory comparison command.

    Args:
        argv: Optional argument sequence. ``None`` reads :data:`sys.argv`.

    Returns:
        Process status code. Zero indicates no ``DIFFERENT`` result; one
        indicates at least one numerically or textually different CSV.

    Notes:
        Missing, schema-different, and shape-different statuses are retained in
        the JSON report. The current exit rule is based specifically on the
        number of files with ``DIFFERENT`` status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-8)
    args = parser.parse_args(argv)
    report = compare_directories(args.reference, args.candidate, args.rtol, args.atol)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print("Comparison report:", output.resolve())
    return 1 if report["summary"].get("DIFFERENT", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
