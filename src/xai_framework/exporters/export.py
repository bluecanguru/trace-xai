
"""Serialise XAI results and trusted reproducibility bundles to disk."""

import os
import json
import joblib
import pandas as pd

try:
    from xai_framework.versioning import runtime_versions
# Support notebooks that import modules directly from src/xai_framework.
except ImportError:
    from versioning import runtime_versions


# Internal helpers

def ensure_directory(path):
    """Create an output directory and any missing parents.

    Args:
        path: Directory path accepted by :func:`os.makedirs`.

    Raises:
        OSError: If the directory cannot be created or conflicts with an
            existing non-directory path.
    """
    os.makedirs(path, exist_ok=True)


def dataframe(X, feature_names):
    """Return a copied or newly constructed feature-aligned DataFrame.

    Args:
        X: Existing DataFrame or two-dimensional array-like matrix.
        feature_names: Ordered column names used when constructing a DataFrame.

    Returns:
        Independent DataFrame suitable for serialisation.

    Raises:
        ValueError: If the number of names does not match the matrix width.
    """
    if isinstance(X, pd.DataFrame):
        return X.copy()

    return pd.DataFrame(
        X,
        columns=feature_names
    )

def save_json(obj, filename):
    """Write an object as indented JSON.

    Args:
        obj: JSON-compatible object. Unsupported leaf values are converted to
            strings by the serializer.
        filename: Destination file path.

    Raises:
        OSError: If the destination cannot be written.
    """
    with open(filename, "w") as f:
        json.dump(
            obj,
            f,
            indent=4,
            default=str
        )


# Model-bundle export

def save_xai_bundle(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    feature_names,
    out_dir,
    metadata=None
):

    """Save the fitted model, aligned inputs, and provenance metadata.

    Args:
        model: Fitted model serialised with Joblib. The resulting pickle must
            be loaded only when its source is trusted.
        X_train: Reference or training feature matrix.
        X_test: Explanation or evaluation feature matrix.
        y_train: Outcomes aligned with ``X_train``.
        y_test: Outcomes aligned with ``X_test``.
        feature_names: Ordered feature or design-term names for both matrices.
        out_dir: Parent result directory in which ``bundle`` is created.
        metadata: Optional experiment metadata merged into the generated model,
            feature, and runtime metadata.

    Raises:
        OSError: If a bundle directory or output file cannot be written.
        ValueError: If a feature matrix cannot be aligned with
            ``feature_names``.
        TypeError: If the fitted model cannot be serialised by Joblib.

    Notes:
        The bundle contains ``model.pkl``, four CSV input files, and
        ``metadata.json``. Loading ``model.pkl`` can execute arbitrary code and
        therefore requires an explicit trust decision.
    """
    bundle_dir = os.path.join(
        out_dir,
        "bundle"
    )

    ensure_directory(bundle_dir)

    X_train = dataframe(
        X_train,
        feature_names
    )

    X_test = dataframe(
        X_test,
        feature_names
    )

    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)

    X_train.to_csv(
        os.path.join(bundle_dir, "X_train.csv"),
        index=False
    )

    X_test.to_csv(
        os.path.join(bundle_dir, "X_test.csv"),
        index=False
    )

    y_train.to_csv(
        os.path.join(bundle_dir, "y_train.csv"),
        index=False
    )

    y_test.to_csv(
        os.path.join(bundle_dir, "y_test.csv"),
        index=False
    )

    joblib.dump(
        model,
        os.path.join(bundle_dir, "model.pkl")
    )

    meta = {
        "model_class":
            model.__class__.__name__,

        "model_module":
            model.__class__.__module__,

        "n_features":
            len(feature_names),

        "feature_names":
            feature_names,

        "runtime":
            runtime_versions()
    }

    if metadata is not None:
        meta.update(metadata)
    save_json(
        meta,
        os.path.join(bundle_dir, "metadata.json")
    )


# Recursive result export

def export_object(obj, out_dir, name):
    """Recursively export one result object in its canonical representation.

    Args:
        obj: Result object to export. DataFrames and Series become CSV files;
            dictionaries become nested directories; lists and scalar values
            become JSON; other objects are attempted with Joblib. ``None`` is
            skipped.
        out_dir: Existing or creatable destination directory.
        name: File stem or nested-directory name.

    Notes:
        Failure to serialise an unsupported fallback object is reported to
        standard output and does not abort export of the remaining results.
    """
    if obj is None:
        return


    if isinstance(obj, pd.DataFrame):
        obj.to_csv(
            os.path.join(
                out_dir,
                f"{name}.csv"
            ),
            index=False
        )
        return


    if isinstance(obj, pd.Series):
        obj.to_csv(
            os.path.join(
                out_dir,
                f"{name}.csv"
            ),
            index=False
        )
        return


    if isinstance(obj, dict):
        folder = os.path.join(
            out_dir,
            name
        )
        ensure_directory(folder)

        for key, value in obj.items():
            export_object(
                value,
                folder,
                key
            )
        return


    if isinstance(obj, list):
        save_json(
            obj,
            os.path.join(
                out_dir,
                f"{name}.json"
            )
        )
        return


    if isinstance(
        obj,
        (
            int,
            float,
            str,
            bool
        )
    ):

        save_json(
            obj,
            os.path.join(
                out_dir,
                f"{name}.json"
            )
        )
        return


    try:
        joblib.dump(
            obj,
            os.path.join(
                out_dir,
                f"{name}.joblib"
            )
        )

    except Exception:
        print(
            f"Could not export '{name}'"
        )


# Public export function

def export_xai_results(
    results,
    out_dir,
    model=None,
    X_train=None,
    X_test=None,
    y_train=None,
    y_test=None,
    feature_names=None,
    metadata=None
):

    """Export framework results and, optionally, a reproducibility bundle.

    Args:
        results: Mapping of canonical result names to serialisable objects.
        out_dir: Destination directory for all exports.
        model: Optional fitted model. When supplied, all aligned inputs and
            ``feature_names`` must also be supplied to create the bundle.
        X_train: Training/reference matrix included in an optional bundle.
        X_test: Test/explanation matrix included in an optional bundle.
        y_train: Training outcomes included in an optional bundle.
        y_test: Test outcomes included in an optional bundle.
        feature_names: Ordered names for bundled matrix columns.
        metadata: Optional experiment metadata included in the bundle.

    Raises:
        OSError: If required directories or files cannot be written.
        ValueError: If bundled matrices cannot be aligned with their names.

    Notes:
        Result objects are exported independently through :func:`export_object`.
        A model bundle is created only when ``model`` is not ``None``.
    """
    ensure_directory(out_dir)


    if model is not None:

        save_xai_bundle(
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=feature_names,
            out_dir=out_dir,
            metadata=metadata

        )


    for name, obj in results.items():

        export_object(
            obj,
            out_dir,
            name
        )

    print(f"Results exported to: {out_dir}")
