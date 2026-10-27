
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
    """Create an output directory and return its resolved path."""
    os.makedirs(path, exist_ok=True)


def dataframe(X, feature_names):
    """Return a feature-aligned DataFrame for export."""
    if isinstance(X, pd.DataFrame):
        return X.copy()

    return pd.DataFrame(
        X,
        columns=feature_names
    )

def save_json(obj, filename):
    """Write JSON data using stable, serialisable representations."""
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

    """Save the trusted fitted model and aligned analysis inputs."""
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
    """Export one supported result object to its canonical format."""
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

    """Export framework results, metadata, and an optional model bundle."""
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
