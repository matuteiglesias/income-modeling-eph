from pathlib import Path
import json
import yaml
import pandas as pd
import numpy as np


PROJECT_ROOT = Path("..").resolve()
RUNS_DIR = PROJECT_ROOT / "reports" / "runs"


def latest_run(pattern: str) -> Path:
    candidates = sorted(RUNS_DIR.glob(pattern), key=lambda p: p.name)
    if not candidates:
        raise FileNotFoundError(f"No runs found for pattern: {pattern}")
    return candidates[-1]


def maybe_read_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def maybe_read_json(path: Path):
    if path.exists():
        return json.loads(path.read_text())
    return None


def maybe_read_yaml(path: Path):
    if path.exists():
        return yaml.safe_load(path.read_text())
    return None


def read_predictions(run_dir: Path, split: str) -> pd.DataFrame:
    path = run_dir / "predictions" / f"{split}_predictions.parquet"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    df["split"] = split
    df["source_run_dir"] = str(run_dir)

    if "residual" not in df.columns:
        df["residual"] = df["y_true"] - df["y_pred"]
    if "abs_error" not in df.columns:
        df["abs_error"] = df["residual"].abs()
    if "squared_error" not in df.columns:
        df["squared_error"] = df["residual"] ** 2

    return df


def read_run_bundle(run_dir: Path, experiment: str, splits=("test", "validation")) -> dict:
    diagnostics = run_dir / "diagnostics"
    metrics = run_dir / "metrics"

    preds = []
    for split in splits:
        df = read_predictions(run_dir, split)
        if not df.empty:
            df["experiment"] = experiment
            df["run_id_from_path"] = run_dir.name
            preds.append(df)

    predictions = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()

    bundle = {
        "experiment": experiment,
        "run_dir": run_dir,
        "predictions": predictions,
        "model_comparison": maybe_read_csv(metrics / "model_comparison.csv"),
        "residual_summary": maybe_read_csv(diagnostics / "residual_summary.csv"),
        "error_by_decile": maybe_read_csv(diagnostics / "error_by_income_decile.csv"),
        "distribution_summary": maybe_read_csv(diagnostics / "prediction_distribution_summary.csv"),
        "compression_summary": maybe_read_csv(diagnostics / "distribution_compression_summary.csv"),
        "metric_gaps": maybe_read_csv(diagnostics / "metric_gaps.csv"),
        "pairwise_errors": maybe_read_csv(diagnostics / "model_pairwise_error_comparison.csv"),
        "feature_columns": maybe_read_json(run_dir / "feature_columns.json"),
        "dataset_card": maybe_read_json(run_dir / "dataset_card.json"),
        "run_manifest": maybe_read_json(run_dir / "run_manifest.json"),
        "config_used": maybe_read_yaml(run_dir / "config_used.yaml"),
    }

    return bundle


def concat_bundles(bundles: dict, key: str) -> pd.DataFrame:
    frames = []

    for experiment, bundle in bundles.items():
        df = bundle.get(key)
        if df is None or len(df) == 0:
            continue

        df = df.copy()
        if "experiment" not in df.columns:
            df.insert(0, "experiment", experiment)
        if "run_dir" not in df.columns:
            df.insert(1, "run_dir", str(bundle["run_dir"]))

        frames.append(df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def add_true_pred_deciles(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    group_cols = ["experiment", "model", "split"]

    out["true_decile"] = (
        out.groupby(group_cols)["y_true"]
        .transform(lambda s: pd.qcut(s, 10, labels=False, duplicates="drop") + 1)
    )

    out["pred_decile"] = (
        out.groupby(group_cols)["y_pred"]
        .transform(lambda s: pd.qcut(s, 10, labels=False, duplicates="drop") + 1)
    )

    out["decile_error"] = out["pred_decile"] - out["true_decile"]
    out["abs_decile_error"] = out["decile_error"].abs()

    return out


def add_stretched_predictions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    group_cols = ["experiment", "model", "split"]

    def stretch_group(g):
        g = g.copy()
        y_true = g["y_true"]
        y_pred = g["y_pred"]

        scale = y_true.std() / y_pred.std()
        g["y_pred_stretched"] = y_true.mean() + (y_pred - y_pred.mean()) * scale
        g["residual_stretched"] = y_true - g["y_pred_stretched"]
        g["abs_error_stretched"] = g["residual_stretched"].abs()
        g["squared_error_stretched"] = g["residual_stretched"] ** 2
        g["stretch_scale"] = scale
        return g

    return out.groupby(group_cols, group_keys=False).apply(stretch_group)


def r2_score_manual(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sse = ((y_true - y_pred) ** 2).sum()
    sst = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - sse / sst