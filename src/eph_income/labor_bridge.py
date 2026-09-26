"""Cross-fitted labor-state bridge shared by EPH training and Census inference.

The bridge deliberately models only two independent decisions:
1) P(active | shared pre-labor covariates)
2) P(unemployed | active, shared pre-labor covariates)

This guarantees coherent employed/unemployed/inactive probabilities without
synthetic hard-state flips and without using observed labor labels as income
predictors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


LABOR_PROBABILITY_COLUMNS = (
    "labor_p_active",
    "labor_p_employed",
    "labor_p_unemployed",
    "labor_p_inactive",
)
VALID_ESTADO = frozenset({0, 1, 2, 3, 4})
MODELLED_ESTADO = frozenset({1, 2, 3, 4})


class LaborBridgeError(ValueError):
    pass


@dataclass(frozen=True)
class LaborFeatureContract:
    categorical: tuple[str, ...]
    numeric: tuple[str, ...]

    @property
    def all_features(self) -> tuple[str, ...]:
        return self.categorical + self.numeric

    def validate(self, frame: pd.DataFrame, context: str) -> None:
        missing = sorted(set(self.all_features) - set(frame.columns))
        if missing:
            raise LaborBridgeError(f"{context} missing labor-bridge features: {missing}")
        overlap = set(self.categorical) & set(self.numeric)
        if overlap:
            raise LaborBridgeError(f"feature contract overlaps categorical/numeric: {sorted(overlap)}")


def _fit_pipeline(model: Pipeline, x: pd.DataFrame, y: np.ndarray, weights: np.ndarray | None):
    kwargs = {}
    if weights is not None:
        kwargs["model__sample_weight"] = weights
    return model.fit(x, y, **kwargs)


def _positive_probability(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(x)
    classes = list(model.named_steps["model"].classes_)
    if 1 not in classes:
        raise LaborBridgeError(f"binary classifier lacks positive class 1: {classes}")
    return np.asarray(probabilities[:, classes.index(1)], dtype=float)


def build_classifier(
    kind: str,
    contract: LaborFeatureContract,
    *,
    random_state: int = 42,
) -> Pipeline:
    """Build a deterministic classifier for the declared shared feature contract."""
    if kind not in {"logistic", "hist_gradient_boosting"}:
        raise LaborBridgeError(f"unsupported labor classifier kind: {kind}")

    if kind == "logistic":
        categorical = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        numeric = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
        preprocess = ColumnTransformer(
            [
                ("categorical", categorical, list(contract.categorical)),
                ("numeric", numeric, list(contract.numeric)),
            ],
            remainder="drop",
        )
        model = LogisticRegression(max_iter=1500, random_state=random_state)
    else:
        categorical = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                (
                    "encode",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        )
        numeric = Pipeline([("impute", SimpleImputer(strategy="median"))])
        preprocess = ColumnTransformer(
            [
                ("categorical", categorical, list(contract.categorical)),
                ("numeric", numeric, list(contract.numeric)),
            ],
            remainder="drop",
        )
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=31,
            min_samples_leaf=50,
            early_stopping=False,
            random_state=random_state,
        )

    return Pipeline([("preprocess", preprocess), ("model", model)])


def labor_targets(frame: pd.DataFrame, *, state_col: str = "ESTADO") -> pd.DataFrame:
    if state_col not in frame:
        raise LaborBridgeError(f"missing labor state column {state_col}")
    state = pd.to_numeric(frame[state_col], errors="coerce")
    if state.isna().any():
        raise LaborBridgeError(f"{state_col} contains missing/non-numeric values")
    state = state.astype(int)
    unknown = sorted(set(state) - VALID_ESTADO)
    if unknown:
        raise LaborBridgeError(f"unsupported {state_col} values: {unknown}")

    eligible = state.isin(MODELLED_ESTADO)
    out = pd.DataFrame(index=frame.index)
    out["eligible"] = eligible
    out["active"] = state.isin([1, 2]).astype(int)
    out["unemployed"] = (state == 2).astype(int)
    return out


class TwoStageLaborBridge:
    """Cross-fit EPH probabilities and fit final models for Census scoring."""

    def __init__(
        self,
        contract: LaborFeatureContract,
        *,
        classifier_kind: str = "logistic",
        cv: int = 5,
        random_state: int = 42,
    ):
        if cv < 2:
            raise LaborBridgeError("cv must be >= 2")
        self.contract = contract
        self.classifier_kind = classifier_kind
        self.cv = cv
        self.random_state = random_state

    def _splitter(
        self,
        active_target: np.ndarray,
        groups: np.ndarray | None,
    ):
        if groups is not None:
            unique = pd.Series(groups).nunique(dropna=False)
            if unique < self.cv:
                raise LaborBridgeError(
                    f"need at least {self.cv} groups for GroupKFold, got {unique}"
                )
            return GroupKFold(n_splits=self.cv).split(
                np.zeros(len(active_target)), active_target, groups
            )
        splitter = StratifiedKFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )
        return splitter.split(np.zeros(len(active_target)), active_target)

    def fit_oof(
        self,
        frame: pd.DataFrame,
        *,
        state_col: str = "ESTADO",
        weight_col: str | None = None,
        group_columns: Sequence[str] | None = ("CODUSU", "NRO_HOGAR"),
    ) -> pd.DataFrame:
        self.contract.validate(frame, "EPH labor bridge")
        targets = labor_targets(frame, state_col=state_col)
        eligible = targets["eligible"]
        work = frame.loc[eligible].copy()
        active = targets.loc[eligible, "active"].to_numpy(dtype=int)
        unemployed = targets.loc[eligible, "unemployed"].to_numpy(dtype=int)

        if len(np.unique(active)) < 2:
            raise LaborBridgeError("activity target needs both active and inactive rows")
        if len(np.unique(unemployed[active == 1])) < 2:
            raise LaborBridgeError("conditional unemployment target needs employed and unemployed rows")

        weights = None
        if weight_col is not None:
            if weight_col not in work:
                raise LaborBridgeError(f"missing labor fit weight {weight_col}")
            weights = pd.to_numeric(work[weight_col], errors="coerce").to_numpy(dtype=float)
            if (
                not np.isfinite(weights).all()
                or (weights < 0).any()
                or float(weights.sum()) <= 0
            ):
                raise LaborBridgeError("labor fit weights must be finite/non-negative/positive total")

        groups = None
        if group_columns:
            present = list(group_columns)
            missing = sorted(set(present) - set(work.columns))
            if missing:
                raise LaborBridgeError(f"labor CV group columns missing: {missing}")
            groups = (
                work[present]
                .astype("string")
                .fillna("<NA>")
                .agg("|".join, axis=1)
                .to_numpy()
            )

        x = work[list(self.contract.all_features)]
        p_active = np.full(len(work), np.nan, dtype=float)
        p_u_given_active = np.full(len(work), np.nan, dtype=float)

        splits = list(self._splitter(active, groups))
        for train_idx, valid_idx in splits:
            active_model = build_classifier(
                self.classifier_kind,
                self.contract,
                random_state=self.random_state,
            )
            _fit_pipeline(
                active_model,
                x.iloc[train_idx],
                active[train_idx],
                None if weights is None else weights[train_idx],
            )
            p_active[valid_idx] = _positive_probability(
                active_model, x.iloc[valid_idx]
            )

            train_active_idx = train_idx[active[train_idx] == 1]
            if len(train_active_idx) == 0 or len(np.unique(unemployed[train_active_idx])) < 2:
                raise LaborBridgeError(
                    "one CV fold lacks both employed/unemployed active training rows"
                )
            unemployment_model = build_classifier(
                self.classifier_kind,
                self.contract,
                random_state=self.random_state,
            )
            _fit_pipeline(
                unemployment_model,
                x.iloc[train_active_idx],
                unemployed[train_active_idx],
                None if weights is None else weights[train_active_idx],
            )
            p_u_given_active[valid_idx] = _positive_probability(
                unemployment_model, x.iloc[valid_idx]
            )

        if not np.isfinite(p_active).all() or not np.isfinite(p_u_given_active).all():
            raise LaborBridgeError("OOF labor probabilities contain gaps")

        final_active = build_classifier(
            self.classifier_kind, self.contract, random_state=self.random_state
        )
        _fit_pipeline(final_active, x, active, weights)
        active_rows = active == 1
        final_unemployment = build_classifier(
            self.classifier_kind, self.contract, random_state=self.random_state
        )
        _fit_pipeline(
            final_unemployment,
            x.loc[active_rows],
            unemployed[active_rows],
            None if weights is None else weights[active_rows],
        )
        self.active_model_ = final_active
        self.unemployment_model_ = final_unemployment
        self.weight_col_ = weight_col
        self.state_col_ = state_col

        result = pd.DataFrame(index=work.index)
        result["labor_p_active"] = p_active
        result["labor_p_unemployed_given_active"] = p_u_given_active
        result["labor_p_unemployed"] = p_active * p_u_given_active
        result["labor_p_employed"] = p_active - result["labor_p_unemployed"]
        result["labor_p_inactive"] = 1.0 - p_active
        result["labor_probability_role"] = "OOF"
        result["observed_active"] = active
        result["observed_unemployed"] = unemployed
        self._validate_probabilities(result)
        return result

    @staticmethod
    def _validate_probabilities(frame: pd.DataFrame) -> None:
        required = {
            "labor_p_active",
            "labor_p_employed",
            "labor_p_unemployed",
            "labor_p_inactive",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise LaborBridgeError(f"probability output missing columns: {missing}")
        values = frame[list(required)].apply(pd.to_numeric, errors="coerce")
        if values.isna().any().any() or ((values < 0) | (values > 1)).any().any():
            raise LaborBridgeError("labor probabilities must be finite and in [0, 1]")
        sums = frame[
            ["labor_p_employed", "labor_p_unemployed", "labor_p_inactive"]
        ].sum(axis=1)
        if not np.allclose(sums, 1.0, atol=1e-12, rtol=0):
            raise LaborBridgeError("employed/unemployed/inactive probabilities must sum to one")

    def predict_probabilities(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "active_model_") or not hasattr(self, "unemployment_model_"):
            raise LaborBridgeError("labor bridge has not been fitted")
        self.contract.validate(frame, "labor bridge inference")
        x = frame[list(self.contract.all_features)]
        p_active = _positive_probability(self.active_model_, x)
        p_u_given_active = _positive_probability(self.unemployment_model_, x)
        out = pd.DataFrame(index=frame.index)
        out["labor_p_active"] = p_active
        out["labor_p_unemployed_given_active"] = p_u_given_active
        out["labor_p_unemployed"] = p_active * p_u_given_active
        out["labor_p_employed"] = p_active - out["labor_p_unemployed"]
        out["labor_p_inactive"] = 1.0 - p_active
        out["labor_probability_role"] = "FULL_FIT_INFERENCE"
        self._validate_probabilities(out)
        return out

    def save(self, path) -> None:
        if not hasattr(self, "active_model_"):
            raise LaborBridgeError("cannot save an unfitted labor bridge")
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise LaborBridgeError("serialized artifact is not a TwoStageLaborBridge")
        return model


def attach_probability_features(
    frame: pd.DataFrame,
    probabilities: pd.DataFrame,
    *,
    key_columns: Sequence[str],
) -> pd.DataFrame:
    """Exact one-to-one join used by the L4 income shadow experiment."""
    missing_frame = sorted(set(key_columns) - set(frame.columns))
    missing_probs = sorted(
        (set(key_columns) | set(LABOR_PROBABILITY_COLUMNS)) - set(probabilities.columns)
    )
    if missing_frame:
        raise LaborBridgeError(f"income frame missing join keys: {missing_frame}")
    if missing_probs:
        raise LaborBridgeError(f"labor probability release missing columns: {missing_probs}")
    if frame.duplicated(list(key_columns)).any():
        raise LaborBridgeError("income frame keys are not unique")
    if probabilities.duplicated(list(key_columns)).any():
        raise LaborBridgeError("labor probability keys are not unique")

    left_keys = set(map(tuple, frame[list(key_columns)].astype(str).to_numpy()))
    right_keys = set(map(tuple, probabilities[list(key_columns)].astype(str).to_numpy()))
    if left_keys != right_keys:
        raise LaborBridgeError(
            f"labor probability coverage mismatch: missing={len(left_keys-right_keys)} "
            f"extra={len(right_keys-left_keys)}"
        )
    joined = frame.merge(
        probabilities[[*key_columns, *LABOR_PROBABILITY_COLUMNS]],
        on=list(key_columns),
        how="left",
        validate="one_to_one",
        sort=False,
    )
    TwoStageLaborBridge._validate_probabilities(joined)
    return joined
