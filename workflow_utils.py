"""Input checks and training-fold preprocessing shared by the research scripts."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def binary_labels(values, source="labels"):
    values = pd.Series(values)
    mapping = {"negative": 0, "neg": 0, "0": 0, "false": 0, "no": 0,
               "positive": 1, "pos": 1, "1": 1, "true": 1, "yes": 1}
    if pd.api.types.is_numeric_dtype(values):
        result = pd.to_numeric(values, errors="coerce")
    else:
        result = values.astype(str).str.strip().str.lower().map(mapping)
    if result.isna().any() or not set(result.unique()).issubset({0, 1}):
        raise ValueError(f"{source}: expected nonmissing binary 0/1 or negative/positive labels.")
    return result.to_numpy(dtype=int)


def validate_ids(frame, column, source):
    if column not in frame:
        raise ValueError(f"{source}: missing ID column {column!r}.")
    raw = frame[column]
    ids = raw.astype(str).str.replace(".svs", "", regex=False).str.strip()
    if raw.isna().any() or ids.eq("").any() or ids.duplicated().any():
        raise ValueError(f"{source}: IDs must be nonmissing and unique after normalization.")
    return ids


def validate_feature_pair(cp_path, qp_path):
    tables = []
    for path in (cp_path, qp_path):
        frame = pd.read_csv(path)
        if frame.shape[1] < 3:
            raise ValueError(f"{path}: expected ID, label and feature columns.")
        label = "label" if "label" in frame else frame.columns[1]
        ids = validate_ids(frame, frame.columns[0], path)
        tables.append(pd.DataFrame({"ID": ids, "label": binary_labels(frame[label], str(path))}))
    shared = tables[0].merge(tables[1], on="ID", validate="one_to_one")
    if shared.empty or not shared.label_x.eq(shared.label_y).all():
        raise ValueError("CP/QP tables must share IDs with consistent labels.")


def numeric_frame(frame):
    result = frame.apply(pd.to_numeric, errors="raise").astype(float)
    if np.isinf(result.to_numpy()).any():
        raise ValueError("Features contain infinity.")
    return result


def impute_from_train(train, other):
    train, other = numeric_frame(train), numeric_frame(other)
    means = train.mean().fillna(0.0)
    return train.fillna(means), other.fillna(means)


def impute_for_fold(frame, train_idx):
    return impute_from_train(frame.iloc[train_idx], frame)[1]


class FeatureScaler:
    def fit(self, frame):
        frame = numeric_frame(frame)
        self.columns_ = list(frame.columns)
        self.means_ = frame.mean().fillna(0.0)
        self.scaler_ = StandardScaler().fit(frame.fillna(self.means_))
        return self

    def transform(self, frame):
        if list(frame.columns) != self.columns_:
            raise ValueError("Feature schema/order differs from training.")
        return self.scaler_.transform(numeric_frame(frame).fillna(self.means_))

    def fit_transform(self, frame):
        return self.fit(frame).transform(frame)


def fit_nested_modality(X_full_df, y_train, train_idx, feature_names, seed,
                        inner_folds, model_factory, selector):
    raw = X_full_df.iloc[train_idx][feature_names]
    y = binary_labels(y_train)
    folds = min(inner_folds, int(np.bincount(y, minlength=2).min()))
    if folds < 2:
        raise ValueError("Nested stacking requires two training samples per class.")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    p_oof = np.full(len(y), np.nan)
    for itr, iva in cv.split(np.zeros(len(y)), y):
        inner_train, _ = impute_from_train(raw.iloc[itr], raw.iloc[iva])
        features = selector(inner_train, y[itr], feature_names, seed)
        if not features:
            raise ValueError("Feature selection returned no features.")
        scaler = FeatureScaler().fit(raw.iloc[itr][features])
        model = model_factory(seed)
        model.fit(scaler.transform(raw.iloc[itr][features]), y[itr])
        p_oof[iva] = model.predict_proba(scaler.transform(raw.iloc[iva][features]))[:, 1]
    outer_train, _ = impute_from_train(raw, raw)
    features = selector(outer_train, y, feature_names, seed)
    scaler = FeatureScaler().fit(raw[features])
    model = model_factory(seed)
    model.fit(scaler.transform(raw[features]), y)

    def predict(frame):
        return model.predict_proba(scaler.transform(frame[features]))[:, 1]

    if not np.isfinite(p_oof).all():
        raise ValueError("Inner-fold probabilities are incomplete.")
    return features, scaler, model, p_oof, predict


def positive_shap_values(values, X):
    if isinstance(values, list):
        values = np.asarray(values[1])
    else:
        values = np.asarray(values)
        if values.ndim == 3:
            if values.shape[:2] != X.shape or values.shape[2] != 2:
                raise ValueError(f"Unexpected SHAP shape: {values.shape}")
            values = values[:, :, 1]
    if values.shape != X.shape:
        raise ValueError(f"SHAP shape {values.shape} differs from features {X.shape}.")
    return values


def measurement_columns(columns):
    blocked = ("ImageNumber", "ObjectNumber", "Group_Index", "Group_Number",
               "Metadata_", "URL_", "FileName_", "PathName_", "ExecutionTime_",
               "ModuleError_", "TimeElapsed_")
    selected = [name for name in columns if not str(name).startswith(blocked)]
    if not selected:
        raise ValueError("No measurement features remain after excluding metadata.")
    return selected
