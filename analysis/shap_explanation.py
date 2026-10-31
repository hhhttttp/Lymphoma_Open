"""Explain saved baseline models on held-out samples without refitting.

RF SHAP explains positive-class probability. PSF SHAP explains the stored logistic
regression's log odds over CP/QP probabilities and scaled clinical variables;
these are not raw-image-feature attributions through the entire stack.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import output_path
from model_artifacts import load_baseline, fold_inputs
from workflow_utils import positive_shap_values

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap


def main():
    saved = load_baseline()
    out = Path(output_path("explainability"))
    out.mkdir(parents=True, exist_ok=True)
    inputs = saved["inputs"]
    model_names = ("CP", "QP", "Clin", "MEFF", "PSF")
    results = {m: [] for m in model_names}
    feature_values = {m: [] for m in model_names}
    bases = {m: [] for m in model_names}
    metadata = inputs[["Merge_ID"]].copy()
    metadata["run_id"] = saved["run_id"]
    metadata["fold"] = 0
    for fold in saved["folds"]:
        te, tr = fold["test_idx"], fold["train_idx"]
        held_out, names = fold_inputs(saved, fold, inputs.iloc[te])
        training, _ = fold_inputs(saved, fold, inputs.iloc[tr])
        metadata.loc[te, "fold"] = fold["fold"]
        for name in model_names:
            X = pd.DataFrame(held_out[name], columns=names[name], index=te)
            model = fold["models"][name]
            if name == "PSF":
                # Explain the already-fitted meta model, not a surrogate RF.
                explainer = shap.LinearExplainer(model, training[name][:100])
                values = np.asarray(explainer.shap_values(X))
                base = float(np.asarray(explainer.expected_value).reshape(-1)[0])
                expected = model.decision_function(X.to_numpy())
            else:
                explainer = shap.TreeExplainer(model)
                values = positive_shap_values(explainer.shap_values(X), X)
                base = float(np.asarray(explainer.expected_value).reshape(-1)[1])
                expected = model.predict_proba(X.to_numpy())[:, 1]
            if values.shape != X.shape or not np.isfinite(values).all():
                raise ValueError(f"Unexpected {name} SHAP output in fold {fold['fold']}.")
            if not np.allclose(base + values.sum(axis=1), expected, atol=1e-5, rtol=1e-4):
                raise ValueError(f"{name} SHAP values do not reconstruct saved model output.")
            results[name].append(pd.DataFrame(values, index=te, columns=names[name]))
            feature_values[name].append(X)
            bases[name].append(pd.Series(base, index=te))
    for name in model_names:
        # Unselected features contribute zero in that fold; schemas may vary.
        values = pd.concat(results[name]).reindex(inputs.index).fillna(0.0)
        X = pd.concat(feature_values[name]).reindex(index=inputs.index, columns=values.columns).fillna(0.0)
        table = metadata.copy()
        table["base_value"] = pd.concat(bases[name]).reindex(inputs.index)
        table.join(values).to_csv(out / f"SHAP_{name}_heldout.csv", index=False)
        metadata.join(X).to_csv(out / f"SHAP_{name}_transformed_inputs.csv", index=False)
        importance = values.abs().mean().sort_values(ascending=False)
        importance.rename("mean_abs_shap").to_csv(out / f"SHAP_{name}_importance.csv")
        shap.summary_plot(values.to_numpy(), X, max_display=35, show=False)
        scale = "meta-model log odds" if name == "PSF" else "positive-class probability"
        plt.title(f"Saved {name}: held-out SHAP ({scale})")
        plt.tight_layout()
        plt.savefig(out / f"SHAP_{name}_heldout.png", dpi=300, bbox_inches="tight")
        plt.close()
    print(f"Explained saved run {saved['run_id']}: {out}")


if __name__ == "__main__":
    main()
