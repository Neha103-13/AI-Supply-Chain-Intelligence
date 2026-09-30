from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "xgboost_late_delivery_model.pkl"
)

FEATURES_PATH = (
    BASE_DIR
    / "models"
    / "model_features.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL_PATH)

model_features = joblib.load(FEATURES_PATH)

preprocessor = model.named_steps["preprocessor"]

classifier = model.named_steps["classifier"]

feature_names = (
    preprocessor.get_feature_names_out()
)

explainer = shap.TreeExplainer(
    classifier
)


print("Model loaded successfully.")
print("Features:", len(model_features))
print("Encoded features:", len(feature_names))


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_delivery_risk(order_data):

    input_df = pd.DataFrame([order_data])

    # Make sure columns are in the same order
    # used during model training
    input_df = input_df[model_features]

    # Prediction
    prediction = int(
        model.predict(input_df)[0]
    )

    # Probability of late delivery
    probability = float(
        model.predict_proba(input_df)[0, 1]
    )

    # Risk category
    if probability >= 0.70:
        risk = "HIGH"

    elif probability >= 0.40:
        risk = "MEDIUM"

    else:
        risk = "LOW"

    return {
        "prediction": prediction,
        "probability": probability,
        "risk": risk
    }


# ============================================================
# SHAP EXPLANATION FUNCTION
# ============================================================

def explain_prediction(order_data, top_n=5):

    input_df = pd.DataFrame([order_data])

    input_df = input_df[model_features]

    # Transform input
    transformed = preprocessor.transform(
        input_df
    )

    # SHAP values
    shap_values = explainer.shap_values(
        transformed
    )

    shap_values = np.asarray(
        shap_values
    ).ravel()

    # Create dataframe
    shap_df = pd.DataFrame({
        "encoded_feature": feature_names,
        "shap_value": shap_values
    })

    # --------------------------------------------------------
    # Convert encoded feature back to original feature
    # --------------------------------------------------------

    def get_original_feature(name):

        name = name.replace(
            "categorical__",
            ""
        )

        name = name.replace(
            "numeric__",
            ""
        )

        for feature in model_features:

            if name == feature:
                return feature

            if name.startswith(
                feature + "_"
            ):
                return feature

        return name


    shap_df["original_feature"] = (
        shap_df["encoded_feature"]
        .apply(get_original_feature)
    )

    # --------------------------------------------------------
    # Aggregate SHAP values
    # --------------------------------------------------------

    explanation = (
        shap_df
        .groupby("original_feature")["shap_value"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    # Positive factors
    positive = (
        explanation[
            explanation > 0
        ]
        .head(top_n)
    )

    # Negative factors
    negative = (
        explanation[
            explanation < 0
        ]
        .sort_values()
        .head(top_n)
    )

    return {
        "positive": positive,
        "negative": negative
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\nPredictor module test successful.")

    print(
        "Prediction function:",
        callable(predict_delivery_risk)
    )

    print(
        "Explanation function:",
        callable(explain_prediction)
    )