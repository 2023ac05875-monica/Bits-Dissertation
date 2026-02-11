import numpy as np
import joblib
import shap
import tensorflow as tf
from flask import Flask, render_template, request

# ---------------------------------------------------------------------------
# Load pre-trained artifacts
# ---------------------------------------------------------------------------
rf = joblib.load("rf_model.pkl")
ae = tf.keras.models.load_model("ae_model.keras", compile=False)
scaler = joblib.load("scaler.pkl")
thresholds = joblib.load("thresholds.pkl")
label_encoders = joblib.load("label_encoders.pkl")

BEST_ALPHA = thresholds["best_alpha"]
BEST_THRESH = thresholds["best_thresh"]

# Feature order must match training exactly
FEATURE_ORDER = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg",
    "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest",
    "isFlaggedFraud",
]

# SHAP explainer for Random Forest
explainer = shap.TreeExplainer(rf)

# Warm-up the autoencoder to avoid first-request latency
ae.predict(np.zeros((1, len(FEATURE_ORDER))), verbose=0)

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Rule engine (single-transaction, no history)
# ---------------------------------------------------------------------------
RULE_DESCRIPTIONS = {
    1: "Amount exceeds 50,000",
    2: "High velocity (>5 txns/hour)",
    3: "Off-hours transaction (00:00-05:00)",
    4: "Type-amount mismatch",
    5: "New beneficiary",
    6: "Geolocation anomaly (>500 km)",
    7: "Behavioural inconsistency (>3 std dev)",
}


def apply_rules_single(step, txn_type, amount):
    """Apply the 7 fraud-detection rules to a single transaction.

    Rules that require historical context (velocity, new beneficiary,
    geolocation, behaviour inconsistency) use safe defaults since we
    have no per-account history at inference time.
    """
    rules = {}
    # Rule 1 - Amount threshold
    rules[1] = int(amount > 50_000)
    # Rule 2 - Velocity (needs history) -> default 0
    rules[2] = 0
    # Rule 3 - Time-of-day (00:00 - 04:59)
    rules[3] = int((step % 24) < 5)
    # Rule 4 - Type-amount mismatch
    rules[4] = int(
        (txn_type == "CASH_OUT" and amount < 10)
        or (txn_type == "TRANSFER" and amount > 80_000)
    )
    # Rule 5 - New beneficiary (needs history) -> default 1 (conservative)
    rules[5] = 1
    # Rule 6 - Geolocation (no location data in form) -> default 0
    rules[6] = 0
    # Rule 7 - Behaviour inconsistency (needs history) -> default 0
    rules[7] = 0

    is_flagged = max(rules.values())
    return rules, is_flagged


# ---------------------------------------------------------------------------
# Label-encoding helpers
# ---------------------------------------------------------------------------
def encode_label(column, value):
    """Encode a categorical value using the saved LabelEncoder.

    For unseen values (nameOrig / nameDest not in training set), fall back
    to a deterministic hash within the encoder's range.
    """
    le = label_encoders[column]
    if value in le.classes_:
        return le.transform([value])[0]
    # Fallback for unseen account IDs
    return hash(value) % len(le.classes_)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1. Parse form inputs
        step = float(request.form["step"])
        txn_type = request.form["type"]
        amount = float(request.form["amount"])
        name_orig = request.form["nameOrig"]
        old_balance_org = float(request.form["oldbalanceOrg"])
        new_balance_orig = float(request.form["newbalanceOrig"])
        name_dest = request.form["nameDest"]
        old_balance_dest = float(request.form["oldbalanceDest"])
        new_balance_dest = float(request.form["newbalanceDest"])

        # 2. Apply rule engine
        rules, is_flagged = apply_rules_single(step, txn_type, amount)

        # 3. Label-encode categoricals
        type_encoded = encode_label("type", txn_type)
        orig_encoded = encode_label("nameOrig", name_orig)
        dest_encoded = encode_label("nameDest", name_dest)

        # 4. Build feature vector (exact training order)
        features = np.array([[
            step,
            type_encoded,
            amount,
            orig_encoded,
            old_balance_org,
            new_balance_orig,
            dest_encoded,
            old_balance_dest,
            new_balance_dest,
            is_flagged,
        ]])

        # 5. Scale
        X_scaled = scaler.transform(features)

        # 6. Random Forest probability
        p_rf = rf.predict_proba(X_scaled)[:, 1][0]

        # 7. Autoencoder reconstruction error
        recon = ae.predict(X_scaled, verbose=0)
        err = np.mean((recon - X_scaled) ** 2)
        # For a single sample min==max, so normalisation is undefined.
        # Use the raw error as p_ae; with alpha=1.0 it doesn't affect
        # the final score anyway.
        p_ae = float(err)

        # 8. Hybrid score
        final_score = BEST_ALPHA * p_rf + (1 - BEST_ALPHA) * p_ae

        # 9. Classify
        prediction = "Fraudulent" if final_score >= BEST_THRESH else "Legitimate"

        # Collect triggered rules
        triggered = [
            f"Rule {k}: {RULE_DESCRIPTIONS[k]}"
            for k, v in rules.items()
            if v == 1
        ]

        # 10. SHAP explanation
        shap_values = explainer.shap_values(X_scaled)
        # Shape is (1, 10, 2) — [sample, feature, class]
        sv = shap_values[0, :, 1]  # fraud class contributions for first sample
        shap_data = []
        for fname, val in zip(FEATURE_ORDER, sv):
            shap_data.append({
                "feature": fname,
                "value": round(float(val), 4),
                "direction": "fraud" if val > 0 else "legit",
                "abs": abs(float(val)),
            })
        # Sort by absolute contribution descending
        shap_data.sort(key=lambda x: x["abs"], reverse=True)
        base_value = round(float(explainer.expected_value[1]), 4)

        return render_template(
            "index.html",
            prediction=prediction,
            final_score=round(float(final_score), 4),
            rf_score=round(float(p_rf), 4),
            ae_score=round(float(p_ae), 6),
            alpha=BEST_ALPHA,
            threshold=round(BEST_THRESH, 4),
            triggered_rules=triggered,
            is_flagged=is_flagged,
            shap_data=shap_data,
            shap_base=base_value,
            # Echo back the submitted values so the form stays filled
            form=request.form,
        )

    except Exception as e:
        return render_template("index.html", error=str(e))


if __name__ == "__main__":
    print(f"Model config: alpha={BEST_ALPHA}, threshold={BEST_THRESH:.4f}")
    print(f"Features: {FEATURE_ORDER}")
    app.run(debug=True, port=5000)
