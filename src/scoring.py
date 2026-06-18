"""Credit scoring and explainability for the Alternative Credit Scoring project.

Converts a model's default-probability prediction into a 0-100 credit score,
risk tier, and lending recommendation - plus "reason codes" that explain which
features drove an individual score, by comparing the applicant's values to the
typical profile of repaid vs. defaulted customers in the training data.

(No SHAP/ELI5/LIME available in this environment, so reason codes are built
from the model's own feature importances combined with how the applicant's
values compare to the two outcome groups - the same idea behind traditional
credit scorecard "reason codes".)
"""

import pandas as pd

# (min_score, risk_level, recommendation, interest_rate_adjustment)
SCORE_TIERS = [
    (75, "EXCELLENT", "Approve at standard terms", "-1% (preferred rate)"),
    (65, "GOOD", "Approve at standard terms", "Standard rate"),
    (55, "FAIR", "Approve with higher interest", "+2-3% (risk premium)"),
    (45, "POOR", "Decline or require collateral", "Not available"),
    (0, "VERY POOR", "Decline", "Not available"),
]


def _tier_for_score(credit_score):
    for min_score, risk_level, recommendation, interest_adjustment in SCORE_TIERS:
        if credit_score >= min_score:
            return risk_level, recommendation, interest_adjustment
    return SCORE_TIERS[-1][1:]


def score_customer(model, customer_row):
    """Score a single customer (1-row DataFrame of engineered features).

    Returns a dict with credit_score (0-100), risk_level, default_probability
    (%), recommendation, and interest_rate_adjustment.
    """
    default_probability = model.predict_proba(customer_row)[0, 1]
    credit_score = (1 - default_probability) * 100
    risk_level, recommendation, interest_adjustment = _tier_for_score(credit_score)

    return {
        "credit_score": round(credit_score, 1),
        "risk_level": risk_level,
        "default_probability": round(default_probability * 100, 1),
        "recommendation": recommendation,
        "interest_rate_adjustment": interest_adjustment,
    }


def explain_score(model, customer_row, X_train, y_train, top_n=3):
    """Return the top `top_n` features driving this customer's score.

    For each feature, this measures how unusual the applicant's value is
    (a z-score against the training population), then weights that by the
    model's overall feature importance and by whether the feature tends to
    rise or fall with default risk (its correlation with `y_train`).
    Positive impact = pushes the score toward higher risk; negative impact
    = protective. `typical_repaid`/`typical_defaulted` are included purely
    for the narrative ("repaid customers typically had X").
    """
    importances = model.feature_importances_
    overall_mean = X_train.mean()
    overall_std = X_train.std()
    risk_direction = X_train.corrwith(y_train)
    good = X_train[y_train == 0]
    bad = X_train[y_train == 1]
    applicant = customer_row.iloc[0]

    rows = []
    for i, feature in enumerate(X_train.columns):
        std = overall_std[feature]
        z = (applicant[feature] - overall_mean[feature]) / std if std > 1e-9 else 0.0
        sign = 1 if risk_direction[feature] >= 0 else -1
        impact = importances[i] * z * sign

        rows.append({
            "feature": feature,
            "applicant_value": applicant[feature],
            "typical_repaid": round(good[feature].mean(), 2),
            "typical_defaulted": round(bad[feature].mean(), 2),
            "impact": impact,
        })

    reasons = pd.DataFrame(rows)
    reasons["abs_impact"] = reasons["impact"].abs()
    return reasons.sort_values("abs_impact", ascending=False).head(top_n).drop(columns="abs_impact")
