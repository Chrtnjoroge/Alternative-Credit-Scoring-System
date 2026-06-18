"""Feature engineering for the Alternative Credit Scoring project.

Builds a model-ready feature set from the raw German Credit Data,
reframed as analogues of Kenyan alternative-data signals:

- Saving accounts   -> mobile savings (e.g. M-Shwari) engagement
- Checking account  -> mobile money (e.g. M-Pesa) transaction activity
- Job               -> income source stability/type
- Housing           -> housing tenure
- Purpose           -> reason for the loan (school fees, business stock, etc.)

Each raw signal is kept distinct rather than pre-combined into a hand-weighted
"score" - the model (Phase 3) learns how to weight these signals from data.
`Sex` is intentionally excluded (protected attribute).
"""

import pandas as pd

SAVINGS_LEVELS = {
    "little": 1,
    "moderate": 2,
    "quite rich": 3,
    "rich": 4,
}

MOBILE_MONEY_LEVELS = {
    "little": 1,
    "moderate": 2,
    "rich": 3,
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the model-ready feature set and target from raw data.

    Returns a new dataframe containing only the engineered features and the
    `loan_default` target - no raw or intermediate columns.
    """
    out = pd.DataFrame(index=df.index)

    # Target
    out["loan_default"] = (df["Risk"] == "bad").astype(int)

    # Numeric signals, used as-is
    out["age"] = df["Age"]
    out["job_level"] = df["Job"]
    out["credit_amount"] = df["Credit amount"]
    out["duration"] = df["Duration"]

    # Mobile savings analogue: separate "has an account at all" from
    # "how active is it" - missing is not assumed to be the worst level
    out["has_savings_account"] = df["Saving accounts"].notna().astype(int)
    out["savings_level"] = df["Saving accounts"].map(SAVINGS_LEVELS).fillna(0).astype(int)

    # Mobile money analogue: same indicator + ordinal pattern
    out["has_mobile_money_account"] = df["Checking account"].notna().astype(int)
    out["mobile_money_level"] = df["Checking account"].map(MOBILE_MONEY_LEVELS).fillna(0).astype(int)

    # Nominal categories with no natural order -> one-hot encode
    housing_dummies = pd.get_dummies(df["Housing"], prefix="housing", drop_first=True).astype(int)
    purpose_dummies = pd.get_dummies(df["Purpose"], prefix="purpose", drop_first=True).astype(int)

    return pd.concat([out, housing_dummies, purpose_dummies], axis=1)
