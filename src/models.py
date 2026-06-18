"""Model training and comparison for the Alternative Credit Scoring project."""

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score

RANDOM_STATE = 42

# Distance/gradient-based models are sensitive to feature scale; tree-based
# models split on thresholds and are scale-invariant.
SCALED_MODELS = {"Logistic Regression", "Support Vector Machine"}


def get_models():
    """Return a fresh dict of model name -> unfitted estimator."""
    return {
        "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
        "Support Vector Machine": SVC(probability=True, random_state=RANDOM_STATE),
    }


def split_data(features_df, target_col="loan_default", test_size=0.2):
    """Split engineered features into train/test sets, stratified on the target."""
    X = features_df.drop(columns=[target_col])
    y = features_df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)


def compare_models(X_train, X_test, y_train, y_test, models=None):
    """Train and evaluate each model. Returns (results_df, fitted_scaler)."""
    if models is None:
        models = get_models()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []
    for name, model in models.items():
        use_scaled = name in SCALED_MODELS
        X_tr = X_train_scaled if use_scaled else X_train
        X_te = X_test_scaled if use_scaled else X_test

        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]
        cv_auc = cross_val_score(model, X_tr, y_train, cv=5, scoring="roc_auc").mean()

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_prob),
            "CV_AUC": cv_auc,
            "Object": model,
            "Predictions": y_pred,
            "Probabilities": y_prob,
        })

    return pd.DataFrame(results), scaler


def get_best_model(results_df, metric="AUC"):
    """Return the row of results_df with the highest value of `metric`."""
    return results_df.loc[results_df[metric].idxmax()]
