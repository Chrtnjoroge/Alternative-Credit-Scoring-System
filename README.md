# Alternative Credit Scoring — Kenya Context

Build a credit score using alternative financial signals (mobile money, savings, employment) instead of traditional credit history. Reach the 60% of Kenyans excluded from formal banking.

---

## The Problem

Kenya's credit gap is real. Traditional banks rely on Credit Reference Bureaus (TransUnion, Metropol, Creditinfo) that only capture formal credit history. This excludes 60% of the population - the unbanked majority who generate daily financial signals through M-Pesa, mobile savings products, utility payments, and informal income sources.

Result: millions of creditworthy people are denied access to credit.

---

## The Solution

This project demonstrates an alternative credit scoring pipeline using signals everyone generates:

- **Mobile money activity** (M-Pesa like transaction patterns)
- **Savings engagement** (M-Shwari like)
- **Employment stability** (job type and consistency)
- **Housing tenure** (rent, own, or free)
- **Loan purpose** (education, business, consumption, etc.)

Instead of a hand-weighted "composite score," the model learns how to weight these signals from historical data — 17 distinct features feeding into 5 different classifiers.

---

## What Works

**Best model:** Gradient Boosting  
**AUC score:** 0.780 (test set) / 0.742 (5-fold cross-validation)  
**Accuracy:** 76%

**Key finding:** Customers without a formal bank account actually default *less* often (10.3% vs 42.6%). The model picks up on this, approving 97.4% of this group with only 1.4% wrongful denial. This validates the core premise: lacking formal credit history is not inherently high-risk.

---

## How to Use

### Setup

```bash
pip install -r requirements.txt
```

### Run the Notebook

```bash
jupyter notebook Alternative_Credit_Scoring_Kenya.ipynb
```

The notebook walks through five phases:

1. **Phase 1:** Load raw German Credit Data (1,000 loan records)
2. **Phase 2:** Engineer 17 features reframed as Kenya alternative-data signals
3. **Phase 3:** Train and compare 5 classifiers (Gradient Boosting wins)
4. **Phase 4:** Score two real customers, show reason codes (why their score is what it is)
5. **Phase 5:** Check for fairness — does the model treat different groups equally?

---

## Project Structure

```
.
├── Alternative_Credit_Scoring_Kenya.ipynb    # Main notebook (run this)
├── german_credit_data.csv                    # Dataset (1,000 loan records)
├── requirements.txt                          # Python dependencies
├── README.md                                 # This file
└── src/
    ├── __init__.py
    ├── data.py                               # Load and inspect raw data
    ├── features.py                           # Engineer 17 features
    ├── models.py                             # Train and compare 5 models
    ├── scoring.py                            # Generate 0–100 credit score 
    └── evaluation.py                         # Fairness checks across groups
```

---

## Key Results

### Model Performance

| Model | Accuracy | AUC | CV AUC |
|-------|----------|-----|--------|
| **Gradient Boosting** | **76%** | **0.780** | **0.742** |
| Random Forest | 75% | 0.774 | 0.738 |
| Support Vector Machine | 74% | 0.771 | 0.735 |
| Logistic Regression | 73% | 0.769 | 0.733 |
| Decision Tree | 68% | 0.686 | 0.657 |

---

### Fairness Findings

**No sex-based discrimination detected.** Male and female applicants are treated similarly across approval rates and error rates.

**Unbanked population is actually lower-risk:** Customers without a mobile money account have a 10.3% default rate versus 42.6% for those with accounts. The model correctly identifies this, approving 97% of the unbanked group with only 1.4% wrongful denial.

---

## The Data

This project uses the **German Credit Dataset** downloaded from Kaggle ([kabure: German Credit Risk - With Target](https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk)). The original data is from the UCI Machine Learning Repository.

Each feature is conceptually reframed as a Kenya alternative-data analogue:

- Saving accounts → M-Shwari engagement
- Checking account → M-Pesa activity
- Job → Income source stability
- Housing → Housing tenure
- Purpose → Loan reason

**Dataset:** 1,000 loan records with 10 raw features (age, sex, job, housing, savings, checking, credit amount, duration, purpose, risk outcome)

**Why German data?** Real Kenyan alternative-data requires partnerships with telecom operators, mobile money providers, or Credit Reference Bureaux. German data lets us build and validate the methodology without that barrier. The same pipeline works with real Kenyan data once available through partnerships.

---

## Credit Scoring Output

For any applicant, the model produces:

- **Credit score** (0–100): Higher = lower risk
- **Risk tier:** Excellent / Good / Fair / Poor / Very Poor
- **Approval recommendation:** Approve at standard terms, approve with higher interest, decline, etc.
- **Interest rate adjustment:** Preferred rate, standard, risk premium, not available
- **Reason codes:** Top 3 features driving this applicant's score (for transparency and dispute resolution)

Example:

```
Credit Score: 68.4
Risk Level: FAIR
Default Probability: 31.6%
Recommendation: Approve with higher interest (+2–3% risk premium)

Top Reasons for This Score:
- has_mobile_money_account: 1 (applicant HAS account) - lower risk
- credit_amount: 3500 KES - moderate concern
- age: 28 years — neutral
```

---

## What Real Deployment Would Require

This is a proof-of-concept. A production system in Kenya would need:

- **Real data sources:** M-Pesa transaction history, utility payment records, SACCO contributions, airtime top-up patterns, CRB records
- **Legal basis:** Data Protection Act 2019 compliance (explicit consent, data minimisation, purpose limitation)
- **Regulatory approval:** Central Bank of Kenya Digital Credit Providers rules (licensing, transparency, dispute handling)
- **Model governance:** Fairness monitoring, reason codes in adverse-action notices, human review path for disputes, drift detection as the population changes

None of this is implemented here. The goal was the modelling pipeline and the thinking behind it.

---

## Technologies

- **Python 3.11+**
- **Pandas** - Data manipulation
- **NumPy** - Numerical operations
- **Scikit-learn** - Machine learning (5 models, metrics, cross-validation)
- **Matplotlib & Seaborn** - Visualisations
- **Jupyter** - Interactive notebook

---

## Next Steps

1. **Adapt to real Kenyan data** - Partner with a digital lender, fintech, or CRB to access actual M-Pesa, utility, or SACCO data
2. **Add more features** - Airtime top-up frequency, utility payment timeliness, informal income patterns
3. **Deploy as a service** - Flask or FastAPI REST endpoint for real-time scoring
4. **Monitor fairness in production** - Quarterly re-runs of the fairness report, drift detection
5. **Gather applicant feedback** - Improve reason codes based on what applicants dispute or question



---

## Licence

This project is released under the MIT Licence.

