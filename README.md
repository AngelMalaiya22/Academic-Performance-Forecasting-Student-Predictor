# 🎓 Academic Performance Forecasting & Student Success Predictor

Machine-learning system that **forecasts a student's final grade (0–20)** and **flags students at risk of failing (G3 < 10)**, so support can be offered early. It works in two modes: an **early-warning** mode that uses only background and behaviour data, and a **mid-term** mode that also uses period grades.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-46E3B7?logo=render&logoColor=white)](https://student-success-predictor-eif3.onrender.com/)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.4-189FDD)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)

👉 **Try it:** https://student-success-predictor-eif3.onrender.com/
*(Hosted on Render's free plan, so the first visit after a period of inactivity can take a minute to wake up.)*

---

## ✨ Features

- **Two prediction tasks:** regression (final grade) and classification (at-risk flag).
- **Two scenarios:** *Early warning* (no grades yet) and *Mid-term forecast* (G1 and G2 known).
- **Full ML pipeline:** EDA → cleaning → feature engineering → encoding and scaling → baseline models → hyperparameter tuning → threshold selection → explainability → fairness check → deployment.
- **Leak-free evaluation:** all preprocessing lives inside a scikit-learn `Pipeline`; model selection uses 5-fold cross-validation on the training set only; the test set is touched once at the end.
- **Explainable:** permutation importance and SHAP for every final model.
- **Deployed:** interactive Streamlit app with risk level and suggested next steps.

## 📊 Dataset

395 students from two Portuguese secondary schools (Mathematics course), 33 columns: demographics, family background, study habits, lifestyle, absences and three period grades (`G1`, `G2`, `G3`). No missing values.
Source: Cortez, P. & Silva, A. (2008), *Using Data Mining to Predict Secondary School Student Performance* (UCI Machine Learning Repository, Student Performance dataset).

**Targets:** `G3` (final grade, 0–20) and `at_risk = 1` if `G3 < 10` (about one third of students).

## 🧪 Method in brief

| Step | What was done |
|---|---|
| Cleaning | No duplicates or missing values. The 38 students with `G3 = 0` (likely dropouts) were **kept**, because they are exactly the students a predictor must catch. |
| Features | `parent_edu`, `alc_total`, `log_absences`; for the mid-term scenario also `grade_trend` and `grade_mean`. |
| Encoding | Ordinal encoding for binary columns, one-hot for `Mjob`, `Fjob`, `reason`, `guardian`, standard scaling for numeric columns. |
| Split | 80/20, stratified on `at_risk` (316 train / 79 test), seed 42. |
| Models | Dummy, Linear/Ridge/Lasso/ElasticNet, KNN, SVM, Decision Tree, Random Forest, Gradient Boosting, XGBoost (Logistic Regression for classification). |
| Tuning | `RandomizedSearchCV`, 5-fold CV, top-3 baselines per scenario. |
| Threshold | Maximum F2 with precision ≥ 0.50, falling back to Youden's J for the weak early model. Chosen on cross-validated training predictions only. |

## 🏆 Results (held-out test set, 79 students)

| | Mid-term (G1 and G2 known) | Early warning (no grades) |
|---|---|---|
| **Best regressor** | Gradient Boosting | XGBoost |
| RMSE / MAE (grade points) | **1.33 / 0.83** | 3.95 / 2.96 |
| R² | **0.92** | 0.26 |
| Baseline RMSE (always predict the mean) | 4.60 | 4.60 |
| **Best classifier** | Logistic Regression | Gradient Boosting |
| ROC-AUC | **0.977** | 0.73 |
| Recall / Precision (at-risk) | 1.00 / 0.70 | 0.50 / 0.65 |
| F1 | 0.83 | 0.57 |

**Takeaways**
- Period grades (`G1`, `G2`) are by far the strongest predictors, so the mid-term model is accurate and catches every at-risk student in the test set, at the cost of some false alarms.
- Background factors alone (past failures, parental education, age, going out, study time) carry real but limited signal. The early model catches about half of at-risk students, so use it as a **screening aid**, not a decision-maker.

## 🖥️ Run locally

```bash
git clone https://github.com/AngelMalaiya22/Academic-Performance-Forecasting-Student-Predictor.git
cd Academic-Performance-Forecasting-Student-Predictor
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Use a Python 3.11 or newer environment. Library versions are pinned in `requirements.txt` because the saved `.joblib` models must be loaded with the same scikit-learn version that trained them.

To re-train everything, open `Academic_Performance_Predictor.ipynb` in Jupyter (it needs `pip install xgboost shap` in addition) and run all cells. It regenerates the `artifacts/` folder.

## ☁️ Deployment

The app runs on **Render** (free web service) using the `render.yaml` Blueprint:

```
streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
```

Pushing to `main` triggers an automatic redeploy.

## 📁 Project structure

```
├── app.py                                # Streamlit web app
├── artifacts/                            # trained pipelines + metadata.json
│   ├── reg_early.joblib   reg_midterm.joblib
│   ├── clf_early.joblib   clf_midterm.joblib
│   └── metadata.json
├── Academic_Performance_Predictor.ipynb  # full ML pipeline notebook
├── student_data_.xls                     # dataset (CSV content)
├── requirements.txt
├── render.yaml
└── .streamlit/config.toml
```

## ⚠️ Limitations and responsible use

- Small dataset (395 students from two schools, one subject), so results may not transfer to other institutions.
- Dropout-type cases (`G3 = 0`) are hard to foresee from earlier data.
- Correlation is not causation: SHAP shows what the model relies on, not what causes poor results.
- The early classifier showed different recall for boys and girls on the test set (small groups, so noisy). Check subgroup performance before any real-world use.
- This is a decision-support prototype. Predictions must never be the only basis for a decision about a student.

## 🔭 Future work

Add the Portuguese-language dataset for more data, repeated cross-validation with confidence intervals, probability calibration, letter-grade (A–F) classification, and drift monitoring after deployment.

## 👤 Author

**Angel Malaiya**, B.Tech CSE (AI & ML), Sharda University
GitHub: [@AngelMalaiya22](https://github.com/AngelMalaiya22)

## 📚 Reference

Cortez, P., & Silva, A. (2008). Using data mining to predict secondary school student performance. In *Proceedings of 5th Future Business Technology Conference (FUBUTEC 2008)*, pp. 5–12.
