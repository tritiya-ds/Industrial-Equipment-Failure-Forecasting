# Industrial Equipment Failure Forecasting

A machine-learning project that analyzes industrial machine operating data and predicts the likelihood of machine failure. It uses the **AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository and compares classification models, with attention to the cost of missed failures and false alarms.

## Project Overview

Unexpected equipment failures can cause downtime, maintenance costs, and safety risks. Predictive maintenance uses equipment measurements to identify patterns associated with failure and help prioritize inspections.

## Objectives

- Explore machine operating data and the distribution of machine failures.
- Prepare the data for machine learning.
- Engineer features from temperature, rotational speed, and torque.
- Train and evaluate classification models.
- Examine feature importance and the effect of decision thresholds on estimated costs.

## Dataset

- **Name:** AI4I 2020 Predictive Maintenance Dataset
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
- **Task:** Binary classification
- **Target:** `Machine failure`

Failure events are much less common than normal observations, so class imbalance is an important consideration.

## Workflow

1. Load the dataset and inspect data types, missing values, and target distribution.
2. Perform exploratory analysis of failure prevalence, numerical features, outliers, and correlations.
3. Engineer `Temperature_Difference` (process temperature − air temperature) and `Power_Estimate` (rotational speed × torque); one-hot encode machine type.
4. Use a stratified 80:20 train/test split and fit the scaler on training data only.
5. Train class-weighted Logistic Regression, tune its regularization with `GridSearchCV`, and train a Multi-Layer Perceptron (MLP).
6. Evaluate models with accuracy, precision, recall, F1-score, and ROC-AUC.
7. Compare estimated costs at different probability thresholds, assuming **$10,000 per false negative** and **$500 per false positive**.
8. Use permutation importance to identify influential features.

## Key Findings

| Analysis | Result |
|---|---:|
| Normal observations | 96.61% |
| Failure observations | 3.39% |
| Rotational-speed outliers identified by IQR | 418 |
| Logistic Regression — 5-fold mean CV ROC-AUC | 0.9207 |
| Logistic Regression — test accuracy | 0.8600 |
| Logistic Regression — reported test ROC-AUC | 0.9864 |
| MLP — test ROC-AUC | 0.9806 |
| Best tested threshold in cost analysis | 0.30 |
| Estimated cost at threshold 0.50 | $186,000 |
| Estimated cost at threshold 0.30 | $132,000 |
| Estimated cost reduction | $54,000 (~29%) |

The tuned Logistic Regression model reported a best cross-validation ROC-AUC of approximately **0.9208** with `C = 10`. The reported permutation-importance analysis ranked **Torque**, **Tool wear**, and **Temperature_Difference** among the most influential features.

> **Evaluation note:** The notebook selected the cost threshold using test-set results. For a more rigorous evaluation, choose the threshold using a validation set or cross-validation, then evaluate it once on the untouched test set. Verify that all reported metrics refer to the intended fitted model and evaluation run before publication.

## Models and Techniques

- Logistic Regression with class weighting
- `GridSearchCV`
- Multi-Layer Perceptron (MLP)
- Stratified train/test split and feature scaling
- Classification metrics and ROC-AUC
- Permutation feature importance
- Cost-sensitive probability-threshold analysis

## Tools and Libraries

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, and Jupyter Notebook / Google Colab.

Install common dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

If the notebook imports additional packages, install them as indicated by its import cells.

## How to Run

### Google Colab

1. Open `Project_9_Industrial_Equipment_Failure_Forecasting (2).ipynb` in Google Colab.
2. Download the dataset from the UCI link above and make it available at the path expected by the notebook.
3. Update the data-loading path if necessary.
4. Run the notebook cells in order and review the visualizations and evaluation results.

### Jupyter Notebook

1. Download or clone the repository.
2. Install the dependencies.
3. Place the dataset at the expected path, updating the notebook path if needed.
4. Start Jupyter:

```bash
jupyter notebook
```

## Suggested Repository Structure

```text
Industrial-Equipment-Failure-Forecasting/
├── Project_9_Industrial_Equipment_Failure_Forecasting (2).ipynb
├── README.md
└── data/
    └── AI4I_2020_Predictive_Maintenance_Dataset.csv
```

This is a suggested structure; keep the actual notebook filename and dataset path consistent with your repository. If you do not include the dataset in the repository, provide download instructions instead.

## Limitations and Future Improvements

- Select the decision threshold using validation data rather than the test set.
- Compare all models using the same test split and a consistent set of metrics.
- Include a confusion matrix and precision-recall analysis because failures are rare.
- Validate the assumed failure and false-alarm costs with operational data.
- Explore probability calibration and additional models.

## References

1. UCI Machine Learning Repository — [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset).
2. scikit-learn — [Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html), [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html), and [Permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html).

## Internship Context

Prepared as part of the Autumn Internship Programme on Data Science at **IDEAS–TIH, Indian Statistical Institute (ISI), Kolkata**.

- **Internship period:** 21 September 2026 – 16 October 2026
- **Project mentor:** Dr. Chandan Biswas

---

If you use or adapt this project, cite the dataset source and describe any changes made to the notebook or evaluation procedure.
