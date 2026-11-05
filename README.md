# Bank Marketing Classification Using Machine Learning

## a. Problem Statement

The objective of this project is to build and compare supervised machine learning models that predict whether a bank customer will subscribe to a term deposit as a result of a marketing campaign. This is a binary classification problem with target variable `y`:

- `yes`: the customer subscribed to a term deposit.
- `no`: the customer did not subscribe to a term deposit.

The project compares five classification algorithms using multiple evaluation metrics. Using more than accuracy helps assess model performance when the target classes are imbalanced and when false-positive and false-negative predictions have different practical implications.

## b. Dataset Description

**Dataset:** Bank Marketing  
**Source:** UCI Machine Learning Repository  
**Dataset URL:** https://archive.ics.uci.edu/dataset/222/bank+marketing

The Bank Marketing dataset contains customer attributes and information related to direct marketing campaigns. The target column is `y`, which indicates whether a customer subscribed to a term deposit. The commonly used full dataset contains 45,211 records and 16 input features, in addition to the target column; verify these counts against the exact dataset file used in the final experiment.

Preprocessing should encode categorical variables and apply any numerical scaling required by the model. The same fitted preprocessing steps must be used during training and evaluation. If the objective is to predict subscription before a call, exclude `duration`, because it is only known after the call and may cause target leakage.

## c. GitHub Repository Link

**GitHub Repository:** https://github.com/Ramya726/bank-marketing-predictive-modeling

The repository should include the application, training code, dependency list, test CSV, saved trained models and preprocessing artifacts, and this README. Confirm that the required files and model artifacts are committed and visible in the repository before submission.

**Streamlit Application:** https://bank-marketing-predictive-modeling-horspkquxtizjemjroke8d.streamlit.app/

## d. Models Used

The five classification models evaluated in this project are:

1. **Logistic Regression** — a probabilistic linear classification model.
2. **Decision Tree** — a rule-based model that can capture nonlinear relationships.
3. **K-Nearest Neighbors (KNN)** — predicts a class based on nearby training examples and is sensitive to feature scaling.
4. **Naive Bayes** — a probabilistic classifier based on Bayes' theorem and conditional-independence assumptions.
5. **Support Vector Machine (SVM)** — learns a decision boundary between classes; its behaviour depends on the selected kernel and hyperparameters.

### Model Comparison Table

The following values are taken from `model_comparison.csv` file.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7548 | 0.7722 | 0.2662 | 0.6238 | 0.3732 | 0.2853 |
| Decision Tree | 0.8251 | 0.7530 | 0.3452 | 0.5520 | 0.4247 | 0.3406 |
| KNN | 0.8891 | 0.7138 | 0.5758 | 0.1975 | 0.2942 | 0.2919 |
| Naive Bayes | 0.8452 | 0.7514 | 0.3708 | 0.4641 | 0.4123 | 0.3271 |
| Support Vector Machine | 0.7578 | 0.7718 | 0.2684 | 0.6200 | 0.3746 | 0.2867 |

### Evaluation Metrics

- **Accuracy:** proportion of all predictions that are correct.
- **AUC:** area under the ROC curve; measures discrimination between positive and negative classes across thresholds.
- **Precision:** proportion of predicted positive cases that are truly positive.
- **Recall:** proportion of actual positive cases correctly identified.
- **F1:** harmonic mean of precision and recall.
- **MCC:** Matthews Correlation Coefficient; summarizes agreement between predicted and actual classes using all four confusion-matrix outcomes. It ranges from -1 to +1.

### Results and Best-Model Observation

From the supplied comparison results:

- **Highest Accuracy:** K-Nearest Neighbors (0.8891)
- **Highest AUC:** Logistic Regression (0.7722)
- **Highest Precision:** K-Nearest Neighbors (0.5758)
- **Highest Recall:** Logistic Regression (0.6238)
- **Highest F1:** Decision Tree (0.4247)
- **Highest MCC:** Decision Tree (0.3406)

A model should not be selected using accuracy alone, especially if the target classes are imbalanced. F1 and MCC can help assess balanced classification performance, while precision and recall show different trade-offs. Based on the supplied results, the model with the highest F1 and/or MCC is a reasonable candidate when balanced performance is the priority; the final choice should reflect the project's objective and should be verified against the final evaluation pipeline.

To run the application locally, install the dependencies from `requirements.txt` and run:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Links

- **GitHub Repository:** https://github.com/Ramya726/bank-marketing-predictive-modeling
- **Streamlit Application:** (https://bank-marketing-predictive-modeling-horspkquxtizjemjroke8d.streamlit.app/)
- **Dataset:** https://archive.ics.uci.edu/dataset/222/bank+marketing