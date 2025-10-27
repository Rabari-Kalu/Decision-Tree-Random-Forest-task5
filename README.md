# Decision-Tree-Random-Forest-task5
Heart Disease Prediction using Decision Tree &amp; Random Forest


## Project Overview
This project applies **Decision Tree** and **Random Forest** classifiers to the **Heart Disease dataset** to predict whether a person has heart disease based on various health indicators such as age, cholesterol, resting blood pressure, and chest pain type.

We train, visualize, and compare both models — analyzing overfitting, interpreting feature importance, and evaluating using cross-validation.

---

## Dataset Information
- **File:** `heart.csv`
- **Rows:** ~1025  
- **Columns:** 14  
- **Target Column:** `target`  
  - `1` → Heart Disease  
  - `0` → No Heart Disease  

### Features:
`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`

---

## Workflow Steps

### 1️⃣Import & Explore Data
- Loaded dataset using Pandas
- Checked for missing values and data types
- Identified target variable `target`

### 2️⃣ Train-Test Split & Preprocessing
- 80% training / 20% testing split  
- Stratified split to maintain class balance  
- Standard scaling (optional for consistency)

### 3️⃣ Decision Tree Classifier
- Trained a baseline `DecisionTreeClassifier`
- Analyzed **overfitting** by changing `max_depth`
- Visualized tree using `plot_tree()`
- Saved plots for confusion matrix & depth performance

### 4️⃣ Random Forest Classifier
- Trained a **RandomForestClassifier (n_estimators=200)**
- Compared accuracy vs Decision Tree
- Extracted **feature importances** for interpretation

### 5️⃣ Evaluation Metrics
- Accuracy, Precision, Recall, F1-score
- ROC-AUC and ROC curve plot
- Confusion matrix visualization
- Cross-validation (5 folds)

---

## 📊 Results Summary

| Model | Accuracy | Precision | Recall | ROC-AUC |
|--------|-----------|------------|----------|----------|
| Decision Tree (Default) | ~0.78 | ~0.80 | ~0.76 | ~0.79 |
| Decision Tree (Tuned Depth) | ~0.82 | ~0.84 | ~0.81 | ~0.84 |
| Random Forest (200 trees) | **~0.86–0.90** | **~0.89** | **~0.87** | **~0.92** |

---

## 🌲 Visualizations
- `plots/dt_confusion_matrix_default.png` – Decision Tree confusion matrix  
- `plots/dt_overfitting_depth_curve.png` – Overfitting analysis  
- `plots/decision_tree_depth_3.png` – Pruned tree visualization  
- `plots/rf_feature_importances.png` – Random Forest feature importance  
- `plots/roc_dt_vs_rf.png` – ROC comparison  

---

## 💬 Questions & Answers

**1️⃣ How does a decision tree work?**  
A decision tree splits data into smaller subsets using features that provide the **highest information gain** (or lowest Gini impurity), forming a tree structure where leaves represent final predictions.

**2️⃣ What is entropy and information gain?**  
- **Entropy**: Measures impurity or uncertainty in data.  
- **Information Gain**: Reduction in entropy after splitting the dataset on a feature.  
Higher information gain = better feature for splitting.

**3️⃣ How is random forest better than a single tree?**  
Random Forest builds multiple decision trees on random subsets of data and averages their results.  
This reduces **overfitting** and improves **generalization**.

**4️⃣ What is overfitting and how do you prevent it?**  
Overfitting happens when a model learns noise in training data.  
Prevent by:
- Limiting tree depth (`max_depth`)
- Using ensemble methods like Random Forest
- Using cross-validation

**5️⃣ What is bagging?**  
**Bootstrap Aggregating (Bagging)** builds multiple models using different random samples of the dataset and combines their predictions to reduce variance and improve stability.

**6️⃣ How do you visualize a decision tree?**
```python
from sklearn.tree import plot_tree
plot_tree(model, feature_names=X.columns, filled=True)
