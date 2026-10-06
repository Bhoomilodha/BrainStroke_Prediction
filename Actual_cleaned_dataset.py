# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 20:30:51 2026

@author: Bhoomi Lodha
"""

# -*- coding: utf-8 -*-
"""
Brain Stroke Prediction - Actual Kaggle Dataset
Dataset: 4981 actual records
"""

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# 2. LOAD ACTUAL DATASET
# ============================================================

file_path = r"C:\Users\rakes\OneDrive\Documents\Kamka\Brain_Stroke_reasearch_paper\Actual _brain_stroke xl file.xlsx"

df = pd.read_excel(file_path)

print("Dataset loaded successfully!")
print("Dataset Shape:", df.shape)


# ============================================================
# 3. FIRST 5 ROWS
# ============================================================

print("\n===== FIRST 5 ROWS =====")
print(df.head())


# ============================================================
# 4. DATASET INFORMATION
# ============================================================

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== DATASET INFO =====")
df.info()


# ============================================================
# 5. CHECK MISSING VALUES
# ============================================================

print("\n===== MISSING VALUES =====")

print(df.isnull().sum())

print("\n===== TOTAL MISSING VALUES =====")

print(df.isnull().sum().sum())


# ============================================================
# 6. MISSING VALUE PERCENTAGE
# ============================================================

missing_percentage = (
    df.isnull().sum() / len(df)
) * 100

print("\n===== MISSING VALUE PERCENTAGE =====")

print(
    missing_percentage[
        missing_percentage > 0
    ]
)


# ============================================================
# 7. CHECK DUPLICATE ROWS
# ============================================================

print("\n===== DUPLICATE ROWS =====")

duplicate_count = df.duplicated().sum()

print(
    "Number of duplicate rows:",
    duplicate_count
)


# ============================================================
# 8. REMOVE DUPLICATE ROWS
# ============================================================

df = df.drop_duplicates()

print(
    "Shape after removing duplicates:",
    df.shape
)


# ============================================================
# 9. CHECK UNIQUE VALUES
# ============================================================

print("\n===== UNIQUE VALUES =====")

categorical_columns = df.select_dtypes(
    include=["object"]
).columns

for col in categorical_columns:

    print("\n", col)

    print(
        df[col].unique()
    )


# ============================================================
# 10. NUMERICAL SUMMARY
# ============================================================

print("\n===== NUMERICAL SUMMARY =====")

print(
    df.describe()
)


# ============================================================
# 11. CHECK STROKE DISTRIBUTION
# ============================================================

print("\n===== STROKE DISTRIBUTION =====")

print(
    df["stroke"].value_counts()
)


print("\n===== STROKE PERCENTAGE =====")

print(
    df["stroke"].value_counts(
        normalize=True
    ) * 100
)


# ============================================================
# 12. GRAPH 1 - AGE VS STROKE
# ============================================================

age_no_stroke = df.loc[
    df["stroke"] == 0,
    "age"
].dropna()

age_stroke = df.loc[
    df["stroke"] == 1,
    "age"
].dropna()


plt.figure(figsize=(8, 5))

plt.boxplot(
    [
        age_no_stroke,
        age_stroke
    ],
    tick_labels=[
        "No Stroke (0)",
        "Stroke (1)"
    ]
)

plt.xlabel("Stroke")
plt.ylabel("Age")
plt.title("Age vs Stroke")

plt.show()


# ============================================================
# 13. GRAPH 2 - AVERAGE GLUCOSE LEVEL VS STROKE
# ============================================================

glucose_no_stroke = df.loc[
    df["stroke"] == 0,
    "avg_glucose_level"
].dropna()

glucose_stroke = df.loc[
    df["stroke"] == 1,
    "avg_glucose_level"
].dropna()


plt.figure(figsize=(8, 5))

plt.boxplot(
    [
        glucose_no_stroke,
        glucose_stroke
    ],
    tick_labels=[
        "No Stroke (0)",
        "Stroke (1)"
    ]
)

plt.xlabel("Stroke")
plt.ylabel("Average Glucose Level")
plt.title("Average Glucose Level vs Stroke")

plt.show()


# ============================================================
# 14. GRAPH 3 - STROKE DISTRIBUTION
# ============================================================

stroke_counts = (
    df["stroke"]
    .value_counts()
    .sort_index()
)


plt.figure(figsize=(7, 5))

plt.bar(
    [
        "No Stroke (0)",
        "Stroke (1)"
    ],
    [
        stroke_counts.get(0, 0),
        stroke_counts.get(1, 0)
    ]
)

plt.xlabel("Stroke")
plt.ylabel("Number of Patients")
plt.title("Stroke Distribution")

plt.show()


# ============================================================
# 15. GRAPH 4 - BMI VS STROKE
# ============================================================

bmi_no_stroke = df.loc[
    df["stroke"] == 0,
    "bmi"
].dropna()

bmi_stroke = df.loc[
    df["stroke"] == 1,
    "bmi"
].dropna()


plt.figure(figsize=(8, 5))

plt.boxplot(
    [
        bmi_no_stroke,
        bmi_stroke
    ],
    tick_labels=[
        "No Stroke (0)",
        "Stroke (1)"
    ]
)

plt.xlabel("Stroke")
plt.ylabel("BMI")
plt.title("BMI vs Stroke")

plt.show()


# ============================================================
# 16. GRAPH 5 - CATEGORICAL VARIABLE: HYPERTENSION
# ============================================================

hypertension_table = pd.crosstab(
    df["hypertension"],
    df["stroke"]
)


hypertension_table.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.xlabel(
    "Hypertension (0 = No, 1 = Yes)"
)

plt.ylabel(
    "Number of Patients"
)

plt.title(
    "Hypertension vs Stroke"
)

plt.xticks(rotation=0)

plt.legend(
    [
        "No Stroke (0)",
        "Stroke (1)"
    ]
)

plt.show()


# ============================================================
# 17. GRAPH 6 - HEART DISEASE VS STROKE
# ============================================================

heart_table = pd.crosstab(
    df["heart_disease"],
    df["stroke"]
)


heart_table.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.xlabel(
    "Heart Disease (0 = No, 1 = Yes)"
)

plt.ylabel(
    "Number of Patients"
)

plt.title(
    "Heart Disease vs Stroke"
)

plt.xticks(rotation=0)

plt.legend(
    [
        "No Stroke (0)",
        "Stroke (1)"
    ]
)

plt.show()


# ============================================================
# 18. GRAPH 7 - GENDER VS STROKE
# ============================================================

gender_table = pd.crosstab(
    df["gender"],
    df["stroke"]
)


gender_table.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.xlabel("Gender")
plt.ylabel("Number of Patients")
plt.title("Gender vs Stroke")

plt.xticks(rotation=0)

plt.legend(
    [
        "No Stroke (0)",
        "Stroke (1)"
    ]
)

plt.show()


# ============================================================
# 19. GRAPH 8 - SMOKING STATUS VS STROKE
# ============================================================

smoking_table = pd.crosstab(
    df["smoking_status"],
    df["stroke"]
)


smoking_table.plot(
    kind="bar",
    figsize=(9, 5)
)

plt.xlabel("Smoking Status")
plt.ylabel("Number of Patients")
plt.title("Smoking Status vs Stroke")

plt.xticks(rotation=30)

plt.legend(
    [
        "No Stroke (0)",
        "Stroke (1)"
    ]
)

plt.show()


# ============================================================
# 20. GRAPH 9 - WORK TYPE VS STROKE
# ============================================================

work_table = pd.crosstab(
    df["work_type"],
    df["stroke"]
)


work_table.plot(
    kind="bar",
    figsize=(9, 5)
)

plt.xlabel("Work Type")
plt.ylabel("Number of Patients")
plt.title("Work Type vs Stroke")

plt.xticks(rotation=30)

plt.legend(
    [
        "No Stroke (0)",
        "Stroke (1)"
    ]
)

plt.show()


# ============================================================
# 21. CORRELATION MATRIX
# ============================================================

numeric_df = df.select_dtypes(
    include=["int64", "float64"]
)

correlation = numeric_df.corr()


plt.figure(figsize=(10, 8))

plt.imshow(
    correlation,
    interpolation="nearest"
)

plt.colorbar()

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=90
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Correlation Matrix")

plt.tight_layout()

plt.show()


# ============================================================
# 22. DEFINE X AND Y
# ============================================================

y = df["stroke"]


X = df.drop(
    columns=["stroke"]
)


print("\n===== DEPENDENT VARIABLE (Y) =====")

print("stroke")


print("\n===== INDEPENDENT VARIABLES (X) =====")

print(
    X.columns.tolist()
)


# ============================================================
# 23. ENCODE CATEGORICAL VARIABLES
# ============================================================

X = pd.get_dummies(
    X,
    drop_first=True,
    dtype=int
)


print("\n===== X SHAPE AFTER ENCODING =====")

print(
    X.shape
)


print("\n===== ENCODED FEATURES =====")

print(
    X.columns.tolist()
)


# ============================================================
# 24. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


print("\n===== DATA SPLIT =====")

print(
    "Training:",
    X_train.shape
)

print(
    "Validation:",
    X_val.shape
)

print(
    "Testing:",
    X_test.shape
)


# ============================================================
# 25. FEATURE SCALING
# ============================================================

scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train
)


X_val_scaled = scaler.transform(
    X_val
)


X_test_scaled = scaler.transform(
    X_test
)


print("\n===== FEATURE SCALING COMPLETED =====")

print(
    "Training data:",
    X_train_scaled.shape
)

print(
    "Validation data:",
    X_val_scaled.shape
)

print(
    "Testing data:",
    X_test_scaled.shape
)


# ============================================================
# 26. FINAL X AND Y SUMMARY
# ============================================================

print("\n======================================")

print("FINAL DATA PREPARATION COMPLETED")

print("======================================")


print("\nDependent Variable (Y):")

print("stroke")


print("\nIndependent Variables (X):")

for column in X.columns:

    print(column)


print("\nDataset ready for ML model training.")

# ============================================================
# 27. IMPORT ML MODELS
# ============================================================

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 28. CREATE ML MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

    "KNN":
        KNeighborsClassifier(
            n_neighbors=5
        ),

    "SVM":
        SVC(
            probability=True,
            random_state=42
        )
}


# ============================================================
# 29. TRAIN MODELS AND EVALUATE
# ============================================================

results = []


for name, model in models.items():

    print("\n======================================")
    print(name)
    print("======================================")

    # Train model
    model.fit(
        X_train_scaled,
        y_train
    )

    # Prediction
    y_pred = model.predict(
        X_test_scaled
    )

    # Probability for ROC-AUC
    y_prob = model.predict_proba(
        X_test_scaled
    )[:, 1]

    # Evaluation metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    # Store results
    results.append({

        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    })

    # Print results
    print(
        "Accuracy :", round(accuracy, 4)
    )

    print(
        "Precision:", round(precision, 4)
    )

    print(
        "Recall   :", round(recall, 4)
    )

    print(
        "F1-Score :", round(f1, 4)
    )

    print(
        "ROC-AUC  :", round(roc_auc, 4)
    )

    # Classification report
    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    # Confusion Matrix
    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )


# ============================================================
# 30. MODEL COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n======================================")
print("MODEL COMPARISON")
print("======================================")

print(
    results_df
)


# ============================================================
# 31. ROUND RESULTS FOR EASY READING
# ============================================================

results_display = results_df.copy()

metric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "ROC-AUC"
]

results_display[
    metric_columns
] = results_display[
    metric_columns
].round(4)


print("\n===== FINAL MODEL RESULTS =====")

print(
    results_display
)


# ============================================================
# 32. FIND BEST MODEL BASED ON ROC-AUC
# ============================================================

best_model = results_df.loc[
    results_df["ROC-AUC"].idxmax()
]


print("\n======================================")
print("BEST MODEL")
print("======================================")

print(
    "Model:",
    best_model["Model"]
)

print(
    "ROC-AUC:",
    round(best_model["ROC-AUC"], 4)
)

print(
    "Accuracy:",
    round(best_model["Accuracy"], 4)
)

print(
    "Precision:",
    round(best_model["Precision"], 4)
)

print(
    "Recall:",
    round(best_model["Recall"], 4)
)

print(
    "F1-Score:",
    round(best_model["F1-Score"], 4)
)

# ============================================================
# 33. CONFUSION MATRICES FOR ALL 5 MODELS
# ============================================================

for name, model in models.items():

    # Prediction
    y_pred = model.predict(X_test_scaled)

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    plt.figure(figsize=(5, 4))

    plt.imshow(cm)

    plt.title(
        "Confusion Matrix - " + name
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    plt.xticks(
        [0, 1],
        ["No Stroke", "Stroke"]
    )

    plt.yticks(
        [0, 1],
        ["No Stroke", "Stroke"]
    )

    # Add values inside matrix
    for i in range(2):
        for j in range(2):

            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.colorbar()

    plt.tight_layout()

    plt.show()


# ============================================================
# 34. ROC CURVE FOR ALL 5 MODELS
# ============================================================

from sklearn.metrics import roc_curve

plt.figure(figsize=(8, 6))


for name, model in models.items():

    # Probability of stroke
    y_prob = model.predict_proba(
        X_test_scaled
    )[:, 1]

    # False Positive Rate and True Positive Rate
    fpr, tpr, thresholds = roc_curve(
        y_test,
        y_prob
    )

    # AUC
    auc = roc_auc_score(
        y_test,
        y_prob
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {auc:.3f})"
    )


# Random classifier line
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)


plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curves - Stroke Prediction Models"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.show()