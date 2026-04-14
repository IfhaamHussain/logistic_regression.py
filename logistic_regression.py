# STEP 1: Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score
# STEP 2: Load dataset
data = load_breast_cancer()
# Convert into DataFrame 
df = pd.DataFrame(data.data, columns=data.feature_names)
# Add target column (0 = malignant, 1 = benign)
df['target'] = data.target
df['target'] = data.target
print("First 5 rows of dataset:")
print(df.head())
# STEP 3: Separate features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']
# STEP 4: Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\nTraining data size:", X_train.shape)
print("Testing data size:", X_test.shape)
# STEP 5: Scaleing the data 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# STEP 6: Create and training the model
model = LogisticRegression()
model.fit(X_train, y_train)
print("\nModel training completed!")
# STEP 7: Makeing predictions
y_pred = model.predict(X_test)
# STEP 8: Evaluate the model
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))
# STEP 9:Check probabilities
y_prob = model.predict_proba(X_test)
print("\nFirst 5 probability predictions:")
print(y_prob[:5])
print("\nPrecision:")
print(precision_score(y_test, y_pred))
print("\nRecall:")
print(recall_score(y_test, y_pred))
print("\nROC-AUC Score:")
print(roc_auc_score(y_test, y_prob[:, 1]))
# STEP 10: Tune decision threshold
print("\n--- Tuning Decision Threshold ---")
thresholds = [0.3, 0.5, 0.7]
for threshold in thresholds:
    y_pred_tuned = (y_prob[:, 1] >= threshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred_tuned)
    precision = precision_score(y_test, y_pred_tuned)
    recall = recall_score(y_test, y_pred_tuned)
    print(f"\nThreshold: {threshold}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")

# STEP 11: Explain Sigmoid Function
print("\n--- Sigmoid Function Explanation ---")
print("Sigmoid: σ(z) = 1 / (1 + e^(-z))")
print("Maps any input to probability range [0, 1]")
print("Default threshold = 0.5")
print("If probability >= threshold: predict 1 (benign)")
print("If probability < threshold: predict 0 (malignant)")
print("\nLower threshold → Higher recall, lower precision")
print("Higher threshold → Higher precision, lower recall")