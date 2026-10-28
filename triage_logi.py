import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Load Logistics Dataset
df = pd.read_csv('synthetic_logistics_delay_triage.csv')

# 2. Feature & Target Separation
# Exclude shipment_id, target label, and directly correlated action column
X = df.drop(columns=['shipment_id', 'delay_label', 'recommended_action'])
y = df['delay_label']

# 3. Define Preprocessing Pipelines
num_cols = ['distance_km', 'cargo_weight_kg', 'origin_congestion', 
            'dest_congestion', 'carrier_reliability', 'scheduled_hours', 'delay_risk_score']
cat_cols = ['transport_mode', 'region', 'weather']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ]
)

# 4. Train-Test Split (80/20 Stratified Split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 5. Build & Train Random Forest Pipeline
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
])

rf_pipeline.fit(X_train, y_train)

# 6. Make Predictions & Evaluate
y_pred = rf_pipeline.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
cv_scores = cross_val_score(rf_pipeline, X_train, y_train, cv=5)

print("="*60)
print(f"PRACTICAL MODEL EVALUATION REPORT")
print("="*60)
print(f"Test Set Accuracy:           {test_acc * 100:.2f}%")
print(f"5-Fold CV Mean Accuracy:     {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 7. Generate Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred, labels=['on_time', 'minor_delay', 'major_delay', 'critical_delay'])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['On Time', 'Minor', 'Major', 'Critical'],
            yticklabels=['On Time', 'Minor', 'Major', 'Critical'])
plt.title('Logistics Delay Triage - Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()




