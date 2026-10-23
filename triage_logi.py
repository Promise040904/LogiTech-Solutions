import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# 1. LOAD THE DATA

FILE = "synthetic_logistics_delay_triage.csv"   # keep this file in the same folder as triage.py
df = pd.read_csv(FILE)

print("Rows and columns:", df.shape)
print("\nMissing values per column:\n", df.isna().sum())
print("\nDelay labels:\n", df["delay_label"].value_counts())


# 2. PICK INPUTS (X) AND TARGET (y)

target = "delay_label"
X = df.drop(columns=["shipment_id", "delay_risk_score", "recommended_action", target])
y = df[target]

text_cols = ["transport_mode", "region", "weather"]
num_cols = [c for c in X.columns if c not in text_cols]

([
    ("text", OneHotEncoder(handle_unknown="ignore"), text_cols),
    ("num", StandardScaler(), num_cols),
])


# 3. SPLIT 80% TRAIN / 20% TEST

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTrain size:", len(X_train), " Test size:", len(X_test))

