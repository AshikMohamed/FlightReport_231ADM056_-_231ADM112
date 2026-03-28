import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_curve,
    precision_score, recall_score
)

print("🚀 Starting model training...")

# STEP 1: LOAD DATA (keep CSV in same folder)
arrivals = pd.read_csv("arrivals.csv")
departures = pd.read_csv("depature.csv")

df_flight = pd.concat([arrivals, departures], ignore_index=True)

print("✅ Data loaded:", df_flight.shape)

# STEP 2: CLEAN DATA
df_flight.columns = df_flight.columns.str.strip()

df_flight['Scheduled departure time'] = pd.to_datetime(
    df_flight['Scheduled departure time'],
    format='%H:%M',
    errors='coerce'
)

df_flight['Dep_Time_Minutes'] = (
    df_flight['Scheduled departure time'].dt.hour * 60 +
    df_flight['Scheduled departure time'].dt.minute
)

df_flight['Arrival Delay (Minutes)'] = pd.to_numeric(
    df_flight['Arrival Delay (Minutes)'],
    errors='coerce'
)

df_flight['Dep_Time_Minutes'] = df_flight['Dep_Time_Minutes'].fillna(0)
df_flight['Arrival Delay (Minutes)'] = df_flight['Arrival Delay (Minutes)'].fillna(0)

# STEP 3: TARGET
df_flight['Delayed'] = (df_flight['Arrival Delay (Minutes)'] > 0).astype(int)

print("✅ Target created")

# STEP 4: FEATURE ENGINEERING
df_flight['Date (MM/DD/YYYY)'] = pd.to_datetime(
    df_flight['Date (MM/DD/YYYY)'],
    errors='coerce'
)

df_flight['DayOfWeek'] = df_flight['Date (MM/DD/YYYY)'].dt.dayofweek
df_flight['DayOfWeek'] = df_flight['DayOfWeek'].fillna(0)

df_flight['Dep_Hour'] = df_flight['Dep_Time_Minutes'] // 60

df_flight['Time_Bucket'] = pd.cut(
    df_flight['Dep_Hour'],
    bins=[0, 6, 12, 18, 24],
    labels=['Night', 'Morning', 'Afternoon', 'Evening']
)

df_flight = pd.get_dummies(
    df_flight,
    columns=['Carrier Code', 'Origin Airport', 'Destination Airport', 'Time_Bucket'],
    drop_first=True
)

print("✅ Feature engineering done")

# STEP 5: FEATURES
features = [col for col in df_flight.columns if col not in [
    'Delayed',
    'Arrival Delay (Minutes)',
    'Scheduled departure time',
    'Actual departure time',
    'Scheduled Arrival Time',
    'Actual Arrival Time',
    'Date (MM/DD/YYYY)',
    'Status',
    'Flight Number',
    'Tail Number'
]]

X = df_flight[features].fillna(0)
y = df_flight['Delayed']

print("✅ Total features:", len(features))

# STEP 6: SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# STEP 7: MODEL
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced'
)

rf_model.fit(X_train, y_train)

print("✅ Model trained")

# STEP 8: PREDICTION
rf_prob = rf_model.predict_proba(X_test)[:, 1]

threshold = 0.6
rf_predictions = (rf_prob > threshold).astype(int)

# STEP 9: EVALUATION
print("\n===== MODEL PERFORMANCE =====")
print(confusion_matrix(y_test, rf_predictions))
print(classification_report(y_test, rf_predictions))
print("Accuracy:", accuracy_score(y_test, rf_predictions))

# STEP 10: SAVE MODEL (IMPORTANT — BEFORE PLOTS)
print("📁 Saving model in:", os.getcwd())

joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(X.columns.tolist(), "features.pkl")

print("✅ Model & features saved successfully!")

# STEP 11: FEATURE IMPORTANCE (SAVE IMAGE)
importances = rf_model.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)

plt.figure()
feat_imp.head(10).plot(kind='barh')
plt.title("Top 10 Feature Importance")
plt.savefig("feature_importance.png")
plt.close()

# STEP 12: ROC CURVE (SAVE IMAGE)
fpr, tpr, _ = roc_curve(y_test, rf_prob)

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.savefig("roc_curve.png")
plt.close()

print("📊 Graphs saved as images!")

# STEP 13: TEST FUNCTION
def predict_flight(dep_time, day):
    data = pd.DataFrame({
        'Dep_Time_Minutes': [dep_time],
        'DayOfWeek': [day]
    })

    data['Dep_Hour'] = dep_time // 60
    data = data.reindex(columns=X.columns, fill_value=0)

    pred = rf_model.predict(data)[0]
    return "Delayed" if pred == 1 else "On Time"


print("🔍 Example Prediction:", predict_flight(800, 2))