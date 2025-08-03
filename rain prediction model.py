import pandas as pd


'''from google.colab import files

# This will pop up a file‑chooser dialog
uploaded = files.upload()'''

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from xgboost import XGBRegressor
df = pd.read_excel("/content/2020.csv.xlsx")
df['time'] = pd.to_datetime(df['time'])
df = df[df['time'] >= '2021-01-01']


df['prcp'] = df['prcp'].dropna()
df['date'] = df['time'].dt.date

daily = df.groupby('date').agg({
    'prcp': 'sum',
    'temp': 'mean',
    'dwpt': 'mean',
    'rhum': 'mean',
    'pres': 'mean'
}).reset_index()
daily['temp'] = (daily['temp']>24.6).astype(int)
daily['Rain'] = (daily['prcp'] > 0.2).astype(int)

daily['day_of_year'] = pd.to_datetime(daily['date']).dt.dayofyear
daily['roll3_prcp'] = daily['prcp'].rolling(window=3, min_periods=1).sum()

print(f"Count of Rainy Days (2021+): {daily['Rain'].sum()}")
print(f"Count of Non-Rainy Days (2021+): {len(daily) - daily['Rain'].sum()}")
print(f"total precipitaion of rainy days(2021+):{daily['prcp'].sum()} mm")


features = ['temp', 'dwpt', 'rhum', 'pres', 'day_of_year', 'roll3_prcp']
X = daily[features]
y = daily['Rain']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


j=RandomForestClassifier(n_estimators=100,n_jobs=4)
j.fit(X_train,y_train)
y_pred = j.predict(X_test)
y_proba = j.predict_proba(X_test)[:, 1]

y_pred = j.predict(X_test)
acc = accuracy_score(y_test, y_pred)

roc = roc_auc_score(y_test, y_pred)

print(f"Test Accuracy: {acc:.5f}")
print(f"ROC AUC Score: {roc:.5f}")



sample = X_test.copy()
sample['Actual'] = y_test

sample['Predicted'] = y_pred
sample['Probability'] = y_proba * 100
print("\nSample Predictions:\n", sample[['Actual', 'Predicted', 'Probability']])