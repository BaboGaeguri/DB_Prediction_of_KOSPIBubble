import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

print("Loading data...")

data = pd.read_csv("test_data.csv")

X = data[['KOSPI_return', 'PER', 'foreign_net_buy', 'base_rate']]
y = data['bubble']

split_idx = int(len(data) * 0.7)

X_train = X.iloc[:split_idx]
X_test  = X.iloc[split_idx:]

y_train = y.iloc[:split_idx]
y_test  = y.iloc[split_idx:]

print("Training model...")

model = XGBClassifier(
    n_estimators=20,
    max_depth=3,
    learning_rate=0.1,
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train, y_train)

print("Predicting...")

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Predictions:", y_pred)

