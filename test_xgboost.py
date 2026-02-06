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
    #결정 나무 20개, 우리는 200~500 생각중, 과적합 주의
    max_depth=3,
    #학계에서는 3~4 권장, 너무 깊으면 과적합 우려
    learning_rate=0.1,
    #0.1 빠르지만 불안정, 0.01 느리자만 안정적
    eval_metric='logloss',
    #얼마나 확신했는가를 평가
    random_state=42
    #결과 변경 방지, 출발점 고정
)

model.fit(X_train, y_train)

print("Predicting...")

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Predictions:", y_pred)

