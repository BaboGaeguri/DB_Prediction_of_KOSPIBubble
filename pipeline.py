"""
pipeline.py - KOSPI 버블 예측 파이프라인

Atsiwo(2025) 3단계 프레임워크를 KOSPI 데이터에 적용
  Step 1: PSY 버블 라벨 로드  (bubble_labels.csv ← PSY_test.py 출력물)
  Step 2: 특성 행렬 구성      (ALL DATA.csv + 뉴스심리지수(실험적 통계).csv merge)
  Step 3: 모형 학습 및 평가   (XGBoost / Logistic Regression / Dummy)

실행 순서:
  1. PSY_test.py 실행 (test mode 해제 후) → bubble_labels.csv 생성
  2. python pipeline.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, average_precision_score
)

try:
    import xgboost as xgb
except ImportError:
    raise ImportError("XGBoost가 설치되어 있지 않습니다.\n  pip install xgboost")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1. 버블 라벨 로드 (PSY_test.py 출력물)
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1. 버블 라벨 로드")
print("=" * 60)

try:
    labels = pd.read_csv("bubble_labels.csv")
    labels["PRD_DE"] = pd.to_datetime(labels["PRD_DE"])
except FileNotFoundError:
    raise FileNotFoundError(
        "bubble_labels.csv 없음.\n"
        "PSY_test.py의 테스트 모드(df.tail(50)) 줄을 제거한 뒤 먼저 실행하세요."
    )

# 데이터 크기 경고
MIN_OBS = 60
if len(labels) < MIN_OBS:
    print(f"\n[경고] 데이터가 {len(labels)}개로 너무 적습니다 (권장: {MIN_OBS}개 이상).")
    print("PSY_test.py의 df.tail(50) 줄을 제거하고 전체 데이터로 재실행하세요.\n")

n_bubble = labels["bubble"].sum()
print(f"기간:          {labels['PRD_DE'].min().strftime('%Y-%m')} ~ {labels['PRD_DE'].max().strftime('%Y-%m')}")
print(f"전체 시점:     {len(labels)}개")
print(f"bubble=1:      {n_bubble}개 ({n_bubble / len(labels) * 100:.1f}%)")
print(f"bubble_up=1:   {labels['bubble_up'].sum()}개")
print(f"bubble_down=1: {labels['bubble_down'].sum()}개")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2. 특성 행렬 구성
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2. 특성 행렬 구성")
print("=" * 60)

# ── 2-1. ALL DATA.csv 로드 ────────────────────────────────────────────────────
all_data = pd.read_csv("ALL DATA.csv")
# 컬럼명 앞뒤 공백 제거
all_data.columns = all_data.columns.str.strip()
all_data["date"] = pd.to_datetime(all_data["date"].str.strip(), format="%Y/%m")
# Bubble 컬럼 제거 (빈 열 — PSY 결과로 대체)
if "Bubble" in all_data.columns:
    all_data = all_data.drop(columns=["Bubble"])
print(f"ALL DATA.csv: {len(all_data)}행, {len(all_data.columns)}컬럼")
print(f"  기간: {all_data['date'].min().strftime('%Y-%m')} ~ {all_data['date'].max().strftime('%Y-%m')}")
print(f"  컬럼: {list(all_data.columns)}")

# ── 2-2. 뉴스심리지수 로드 ───────────────────────────────────────────────────
news = pd.read_csv(
    "뉴스심리지수(실험적 통계).csv",
    header=None,
    names=["date", "news_sentiment"]
)
news["date"] = pd.to_datetime(news["date"], format="%Y/%m")
print(f"\n뉴스심리지수: {len(news)}행")
print(f"  기간: {news['date'].min().strftime('%Y-%m')} ~ {news['date'].max().strftime('%Y-%m')}")

# ── 2-3. 날짜 기준 merge ─────────────────────────────────────────────────────
# bubble_labels는 PRD_DE 컬럼, all_data/news는 date 컬럼
labels_renamed = labels.rename(columns={"PRD_DE": "date"})

df = labels_renamed.merge(all_data, on="date", how="inner")
df = df.merge(news, on="date", how="inner")
df = df.sort_values("date").reset_index(drop=True)

print(f"\nmerge 후: {len(df)}행")
print(f"  기간: {df['date'].min().strftime('%Y-%m')} ~ {df['date'].max().strftime('%Y-%m')}")

# ── 2-4. 특성 목록 확정 & 결측치 제거 ────────────────────────────────────────
LABEL_COLS   = ["bubble", "bubble_up", "bubble_down"]
EXCLUDE_COLS = ["date", "PD_ratio"] + LABEL_COLS

# ALL DATA.csv 특성 + 뉴스심리지수
MACRO_FEATURES = [
    "Return",
    "PER",
    "Dividend Yield",
    "BaseRate",
    "M2",
    "GDP G.R",
    "CPI",
    "Long-term interest rate",
    "Foreign Net Buy",
    "Institutional Net buy",
    "news_sentiment",
]

# 실제 존재하는 컬럼만 선택
feature_cols = [c for c in MACRO_FEATURES if c in df.columns]
missing_features = [c for c in MACRO_FEATURES if c not in df.columns]
if missing_features:
    print(f"\n[경고] 누락된 특성 컬럼: {missing_features}")

df = df.dropna(subset=feature_cols).reset_index(drop=True)

print(f"\n특성 수: {len(feature_cols)}개")
print(f"유효 시점: {len(df)}개  "
      f"({df['date'].min().strftime('%Y-%m')} ~ {df['date'].max().strftime('%Y-%m')})")
print(f"특성 목록: {feature_cols}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3-1. 시계열 기반 학습/검증 분할
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3. 시계열 기반 학습/검증 분할 (시간 순서 유지)")
print("=" * 60)

TRAIN_RATIO = 0.7  # 앞 70% 학습 / 뒤 30% 평가

split_idx = int(len(df) * TRAIN_RATIO)
X      = df[feature_cols].values
Y      = df[LABEL_COLS].values          # shape: (T, 3)
dates  = df["date"]

X_train, X_test = X[:split_idx], X[split_idx:]
Y_train, Y_test = Y[:split_idx], Y[split_idx:]

print(f"학습 구간: ~ {dates.iloc[split_idx - 1].strftime('%Y-%m')}  ({split_idx}개월)")
print(f"평가 구간:   {dates.iloc[split_idx].strftime('%Y-%m')} ~ "
      f"{dates.iloc[-1].strftime('%Y-%m')}  ({len(df) - split_idx}개월)")
print(f"학습 버블 비율: {Y_train[:, 0].mean() * 100:.1f}%")
print(f"평가 버블 비율: {Y_test[:, 0].mean() * 100:.1f}%")

# 정규화 (Logistic Regression용; XGBoost는 불필요하나 인터페이스 통일)
scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train)
X_test_sc   = scaler.transform(X_test)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3-2. 모형 정의
# ═══════════════════════════════════════════════════════════════════════════════
# MultiOutputClassifier: 각 라벨(bubble, bubble_up, bubble_down)에
# 독립적인 이진 분류기를 학습 — multilabel 구조 구현

models = {
    "XGBoost": (
        MultiOutputClassifier(
            xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                verbosity=0,
                eval_metric="logloss",
            )
        ),
        False,   # scaled 여부 (XGBoost는 원본 스케일 사용)
    ),
    "Logistic Regression": (
        MultiOutputClassifier(
            LogisticRegression(max_iter=1000, random_state=42)
        ),
        True,
    ),
    "Dummy (Always 0)": (
        MultiOutputClassifier(
            DummyClassifier(strategy="most_frequent", random_state=42)
        ),
        True,
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3-3. 학습 및 예측
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3-3. 모형 학습")
print("=" * 60)

preds = {}
probas = {}

for name, (model, use_scaled) in models.items():
    print(f"  [{name}] 학습 중...")
    Xtr = X_train_sc if use_scaled else X_train
    Xte = X_test_sc  if use_scaled else X_test

    model.fit(Xtr, Y_train)
    preds[name] = model.predict(Xte)

    # predict_proba → list of (n_samples, n_classes) per label
    raw_proba = model.predict_proba(Xte)
    proba_mat = np.zeros((len(Xte), len(LABEL_COLS)))
    for j, p_arr in enumerate(raw_proba):
        proba_mat[:, j] = p_arr[:, 1] if p_arr.shape[1] >= 2 else p_arr[:, 0]
    probas[name] = proba_mat


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4. 성능 평가 (out-of-sample)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4. 성능 평가 (평가 구간 기준)")
print("=" * 60)


def evaluate(y_true_all, y_pred_all, y_prob_all, label_names):
    """라벨별 PR-AUC / F1 / Balanced Accuracy를 딕셔너리로 반환"""
    row = {}
    for i, label in enumerate(label_names):
        yt  = y_true_all[:, i]
        yp  = y_pred_all[:, i]
        ypr = y_prob_all[:, i]
        row[f"{label}_PRAUC"] = (
            average_precision_score(yt, ypr)
            if len(np.unique(yt)) >= 2 else np.nan
        )
        row[f"{label}_F1"] = f1_score(yt, yp, average="binary", zero_division=0)
        row[f"{label}_BA"] = balanced_accuracy_score(yt, yp)
    row["F1_macro"] = f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)
    return row


records = []
for name in models:
    row = {"Model": name}
    row.update(evaluate(Y_test, preds[name], probas[name], LABEL_COLS))
    records.append(row)

results = pd.DataFrame(records).set_index("Model")

# 출력
print("\n[bubble 이진 분류]")
print(results[["bubble_PRAUC", "bubble_F1", "bubble_BA"]].round(4).to_string())

print("\n[bubble_up / bubble_down]")
up_down_cols = [c for c in results.columns if "bubble_up" in c or "bubble_down" in c]
print(results[up_down_cols].round(4).to_string())

print("\n[전체 Multilabel F1-macro]")
print(results[["F1_macro"]].round(4).to_string())

results.to_csv("model_results.csv", encoding="utf-8-sig")
print("\n결과 저장: model_results.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5. 시각화
# ═══════════════════════════════════════════════════════════════════════════════
test_dates = dates.iloc[split_idx:].values

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
for i, label in enumerate(LABEL_COLS):
    axes[i].plot(test_dates, Y_test[:, i],           'k-',  lw=1.5, label="실제")
    axes[i].plot(test_dates, preds["XGBoost"][:, i], 'r--', lw=1.0, alpha=0.8, label="XGBoost 예측")
    axes[i].set_title(f"{label}  (평가 구간)")
    axes[i].set_ylabel("Label")
    axes[i].legend(loc="upper left", fontsize=8)
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pipeline_result.png", dpi=150)
print("시각화 저장: pipeline_result.png")
