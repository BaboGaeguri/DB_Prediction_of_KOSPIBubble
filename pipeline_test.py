"""
pipeline_test.py - KOSPI 버블 예측 파이프라인 (빠른 테스트용)

전체 pipeline을 단일 스크립트로 실행 (PSY_test.py 별도 실행 불필요)
5분 이내 완료를 위한 축소 설정:
  - PSY 데이터: KRX_배당수익률.csv tail(60) 사용
  - Monte Carlo: n_sim=20
  - XGBoost: n_estimators=50

정식 실행은 PSY_test.py (전체 데이터, n_sim=500) + pipeline.py 사용
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool, cpu_count

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
    raise ImportError("pip install xgboost")


# ── PSY 함수 (PSY_test.py와 동일) ────────────────────────────────────────────
def adf_stat(y, lag=0):
    dy = np.diff(y)
    y_lag = y[:-1]
    if lag > 0:
        n = len(dy) - lag
        if n <= lag + 2:
            return np.nan
        X = np.empty((n, lag + 2))
        X[:, 0] = 1.0
        X[:, 1] = y_lag[lag:]
        for i in range(lag):
            X[:, i + 2] = dy[lag - i - 1:len(dy) - i - 1]
        dy = dy[lag:]
    else:
        n = len(dy)
        if n <= 2:
            return np.nan
        X = np.column_stack([np.ones(n), y_lag])
    n, k = X.shape
    if n <= k:
        return np.nan
    XtX = X.T @ X
    Xty = X.T @ dy
    try:
        beta = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        return np.nan
    residuals = dy - X @ beta
    sigma2 = np.dot(residuals, residuals) / (n - k)
    try:
        cov = sigma2 * np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return np.nan
    se = np.sqrt(cov[1, 1])
    return beta[1] / se if se > 0 else np.nan


def bsadf(y, r0=None, lag=0):
    T = len(y)
    if r0 is None:
        r0 = 0.01 + 1.8 / np.sqrt(T)
    r0_obs = int(np.floor(r0 * T))
    adf_stats = []
    for end in range(r0_obs, T):
        sup_adf = -np.inf
        for start in range(0, end - r0_obs + 1):
            y_sub = y[start:end + 1]
            if len(y_sub) > lag + 2:
                stat = adf_stat(y_sub, lag=lag)
                if stat is not None and not np.isnan(stat) and stat > sup_adf:
                    sup_adf = stat
        adf_stats.append(sup_adf)
    return adf_stats


def gsadf(y, r0=None, lag=0):
    T = len(y)
    if r0 is None:
        r0 = 0.01 + 1.8 / np.sqrt(T)
    r0_obs = int(np.floor(r0 * T))
    sup_adf = -np.inf
    for start in range(T - r0_obs + 1):
        for end in range(start + r0_obs, T + 1):
            y_sub = y[start:end]
            if len(y_sub) > lag + 2:
                stat = adf_stat(y_sub, lag=lag)
                if stat is not None and not np.isnan(stat) and stat > sup_adf:
                    sup_adf = stat
    return sup_adf if sup_adf > -np.inf else np.nan


def _sim_worker(args):
    T, r0 = args
    np.random.seed()
    y_sim = np.cumsum(np.random.randn(T))
    return gsadf(y_sim, r0=r0)


def psy_critical_values(T, r0=None, n_sim=20, significance=0.05):
    if r0 is None:
        r0 = 0.01 + 1.8 / np.sqrt(T)
    n_cores = max(1, cpu_count() - 1)
    print(f"  (코어: {n_cores}, 시뮬: {n_sim}회)")
    with Pool(n_cores) as pool:
        results = pool.map(_sim_worker, [(T, r0)] * n_sim)
    gsadf_sim = [r for r in results if r is not None and not np.isnan(r)]
    return np.percentile(gsadf_sim, (1 - significance) * 100)


# ════════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':

    # ── STEP 1. PSY 검정 → 버블 라벨 생성 ────────────────────────────────────
    print("=" * 60)
    print("STEP 1. PSY 검정 (전체 데이터, n_sim=20 경량 Monte Carlo)")
    print("=" * 60)

    krx = pd.read_csv("KRX_배당수익률.csv")
    krx = krx[krx["C1_NM"] == "KOSPI"].copy()
    krx["PRD_DE"] = pd.to_datetime(krx["PRD_DE"], format="%Y%m")
    krx = krx.sort_values("PRD_DE").reset_index(drop=True)

    # 전체 데이터 사용 (n_sim=20으로 속도 확보)

    krx["PD_ratio"] = 100 / krx["DT"]
    y = krx["PD_ratio"].values
    T = len(y)
    r0 = 0.01 + 1.8 / np.sqrt(T)

    print(f"PSY 데이터: {krx['PRD_DE'].min().strftime('%Y-%m')} ~ {krx['PRD_DE'].max().strftime('%Y-%m')} ({T}개월)")
    print(f"r0={r0:.4f}, 최소 윈도우={int(np.floor(r0 * T))}개월\n")

    print("GSADF 계산 중...")
    gsadf_stat = gsadf(y, r0=r0)
    print(f"GSADF 통계량: {gsadf_stat:.4f}")

    print("\n임계값 계산 중 (Monte Carlo)...")
    cv_95 = psy_critical_values(T, r0=r0, n_sim=20, significance=0.05)
    cv_99 = psy_critical_values(T, r0=r0, n_sim=20, significance=0.01)
    print(f"95% CV: {cv_95:.4f}  |  99% CV: {cv_99:.4f}")

    if gsadf_stat > cv_99:
        print(f"=> 1% 유의수준 버블 존재 (강한 증거)")
    elif gsadf_stat > cv_95:
        print(f"=> 5% 유의수준 버블 존재")
    else:
        print(f"=> 버블 존재 증거 없음")

    print("\nBSADF 시계열 계산 중...")
    bsadf_stats = bsadf(y, r0=r0)
    r0_obs = int(np.floor(r0 * T))
    bsadf_dates = krx["PRD_DE"].values[r0_obs:]

    # 버블 라벨 생성 (통계적 CV 기준)
    bsadf_arr = np.array(bsadf_stats)
    bubble_label = (bsadf_arr > cv_95).astype(int)

    # ── 테스트 폴백: n_sim=20의 Monte Carlo CV가 불안정할 경우 ──────────────────
    # GSADF가 CV보다 낮아 버블 미검출 시 → BSADF 상위 20%를 임시 임계값으로 사용
    # (통계적으로 유효하지 않음 — 정식 실행에서는 n_sim=500 PSY_test.py 사용)
    if bubble_label.sum() == 0:
        test_cv = np.percentile(bsadf_arr, 80)  # 상위 20%
        bubble_label = (bsadf_arr > test_cv).astype(int)
        print(f"\n[테스트 모드] PSY 95% CV ({cv_95:.4f})로 버블 미검출")
        print(f"  → 테스트용 BSADF 상위 20% 임계값({test_cv:.4f}) 사용")
        print(f"  → 정식 실행: PSY_test.py (n_sim=500) 결과를 사용하세요\n")

    label_full = np.zeros(T, dtype=int)
    label_full[r0_obs:] = bubble_label
    krx["bubble"] = label_full

    tau = 3
    pd_values = krx["PD_ratio"].values
    bubble_up   = np.zeros(T, dtype=int)
    bubble_down = np.zeros(T, dtype=int)
    for t in range(T):
        if label_full[t] == 1:
            if t + tau < T:
                future_avg = np.mean(pd_values[t + 1:t + tau + 1])
                if future_avg > pd_values[t]:
                    bubble_up[t] = 1
                else:
                    bubble_down[t] = 1
            else:
                bubble_down[t] = 1
    krx["bubble_up"]   = bubble_up
    krx["bubble_down"] = bubble_down

    labels = krx[["PRD_DE", "PD_ratio", "bubble", "bubble_up", "bubble_down"]].copy()
    labels = labels.rename(columns={"PRD_DE": "date"})

    print(f"\nbubble=1:      {label_full.sum()}개월 / {T}개월")
    print(f"bubble_up=1:   {bubble_up.sum()}개월")
    print(f"bubble_down=1: {bubble_down.sum()}개월")

    # ── STEP 2. 특성 행렬 구성 ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2. 특성 행렬 구성")
    print("=" * 60)

    all_data = pd.read_csv("ALL DATA.csv")
    all_data.columns = all_data.columns.str.strip()
    all_data["date"] = pd.to_datetime(all_data["date"].str.strip(), format="%Y/%m")
    if "Bubble" in all_data.columns:
        all_data = all_data.drop(columns=["Bubble"])

    news = pd.read_csv(
        "뉴스심리지수(실험적 통계).csv",
        header=None,
        names=["date", "news_sentiment"]
    )
    news["date"] = pd.to_datetime(news["date"], format="%Y/%m")

    df = labels.merge(all_data, on="date", how="inner")
    df = df.merge(news, on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)

    print(f"merge 후: {len(df)}행  "
          f"({df['date'].min().strftime('%Y-%m')} ~ {df['date'].max().strftime('%Y-%m')})")

    LABEL_COLS = ["bubble", "bubble_up", "bubble_down"]
    MACRO_FEATURES = [
        "Return", "PER", "Dividend Yield", "BaseRate", "M2",
        "GDP G.R", "CPI", "Long-term interest rate",
        "Foreign Net Buy", "Institutional Net buy", "news_sentiment",
    ]
    feature_cols = [c for c in MACRO_FEATURES if c in df.columns]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    print(f"특성: {len(feature_cols)}개 | 유효 시점: {len(df)}개")

    if len(df) < 20:
        print("\n[경고] 머신러닝에 필요한 데이터가 너무 적습니다.")
        print("PSY 검정 데이터 범위와 ALL DATA.csv 날짜 범위가 겹치는지 확인하세요.")
        import sys; sys.exit(1)

    # ── STEP 3. 학습/검증 분할 ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3. 시계열 분할 (70/30)")
    print("=" * 60)

    TRAIN_RATIO = 0.7
    split_idx = int(len(df) * TRAIN_RATIO)
    X     = df[feature_cols].values
    Y     = df[LABEL_COLS].values
    dates = df["date"]

    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    print(f"학습: ~ {dates.iloc[split_idx - 1].strftime('%Y-%m')} ({split_idx}개월) | "
          f"버블비율={Y_train[:, 0].mean() * 100:.1f}%")
    print(f"평가:   {dates.iloc[split_idx].strftime('%Y-%m')} ~ "
          f"{dates.iloc[-1].strftime('%Y-%m')} ({len(df) - split_idx}개월) | "
          f"버블비율={Y_test[:, 0].mean() * 100:.1f}%")

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── STEP 4. 모형 학습 ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4. 모형 학습 (테스트: n_estimators=50)")
    print("=" * 60)

    models = {
        "XGBoost": (
            MultiOutputClassifier(
                xgb.XGBClassifier(
                    n_estimators=50, max_depth=4, learning_rate=0.1,
                    random_state=42, verbosity=0, eval_metric="logloss",
                )
            ),
            False,
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

    # 학습 데이터 클래스 분포 확인
    for i, label in enumerate(LABEL_COLS):
        n_pos = Y_train[:, i].sum()
        if n_pos == 0:
            print(f"  [경고] '{label}' 학습 데이터에 양성(1) 샘플 없음 → "
                  "PSY 기간을 늘려야 신뢰할 수 있는 결과를 얻을 수 있습니다.")

    preds  = {}
    probas = {}
    for name, (model, use_scaled) in models.items():
        print(f"  [{name}] 학습 중...")
        Xtr = X_train_sc if use_scaled else X_train
        Xte = X_test_sc  if use_scaled else X_test
        try:
            model.fit(Xtr, Y_train)
        except ValueError as e:
            print(f"    [스킵] {name} 학습 실패 (단일 클래스 문제): {e}")
            # 학습 실패 시 모두 0 예측으로 대체
            preds[name]  = np.zeros((len(Xte), len(LABEL_COLS)), dtype=int)
            probas[name] = np.zeros((len(Xte), len(LABEL_COLS)))
            continue
        preds[name] = model.predict(Xte)
        raw_proba   = model.predict_proba(Xte)
        proba_mat   = np.zeros((len(Xte), len(LABEL_COLS)))
        for j, p_arr in enumerate(raw_proba):
            proba_mat[:, j] = p_arr[:, 1] if p_arr.shape[1] >= 2 else p_arr[:, 0]
        probas[name] = proba_mat

    # ── STEP 5. 성능 평가 ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5. 성능 평가 (out-of-sample)")
    print("=" * 60)

    def evaluate(y_true_all, y_pred_all, y_prob_all, label_names):
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

    print("\n[bubble 이진 분류]")
    print(results[["bubble_PRAUC", "bubble_F1", "bubble_BA"]].round(4).to_string())

    print("\n[bubble_up / bubble_down]")
    up_down_cols = [c for c in results.columns if "bubble_up" in c or "bubble_down" in c]
    if up_down_cols:
        print(results[up_down_cols].round(4).to_string())

    print("\n[전체 Multilabel F1-macro]")
    print(results[["F1_macro"]].round(4).to_string())

    results.to_csv("model_results_test.csv", encoding="utf-8-sig")
    print("\n결과 저장: model_results_test.csv")

    # ── STEP 6. 시각화 ────────────────────────────────────────────────────────
    test_dates = dates.iloc[split_idx:].values

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for i, label in enumerate(LABEL_COLS):
        axes[i].plot(test_dates, Y_test[:, i],           'k-',  lw=1.5, label="실제")
        axes[i].plot(test_dates, preds["XGBoost"][:, i], 'r--', lw=1.0, alpha=0.8,
                     label="XGBoost 예측")
        axes[i].set_title(f"{label}  (평가 구간, 테스트 모드)")
        axes[i].set_ylabel("Label")
        axes[i].legend(loc="upper left", fontsize=8)
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("pipeline_result_test.png", dpi=150)
    print("시각화 저장: pipeline_result_test.png")
    print("\n[완료] 정식 실행은 PSY_test.py + pipeline.py 사용")
