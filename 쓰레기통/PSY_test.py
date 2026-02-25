import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count


# ── NumPy 기반 고속 OLS (statsmodels 대체) ───────────────────────────────────
def adf_stat(y, lag=0):
    """
    ADF 통계량 계산 - NumPy 직접 OLS (statsmodels 대비 5~10배 빠름)
    H1: 폭발적 루트 존재 (우측 검정)
    """
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

    # β = (XᵀX)⁻¹Xᵀy
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
    """
    BSADF (Backward Supremum ADF) 시계열 계산
    각 시점 t에서 가능한 모든 시작점에 대해 ADF를 계산하고 supremum 반환
    """
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
    """
    GSADF (Generalized Supremum ADF) 통계량 계산
    Phillips, Shi, Yu (2015) - 모든 가능한 윈도우의 ADF 최댓값
    """
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


# ── Monte Carlo 병렬화 ────────────────────────────────────────────────────────
def _sim_worker(args):
    """각 프로세스에서 독립적으로 실행되는 단일 시뮬레이션"""
    T, r0 = args
    np.random.seed()  # 프로세스마다 다른 시드 보장
    y_sim = np.cumsum(np.random.randn(T))
    return gsadf(y_sim, r0=r0)


def psy_critical_values(T, r0=None, n_sim=1000, significance=0.05):
    """
    Monte Carlo 시뮬레이션으로 GSADF 임계값 계산
    멀티프로세싱으로 병렬 실행 (CPU 코어 수배 속도 향상)
    """
    if r0 is None:
        r0 = 0.01 + 1.8 / np.sqrt(T)

    n_cores = max(1, cpu_count() - 1)  # 코어 하나 여유
    print(f"  (사용 코어 수: {n_cores}, 시뮬레이션: {n_sim}회)")

    args_list = [(T, r0)] * n_sim
    with Pool(n_cores) as pool:
        results = pool.map(_sim_worker, args_list)

    gsadf_sim = [r for r in results if r is not None and not np.isnan(r)]
    return np.percentile(gsadf_sim, (1 - significance) * 100)


# ── 메인 실행 (Windows 멀티프로세싱 필수 가드) ───────────────────────────────
if __name__ == '__main__':

    # 데이터 로드
    df = pd.read_csv("KRX_배당수익률.csv")
    df = df[df["C1_NM"] == "KOSPI"].copy()
    df["PRD_DE"] = pd.to_datetime(df["PRD_DE"], format="%Y%m")
    df = df.sort_values("PRD_DE").reset_index(drop=True)

    # P/D ratio 계산 (DT는 D/P in %, 역수 취함)
    df["PD_ratio"] = 100 / df["DT"]

    print(f"데이터 기간: {df['PRD_DE'].min()} ~ {df['PRD_DE'].max()}")
    print(f"총 관측치: {len(df)}개\n")
    print(df[["PRD_DE", "DT", "PD_ratio"]].head(10))

    y = df["PD_ratio"].values
    T = len(y)
    r0 = 0.01 + 1.8 / np.sqrt(T)

    print("\n" + "="*50)
    print("PSY Test (Phillips, Shi, Yu 2015)")
    print("="*50)
    print(f"\n최소 윈도우 비율 (r0): {r0:.4f}")
    print(f"최소 윈도우 크기: {int(np.floor(r0 * T))}개월")

    # GSADF 통계량 계산
    print("\nGSADF 통계량 계산 중...")
    gsadf_stat = gsadf(y, r0=r0)
    print(f"GSADF 통계량: {gsadf_stat:.4f}")

    # 임계값 계산 (병렬 Monte Carlo)
    print("\n임계값 계산 중 (Monte Carlo, 병렬)...")
    cv_95 = psy_critical_values(T, r0=r0, n_sim=500, significance=0.05)
    cv_99 = psy_critical_values(T, r0=r0, n_sim=500, significance=0.01)

    print(f"\n95% 임계값: {cv_95:.4f}")
    print(f"99% 임계값: {cv_99:.4f}")

    # 버블 판정
    print("\n" + "="*50)
    print("결과")
    print("="*50)
    if gsadf_stat > cv_99:
        print(f"GSADF ({gsadf_stat:.4f}) > 99% CV ({cv_99:.4f})")
        print("=> 1% 유의수준에서 버블 존재 (강한 증거)")
    elif gsadf_stat > cv_95:
        print(f"GSADF ({gsadf_stat:.4f}) > 95% CV ({cv_95:.4f})")
        print("=> 5% 유의수준에서 버블 존재")
    else:
        print(f"GSADF ({gsadf_stat:.4f}) <= 95% CV ({cv_95:.4f})")
        print("=> 버블 존재 증거 없음")

    # BSADF 시계열 계산
    print("\nBSADF 시계열 계산 중...")
    bsadf_stats = bsadf(y, r0=r0)
    r0_obs = int(np.floor(r0 * T))
    bsadf_dates = df["PRD_DE"].values[r0_obs:]

    # ── 버블 라벨 생성 ────────────────────────────────────────────────────────

    # 이진 라벨: BSADF > 95% CV → bubble=1
    bubble_label = (np.array(bsadf_stats) > cv_95).astype(int)
    label_full = np.zeros(T, dtype=int)
    label_full[r0_obs:] = bubble_label
    df["bubble"] = label_full

    # 다중라벨: bubble_up / bubble_down (Atsiwo 2025, Eq.2-3)
    tau = 3  # 향후 관측 윈도우 (개월)
    pd_values = df["PD_ratio"].values
    bubble_up = np.zeros(T, dtype=int)
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
                # 향후 데이터 부족 → bubble_down으로 처리
                bubble_down[t] = 1

    df["bubble_up"] = bubble_up
    df["bubble_down"] = bubble_down

    df[["PRD_DE", "PD_ratio", "bubble", "bubble_up", "bubble_down"]].to_csv(
        "bubble_labels.csv", index=False, encoding="utf-8-sig"
    )
    print(f"\n버블 라벨 저장 완료:")
    print(f"  bubble=1:      {label_full.sum()}개월 / 전체 {T}개월")
    print(f"  bubble_up=1:   {bubble_up.sum()}개월")
    print(f"  bubble_down=1: {bubble_down.sum()}개월")

    # ── 결과 시각화 ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(df["PRD_DE"], df["PD_ratio"], 'b-', label="P/D Ratio")
    axes[0].set_title("KOSPI P/D Ratio")
    axes[0].set_ylabel("P/D Ratio")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(bsadf_dates, bsadf_stats, 'b-', label="BSADF")
    axes[1].axhline(y=cv_95, color='r', linestyle='--', label=f"95% CV ({cv_95:.2f})")
    axes[1].axhline(y=cv_99, color='darkred', linestyle='--', label=f"99% CV ({cv_99:.2f})")
    axes[1].fill_between(bsadf_dates, cv_95, np.array(bsadf_stats),
                         where=np.array(bsadf_stats) > cv_95,
                         color='red', alpha=0.3, label="Bubble Period")
    axes[1].set_title("BSADF Statistics (Bubble Detection)")
    axes[1].set_ylabel("BSADF Statistic")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("PSY_test_result.png", dpi=150)
    print("\n결과가 'PSY_test_result.png'로 저장되었습니다.")
    plt.show()
