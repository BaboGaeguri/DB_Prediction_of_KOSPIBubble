import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 데이터 로드
df = pd.read_csv("KRX_배당수익률.csv")

# KOSPI 데이터만 필터링 (필요시)
df = df[df["C1_NM"] == "KOSPI"].copy()

# 날짜 정렬
df["PRD_DE"] = pd.to_datetime(df["PRD_DE"], format="%Y%m")
df = df.sort_values("PRD_DE").reset_index(drop=True)

# 테스트용: 최근 50개월만 사용 (전체 데이터 사용 시 아래 줄 제거)
df = df.tail(50).reset_index(drop=True)

# P/D ratio 계산 (DT는 D/P in %, 역수 취함)
df["PD_ratio"] = 100 / df["DT"]

print(f"데이터 기간: {df['PRD_DE'].min()} ~ {df['PRD_DE'].max()}")
print(f"총 관측치: {len(df)}개\n")
print(df[["PRD_DE", "DT", "PD_ratio"]].head(10))


def adf_stat(y, lag=0):
    """
    ADF 통계량 계산 (no constant, no trend)
    H0: unit root (비정상), H1: stationary
    """
    T = len(y)
    dy = np.diff(y)
    y_lag = y[:-1]

    # lag 차분 추가
    if lag > 0:
        X = np.column_stack([y_lag[lag:]] + [dy[lag-i-1:-i-1] if i < lag-1 else dy[:-1][lag-1:] for i in range(lag)])
        dy = dy[lag:]
    else:
        X = y_lag.reshape(-1, 1)

    # OLS 회귀
    X = add_constant(X)
    model = OLS(dy, X).fit()

    # t-statistic for rho (첫 번째 계수)
    t_stat = model.tvalues[1]
    return t_stat


def bsadf(y, r0=None, lag=0):
    """
    BSADF (Backward Supremum ADF) 통계량 계산
    각 시점 t에서 [r0*T, t]까지의 윈도우로 ADF 테스트 수행 후 supremum 반환
    """
    T = len(y)
    if r0 is None:
        r0 = 0.01 + 1.8 / np.sqrt(T)
    r0_obs = int(np.floor(r0 * T))

    adf_stats = []
    for r1 in range(r0_obs, T):
        # r1 시점에서 backward로 ADF 계산
        sup_adf = -np.inf
        for r2 in range(0, r1 - r0_obs + 1):
            y_sub = y[r2:r1+1]
            if len(y_sub) > lag + 2:
                try:
                    stat = adf_stat(y_sub, lag=lag)
                    if stat > sup_adf:
                        sup_adf = stat
                except:
                    pass
        adf_stats.append(sup_adf)

    return adf_stats


def gsadf(y, r0=None, lag=0):
    """
    GSADF (Generalized Supremum ADF) 통계량 계산
    Phillips, Shi, Yu (2015) 방법론
    """
    T = len(y)
    if r0 is None:
        r0 = 0.01 + 1.8 / np.sqrt(T)

    r0_obs = int(np.floor(r0 * T))

    # 모든 가능한 윈도우에서 ADF 계산
    all_adf = []
    for r2 in range(T - r0_obs + 1):
        for r1 in range(r2 + r0_obs, T + 1):
            y_sub = y[r2:r1]
            if len(y_sub) > lag + 2:
                try:
                    stat = adf_stat(y_sub, lag=lag)
                    all_adf.append(stat)
                except:
                    pass

    return np.max(all_adf) if all_adf else np.nan


def psy_critical_values(T, r0=None, n_sim=1000, significance=0.05):
    """
    Monte Carlo 시뮬레이션으로 GSADF 임계값 계산
    """
    if r0 is None:
        r0 = 0.01 + 1.8 / np.sqrt(T)

    gsadf_sim = []
    for _ in range(n_sim):
        # 랜덤워크 생성 (H0: unit root)
        y_sim = np.cumsum(np.random.randn(T))
        gsadf_stat = gsadf(y_sim, r0=r0)
        if not np.isnan(gsadf_stat):
            gsadf_sim.append(gsadf_stat)

    cv = np.percentile(gsadf_sim, (1 - significance) * 100)
    return cv


# PSY Test 실행
print("\n" + "="*50)
print("PSY Test (Phillips, Shi, Yu 2015)")
print("="*50)

y = df["PD_ratio"].values
T = len(y)

# 최소 윈도우 크기 설정
r0 = 0.01 + 1.8 / np.sqrt(T)
print(f"\n최소 윈도우 비율 (r0): {r0:.4f}")
print(f"최소 윈도우 크기: {int(np.floor(r0 * T))}개월")

# GSADF 통계량 계산
print("\nGSADF 통계량 계산 중...")
gsadf_stat = gsadf(y, r0=r0)
print(f"GSADF 통계량: {gsadf_stat:.4f}")

# 임계값 계산 (Monte Carlo)
print("\n임계값 계산 중 (Monte Carlo 시뮬레이션)...")
cv_95 = psy_critical_values(T, r0=r0, n_sim=10, significance=0.05)
cv_99 = psy_critical_values(T, r0=r0, n_sim=10, significance=0.01)

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

# BSADF 시계열 계산 (버블 기간 탐지용)
print("\nBSADF 시계열 계산 중...")
bsadf_stats = bsadf(y, r0=r0)
bsadf_dates = df["PRD_DE"].values[int(np.floor(r0 * T)):]

# 버블 라벨 생성: BSADF > 95% CV이면 1, 아니면 0
bubble_label = (np.array(bsadf_stats) > cv_95).astype(int)

# 전체 시점에 맞춰 라벨 구성 (BSADF가 없는 초기 구간은 0으로 설정)
r0_obs = int(np.floor(r0 * T))
label_full = np.zeros(T, dtype=int)
label_full[r0_obs:] = bubble_label

df["bubble"] = label_full
df[["PRD_DE", "PD_ratio", "bubble"]].to_csv("bubble_labels.csv", index=False, encoding="utf-8-sig")
print(f"\n버블 라벨 저장 완료: bubble=1 구간 {bubble_label.sum()}개월 / 전체 {T}개월")

# 결과 시각화
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# P/D ratio 시계열
axes[0].plot(df["PRD_DE"], df["PD_ratio"], 'b-', label="P/D Ratio")
axes[0].set_title("KOSPI P/D Ratio")
axes[0].set_ylabel("P/D Ratio")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# BSADF 통계량 & 임계값
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

print("\n결과가 'PSY_test_result.png'로 저장되었습니다.")
