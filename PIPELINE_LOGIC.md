# KOSPI 버블 예측 파이프라인 작동 로직

## 1. 전체 구조

파이프라인은 두 가지 실행 방식이 있다.

```
[정식 실행]                          [테스트 실행]
PSY_test.py                          pipeline_test.py
    ↓ bubble_labels.csv                  (PSY + ML 통합)
pipeline.py                              ↓
    ↓                                model_results_test.csv
model_results.csv                    pipeline_result_test.png
pipeline_result.png
```

---

## 2. 데이터 흐름

```
입력 파일
├── ALL DATA.csv             → PER (PSY 검정용) + 거시·수급 변수 10개 (특성 행렬)
└── 뉴스심리지수(실험적 통계).csv → 뉴스심리지수 (특성 행렬)

STEP 1: PSY 검정
    ALL DATA.csv → PER 시계열
        → BSADF 시계열 계산
        → Monte Carlo 임계값(CV_95 또는 CV_99/CV_90) 비교
        → bubble / bubble_up / bubble_down 라벨 생성

STEP 2: 특성 행렬 구성
    bubble 라벨 + ALL DATA + 뉴스심리지수
        → 날짜 기준 inner join
        → 결측치 제거
        → 최종 데이터셋 (252개월, 2005/01~2025/12)

STEP 3: 학습/평가 분할
    앞 70% → 학습 (2005/01~2019/08, 176개월)
    뒤 30% → 평가 (2019/09~2025/12, 76개월)

STEP 4: 모형 학습
    XGBoost / Logistic Regression / Dummy

STEP 5: 성능 평가
    PR-AUC / F1 / Balanced Accuracy

출력 파일
├── model_results.csv        → 성능 지표 테이블
└── pipeline_result.png      → 예측 vs 실제 시각화
```

---

## 3. STEP별 상세 로직

### STEP 1. PSY 검정 → 버블 라벨 생성

#### 1-1. 데이터 준비
```
ALL DATA.csv
    → PER(주가수익비율) 직접 사용 (버블 시 폭발적 상승 경향)
    → T = 252개월 (2005/01~2025/12)

[변경 이유] D.Y.csv 배당수익률 기반 P/D ratio는 배당금이 주가와 함께 오르면
            수익률이 안정적으로 유지되어 PSY 폭발적 성장 감지 불가.
            PER은 주가 버블 시 이익 대비 주가가 빠르게 오르므로 PSY에 적합.
```

#### 1-2. GSADF 통계량 계산
```
모든 가능한 부분 구간 [start, end]에 대해 ADF 검정 수행
    → 각 구간의 ADF t-통계량 중 최대값 = GSADF
    → 조건: 구간 길이 ≥ r0_obs (최소 윈도우)

r0 설정: r0 = 15 / T  →  r0_obs = 15개월 고정
[참고] PSY 권장 공식(r0 = 0.01 + 1.8/sqrt(T))은 T=252에서 r0_obs=29개월을 산출하여
       KOSPI 중단기 버블 에피소드 검출에 부적합. 15개월 고정으로 대체.
```

#### 1-3. Monte Carlo 임계값 계산
```
랜덤워크(귀무가설) 시뮬레이션 n_sim회 반복
    → 각 시뮬레이션에서 GSADF 계산
    → 상위 5% 분위수 = CV_95 (95% 임계값)
    → 병렬처리: multiprocessing.Pool (CPU 코어 수 - 1)
```

#### 1-4. BSADF 시계열 계산 (버블 날짜 특정용)
```
각 시점 t에서:
    BSADF_t = sup{ADF(start→t)} for all start in [0, t-r0_obs]
    (끝점 t를 고정하고, 시작점을 변화시키며 최대 ADF 탐색)

결과: 길이 (T - r0_obs)의 시계열
→ BSADF_t > CV_95 이면 bubble_t = 1
```

#### 1-5. 버블 라벨 생성

**bubble** (이진 분류):
```
bubble_t = 1  if BSADF_t > CV_95
bubble_t = 0  otherwise
```

**bubble_up / bubble_down** (방향 분류, τ=3개월):
```
bubble_t = 1인 시점에서:
    future_avg = mean(PD_ratio[t+1 : t+τ+1])

    bubble_up_t   = 1  if future_avg > PD_ratio[t]  (향후 P/D ratio 상승 예상)
    bubble_down_t = 1  if future_avg ≤ PD_ratio[t]  (향후 P/D ratio 하락 예상)
```

> **테스트 폴백**: GSADF < CV_95로 버블 미검출 시 BSADF 상위 20%를 임시 임계값으로 사용
> (통계적으로 유효하지 않음 — 파이프라인 동작 확인 목적)

---

### STEP 2. 특성 행렬 구성

#### 사용 변수 (11개)

| 변수명 | 출처 | 분류 |
|--------|------|------|
| Return | ALL DATA.csv | 시장 수익률 |
| PER | ALL DATA.csv | 밸류에이션 |
| Dividend Yield | ALL DATA.csv | 밸류에이션 |
| BaseRate | ALL DATA.csv | 거시경제 |
| M2 | ALL DATA.csv | 거시경제 |
| GDP G.R | ALL DATA.csv | 거시경제 |
| CPI | ALL DATA.csv | 거시경제 |
| Long-term interest rate | ALL DATA.csv | 거시경제 |
| Foreign Net Buy | ALL DATA.csv | 수급 |
| Institutional Net buy | ALL DATA.csv | 수급 |
| news_sentiment | 뉴스심리지수.csv | 감성 |

#### Merge 로직
```
labels (bubble 라벨, date 기준)
    inner join ALL DATA.csv (date 기준)
    inner join 뉴스심리지수.csv (date 기준)
    → 세 파일의 교집합 기간만 사용
    → 결측치(NaN) 행 제거
    → 최종: 252개월 (2005/01~2025/12)
```

---

### STEP 3. 학습 / 평가 분할

```
시간 순서 유지 (무작위 분할 금지)

전체 252개월
    앞 70% (176개월): 학습  → 2005/01 ~ 2019/08
    뒤 30%  (76개월): 평가  → 2019/09 ~ 2025/12

정규화:
    StandardScaler → 학습 데이터로 fit, 평가 데이터에 transform
    XGBoost: 원본 스케일 사용 (정규화 불필요)
    Logistic Regression: 정규화 적용
```

---

### STEP 3-2. XGBoost 하이퍼파라미터 튜닝

```
RandomizedSearchCV + TimeSeriesSplit(n_splits=3)

탐색 공간:
    max_depth:        [2, 3, 4]
    n_estimators:     [100, 200, 300]
    learning_rate:    [0.05, 0.1]
    scale_pos_weight: [spw×0.5, spw, spw×1.5]
        (spw = (1 - 버블비율) / 버블비율  ← 클래스 불균형 보정)

n_iter=12 (무작위 12가지 조합 탐색)
scoring: F1-macro (make_scorer)
→ 최적 파라미터로 새 XGBoost 생성 후 STEP 4에서 학습
```

---

### STEP 4. 모형 학습

#### MultiOutputClassifier 구조
```
각 라벨(bubble, bubble_up, bubble_down)에 대해
독립적인 이진 분류기를 학습하는 Wrapper

→ 라벨 3개 × 분류기 1개 = 분류기 3개가 내부적으로 학습
→ 예측 시 3개 라벨 동시 출력
```

#### 모형별 설정

| 모형 | 주요 파라미터 | 정규화 |
|------|-------------|--------|
| XGBoost (Tuned) | RandomizedSearchCV로 최적화 (max_depth/n_estimators/lr/scale_pos_weight) | 미적용 |
| Logistic Regression | max_iter=1000 | 적용 |
| Dummy (Always 0) | strategy="most_frequent" | 무관 |

---

### STEP 5. 성능 평가

평가 구간(뒤 30%)의 실제값과 예측값 비교

#### 평가 지표

| 지표 | 계산 방식 | 선택 이유 |
|------|----------|---------|
| **PR-AUC** | Precision-Recall 곡선 아래 면적 | 불균형 데이터에서 정밀도-재현율 균형 |
| **F1 (binary)** | 2×P×R / (P+R) | 버블 클래스 감지 능력 |
| **Balanced Accuracy** | (TPR + TNR) / 2 | 클래스별 정확도 평균 |
| **F1-macro** | 라벨별 F1 평균 | 전체 multilabel 성능 |

#### 기준 모형 (Dummy)과 비교

```
Dummy (Always 0) 기준값:
    PR-AUC ≈ 버블 비율 (랜덤 기댓값)
    F1 = 0 (버블을 한 번도 예측 안 함)
    Balanced Accuracy = 0.5 (정의에 의해)

→ 머신러닝 모형이 이 기준값을 유의미하게 초과해야 H1 지지
```

---

## 4. 두 스크립트 차이점

| 항목 | pipeline_test.py | pipeline.py |
|------|-----------------|-------------|
| PSY 포함 여부 | 포함 (내장) | 포함 (내장, n_sim=500) |
| 데이터 | ALL DATA.csv → PER (T=252) | ALL DATA.csv → PER (T=252) |
| n_sim | 20 (경량) | 500 (정식) |
| XGBoost 튜닝 | RandomizedSearchCV (n_iter=12) | RandomizedSearchCV (n_iter=12) |
| 버블 미검출 | 폴백 로직 있음 (BSADF 상위 20%) | 없음 (RuntimeError 발생) |
| 출력 파일 | model_results_test.csv | model_results.csv |
| 소요 시간 | ~1~2분 | ~15~20분 |

---

## 5. 핵심 PSY 함수 설명

### adf_stat(y, lag=0)
```
입력: 시계열 y, lag 수
출력: ADF t-통계량 (우측 검정)

내부 로직:
    Δy_t = μ + ρ·y_{t-1} + Σφ·Δy_{t-j} + ε_t 추정
    H0: ρ = 0 (단위근)
    H1: ρ > 0 (폭발적 성장)
    NumPy 직접 OLS (statsmodels 미사용 → 5~10배 빠름)
```

### bsadf(y, r0)
```
입력: 전체 시계열, 최소 윈도우 비율 r0
출력: 길이 (T - r0_obs)의 BSADF 시계열

각 시점 t에서:
    for start in [0, t - r0_obs]:
        ADF(y[start:t+1]) 계산
    BSADF_t = max(ADF) over all start
```

### gsadf(y, r0)
```
입력: 전체 시계열, 최소 윈도우 비율 r0
출력: 스칼라 (전체 기간의 최대 BSADF)

GSADF = max(BSADF_t) for t in [r0_obs, T]
→ 전역적 버블 존재 여부 판단에 사용
```

### psy_critical_values(T, r0, n_sim)
```
입력: 데이터 길이 T, 최소 윈도우 r0, 시뮬레이션 횟수
출력: 95% (또는 99%) 임계값

귀무가설(랜덤워크) 하의 분포 추정:
    for i in range(n_sim):
        y_sim = cumsum(randn(T))  # 랜덤워크 생성
        gsadf_sim[i] = gsadf(y_sim, r0)
    CV_95 = percentile(gsadf_sim, 95)

병렬처리: multiprocessing.Pool (Windows: if __name__ == '__main__' 필수)
```

---

## 6. r0_obs 파라미터 설정 근거

### PSY 권장 공식 대비 고정값 선택 이유
```
PSY 권장 공식: r0 = 0.01 + 1.8 / sqrt(T)
  T=252 → r0_obs = 29개월
  → 코스피 중단기 버블(수개월~수년)보다 최소 윈도우가 커서 검출 불가

채택값: r0 = 15 / T  →  r0_obs = 15개월 고정
  근거:
    - 통계적 하한: ADF 신뢰성을 위해 최소 ~15개월 관측 필요
    - 검출 상한:  목표 버블(중단기, 36~60개월)의 절반 이하로 설정
    - 15개월 < 36개월(최단 목표 버블)  →  두 조건 동시 충족
    - Mulenga & Ji(2022) 등 선행 연구에서 PSY 공식 대신 고정 r0 사용 사례 존재
```
