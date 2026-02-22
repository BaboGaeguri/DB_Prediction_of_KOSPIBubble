# 3.2 실험 설계 및 프레임워크 구성

본 연구는 PSY 절차를 통해 정의된 코스피 버블 상태를 머신러닝 분류모형으로 식별하는 프레임워크를 구축한다. 전체 분석 절차는 다음 네 단계로 구성된다.

1. 버블 라벨 생성
2. 분류용 데이터셋 구성
3. 학습·검증 전략 설정
4. 기준모형과의 성능 비교

---

## 3.2.1 버블 라벨 생성

Phillips, Shi, and Yu(2015)의 PSY 절차를 적용하여 코스피 지수의 버블 형성 및 붕괴 시점을 식별한다. PSY 검정 결과에 따라 각 시점 $t$에 대해 버블 상태 여부를 이진 변수로 정의한 뒤, 버블 구간 내에서 방향성을 추가로 분류하는 다중라벨(multilabel) 구조를 적용한다.

### 이진 분류 (bubble)

- 버블 구간: $bubble_t = 1$
- 비버블 구간: $bubble_t = 0$

### 다중라벨 분류 (bubble_up / bubble_down)

Atsiwo(2025)의 방법론을 참고하여, 버블로 식별된 구간($bubble_t = 1$)을 자산가격 상승기(bubble up)와 붕괴기(bubble down)로 세분화한다.

$$\text{bubble\_up}_t = \begin{cases} 1, & \text{if } bubble_t = 1 \text{ and } \frac{Y_{t+1} + \cdots + Y_{t+\tau}}{\tau} > Y_t \\ 0, & \text{otherwise} \end{cases}$$

$$\text{bubble\_down}_t = \begin{cases} 1, & \text{if } bubble_t = 1 \text{ and } \frac{Y_{t+1} + \cdots + Y_{t+\tau}}{\tau} \leq Y_t \\ 0, & \text{otherwise} \end{cases}$$

여기서 $\tau$는 향후 관측 윈도우 크기(기본값: 3개월), $Y_t$는 시점 $t$의 P/D ratio이다. 한 시점이 bubble이면서 동시에 bubble_up 또는 bubble_down 중 하나로 분류되므로 multilabel 구조에 해당한다.

### 구현 세부사항

버블 라벨은 BSADF 시계열과 Monte Carlo 시뮬레이션 임계값을 비교하여 생성한다.

1. 각 시점 $t$에서 BSADF 통계량을 계산
2. Monte Carlo 시뮬레이션으로 임계값(95% CV) 산출
3. $BSADF_t > CV_{95}$ 이면 $bubble_t = 1$, 아니면 $bubble_t = 0$
4. $bubble_t = 1$인 시점에서 향후 $\tau$개월 평균 P/D ratio와 현재값을 비교하여 bubble_up / bubble_down 분류
5. 생성된 라벨(bubble, bubble_up, bubble_down)을 CSV로 저장하여 이후 분류 모델의 입력으로 사용

### 성능 최적화

Monte Carlo 시뮬레이션은 계산 비용이 크므로 다음 두 가지 최적화를 적용한다.

- **NumPy 직접 OLS**: `statsmodels.OLS` 대신 $\hat{\beta} = (X^\top X)^{-1} X^\top y$ 공식을 NumPy로 직접 계산하여 객체 생성 오버헤드를 제거 (약 5~10배 속도 향상)
- **멀티프로세싱 병렬화**: Monte Carlo 시뮬레이션 $n_{sim}$회는 각 시뮬레이션이 독립적이므로 `multiprocessing.Pool`로 CPU 코어 수에 비례하여 병렬 실행 (Windows 환경에서 `if __name__ == '__main__':` 가드 필수)

### ⚠️ PSY 검정 파라미터 이슈 (미해결)

테스트 실행 결과, KOSPI P/D ratio 2004~2025 전체 데이터(T=264)에서 PSY 검정이 버블을 검출하지 못하는 문제가 확인되었다.

**원인**: PSY 권장 공식 $r_0 = 0.01 + 1.8/\sqrt{T}$를 따를 경우 T=264에서 최소 윈도우 $r_{0,obs} = 31$개월로 설정된다. 이로 인해 31개월보다 짧은 버블 에피소드가 평균에 희석되어 검출되지 않는다.

**실측값** (T=264, $n_{sim}$=20):
- GSADF = 0.9257
- CV_95 ≈ 1.83 → 버블 미검출

**T=60 (2021~2025)으로 단독 실행 시**: GSADF=1.95 > CV_95=1.58 → 버블 1개월 검출
→ 동일 데이터라도 T에 따라 $r_{0,obs}$가 달라져 검출 여부가 바뀌는 구조적 문제

**검토 필요 사항** (연구팀 결정):
1. $r_{0,obs}$를 고정값(예: 18개월)으로 설정
2. 검정 변수를 log(P/D ratio) 또는 KOSPI 지수 수준으로 변경
3. 대안적 버블 식별 방법론 검토

---

## 3.2.2 분류용 데이터셋 구성

설명변수 행렬 $X_t$는 시점 $t$까지 관측 가능한 거시·금융 변수들로 구성한다. 모든 설명변수는 시점 정렬을 엄격히 유지하며, 타깃 변수 $y_t$와 동일한 시점을 기준으로 정렬한다.

### 데이터 구성 및 파일 흐름

```
KRX_배당수익률.csv (2004/01~2025/12, 264개월)
  → PSY_test.py → bubble_labels.csv (bubble / bubble_up / bubble_down)
                         ↓
ALL DATA.csv (2005/01~2025/12, 252개월)  ─┐
뉴스심리지수(실험적 통계).csv (2005/01~2025/12) ─┤→ pipeline.py → model_results.csv
bubble_labels.csv                          ─┘
```

### 최종 특성 행렬 (X) — 11개 변수

| 변수 | 출처 | 분류 |
|------|------|------|
| Return | ALL DATA.csv | 시장 수익률 |
| PER | ALL DATA.csv | 밸류에이션 |
| Dividend Yield | ALL DATA.csv | 밸류에이션 (PBR 대체) |
| BaseRate | ALL DATA.csv | 거시경제 |
| M2 | ALL DATA.csv | 거시경제 |
| GDP G.R | ALL DATA.csv | 거시경제 (분기→월 반복) |
| CPI | ALL DATA.csv | 거시경제 |
| Long-term interest rate | ALL DATA.csv | 거시경제 |
| Foreign Net Buy | ALL DATA.csv | 투자자 수급 |
| Institutional Net buy | ALL DATA.csv | 투자자 수급 |
| 뉴스심리지수 | 뉴스심리지수(실험적 통계).csv | 감성 변수 |

### 라벨 (Y) — bubble_labels.csv에서 merge

| 라벨 | 설명 |
|------|------|
| `bubble` | 이진 분류 (PSY 검정 결과) |
| `bubble_up` | 버블 구간 中 상승기 |
| `bubble_down` | 버블 구간 中 붕괴기 |

최종 데이터셋 구조: $(X_t, y_t)$ — **"시점 $t$의 정보로 시점 $t$의 버블 상태를 식별하는"** 분류 문제

---

## 3.2.3 학습 및 검증 전략

시계열 데이터의 특성을 고려하여 무작위 분할 대신 **시간 순서를 유지한 분할 방식**을 적용한다.

- 전체 표본: 2005/01~2025/12 (252개월, inner join 기준)
- 학습 구간 (앞 70%): 2005/01~2019/08 (176개월)
- 평가 구간 (뒤 30%): 2019/09~2025/12 (76개월)

---

## 3.2.4 예측 모형 및 비교 모형

| 모형 | 설명 |
|------|------|
| **XGBoost** (주요 모형) | 비선형 관계 및 변수 간 상호작용을 학습하는 그래디언트 부스팅 기반 앙상블 모형 |
| 단순 기준모형 | 항상 비버블(0)을 예측 |
| 로지스틱 회귀 | 전통적 통계모형 비교용 |

---

## 3.2.5 평가 지표

버블 구간은 전체 기간 중 일부에만 나타나는 사건일 가능성이 높으므로, 단순 정확도 대신 불균형 분류에 적합한 지표를 활용한다.

| 지표 | 용도 |
|------|------|
| **PR-AUC** | 불균형 데이터에서의 정밀도-재현율 성능 |
| **F1-macro** | 클래스 균형 고려한 F1 점수 |
| **Balanced Accuracy** | 클래스별 정확도의 평균 |

모형의 성능은 평가 구간(out-of-sample)에서 산출된 **분류 결과**를 기준으로 비교하며, 이를 통해 가설 1(H1)을 검정한다.

---

## 3.2.6 테스트 실행 결과 (pipeline_test.py, 2026-02-22)

PSY 버블 미검출로 인해 BSADF 상위 20% 임계값(테스트용 폴백)을 적용한 결과:

- 데이터: 2005/01~2025/12 (252개월), 학습 176개월 / 평가 76개월
- 버블 비율: 학습 19.3% / 평가 17.1% (테스트 폴백 기준)

| 모형 | bubble PR-AUC | bubble F1 | Balanced Acc | F1-macro |
|------|--------------|-----------|--------------|---------|
| **XGBoost** | **0.4218** | **0.5000** | **0.7204** | **0.2692** |
| Logistic Regression | 0.3252 | 0.2973 | 0.5263 | 0.2675 |
| Dummy (Always 0) | 0.1711 | 0.0000 | 0.5000 | 0.0000 |

XGBoost > Logistic Regression > Dummy 순서로 성능 확인. **단, PSY 버블 라벨 이슈 해결 후 재실행 필요.**

---

## 스크립트 구성

| 파일 | 역할 | 상태 |
|------|------|------|
| `PSY_test.py` | 전체 데이터 PSY 검정 → `bubble_labels.csv` 생성 | 테스트 모드 해제 완료 (n_sim=500) |
| `pipeline.py` | 정식 파이프라인 (bubble_labels.csv + ALL DATA + 뉴스심리지수 → 모델 학습) | 완성 |
| `pipeline_test.py` | 단일 스크립트 테스트 (PSY 내장, n_sim=20) | 완성 |

**실행 순서 (정식)**: `python PSY_test.py` → `python pipeline.py`
**실행 순서 (테스트)**: `python pipeline_test.py` (약 1~2분 소요)
