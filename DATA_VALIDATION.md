# DATA_VALIDATION.md
# 데이터 정합성 점검 및 파이프라인 흐름 확정

---

## 1. 파일 현황

| 파일 | 기간 | 관측치 | 상태 |
|------|------|--------|------|
| `KRX_배당수익률.csv` | 2004/01 ~ 2025/12 | 264개월 | ✅ 존재 |
| `ALL DATA.csv` | 2005/01 ~ 2025/12 | 252개월 | ✅ 존재 |
| `뉴스심리지수(실험적 통계).csv` | 2005/01 ~ 2025/12 | 252개월 | ✅ 존재 |
| `데이터 및 변수 (Data and Variables).docx` | — | — | ✅ 존재 |
| `bubble_labels.csv` | PSY_test.py 출력물 | — | ⏳ 미생성 (PSY_test.py 실행 필요) |

---

## 2. 파일 간 맥락 정합성 확인

### 2.1 docx ↔ ALL DATA.csv 변수 대응

| docx 원안 변수 | ALL DATA.csv 실제 컬럼 | 상태 |
|---------------|----------------------|------|
| KOSPI Return | `Return` | ✅ |
| PER | `PER` | ✅ |
| **PBR** | **`Dividend Yield`** | ⚠️ **대체됨** (친구 확인: PBR 수집 어려움 → 배당수익률로 교체) |
| Base Rate | `BaseRate` | ✅ |
| M2 | `M2` | ✅ |
| GDP Growth Rate | `GDP G.R` | ✅ (단, 아래 주의사항 참고) |
| CPI | `CPI` | ✅ |
| Long-term Interest Rate | `Long-term interest rate` | ✅ |
| Foreign Net Buy | `Foreign Net Buy` | ✅ |
| Institutional Net Buy | `Institutional Net buy` | ✅ |
| (해당 없음) | `Bubble` | ⏳ **공란** — PSY_test.py 결과로 채워야 함 |

### 2.2 뉴스심리지수(실험적 통계).csv

- **출처**: 정부 기관 제공 (한국언론진흥재단 또는 한국은행 추정) 실험적 통계
- **형식**: 날짜(YYYY/MM), 지수값 (열 이름 없음)
- **기준**: 100 = 중립, 100 초과 = 긍정, 100 미만 = 부정
- **역할**: Atsiwo(2025)의 finbert-lc 감성 점수에 대응하는 한국판 감성 변수
- **기간**: 2005/01 ~ 2025/12 → ALL DATA.csv와 완전 일치 ✅

### 2.3 KRX_배당수익률.csv ↔ ALL DATA.csv 배당수익률 관계

- KRX 파일: PSY test 입력용 (P/D ratio 계산 원천, 2004/01부터 시작)
- ALL DATA의 `Dividend Yield`: 동일한 KOSPI 배당수익률 (2005/01부터)
- **KRX 데이터가 1년 더 앞서는 이유**: PSY 검정의 최소 윈도우(r0) 확보를 위한 warm-up 기간
- **정합성**: 두 파일의 배당수익률 값이 동일한 출처에서 왔는지 교차 검증 필요 (아래 주의사항 참고)

---

## 3. 알려진 데이터 처리 이슈

| 이슈 | 내용 | 처리 방식 | 영향 |
|------|------|-----------|------|
| **PBR → Dividend Yield 대체** | PBR 월별 데이터 수집 어려움 | `Dividend Yield` 컬럼으로 대체 | 연구 설계 문서(docx) 업데이트 필요 |
| **GDP 분기→월 변환** | GDP 성장률은 분기 발표 | 해당 분기 값을 월별로 반복 입력 | 정보 손실 없음, 일반적 관행 |
| **Bubble 컬럼 공란** | ALL DATA.csv의 `Bubble` 열이 비어있음 | PSY_test.py 실행 후 bubble_labels.csv로 대체 | pipeline.py에서 별도 merge |

---

## 4. 파이프라인 흐름 확정

```
[데이터 수집]
  KRX_배당수익률.csv          ALL DATA.csv          뉴스심리지수(실험적 통계).csv
  (KOSPI 배당수익률, 264개월)  (전체 변수, 252개월)   (감성지수, 252개월)
         │                          │                          │
         ▼                          │                          │
  [PSY_test.py]                     │                          │
  - P/D ratio 계산                  │                          │
  - BSADF 검정                      │                          │
  - Monte Carlo 임계값               │                          │
  - bubble/bubble_up/bubble_down 생성│                          │
         │                          │                          │
         ▼                          ▼                          ▼
  bubble_labels.csv ──────► [pipeline.py] ◄─────────────────────
  (date, PD_ratio,            - 날짜 기준 merge (2005/01~2025/12)
   bubble, bubble_up,         - 특성 행렬 X 구성
   bubble_down)               - 라벨 Y (bubble/bubble_up/bubble_down)
                              - 시계열 70/30 분할
                              - XGBoost / LogReg / Dummy 학습
                              - PR-AUC / F1-macro / Balanced Accuracy 평가
                                     │
                                     ▼
                              model_results.csv
                              pipeline_result.png
```

---

## 5. pipeline.py 수정 필요 사항

현재 `pipeline.py`는 `bubble_labels.csv`에서 P/D ratio 파생 특성만 생성합니다.
**`ALL DATA.csv`와 `뉴스심리지수(실험적 통계).csv`를 실제 특성 행렬로 사용하도록 업데이트가 필요합니다.**

### 최종 특성 행렬 (X)

| 변수 | 출처 | 분류 |
|------|------|------|
| Return | ALL DATA.csv | 시장 밸류에이션 |
| PER | ALL DATA.csv | 시장 밸류에이션 |
| Dividend Yield | ALL DATA.csv | 시장 밸류에이션 (PBR 대체) |
| BaseRate | ALL DATA.csv | 거시경제 |
| M2 | ALL DATA.csv | 거시경제 |
| GDP G.R | ALL DATA.csv | 거시경제 |
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

---

## 6. 실행 전 체크리스트

- [ ] PSY_test.py의 테스트 모드 해제 (`df.tail(50)` 줄 제거, `n_sim` 증가)
- [ ] `python PSY_test.py` 실행 → `bubble_labels.csv` 생성 확인
- [ ] `bubble_labels.csv` 날짜 범위가 2005/01~2025/12 포함하는지 확인
- [ ] `python pipeline.py` 실행
- [ ] `model_results.csv` 출력 확인

---

## 7. 주의사항 및 향후 과제

1. **배당수익률 교차 검증**: `KRX_배당수익률.csv`의 DT 값과 `ALL DATA.csv`의 `Dividend Yield` 값이 동일한지 샘플 시점에서 수동 확인 권장
2. **뉴스심리지수 열 이름 없음**: CSV 파일에 헤더가 없으므로 pipeline.py에서 `names=['date', 'news_sentiment']` 지정 필요
3. **GDP 분기→월 처리**: 분기 값이 3개월 반복 입력된 구조이므로, 일부 변동성 분석 시 주의
4. **PBR 대체 처리**: docx 연구 설계 문서에서 PBR → Dividend Yield 대체 내용을 반영하여 논문 기술 업데이트 필요
