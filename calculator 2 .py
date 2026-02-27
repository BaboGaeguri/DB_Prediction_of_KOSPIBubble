import pandas as pd
import numpy as np

# ==============================
# 1. 데이터 불러오기
# ==============================

df = pd.read_csv("ALLDATA.csv")   # ← 파일명 수정

# 컬럼 공백 제거 (혹시 모를 오류 방지)
df.columns = df.columns.str.strip()

# 날짜 정렬 (첫 번째 컬럼이 날짜라고 가정)
date_col = df.columns[0]
df[date_col] = pd.to_datetime(df[date_col], format='%Y/%m')
df = df.sort_values(date_col)

# ==============================
# 2. 분석할 변수 선택
# ==============================

cols = ['Return','PER','PBR','BaseRate','M2',
        'GDP G.R','CPI','Long-term interest rate',
        'Foreign Net Buy','Institutional Net buy','Sentiment','log(kospi)']

df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

# ==============================
# 3. 기초통계량 계산
# ==============================

desc = df[cols].describe().T
desc['Skewness'] = df[cols].skew()
desc['Kurtosis'] = df[cols].kurtosis()
desc['N'] = df[cols].count()

table2 = desc[['N','mean','std','min','max','Skewness','Kurtosis']]
table2.columns = ['N','Mean','Std','Min','Max','Skewness','Kurtosis']

# ==============================
# 4. CSV 파일로 저장
# ==============================

table2.to_csv("Table2_Descriptive_Statistics.csv", encoding='utf-8-sig')

print("✅ CSV 파일 저장 완료: Table2_Descriptive_Statistics.csv")