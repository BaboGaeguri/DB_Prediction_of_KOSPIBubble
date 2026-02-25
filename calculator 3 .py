import pandas as pd

# CSV 불러오기
df = pd.read_csv("ALL DATA.csv")

# date 제거 (상관계수 계산에 필요 없음)
df = df.drop(columns=['date '])

# 상관계수 계산
corr_matrix = df.corr()

# 출력
print(corr_matrix)

# CSV로 저장
corr_matrix.to_csv("correlation_matrix.csv")