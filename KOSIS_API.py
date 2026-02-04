import json
import os
import pandas as pd
from urllib.request import urlopen
from urllib.parse import urlencode
from dotenv import load_dotenv

# .env 파일에서 API 키 불러오기
load_dotenv()
API_KEY = os.getenv("KOSIS_KEY")

# KRX 데이터 가져오기
ORG_ID = "343"                    # 기관ID (한국거래소)
TBL_ID = "DT_343_2010_S0032"      # 통계표ID

params = {
    "method": "getList",
    "apiKey": API_KEY,
    "itmId": "13103792819T1+",
    "objL1": "ALL",
    "objL2": "",
    "objL3": "",
    "objL4": "",
    "objL5": "",
    "objL6": "",
    "objL7": "",
    "objL8": "",
    "format": "json",
    "jsonVD": "Y",
    "prdSe": "M",
    "newEstPrdCnt": "1000",
    "orgId": ORG_ID,
    "tblId": TBL_ID,
}

base_url = "https://kosis.kr/openapi/Param/statisticsParameterData.do"
data_url = f"{base_url}?{urlencode(params)}"

print("=== KRX 데이터 가져오는 중... ===\n")

with urlopen(data_url) as response:
    data = json.loads(response.read().decode('utf-8'))

# 결과 확인
if isinstance(data, list) and len(data) > 0:
    df = pd.DataFrame(data)
    print(f"총 {len(df)}개 데이터 로드 완료\n")
    print("=== 컬럼 목록 ===")
    print(df.columns.tolist())
    print("\n=== 데이터 미리보기 ===")
    print(df.info())
    print(df.head())

    # CSV로 저장
    df.to_csv("KRX_배당수익률.csv", index=False, encoding="utf-8-sig")
    print("\n✓ 'KRX_배당수익률.csv' 파일로 저장되었습니다.")
else:
    print("데이터를 가져오지 못했습니다.")
    print("응답:", data)

