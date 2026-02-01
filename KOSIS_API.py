import requests
import pandas as pd

from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('KOSIS_KEY')
base_url = "https://kosis.kr/openapi/Param/statisticsParameterData.do"

params = {
    "method": "getList",
    "apiKey": api_key,
    "itmId": "T1",           # 항목 ID
    "objL1": "ALL",          # 분류값
    "objL2": "",
    "objL3": "",
    "objL4": "",
    "objL5": "",
    "objL6": "",
    "objL7": "",
    "objL8": "",
    "format": "json",        # 응답 형식
    "jsonVD": "Y",
    "prdSe": "M",            # 수록주기 (M:월, Y:년, Q:분기)
    "startPrdDe": "202001",  # 시작 시점
    "endPrdDe": "202412",    # 종료 시점
    "orgId": "101",          # 기관 ID
    "tblId": "DT_1J22112"    # 통계표 ID
}

response = requests.get(base_url, params=params)
data = response.json()
df = pd.DataFrame(data)
