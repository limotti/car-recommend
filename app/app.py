"""
app.py - 차량 추천 Flask 서버 (v4 모델 기반)
폴백 전략:
  Level 0: ML 예측 결과가 예산 내  → 정상 추천
  Level 1: 20% 완화 후 재예측      → "조건 일부 완화" 안내
  Level 2: 30% 완화 후 재예측      → "조건 일부 완화" 안내
  Level 3: 규칙 기반 TOP3          → "근사 추천" 경고
"""
import os
import pickle
import threading
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from uuid import uuid4
from google import genai
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder="../templates")

# ── 경로 ───────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "model"

# ── 상수 ───────────────────────────────────────────────
NUMERIC_COLS = ["월소득", "초기자금", "가족수"]
CAT_COLS     = ["주용도", "연료선호", "운전경력", "거주지역"]
FEATURE_ORDERS = {
    "주용도":   ["출퇴근", "레저", "가족용", "업무"],
    "연료선호": ["가솔린", "하이브리드", "전기", "디젤", "상관없음"],
    "운전경력": ["1년미만", "1~3년", "3년이상"],
    "거주지역": ["도심", "지방"],
}
FUEL_SUFFIX = {0: "가솔린", 1: "하이브리드", 2: "전기", 3: "디젤"}

FAMILY_MAP = {
    "아반떼_가솔린": "아반떼",          "아반떼_하이브리드": "아반떼",
    "쏘나타_가솔린": "쏘나타",          "쏘나타_하이브리드": "쏘나타",
    "그랜저_가솔린": "그랜저",          "그랜저_하이브리드": "그랜저",
    "코나_가솔린":   "코나",            "코나_전기": "코나",
    "산타페_가솔린": "산타페",          "산타페_하이브리드": "산타페",
    "팰리세이드_가솔린": "팰리세이드",  "팰리세이드_하이브리드": "팰리세이드",
    "아이오닉6_전기": "아이오닉6",
    "K5_가솔린": "K5",                  "K5_하이브리드": "K5",
    "K8_가솔린": "K8",                  "K8_하이브리드": "K8",
    "셀토스_가솔린": "셀토스",
    "스포티지_가솔린": "스포티지",      "스포티지_하이브리드": "스포티지",
    "스포티지_디젤": "스포티지",
    "EV6_전기": "EV6",
    "G80_가솔린": "G80",                "G80_전기": "G80",
    "GV80_가솔린": "GV80",
}
MULTI_FUEL = {
    "아반떼":    ["가솔린", "하이브리드"],
    "쏘나타":    ["가솔린", "하이브리드"],
    "그랜저":    ["가솔린", "하이브리드"],
    "코나":      ["가솔린", "전기"],
    "산타페":    ["가솔린", "하이브리드"],
    "팰리세이드": ["가솔린", "하이브리드"],
    "K5":        ["가솔린", "하이브리드"],
    "K8":        ["가솔린", "하이브리드"],
    "스포티지":  ["가솔린", "하이브리드", "디젤"],
    "G80":       ["가솔린", "전기"],
}
SINGLE_FUEL = {"아이오닉6": "전기", "셀토스": "가솔린", "EV6": "전기", "GV80": "가솔린"}

ENG_COLS = ["자금여력비", "월소득_구간", "자금_구간", "전기차가능", "대형차가능",
            "하이브리드적합", "소형차적합", "중형차적합", "대형SUV적합", "프리미엄적합"]

# ── 차종별 최소 초기자금 기준 (최저가의 약 30%) ─────────
CAR_MIN_BUDGET = {
    "아반떼_가솔린":           510,
    "아반떼_하이브리드":       690,
    "쏘나타_가솔린":           780,
    "쏘나타_하이브리드":       930,
    "그랜저_가솔린":          1110,
    "그랜저_하이브리드":      1260,
    "코나_가솔린":             660,
    "코나_전기":              1410,
    "산타페_가솔린":           960,
    "산타페_하이브리드":      1170,
    "팰리세이드_가솔린":      1140,
    "팰리세이드_하이브리드":  1350,
    "아이오닉6_전기":         1560,
    "K5_가솔린":               750,
    "K5_하이브리드":           900,
    "K8_가솔린":              1050,
    "K8_하이브리드":          1230,
    "셀토스_가솔린":           600,
    "스포티지_가솔린":         750,
    "스포티지_하이브리드":     960,
    "스포티지_디젤":           840,
    "EV6_전기":               1500,
    "G80_가솔린":             1950,
    "G80_전기":               2550,
    "GV80_가솔린":            2100,
}

# ── 차종 메타 정보 ──────────────────────────────────────
CAR_INFO = {
    "아반떼_가솔린":           {"brand": "현대", "type": "준중형 세단",            "price": "약 1,700~2,300만원"},
    "아반떼_하이브리드":       {"brand": "현대", "type": "준중형 세단 하이브리드", "price": "약 2,300~2,700만원"},
    "쏘나타_가솔린":           {"brand": "현대", "type": "중형 세단",              "price": "약 2,600~3,400만원"},
    "쏘나타_하이브리드":       {"brand": "현대", "type": "중형 세단 하이브리드",   "price": "약 3,100~3,700만원"},
    "그랜저_가솔린":           {"brand": "현대", "type": "대형 세단",              "price": "약 3,700~5,200만원"},
    "그랜저_하이브리드":       {"brand": "현대", "type": "대형 세단 하이브리드",   "price": "약 4,200~5,500만원"},
    "코나_가솔린":             {"brand": "현대", "type": "소형 SUV",               "price": "약 2,200~2,900만원"},
    "코나_전기":               {"brand": "현대", "type": "소형 전기 SUV",          "price": "약 4,700~5,200만원"},
    "산타페_가솔린":           {"brand": "현대", "type": "중형 SUV",               "price": "약 3,200~4,400만원"},
    "산타페_하이브리드":       {"brand": "현대", "type": "중형 SUV 하이브리드",    "price": "약 3,900~4,900만원"},
    "팰리세이드_가솔린":       {"brand": "현대", "type": "대형 SUV",               "price": "약 3,800~5,000만원"},
    "팰리세이드_하이브리드":   {"brand": "현대", "type": "대형 SUV 하이브리드",    "price": "약 4,500~5,500만원"},
    "아이오닉6_전기":          {"brand": "현대", "type": "중형 전기 세단",          "price": "약 5,200~6,200만원"},
    "K5_가솔린":               {"brand": "기아", "type": "중형 세단",              "price": "약 2,500~3,300만원"},
    "K5_하이브리드":           {"brand": "기아", "type": "중형 세단 하이브리드",   "price": "약 3,000~3,600만원"},
    "K8_가솔린":               {"brand": "기아", "type": "대형 세단",              "price": "약 3,500~5,000만원"},
    "K8_하이브리드":           {"brand": "기아", "type": "대형 세단 하이브리드",   "price": "약 4,100~5,200만원"},
    "셀토스_가솔린":           {"brand": "기아", "type": "소형 SUV",               "price": "약 2,000~2,700만원"},
    "스포티지_가솔린":         {"brand": "기아", "type": "중형 SUV",               "price": "약 2,500~3,300만원"},
    "스포티지_하이브리드":     {"brand": "기아", "type": "중형 SUV 하이브리드",    "price": "약 3,200~3,900만원"},
    "스포티지_디젤":           {"brand": "기아", "type": "중형 SUV 디젤",          "price": "약 2,800~3,500만원"},
    "EV6_전기":                {"brand": "기아", "type": "중형 전기 SUV",          "price": "약 5,000~6,200만원"},
    "G80_가솔린":              {"brand": "제네시스", "type": "대형 세단",           "price": "약 6,500~9,000만원"},
    "G80_전기":                {"brand": "제네시스", "type": "대형 전기 세단",      "price": "약 8,500~1억원"},
    "GV80_가솔린":             {"brand": "제네시스", "type": "대형 SUV",            "price": "약 7,000~1억원"},
}

# ── 차종별 공식 URL ────────────────────────────────────
CAR_URLS = {
    "아반떼_가솔린":         "https://www.hyundai.com/kr/ko/e/vehicles/avante/intro",
    "아반떼_하이브리드":     "https://www.hyundai.com/kr/ko/e/vehicles/avante/intro",
    "쏘나타_가솔린":         "https://www.hyundai.com/kr/ko/e/vehicles/sonata/intro",
    "쏘나타_하이브리드":     "https://www.hyundai.com/kr/ko/e/vehicles/sonata/intro",
    "그랜저_가솔린":         "https://www.hyundai.com/kr/ko/e/vehicles/grandeur/intro",
    "그랜저_하이브리드":     "https://www.hyundai.com/kr/ko/e/vehicles/grandeur/intro",
    "코나_가솔린":           "https://www.hyundai.com/kr/ko/e/vehicles/kona/intro",
    "코나_전기":             "https://www.hyundai.com/kr/ko/e/vehicles/kona-electric/intro",
    "산타페_가솔린":         "https://www.hyundai.com/kr/ko/e/vehicles/santafe/intro",
    "산타페_하이브리드":     "https://www.hyundai.com/kr/ko/e/vehicles/santafe/intro",
    "팰리세이드_가솔린":     "https://www.hyundai.com/kr/ko/e/vehicles/palisade/intro",
    "팰리세이드_하이브리드": "https://www.hyundai.com/kr/ko/e/vehicles/palisade/intro",
    "아이오닉6_전기":        "https://www.hyundai.com/kr/ko/e/vehicles/ioniq6/intro",
    "K5_가솔린":             "https://www.kia.com/kr/vehicles/k5/features.html",
    "K5_하이브리드":         "https://www.kia.com/kr/vehicles/k5/features.html",
    "K8_가솔린":             "https://www.kia.com/kr/vehicles/k8/features.html",
    "K8_하이브리드":         "https://www.kia.com/kr/vehicles/k8/features.html",
    "셀토스_가솔린":         "https://www.kia.com/kr/vehicles/seltos/features.html",
    "스포티지_가솔린":       "https://www.kia.com/kr/vehicles/sportage/features.html",
    "스포티지_하이브리드":   "https://www.kia.com/kr/vehicles/sportage/features.html",
    "스포티지_디젤":         "https://www.kia.com/kr/vehicles/sportage/features.html",
    "EV6_전기":              "https://www.kia.com/kr/vehicles/ev6/features.html",
    "G80_가솔린":            "https://www.genesis.com/kr/ko/models/sedan/g80/highlights.html",
    "G80_전기":              "https://www.genesis.com/kr/ko/models/luxury-sedan-genesis/electrified-g80/highlights.html",
    "GV80_가솔린":           "https://www.genesis.com/kr/ko/models/suv/gv80/highlights.html",
}

# ── 차종별 연비 (가솔린·하이브리드·디젤: km/L, 전기: km/kWh) ─
CAR_FUEL_EFFICIENCY = {
    "아반떼_가솔린":          14.0,
    "아반떼_하이브리드":      22.5,
    "쏘나타_가솔린":          13.0,
    "쏘나타_하이브리드":      20.0,
    "그랜저_가솔린":          11.0,
    "그랜저_하이브리드":      17.0,
    "코나_가솔린":            14.0,
    "코나_전기":               6.3,
    "산타페_가솔린":          10.5,
    "산타페_하이브리드":      14.5,
    "팰리세이드_가솔린":       9.5,
    "팰리세이드_하이브리드":  13.0,
    "아이오닉6_전기":          6.3,
    "K5_가솔린":              13.5,
    "K5_하이브리드":          21.0,
    "K8_가솔린":              11.5,
    "K8_하이브리드":          17.5,
    "셀토스_가솔린":          14.5,
    "스포티지_가솔린":        13.0,
    "스포티지_하이브리드":    15.5,
    "스포티지_디젤":          15.0,
    "EV6_전기":                5.5,
    "G80_가솔린":             10.0,
    "G80_전기":                4.8,
    "GV80_가솔린":             9.5,
}

# ── 차종별 월 보험료 (만원, 30대 평균) ──────────────────────
CAR_INSURANCE = {
    "아반떼_가솔린":           6.0,  "아반떼_하이브리드":       6.0,
    "쏘나타_가솔린":           7.0,  "쏘나타_하이브리드":       7.0,
    "그랜저_가솔린":           9.0,  "그랜저_하이브리드":       9.0,
    "코나_가솔린":             7.0,  "코나_전기":               8.0,
    "산타페_가솔린":           8.0,  "산타페_하이브리드":       8.0,
    "팰리세이드_가솔린":      10.0,  "팰리세이드_하이브리드":  10.0,
    "아이오닉6_전기":          8.0,
    "K5_가솔린":               7.0,  "K5_하이브리드":           7.0,
    "K8_가솔린":               9.0,  "K8_하이브리드":           9.0,
    "셀토스_가솔린":           7.0,
    "스포티지_가솔린":         8.0,  "스포티지_하이브리드":     8.0,  "스포티지_디젤":  8.0,
    "EV6_전기":                8.0,
    "G80_가솔린":             12.0,  "G80_전기":               12.0,
    "GV80_가솔린":            12.0,
}

# ── 차종 패밀리별 승차 정원 ──────────────────────────────────
CAR_SEATING = {
    "아반떼": 5, "쏘나타": 5, "그랜저": 5, "코나": 5,
    "산타페": 7, "팰리세이드": 7, "아이오닉6": 5,
    "K5": 5, "K8": 5, "셀토스": 5, "스포티지": 5,
    "EV6": 5, "G80": 5, "GV80": 7,
}

# ── og:image 크롤링 + 캐싱 ──────────────────────────────
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9",
}
car_images: dict = {}   # {label: image_url | None}


def _fetch_og_image(url: str) -> str | None:
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=8, allow_redirects=True)
        soup = BeautifulSoup(resp.text, "html.parser")
        tag  = (soup.find("meta", property="og:image") or
                soup.find("meta", attrs={"name": "og:image"}))
        if tag and tag.get("content"):
            img = tag["content"]
            # 상대 경로 → 절대 URL 변환
            if img.startswith("/"):
                from urllib.parse import urlparse
                p = urlparse(url)
                img = f"{p.scheme}://{p.netloc}{img}"
            return img
    except Exception:
        pass
    return None


def load_car_images():
    """앱 시작 시 1회 — 고유 URL 병렬 크롤링 후 캐싱"""
    global car_images
    unique_urls: dict[str, list[str]] = {}
    for label, url in CAR_URLS.items():
        unique_urls.setdefault(url, []).append(label)

    url_to_img: dict[str, str | None] = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_fetch_og_image, url): url for url in unique_urls}
        for fut in as_completed(futures):
            url = futures[fut]
            url_to_img[url] = fut.result()

    for label, url in CAR_URLS.items():
        car_images[label] = url_to_img.get(url)

    ok = sum(1 for v in car_images.values() if v)
    print(f"[images] {ok}/{len(car_images)}개 og:image 로드 완료")


# 디버그 모드 리로더 자식 프로세스에서만 1회 실행
if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not os.environ.get("FLASK_DEBUG"):
    threading.Thread(target=load_car_images, daemon=True).start()

# ── 챗봇 설정 ──────────────────────────────────────────
CHATBOT_KEY = os.environ.get("CHATBOT_KEY", "")   # Render 환경변수 또는 로컬 .env

SYSTEM_PROMPT_TEMPLATE = """\
너는 현대/기아/제네시스 전문 자동차 상담사야.
ML 모델이 사용자 조건(소득/자금/가족수/용도/연료/경력/지역)을
분석해서 차종을 추천했어.
너는 ML이 수치로 판단하기 어려운 부분을 보완하는 역할이야.

## 네가 중점적으로 다뤄야 할 영역
1. 디자인/색상/스타일
   - "스포티한 느낌 원해요" → 색상/외관 옵션 안내
   - "패밀리카 느낌 나는 색은?" → 색상 추천

2. 감성/취향
   - "운전하는 느낌이 어때요?" → 승차감/핸들링 설명
   - "고급스러운 느낌 나나요?" → 인테리어/소재 설명

3. 세부 스펙 상담
   - "트렁크 얼마나 커요?" → 적재공간 안내
   - "애기 카시트 들어가요?" → 공간/안전 옵션 안내

4. 비교 상담
   - "산타페랑 팰리세이드 차이가 뭐예요?" → 비교 설명
   - "비슷한 다른 차 있어요?" → 대안 제시

5. 구매 실무
   - "할부 어떻게 해요?" → 일반적인 할부 안내
   - "언제 사는 게 좋아요?" → 시즌/프로모션 일반 안내

## 네가 하지 말아야 할 것
- 수치 기반 추천 변경 (그건 ML 모델 역할)
- 가격 재계산 (이미 ML이 처리함)
- 현대/기아/제네시스 외 브랜드 추천

## 말투
- 친근하고 전문적으로
- 너무 길지 않게 (3~5문장)
- 이모지 가끔 활용

현재 ML이 추천한 차종: {추천차종}
사용자 조건: 월소득 {월소득}만원, 가족수 {가족수}명, 용도 {용도}\
"""

# 서버 사이드 세션 저장소 {session_id: [history]}
chat_sessions: dict = {}

def get_fallback_info(inp: dict, level: int) -> dict:
    """
    폴백 발동 이유 분석 → 사용자 친화적 메시지 + 챗봇용 컨텍스트 반환
    Returns: {type, message, affordable_price}
    """
    if level == 0:
        return {"type": "none", "message": None, "affordable_price": None}

    budget    = inp["초기자금"]
    income    = inp["월소득"]
    fuel_pref = inp["연료선호"]

    # 월소득×0.3×72개월(할부) + 초기 보유 자금 = 구매 가능 최대 차값 (100만원 단위 반올림)
    affordable = round((income * 0.3 * 72 + budget) / 100) * 100

    if level in (1, 2):
        return {
            "type": "budget",
            "affordable_price": affordable,
            "message": (
                f"입력하신 조건으로는 정확한 매칭 차종이 없습니다.\n"
                f"💡 이유: 초기 보유 자금 {budget:,}만원 + 월소득 {income:,}만원 기준\n"
                f"   구매 가능 가격대가 약 {affordable:,}만원으로,\n"
                f"   조건에 가장 가까운 차종을 추천드립니다."
            ),
        }

    # level == 3: 연료 미매칭 여부 판단
    if fuel_pref != "상관없음":
        fuel_cars_ok = [
            lbl for lbl in CAR_MIN_BUDGET
            if lbl.endswith(f"_{fuel_pref}") and CAR_MIN_BUDGET[lbl] <= budget
        ]
        if not fuel_cars_ok:
            return {
                "type": "fuel",
                "affordable_price": affordable,
                "message": (
                    f"선호하신 {fuel_pref} 차종 중 조건에 맞는 차량이 없어\n"
                    f"   가장 가까운 대안을 추천드립니다."
                ),
            }

    return {
        "type": "complex",
        "affordable_price": affordable,
        "message": (
            "모든 조건을 동시에 충족하는 차종이 없어\n"
            "   가장 근접한 차종을 추천드립니다."
        ),
    }

# TOP3 룰 기반 스코어링용 그룹
_SUV_FAM      = {"코나", "산타페", "팰리세이드", "셀토스", "스포티지", "EV6", "GV80"}
_COMMUTE_FAM  = {"아반떼", "쏘나타", "K5", "셀토스", "아이오닉6", "EV6"}
_PREMIUM_FAM  = {"그랜저", "K8", "G80", "GV80"}


# ── 월 유지비 계산 ────────────────────────────────────
def get_monthly_maintenance(label: str) -> float:
    """
    월 유지비 = 연료비 + 보험료 + 소모품  (만원, 소수점 1자리)
    연료비 기준: 월 1,000km
      가솔린·하이브리드·디젤: 1000 ÷ 연비(km/L) × 1,700원 ÷ 10,000
      전기:                   1000 ÷ 연비(km/kWh) × 200원 ÷ 10,000
    """
    fuel         = label.split("_")[-1]
    efficiency   = CAR_FUEL_EFFICIENCY.get(label, 12.0)
    insurance    = CAR_INSURANCE.get(label, 8.0)

    if fuel == "전기":
        fuel_cost    = 1000 / efficiency * 200  / 10000
        consumables  = 1.0
    elif fuel == "하이브리드":
        fuel_cost    = 1000 / efficiency * 1700 / 10000
        consumables  = 2.5
    else:                                          # 가솔린 / 디젤
        fuel_cost    = 1000 / efficiency * 1700 / 10000
        consumables  = 3.0

    return round(fuel_cost + insurance + consumables, 1)


# ── 추천 이유 생성 (동적·케이스별) ────────────────────
def generate_reasons(inp: dict, label: str) -> list:
    """
    입력 조건 + 추천 차종 기반으로 추천 이유 TOP3 반환
    우선순위: 예산(1) → 연료선호(2) → 유지비(3) → 가족수(4) → 전기+도심(5) → 초보(6)
    """
    budget    = inp["초기자금"]
    income    = inp["월소득"]
    family_n  = inp["가족수"]
    fuel_pref = inp["연료선호"]
    region    = inp["거주지역"]
    career    = inp["운전경력"]

    fam_name         = FAMILY_MAP.get(label, "")
    fuel             = label.split("_")[-1]
    min_bud          = CAR_MIN_BUDGET.get(label, 999)
    actual_min_price = round(min_bud / 0.3)        # 차량 최저가(만원) ≈ CAR_MIN_BUDGET / 0.3
    seating          = CAR_SEATING.get(fam_name, 5)
    monthly_maint    = get_monthly_maintenance(label)

    cands = []   # (priority, reason_text)

    # ── 우선순위 1: 예산 ──────────────────────────────
    if budget >= actual_min_price:
        # 케이스 1: 초기자금으로 바로 구매 가능
        cands.append((1, f"초기 보유 자금 {budget:,}만원으로 구매 가능한 차종입니다"))
    else:
        remaining   = actual_min_price - budget
        installment = round(remaining / 72, 1)
        if fam_name in _PREMIUM_FAM:
            # 케이스 4: 프리미엄 차종
            cands.append((1, (
                f"초기자금 {budget:,}만원 + 월 {installment}만원 할부로 "
                f"월소득 {income:,}만원 기준 감당 가능한 프리미엄 차종입니다"
            )))
        else:
            # 케이스 2: 할부 필요
            cands.append((1, (
                f"월소득 {income:,}만원 기준 월 {installment}만원 할부로 "
                f"72개월 구매 가능한 차종입니다"
            )))

    # ── 우선순위 2: 연료 선호 일치 ───────────────────
    if fuel_pref != "상관없음" and fuel_pref == fuel:
        cands.append((2, f"선호하신 {fuel} 차종입니다"))

    # ── 우선순위 3: 월 유지비 비율 ───────────────────
    if income > 0:
        ratio_pct = round(monthly_maint / income * 100, 1)
        cands.append((3, (
            f"월 유지비 {monthly_maint}만원으로 월소득의 {ratio_pct}% 수준입니다"
        )))

    # ── 우선순위 4: 가족 수 기준 (3명 이상일 때) ─────
    if family_n >= 3:
        cands.append((4, f"가족 {family_n}명 기준 {seating}인승으로 적합합니다"))

    # ── 우선순위 5: 전기차 + 도심 ────────────────────
    if fuel == "전기" and region == "도심":
        cands.append((5, "도심 거주 기준 전기차 충전 인프라 활용에 유리합니다"))

    # ── 우선순위 6: 초보 운전자 + 소형 ──────────────
    if career == "1년미만" and fam_name in {"아반떼", "코나", "셀토스"}:
        cands.append((6, "초보 운전자에게 적합한 크기의 차종입니다"))

    # priority 순 정렬 → 중복 제거 → TOP3
    cands.sort(key=lambda x: x[0])
    seen, result = set(), []
    for _, reason in cands:
        if reason not in seen:
            seen.add(reason)
            result.append(reason)
        if len(result) == 3:
            break

    if not result:
        result.append(f"현재 조건과 가장 가까운 {fam_name} 추천")

    return result[:3]


# ── 모델 로드 (앱 시작 시 1회) ────────────────────────
def load_models():
    with open(MODEL_DIR / "stage1_model.pkl", "rb") as f:
        s1 = pickle.load(f)
    with open(MODEL_DIR / "fuel_clfs.pkl", "rb") as f:
        fuel_clfs = pickle.load(f)
    with open(MODEL_DIR / "encoder.pkl", "rb") as f:
        enc = pickle.load(f)
    with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
        le_full = pickle.load(f)
    with open(MODEL_DIR / "label_encoder_family.pkl", "rb") as f:
        le_family = pickle.load(f)
    with open(MODEL_DIR / "col_order_base.pkl", "rb") as f:
        col_order = pickle.load(f)
    return s1, fuel_clfs, enc, le_full, le_family, col_order

s1_model, fuel_clfs, encoder, le_full, le_family, col_order = load_models()


# ── 피처 엔지니어링 ────────────────────────────────────
def add_features(df):
    df = df.copy()
    df["자금여력비"]     = df["초기자금"] / (df["월소득"] + 1)
    df["월소득_구간"]    = pd.cut(df["월소득"],   bins=[0,250,400,550,np.inf],   labels=[0,1,2,3]).astype(int)
    df["자금_구간"]      = pd.cut(df["초기자금"], bins=[0,500,1500,3000,np.inf], labels=[0,1,2,3]).astype(int)
    df["전기차가능"]     = ((df["거주지역"] == "도심") & (df["초기자금"] > 3500)).astype(int)
    df["대형차가능"]     = ((df["초기자금"] > 3000) & (df["가족수"] >= 3)).astype(int)
    df["하이브리드적합"] = ((df["월소득"] > 350) & (df["주용도"].isin(["출퇴근", "가족용"]))).astype(int)
    df["소형차적합"]     = ((df["초기자금"] <= 3200) & (df["월소득"] <= 380)).astype(int)
    df["중형차적합"]     = (df["초기자금"].between(2500, 5000) & df["월소득"].between(280, 580)).astype(int)
    df["대형SUV적합"]    = ((df["초기자금"] >= 3800) & (df["가족수"] >= 3)).astype(int)
    df["프리미엄적합"]   = ((df["초기자금"] >= 5500) & (df["월소득"] >= 500)).astype(int)
    return df


def build_X(df):
    df = add_features(df)
    df_cat = pd.DataFrame(encoder.transform(df[CAT_COLS]), columns=CAT_COLS)
    X = pd.concat([df[NUMERIC_COLS + ENG_COLS].reset_index(drop=True),
                   df_cat.reset_index(drop=True)], axis=1)
    return X[col_order]


def postprocess(pred_label, fuel_enc):
    """연료선호 명시 시 강제 보정"""
    if fuel_enc == 4:
        return pred_label
    target_fuel = FUEL_SUFFIX[fuel_enc]
    family = FAMILY_MAP.get(pred_label)
    if family and target_fuel in MULTI_FUEL.get(family, []):
        return f"{family}_{target_fuel}"
    return pred_label


def predict_single(input_dict):
    """ML 파이프라인으로 단일 추천 차종 반환"""
    df  = pd.DataFrame([input_dict])
    X   = build_X(df)

    family = le_family.inverse_transform(s1_model.predict(X))[0]

    if family in SINGLE_FUEL:
        label = f"{family}_{SINGLE_FUEL[family]}"
    elif family in fuel_clfs:
        clf, fuel_le = fuel_clfs[family]
        fuel  = fuel_le.inverse_transform(clf.predict(X))[0]
        label = f"{family}_{fuel}"
    else:
        label = f"{family}_가솔린"

    fuel_enc = int(encoder.transform(df[CAT_COLS])[0][CAT_COLS.index("연료선호")])
    return postprocess(label, fuel_enc)


# ── 폴백 Level 3: 규칙 기반 TOP3 스코어링 ──────────────
def _score_car(label, inp):
    """원본 예산 제약 없이 적합도 점수 계산 (높을수록 좋음)"""
    budget    = inp["초기자금"]
    income    = inp["월소득"]
    family_n  = inp["가족수"]
    purpose   = inp["주용도"]
    fuel_pref = inp["연료선호"]
    region    = inp["거주지역"]

    min_bud  = CAR_MIN_BUDGET.get(label, 9999)
    fam_name = FAMILY_MAP.get(label, "")
    fuel     = label.split("_")[1]
    score    = 0.0

    # 1. 예산 근접도 (가장 중요)
    ratio = budget / max(min_bud, 1)
    if ratio >= 1.0:    score += 10
    elif ratio >= 0.85: score += 7
    elif ratio >= 0.70: score += 4
    elif ratio >= 0.55: score += 2
    else:               score += 0

    # 2. 월 부담률 (차 최저가 / 72개월 / 월소득)
    monthly_ratio = (min_bud / 72) / max(income, 1)
    if monthly_ratio <= 0.30:   score += 5
    elif monthly_ratio <= 0.40: score += 3
    elif monthly_ratio <= 0.50: score += 1

    # 3. 연료 선호 일치
    if fuel_pref == "상관없음":
        score += 1
    elif fuel_pref == fuel:
        score += 4
    else:
        score -= 3

    # 4. 가족 수 + 차종 유형
    if family_n >= 4 and fam_name in _SUV_FAM:
        score += 3
    elif family_n <= 2 and fam_name not in _SUV_FAM:
        score += 2

    # 5. 용도 매칭
    if purpose == "출퇴근" and fam_name in _COMMUTE_FAM:
        score += 2
    elif purpose == "가족용" and fam_name in _SUV_FAM:
        score += 2
    elif purpose == "레저" and fam_name in _SUV_FAM:
        score += 1
    elif purpose == "업무" and fam_name in _PREMIUM_FAM:
        score += 1

    # 6. 전기차 + 지방 페널티
    if fuel == "전기" and region == "지방":
        score -= 2

    return score


def _top3_rule_based(inp):
    scored = [(lbl, _score_car(lbl, inp)) for lbl in CAR_MIN_BUDGET]
    scored.sort(key=lambda x: -x[1])
    return [lbl for lbl, _ in scored[:3]]


# ── 폴백 메인 로직 ─────────────────────────────────────
def predict_with_fallback(inp):
    """
    반환: (labels: list[str], level: int)
      level 0 → 정상 (labels 1개)
      level 1 → 20% 완화 (labels 1개)
      level 2 → 30% 완화 (labels 1개)
      level 3 → TOP3 규칙 기반 (labels 3개)
    """
    budget = inp["초기자금"]
    income = inp["월소득"]

    # 1차: 원본 조건
    label = predict_single(inp)
    if CAR_MIN_BUDGET.get(label, 0) <= budget:
        return [label], 0

    # 2차: 20% 완화
    inp2  = {**inp, "초기자금": int(budget * 1.2), "월소득": int(income * 1.2)}
    label2 = predict_single(inp2)
    if CAR_MIN_BUDGET.get(label2, 0) <= budget * 1.2:
        return [label2], 1

    # 3차: 30% 완화
    inp3  = {**inp, "초기자금": int(budget * 1.3), "월소득": int(income * 1.3)}
    label3 = predict_single(inp3)
    if CAR_MIN_BUDGET.get(label3, 0) <= budget * 1.3:
        return [label3], 2

    # 최종: 규칙 기반 TOP3
    return _top3_rule_based(inp), 3


# ── 라우트 ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data     = request.get_json()
        required = ["월소득", "초기자금", "가족수", "주용도", "연료선호", "운전경력", "거주지역"]
        for k in required:
            if k not in data:
                return jsonify({"error": f"'{k}' 항목이 누락되었습니다."}), 400

        inp = {
            "월소득":   int(data["월소득"]),
            "초기자금": int(data["초기자금"]),
            "가족수":   int(data["가족수"]),
            "주용도":   str(data["주용도"]),
            "연료선호": str(data["연료선호"]),
            "운전경력": str(data["운전경력"]),
            "거주지역": str(data["거주지역"]),
        }

        # ── 0 입력 처리: 최소 조건 추천 ──────────────────
        if inp["월소득"] == 0 or inp["초기자금"] == 0:
            cheapest = min(CAR_MIN_BUDGET, key=CAR_MIN_BUDGET.get)  # 셀토스_가솔린
            info = CAR_INFO.get(cheapest, {})
            return jsonify({
                "results": [{
                    "recommendation": cheapest,
                    "brand":        info.get("brand", ""),
                    "type":         info.get("type",  ""),
                    "price":        info.get("price", ""),
                    "car_image":    car_images.get(cheapest),
                    "official_url": CAR_URLS.get(cheapest, ""),
                    "reasons":      [
                        "현재 조건 기준 구매 가능한 최저가 차종",
                        "첫 차로 적합한 소형 SUV",
                        "유지비 부담이 가장 적은 모델",
                    ],
                }],
                "fallback_level":   3,
                "fallback_type":    "zero_input",
                "fallback_message": (
                    "최소 조건으로 추천 가능한 차종을 안내합니다.\n"
                    "소득 또는 초기 자금이 입력되지 않아 구매 가능한 최저가 차종을 기준으로 추천드려요."
                ),
                "affordable_price": None,
                "recommendation":   cheapest,
                "brand":            info.get("brand", ""),
                "type":             info.get("type",  ""),
                "price":            info.get("price", ""),
            })

        labels, level = predict_with_fallback(inp)
        fb            = get_fallback_info(inp, level)

        results = []
        for lbl in labels:
            info = CAR_INFO.get(lbl, {})
            results.append({
                "recommendation": lbl,
                "brand":        info.get("brand", ""),
                "type":         info.get("type",  ""),
                "price":        info.get("price", ""),
                "car_image":    car_images.get(lbl),          # og:image (None이면 fallback)
                "official_url": CAR_URLS.get(lbl, ""),
                "reasons":      generate_reasons(inp, lbl),  # 추천 이유 TOP3
            })

        return jsonify({
            "results":           results,
            "fallback_level":    level,
            "fallback_type":     fb["type"],
            "fallback_message":  fb["message"],
            "affordable_price":  fb["affordable_price"],
            # 하위 호환
            "recommendation":    results[0]["recommendation"],
            "brand":             results[0]["brand"],
            "type":              results[0]["type"],
            "price":             results[0]["price"],
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data              = request.get_json()
        user_msg          = (data.get("message") or "").strip()
        session_id        = data.get("session_id", "")
        car               = data.get("recommendation", "미정")
        income            = data.get("월소득", "")
        family            = data.get("가족수", "")
        purpose           = data.get("용도", "")
        fallback_level    = int(data.get("fallback_level", 0))
        affordable_price  = data.get("affordable_price")
        fallback_type     = data.get("fallback_type", "none")

        if not user_msg:
            return jsonify({"error": "메시지가 비어있습니다."}), 400

        # 세션 히스토리
        if session_id not in chat_sessions:
            session_id = str(uuid4())
            chat_sessions[session_id] = []
        history = chat_sessions[session_id]

        # 유저 메시지 추가
        history.append({"role": "user", "parts": [{"text": user_msg}]})

        # 기본 시스템 프롬프트
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            추천차종=car, 월소득=income, 가족수=family, 용도=purpose
        )

        # 폴백 발동 시 추가 컨텍스트 주입
        if fallback_level > 0:
            price_hint = f"약 {affordable_price:,}만원" if affordable_price else "미산출"
            system_prompt += (
                f"\n\n## 폴백 추천 상황\n"
                f"이 추천은 사용자 조건에 완전히 맞지 않는 폴백 추천이야.\n"
                f"폴백 유형: {fallback_type} "
                f"(budget=자금부족 / fuel=연료미매칭 / complex=복합미매칭)\n"
                f"사용자 예산 기준 구매 가능 가격대: {price_hint}\n"
                f"사용자가 조건 조정 방법을 물어보면 구체적으로 안내해줘.\n"
                f"예: '초기 보유 자금을 N만원 더 높이면 {{차종}} 추천 가능합니다.'"
            )

        # Gemini 호출
        client   = genai.Client(api_key=CHATBOT_KEY)
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=history,
            config={"system_instruction": system_prompt},
        )
        reply = response.text

        # 어시스턴트 응답 저장 + 최대 10턴(20개) 유지
        history.append({"role": "model", "parts": [{"text": reply}]})
        chat_sessions[session_id] = history[-20:]

        return jsonify({
            "reply":          reply,
            "session_id":     session_id,
            "history_length": len(chat_sessions[session_id]) // 2,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chat/reset", methods=["POST"])
def chat_reset():
    data = request.get_json() or {}
    sid  = data.get("session_id", "")
    if sid in chat_sessions:
        del chat_sessions[sid]
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
