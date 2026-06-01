from google import genai
import pandas as pd
import json
import time
import re
from pathlib import Path

# ===== Gemini API Key =====
AUGMENTATION_KEY = "AIzaSyD_b_LfCDtRvkgkvPYiI5ug1OYS5IHpkEk"
# ==========================

DATA_PATH = Path("data/synthetic_car_data.csv")
OUTPUT_PATH = Path("data/augmented_car_data.csv")
SAMPLES_PER_LABEL = 40
MODEL_NAME = "gemini-2.5-flash-lite"

LABELS = [
    "아반떼_가솔린", "아반떼_하이브리드", "쏘나타_가솔린", "쏘나타_하이브리드",
    "그랜저_가솔린", "그랜저_하이브리드", "코나_가솔린", "코나_전기",
    "산타페_가솔린", "산타페_하이브리드", "팰리세이드_가솔린", "팰리세이드_하이브리드",
    "아이오닉6_전기", "K5_가솔린", "K5_하이브리드", "K8_가솔린", "K8_하이브리드",
    "셀토스_가솔린", "스포티지_가솔린", "스포티지_하이브리드", "스포티지_디젤",
    "EV6_전기", "G80_가솔린", "G80_전기", "GV80_가솔린",
]

LABEL_CONTEXT = {
    "아반떼_가솔린":         "현대 소형 세단, 가격 1800~2500만원, 연비 14~16km/L, 사회초년생 1순위",
    "아반떼_하이브리드":     "현대 소형 세단 하이브리드, 가격 2300~2900만원, 연비 21~24km/L, 도심 절약형",
    "쏘나타_가솔린":         "현대 중형 세단, 가격 2700~3500만원, 가족용, 업무용 무난",
    "쏘나타_하이브리드":     "현대 중형 세단 하이브리드, 가격 3100~3800만원, 장거리 연비 우수",
    "그랜저_가솔린":         "현대 준대형 세단, 가격 3800~5000만원, 비즈니스, 격식",
    "그랜저_하이브리드":     "현대 준대형 세단 하이브리드, 가격 4200~5500만원, 고급+연비",
    "코나_가솔린":           "현대 소형 SUV, 가격 2200~2900만원, 도심 주차 편리, 1~2인 최적",
    "코나_전기":             "현대 소형 전기 SUV, 가격 4700~5200만원, 주행거리 400km+, 도심 충전 필요",
    "산타페_가솔린":         "현대 중형 SUV, 가격 3500~4500만원, 4~5인 가족 레저",
    "산타페_하이브리드":     "현대 중형 SUV 하이브리드, 가격 3900~5000만원, 연비+공간",
    "팰리세이드_가솔린":     "현대 대형 SUV 3열, 가격 4000~5500만원, 5인 이상 대가족",
    "팰리세이드_하이브리드": "현대 대형 SUV 하이브리드 3열, 가격 4500~6000만원, 대가족+연비",
    "아이오닉6_전기":        "현대 중형 전기 세단, 가격 5000~6500만원, 주행거리 500km+, 얼리어답터",
    "K5_가솔린":             "기아 중형 세단, 가격 2600~3400만원, 스포티 디자인",
    "K5_하이브리드":         "기아 중형 세단 하이브리드, 가격 3000~3700만원",
    "K8_가솔린":             "기아 준대형 세단, 가격 3700~5000만원, 고급감, 업무",
    "K8_하이브리드":         "기아 준대형 세단 하이브리드, 가격 4100~5500만원",
    "셀토스_가솔린":         "기아 소형 SUV, 가격 2300~3000만원, 젊은 층, 도심+레저 겸용",
    "스포티지_가솔린":       "기아 준중형 SUV, 가격 2800~3600만원, 다목적 실용",
    "스포티지_하이브리드":   "기아 준중형 SUV 하이브리드, 가격 3200~4000만원",
    "스포티지_디젤":         "기아 준중형 SUV 디젤, 가격 3000~3700만원, 장거리, 지방 경제성",
    "EV6_전기":              "기아 중형 전기 크로스오버, 가격 5200~7000만원, 스포티, 급속충전",
    "G80_가솔린":            "제네시스 준대형 세단, 가격 6000~8000만원, 최고급 비즈니스",
    "G80_전기":              "제네시스 준대형 전기 세단, 가격 8500만~1억, 최고급 전기",
    "GV80_가솔린":           "제네시스 대형 SUV, 가격 7000만~1억, 최고급 SUV",
}

FEATURE_SPEC = """- 월소득: 150~800 사이 정수 (만원)
- 초기자금: 100~5000 사이 정수 (만원)
- 가족수: 1~7 사이 정수
- 주용도: ["출퇴근", "레저", "가족용", "업무"] 중 하나
- 연료선호: ["가솔린", "하이브리드", "전기", "디젤", "상관없음"] 중 하나
- 운전경력: ["1년미만", "1~3년", "3년이상"] 중 하나
- 거주지역: ["도심", "지방"] 중 하나"""


def build_prompt(label, n, existing_df):
    context = LABEL_CONTEXT[label]
    examples = existing_df[existing_df["추천차종"] == label].head(3)
    example_str = ""
    if not examples.empty:
        rows = examples.drop(columns=["추천차종"]).to_dict("records")
        example_str = "\n기존 샘플 예시(참고용, 그대로 복사 금지):\n" + json.dumps(rows, ensure_ascii=False) + "\n"

    return f"""당신은 자동차 구매 성향 데이터 생성 전문가입니다.

추천 차종: {label}
차종 정보: {context}
{example_str}
위 차종을 구매하기에 현실적으로 적합한 사용자 프로필 {n}개를 생성하세요.

변수 규칙:
{FEATURE_SPEC}

다양성 지침:
1. 월소득, 초기자금은 차종 가격대에 맞되 넓은 범위로 분포
2. 주용도, 연료선호는 차종 특성과 자연스럽게 어울리는 조합 우선
3. 운전경력과 가족수는 균등하게 분포
4. 도심/지방 비율을 50:50에 가깝게

반드시 JSON 배열만 출력하세요. 다른 텍스트, 설명, 마크다운 없이.

[
  {{"월소득": 정수, "초기자금": 정수, "가족수": 정수, "주용도": "값", "연료선호": "값", "운전경력": "값", "거주지역": "값"}},
  ...
]"""


def parse_json(text):
    text = text.strip()
    m = re.search(r'``(?:json)?\s*([\s\S]*?)\s*``', text)
    if m:
        text = m.group(1)
    m = re.search(r'\[[\s\S]*\]', text)
    if m:
        text = m.group(0)
    return json.loads(text)


def validate_row(row):
    try:
        assert 150 <= int(row["월소득"]) <= 800
        assert 100 <= int(row["초기자금"]) <= 5000
        assert 1 <= int(row["가족수"]) <= 7
        assert row["주용도"] in ("출퇴근", "레저", "가족용", "업무")
        assert row["연료선호"] in ("가솔린", "하이브리드", "전기", "디젤", "상관없음")
        assert row["운전경력"] in ("1년미만", "1~3년", "3년이상")
        assert row["거주지역"] in ("도심", "지방")
        return True
    except (KeyError, AssertionError, ValueError, TypeError):
        return False


def augment_data():
    client = genai.Client(api_key=AUGMENTATION_KEY)

    df_orig = pd.read_csv(DATA_PATH)
    print(f"원본 데이터: {len(df_orig)}행\n")

    all_new_rows = []
    failed = []

    for idx, label in enumerate(LABELS, 1):
        print(f"[{idx:02d}/{len(LABELS)}] {label} ...", end=" ", flush=True)
        prompt = build_prompt(label, SAMPLES_PER_LABEL, df_orig)

        for attempt in range(3):
            try:
                resp = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                )
                samples = parse_json(resp.text)
                valid = [r for r in samples if validate_row(r)]
                for r in valid:
                    r["추천차종"] = label
                    all_new_rows.append(r)
                print(f"{len(valid)}/{len(samples)} 유효")
                time.sleep(1.5)
                break
            except Exception as e:
                wait = 5 * (attempt + 1)
                print(f"재시도 {attempt+1}/3 ({e}) {wait}s 대기...", end=" ", flush=True)
                time.sleep(wait)
        else:
            print("실패")
            failed.append(label)

    df_new = pd.DataFrame(all_new_rows)
    col_order = ["월소득", "초기자금", "가족수", "주용도", "연료선호", "운전경력", "거주지역", "추천차종"]
    df_new = df_new[col_order]

    df_combined = pd.concat([df_orig, df_new], ignore_index=True)
    df_combined.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\n{'='*50}")
    print(f"원본  : {len(df_orig):>6}행")
    print(f"증강  : {len(df_new):>6}행")
    print(f"합계  : {len(df_combined):>6}행")
    print(f"저장  : {OUTPUT_PATH}")
    if failed:
        print(f"실패 라벨 ({len(failed)}개): {failed}")
    print("\n[ 라벨별 증강 분포 ]")
    print(df_new["추천차종"].value_counts().to_string())
    return df_combined


if __name__ == "__main__":
    augment_data()
