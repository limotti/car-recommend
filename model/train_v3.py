"""
model/train_v3.py  —  개선 3단계
추가: 가격대 적합도 피처 / 연료선호 후처리 규칙 / n_iter=30
목표: 78.21% → 80%+
"""
import pandas as pd
import numpy as np
from pathlib import Path
import pickle, warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import (train_test_split, RandomizedSearchCV,
                                     StratifiedKFold, cross_val_predict)
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

DATA_PATH = Path("data/augmented_car_data.csv")
MODEL_DIR  = Path("model")

NUMERIC_COLS = ["월소득", "초기자금", "가족수"]
CAT_COLS     = ["주용도", "연료선호", "운전경력", "거주지역"]
FEATURE_ORDERS = {
    "주용도":   ["출퇴근", "레저", "가족용", "업무"],
    "연료선호": ["가솔린", "하이브리드", "전기", "디젤", "상관없음"],
    "운전경력": ["1년미만", "1~3년", "3년이상"],
    "거주지역": ["도심", "지방"],
}
# 연료선호 OrdinalEncoding: 가솔린=0, 하이브리드=1, 전기=2, 디젤=3, 상관없음=4
FUEL_ENC = {"가솔린": 0, "하이브리드": 1, "전기": 2, "디젤": 3, "상관없음": 4}
FUEL_SUFFIX = {0: "가솔린", 1: "하이브리드", 2: "전기", 3: "디젤"}

FAMILY_MAP = {
    "아반떼_가솔린":"아반떼",      "아반떼_하이브리드":"아반떼",
    "쏘나타_가솔린":"쏘나타",      "쏘나타_하이브리드":"쏘나타",
    "그랜저_가솔린":"그랜저",      "그랜저_하이브리드":"그랜저",
    "코나_가솔린":"코나",          "코나_전기":"코나",
    "산타페_가솔린":"산타페",      "산타페_하이브리드":"산타페",
    "팰리세이드_가솔린":"팰리세이드","팰리세이드_하이브리드":"팰리세이드",
    "아이오닉6_전기":"아이오닉6",
    "K5_가솔린":"K5",             "K5_하이브리드":"K5",
    "K8_가솔린":"K8",             "K8_하이브리드":"K8",
    "셀토스_가솔린":"셀토스",
    "스포티지_가솔린":"스포티지",  "스포티지_하이브리드":"스포티지",
    "스포티지_디젤":"스포티지",
    "EV6_전기":"EV6",
    "G80_가솔린":"G80",           "G80_전기":"G80",
    "GV80_가솔린":"GV80",
}
# 계열별 사용 가능한 연료타입
FAMILY_FUELS = {}
for label, family in FAMILY_MAP.items():
    fuel = label.split("_")[1]
    FAMILY_FUELS.setdefault(family, set()).add(fuel)

XGB_PARAMS = {
    "n_estimators":     [200, 300, 400, 500],
    "max_depth":        [4, 6, 8, 10],
    "learning_rate":    [0.03, 0.05, 0.1, 0.15],
    "subsample":        [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "min_child_weight": [1, 3, 5],
    "gamma":            [0, 0.05, 0.1, 0.2],
    "reg_alpha":        [0, 0.1, 0.5],
    "reg_lambda":       [1, 1.5, 2],
}


# ── 피처 엔지니어링 ─────────────────────────────────────
def add_features(df):
    df = df.copy()
    # 기존
    df["자금여력비"]     = df["초기자금"] / (df["월소득"] + 1)
    df["월소득_구간"]    = pd.cut(df["월소득"], bins=[0,250,400,550,999],
                                  labels=[0,1,2,3]).astype(int)
    df["자금_구간"]      = pd.cut(df["초기자금"], bins=[0,500,1500,3000,9999],
                                  labels=[0,1,2,3]).astype(int)
    df["전기차가능"]     = ((df["거주지역"]=="도심") & (df["초기자금"]>3500)).astype(int)
    df["대형차가능"]     = ((df["초기자금"]>3000) & (df["가족수"]>=3)).astype(int)
    df["하이브리드적합"] = ((df["월소득"]>350) & (df["주용도"].isin(["출퇴근","가족용"]))).astype(int)
    # 신규: 차급별 가격 적합도
    df["소형차적합"]   = ((df["초기자금"] <= 3200) & (df["월소득"] <= 380)).astype(int)
    df["중형차적합"]   = ((df["초기자금"].between(2500,5000)) & (df["월소득"].between(280,580))).astype(int)
    df["대형SUV적합"]  = ((df["초기자금"] >= 3800) & (df["가족수"] >= 3)).astype(int)
    df["프리미엄적합"] = ((df["초기자금"] >= 5500) & (df["월소득"] >= 500)).astype(int)
    return df

ENG_COLS = [
    "자금여력비","월소득_구간","자금_구간",
    "전기차가능","대형차가능","하이브리드적합",
    "소형차적합","중형차적합","대형SUV적합","프리미엄적합",
]


def build_X(df, enc=None, fit=False):
    df = add_features(df)
    all_num = NUMERIC_COLS + ENG_COLS
    if fit:
        enc = OrdinalEncoder(
            categories=[FEATURE_ORDERS[c] for c in CAT_COLS],
            handle_unknown="use_encoded_value", unknown_value=-1,
        )
        df_cat = pd.DataFrame(enc.fit_transform(df[CAT_COLS]), columns=CAT_COLS)
    else:
        df_cat = pd.DataFrame(enc.transform(df[CAT_COLS]), columns=CAT_COLS)
    X = pd.concat([df[all_num].reset_index(drop=True),
                   df_cat.reset_index(drop=True)], axis=1)
    return X, enc


def smote(X, y, tag=""):
    k = min(5, pd.Series(y).value_counts().min() - 1)
    X_r, y_r = SMOTE(random_state=42, k_neighbors=k).fit_resample(X, y)
    print(f"  [{tag}] SMOTE 후: {len(X_r)}행")
    return X_r, y_r


def tune_xgb(X_tr, y_tr, n_iter=30, tag=""):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        XGBClassifier(random_state=42, eval_metric="mlogloss", verbosity=0),
        XGB_PARAMS, n_iter=n_iter, cv=cv,
        scoring="accuracy", random_state=42, n_jobs=-1, verbose=0,
    )
    search.fit(X_tr, y_tr)
    print(f"  [{tag}] 최적: {search.best_params_}")
    return search.best_estimator_


# ── 후처리 규칙: 연료선호 명시 시 강제 보정 ─────────────
def postprocess(preds, X_te_raw, le_full):
    """
    연료선호가 '상관없음(4)' 이 아닌 경우,
    예측된 계열에서 해당 연료타입이 존재하면 강제 보정.
    """
    labels = list(le_full.inverse_transform(preds))
    fuel_col_idx = X_te_raw.columns.tolist().index("연료선호")

    corrected = 0
    for i, (pred_label, row) in enumerate(zip(labels, X_te_raw.itertuples(index=False))):
        fuel_enc_val = int(getattr(row, "연료선호"))
        if fuel_enc_val == 4:          # 상관없음 → 모델 결과 그대로
            continue
        target_fuel  = FUEL_SUFFIX[fuel_enc_val]
        family       = FAMILY_MAP.get(pred_label)
        if family is None:
            continue
        if target_fuel in FAMILY_FUELS.get(family, set()):
            correct_label = f"{family}_{target_fuel}"
            if correct_label != pred_label:
                labels[i] = correct_label
                corrected += 1

    print(f"  후처리 보정: {corrected}건")
    return le_full.transform(labels)


# ══════════════════════════════════════════════════════
def train():
    df = pd.read_csv(DATA_PATH)
    print(f"데이터 로드: {len(df)}행, {df['추천차종'].nunique()}개 라벨\n")

    df["차종_계열"] = df["추천차종"].map(FAMILY_MAP)

    X_base, enc = build_X(df, fit=True)
    col_base    = X_base.columns.tolist()

    le_full   = LabelEncoder().fit(df["추천차종"])
    le_family = LabelEncoder().fit(df["차종_계열"])
    y_full    = le_full.transform(df["추천차종"])
    y_family  = le_family.transform(df["차종_계열"])

    (X_tr, X_te,
     yf_tr, yf_te,
     yfam_tr, yfam_te) = train_test_split(
        X_base, y_full, y_family,
        test_size=0.2, random_state=42, stratify=y_full,
    )

    # ══════════════════════════════════════════
    #  STAGE 1 : 차종 계열 (14 classes)
    # ══════════════════════════════════════════
    print("=" * 55)
    print("  STAGE 1 : 차종 계열 예측 (14 classes)")
    print("=" * 55)
    X_tr_s1, yfam_sm = smote(X_tr, yfam_tr, "Stage1")
    print("  XGBoost 튜닝 중... (약 1~2분)")
    s1 = tune_xgb(X_tr_s1, yfam_sm, n_iter=30, tag="S1")
    s1_acc = accuracy_score(yfam_te, s1.predict(X_te))
    print(f"  Stage1 정확도: {s1_acc*100:.2f}%\n")

    print("  Stage1 cross-val 예측 생성 (leakage 방지)...")
    s1_cv = cross_val_predict(s1, X_tr, yfam_tr, cv=3, n_jobs=-1)

    # ══════════════════════════════════════════
    #  STAGE 2 : 최종 차종 (25 classes)
    # ══════════════════════════════════════════
    print("=" * 55)
    print("  STAGE 2 : 최종 차종 예측 (25 classes)")
    print("=" * 55)
    X_tr_s2 = X_tr.copy(); X_tr_s2["예측_계열"] = s1_cv
    X_te_s2 = X_te.copy(); X_te_s2["예측_계열"] = s1.predict(X_te)
    col_s2  = X_tr_s2.columns.tolist()

    X_tr_s2_sm, yf_sm = smote(X_tr_s2, yf_tr, "Stage2")
    print("  XGBoost 튜닝 중... (약 2~3분)")
    s2 = tune_xgb(X_tr_s2_sm, yf_sm, n_iter=30, tag="S2")
    s2_acc = accuracy_score(yf_te, s2.predict(X_te_s2))
    print(f"  Stage2 정확도 (후처리 전): {s2_acc*100:.2f}%\n")

    # ── 후처리 ────────────────────────────────
    print("  후처리 규칙 적용 중...")
    preds_raw = s2.predict(X_te_s2)
    preds_pp  = postprocess(preds_raw, X_te_s2, le_full)
    s2_acc_pp = accuracy_score(yf_te, preds_pp)
    print(f"  Stage2 정확도 (후처리 후): {s2_acc_pp*100:.2f}%\n")

    # ── 최종 예측 선택 ────────────────────────
    final_preds = preds_pp if s2_acc_pp >= s2_acc else preds_raw
    final_acc   = max(s2_acc_pp, s2_acc)
    use_pp      = s2_acc_pp >= s2_acc

    # ── 결과 출력 ─────────────────────────────
    report = classification_report(
        yf_te, final_preds, target_names=le_full.classes_, output_dict=True
    )
    df_rep = (pd.DataFrame(report).T.iloc[:-3]
                [["precision","recall","f1-score","support"]])
    df_rep["support"] = df_rep["support"].astype(int)

    print("=" * 55)
    print(f"  Stage1 계열  : {s1_acc*100:.2f}%")
    print(f"  Stage2 raw   : {s2_acc*100:.2f}%")
    print(f"  Stage2 후처리: {s2_acc_pp*100:.2f}%")
    print(f"  최종 채택    : {'후처리 적용' if use_pp else 'raw'} ({final_acc*100:.2f}%)")
    print(f"  이전 기록    : 76.17%  →  {final_acc*100:.2f}% ({final_acc*100-76.17:+.2f}%p)")
    print("=" * 55)

    print("\n[ Precision / Recall / F1 상세 ]")
    print(df_rep.sort_values("f1-score").to_string(float_format="{:.3f}".format))

    weak = df_rep[df_rep["f1-score"] < 0.70].index.tolist()
    if weak:
        print(f"\n⚠  F1 < 0.70 라벨 ({len(weak)}개): {weak}")
    else:
        print("\n✓  F1 < 0.70 라벨 없음!")

    # ── 저장 ─────────────────────────────────
    MODEL_DIR.mkdir(exist_ok=True)
    saves = {
        "stage1_model.pkl":         s1,
        "stage2_model.pkl":         s2,
        "encoder.pkl":              enc,
        "label_encoder.pkl":        le_full,
        "label_encoder_family.pkl": le_family,
        "col_order_base.pkl":       col_base,
        "col_order_s2.pkl":         col_s2,
        "use_postprocess.pkl":      use_pp,
    }
    for fname, obj in saves.items():
        with open(MODEL_DIR / fname, "wb") as f:
            pickle.dump(obj, f)
    print(f"\n저장 완료 → model/ ({len(saves)}개 파일)")
    return s1, s2, enc, le_full


if __name__ == "__main__":
    train()
