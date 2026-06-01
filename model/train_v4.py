"""
train_v4.py — 계열별 연료타입 전문 분류기
Stage 1 : 차종 계열 (14 classes) - XGBoost
Stage 2 : 계열별 전용 연료 분류기 (binary/multiclass per family)
후처리  : 연료선호 명시 시 강제 보정
목표    : 78.21% → 80%+
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
FUEL_SUFFIX = {0:"가솔린", 1:"하이브리드", 2:"전기", 3:"디젤"}

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
# 연료 선택지가 여러 개인 계열만 Stage2 필요
MULTI_FUEL = {
    "아반떼":   ["가솔린","하이브리드"],
    "쏘나타":   ["가솔린","하이브리드"],
    "그랜저":   ["가솔린","하이브리드"],
    "코나":     ["가솔린","전기"],
    "산타페":   ["가솔린","하이브리드"],
    "팰리세이드":["가솔린","하이브리드"],
    "K5":       ["가솔린","하이브리드"],
    "K8":       ["가솔린","하이브리드"],
    "스포티지": ["가솔린","하이브리드","디젤"],
    "G80":      ["가솔린","전기"],
}
SINGLE_FUEL = {"아이오닉6":"전기","셀토스":"가솔린","EV6":"전기","GV80":"가솔린"}

# Stage1 하이퍼파라미터 탐색 공간
S1_PARAMS = {
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


def add_features(df):
    df = df.copy()
    df["자금여력비"]     = df["초기자금"] / (df["월소득"] + 1)
    df["월소득_구간"]    = pd.cut(df["월소득"], bins=[0,250,400,550,999], labels=[0,1,2,3]).astype(int)
    df["자금_구간"]      = pd.cut(df["초기자금"], bins=[0,500,1500,3000,9999], labels=[0,1,2,3]).astype(int)
    df["전기차가능"]     = ((df["거주지역"]=="도심") & (df["초기자금"]>3500)).astype(int)
    df["대형차가능"]     = ((df["초기자금"]>3000) & (df["가족수"]>=3)).astype(int)
    df["하이브리드적합"] = ((df["월소득"]>350) & (df["주용도"].isin(["출퇴근","가족용"]))).astype(int)
    df["소형차적합"]     = ((df["초기자금"]<=3200) & (df["월소득"]<=380)).astype(int)
    df["중형차적합"]     = (df["초기자금"].between(2500,5000) & df["월소득"].between(280,580)).astype(int)
    df["대형SUV적합"]    = ((df["초기자금"]>=3800) & (df["가족수"]>=3)).astype(int)
    df["프리미엄적합"]   = ((df["초기자금"]>=5500) & (df["월소득"]>=500)).astype(int)
    return df

ENG_COLS = ["자금여력비","월소득_구간","자금_구간","전기차가능","대형차가능",
            "하이브리드적합","소형차적합","중형차적합","대형SUV적합","프리미엄적합"]


def build_X(df, enc=None, fit=False):
    df = add_features(df)
    if fit:
        enc = OrdinalEncoder(
            categories=[FEATURE_ORDERS[c] for c in CAT_COLS],
            handle_unknown="use_encoded_value", unknown_value=-1,
        )
        df_cat = pd.DataFrame(enc.fit_transform(df[CAT_COLS]), columns=CAT_COLS)
    else:
        df_cat = pd.DataFrame(enc.transform(df[CAT_COLS]), columns=CAT_COLS)
    X = pd.concat([df[NUMERIC_COLS+ENG_COLS].reset_index(drop=True),
                   df_cat.reset_index(drop=True)], axis=1)
    return X, enc


def smote_safe(X, y, tag=""):
    k = min(5, pd.Series(y).value_counts().min() - 1)
    if k < 1:
        print(f"  [{tag}] 샘플 부족 - SMOTE 생략")
        return X, y
    X_r, y_r = SMOTE(random_state=42, k_neighbors=k).fit_resample(X, y)
    print(f"  [{tag}] SMOTE 후: {len(X_r)}행")
    return X_r, y_r


def tune_xgb(X_tr, y_tr, params, n_iter=30, tag=""):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        XGBClassifier(random_state=42, eval_metric="mlogloss", verbosity=0),
        params, n_iter=n_iter, cv=cv,
        scoring="accuracy", random_state=42, n_jobs=-1, verbose=0,
    )
    search.fit(X_tr, y_tr)
    print(f"  [{tag}] best: {search.best_params_}")
    return search.best_estimator_


def build_fuel_clf(X_tr, y_fuel_tr, family, fuel_le):
    """계열별 연료 분류기 - 빠른 고정 파라미터 + SMOTE"""
    X_sm, y_sm = smote_safe(X_tr, y_fuel_tr, family)
    clf = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="mlogloss", verbosity=0
    )
    clf.fit(X_sm, y_sm)
    tr_acc = accuracy_score(y_fuel_tr, clf.predict(X_tr))
    print(f"    [{family}] 연료 분류기 학습 완료 (train acc: {tr_acc*100:.1f}%)")
    return clf


# ── 후처리 규칙 ────────────────────────────────────────
def postprocess(pred_labels, X_te_orig):
    fuel_idx = list(X_te_orig.columns).index("연료선호")
    corrected = 0
    result = list(pred_labels)
    for i, label in enumerate(result):
        fuel_enc = int(X_te_orig.iloc[i, fuel_idx])
        if fuel_enc == 4:
            continue
        target_fuel = FUEL_SUFFIX[fuel_enc]
        family = FAMILY_MAP.get(label)
        if family and target_fuel in MULTI_FUEL.get(family, []):
            new_label = f"{family}_{target_fuel}"
            if new_label != label:
                result[i] = new_label
                corrected += 1
    print(f"  후처리 보정: {corrected}건")
    return result


# ══════════════════════════════════════════════════════
def train():
    df = pd.read_csv(DATA_PATH)
    print(f"데이터 로드: {len(df)}행, {df['추천차종'].nunique()}개 라벨\n")

    df["차종_계열"] = df["추천차종"].map(FAMILY_MAP)
    df["연료타입"]  = df["추천차종"].str.split("_").str[1]

    X_base, enc = build_X(df, fit=True)
    col_base    = X_base.columns.tolist()
    le_full     = LabelEncoder().fit(df["추천차종"])
    le_family   = LabelEncoder().fit(df["차종_계열"])
    y_full      = le_full.transform(df["추천차종"])
    y_family    = le_family.transform(df["차종_계열"])

    (X_tr, X_te,
     yf_tr, yf_te,
     yfam_tr, yfam_te,
     idx_tr, idx_te) = train_test_split(
        X_base, y_full, y_family, np.arange(len(df)),
        test_size=0.2, random_state=42, stratify=y_full,
    )
    df_tr = df.iloc[idx_tr].reset_index(drop=True)

    # ══════════════════════════════════════════
    #  STAGE 1 : 차종 계열 (14 classes)
    # ══════════════════════════════════════════
    print("=" * 55)
    print("  STAGE 1 : 차종 계열 예측 (14 classes)")
    print("=" * 55)
    X_tr_s1, yfam_sm = smote_safe(X_tr, yfam_tr, "Stage1")
    print("  XGBoost 튜닝 중... (약 2분)")
    s1 = tune_xgb(X_tr_s1, yfam_sm, S1_PARAMS, n_iter=30, tag="S1")
    s1_acc = accuracy_score(yfam_te, s1.predict(X_te))
    print(f"  Stage1 정확도: {s1_acc*100:.2f}%\n")

    # ══════════════════════════════════════════
    #  STAGE 2 : 계열별 연료 전문 분류기
    # ══════════════════════════════════════════
    print("=" * 55)
    print("  STAGE 2 : 계열별 연료 전문 분류기")
    print("=" * 55)
    fuel_clfs = {}   # {family: (clf, fuel_le)}
    for family, fuels in MULTI_FUEL.items():
        mask = df_tr["차종_계열"] == family
        X_fam = X_tr[mask.values]
        y_fam = df_tr.loc[mask, "연료타입"].values
        if len(X_fam) < 10:
            print(f"  [{family}] 데이터 부족 ({len(X_fam)}행) - 스킵")
            continue
        fuel_le = LabelEncoder().fit(fuels)
        y_enc   = fuel_le.transform(y_fam)
        clf     = build_fuel_clf(X_fam, y_enc, family, fuel_le)
        fuel_clfs[family] = (clf, fuel_le)

    # ══════════════════════════════════════════
    #  추론 : Stage1 계열 → Stage2 연료 → 최종
    # ══════════════════════════════════════════
    print("\n  테스트셋 추론 중...")
    pred_families = le_family.inverse_transform(s1.predict(X_te))
    pred_labels   = []

    for i, family in enumerate(pred_families):
        if family in SINGLE_FUEL:
            pred_labels.append(f"{family}_{SINGLE_FUEL[family]}")
        elif family in fuel_clfs:
            clf, fuel_le = fuel_clfs[family]
            fuel_pred = fuel_le.inverse_transform(clf.predict(X_te[i:i+1]))[0]
            pred_labels.append(f"{family}_{fuel_pred}")
        else:
            pred_labels.append(f"{family}_가솔린")  # fallback

    # 후처리
    pred_labels = postprocess(pred_labels, X_te)
    y_pred = le_full.transform(pred_labels)
    final_acc = accuracy_score(yf_te, y_pred)

    # ── 결과 출력 ─────────────────────────────
    report = classification_report(
        yf_te, y_pred, target_names=le_full.classes_, output_dict=True
    )
    df_rep = (pd.DataFrame(report).T.iloc[:-3]
                [["precision","recall","f1-score","support"]])
    df_rep["support"] = df_rep["support"].astype(int)

    print("\n" + "=" * 55)
    print(f"  Stage1 계열  : {s1_acc*100:.2f}%")
    print(f"  최종 정확도  : {final_acc*100:.2f}%")
    print(f"  이전 기록    : 78.21% → {final_acc*100:.2f}% ({final_acc*100-78.21:+.2f}%p)")
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
        "fuel_clfs.pkl":            fuel_clfs,
        "encoder.pkl":              enc,
        "label_encoder.pkl":        le_full,
        "label_encoder_family.pkl": le_family,
        "col_order_base.pkl":       col_base,
    }
    for fname, obj in saves.items():
        with open(MODEL_DIR / fname, "wb") as f:
            pickle.dump(obj, f)
    print(f"\n저장 완료 → model/ ({len(saves)}개 파일)")
    return s1, fuel_clfs, enc, le_full


if __name__ == "__main__":
    train()
