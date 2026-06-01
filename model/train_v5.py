"""
train_v5.py - LightGBM + XGBoost 앙상블 + 하이브리드 프리미엄 피처
Stage1: XGBoost + LightGBM Soft Voting
Stage2: XGBoost + LightGBM Soft Voting (global, v2 방식 유지)
신규피처: 하이브리드_감당가능, 전기차_감당가능
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
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

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
FAMILY_FUELS = {}
for lbl, fam in FAMILY_MAP.items():
    FAMILY_FUELS.setdefault(fam, set()).add(lbl.split("_")[1])

# 계열별 하이브리드/전기 최소 초기자금 (만원, 차량가의 약 30% 자기부담 가정)
HYBRID_MIN_BUDGET = {
    "아반떼": 690,   "쏘나타": 930,  "그랜저": 1260,
    "산타페": 1170,  "팰리세이드": 1350,
    "K5":    900,   "K8":    1230,  "스포티지": 960,
    "코나":  1410,   "G80":   2550,
}

XGB_PARAMS = {
    "n_estimators":     [300, 400, 500],
    "max_depth":        [4, 6, 8],
    "learning_rate":    [0.03, 0.05, 0.1],
    "subsample":        [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "min_child_weight": [1, 3],
    "gamma":            [0, 0.1],
    "reg_alpha":        [0, 0.1],
    "reg_lambda":       [1, 1.5],
}
LGBM_PARAMS = {
    "n_estimators":   [300, 400, 500],
    "max_depth":      [4, 6, 8, -1],
    "learning_rate":  [0.03, 0.05, 0.1],
    "num_leaves":     [31, 63, 127],
    "subsample":      [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "min_child_samples": [10, 20, 30],
    "reg_alpha":      [0, 0.1],
    "reg_lambda":     [0, 0.1],
}


def add_features(df):
    df = df.copy()
    df["자금여력비"]      = df["초기자금"] / (df["월소득"] + 1)
    df["월소득_구간"]     = pd.cut(df["월소득"],   bins=[0,250,400,550,999],  labels=[0,1,2,3]).astype(int)
    df["자금_구간"]       = pd.cut(df["초기자금"], bins=[0,500,1500,3000,9999], labels=[0,1,2,3]).astype(int)
    df["전기차가능"]      = ((df["거주지역"]=="도심") & (df["초기자금"]>3500)).astype(int)
    df["대형차가능"]      = ((df["초기자금"]>3000) & (df["가족수"]>=3)).astype(int)
    df["하이브리드적합"]  = ((df["월소득"]>350) & (df["주용도"].isin(["출퇴근","가족용"]))).astype(int)
    df["소형차적합"]      = ((df["초기자금"]<=3200) & (df["월소득"]<=380)).astype(int)
    df["중형차적합"]      = (df["초기자금"].between(2500,5000) & df["월소득"].between(280,580)).astype(int)
    df["대형SUV적합"]     = ((df["초기자금"]>=3800) & (df["가족수"]>=3)).astype(int)
    df["프리미엄적합"]    = ((df["초기자금"]>=5500) & (df["월소득"]>=500)).astype(int)
    # ── 신규: 계열별 하이브리드 감당 가능 여부 ──────────
    for fam, budget in HYBRID_MIN_BUDGET.items():
        df[f"HYB_{fam}"] = (df["초기자금"] >= budget).astype(int)
    return df

ENG_COLS = (["자금여력비","월소득_구간","자금_구간","전기차가능","대형차가능",
             "하이브리드적합","소형차적합","중형차적합","대형SUV적합","프리미엄적합"]
            + [f"HYB_{f}" for f in HYBRID_MIN_BUDGET])


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
    X_r, y_r = SMOTE(random_state=42, k_neighbors=k).fit_resample(X, y)
    print(f"  [{tag}] SMOTE 후: {len(X_r)}행")
    return X_r, y_r


def tune(estimator, params, X_tr, y_tr, n_iter=25, tag=""):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    s  = RandomizedSearchCV(estimator, params, n_iter=n_iter, cv=cv,
                            scoring="accuracy", random_state=42,
                            n_jobs=-1, verbose=0)
    s.fit(X_tr, y_tr)
    print(f"  [{tag}] CV best: {s.best_score_*100:.2f}%  params: {s.best_params_}")
    return s.best_estimator_


def postprocess(pred_labels, X_te):
    fuel_idx = list(X_te.columns).index("연료선호")
    corrected, result = 0, list(pred_labels)
    for i, lbl in enumerate(result):
        fenc = int(X_te.iloc[i, fuel_idx])
        if fenc == 4:
            continue
        tfuel  = FUEL_SUFFIX[fenc]
        family = FAMILY_MAP.get(lbl)
        if family and tfuel in FAMILY_FUELS.get(family, set()):
            nl = f"{family}_{tfuel}"
            if nl != lbl:
                result[i] = nl; corrected += 1
    print(f"  후처리 보정: {corrected}건")
    return result


def train():
    df = pd.read_csv(DATA_PATH)
    print(f"데이터 로드: {len(df)}행\n")
    df["차종_계열"] = df["추천차종"].map(FAMILY_MAP)

    X_base, enc = build_X(df, fit=True)
    col_base     = X_base.columns.tolist()
    le_full      = LabelEncoder().fit(df["추천차종"])
    le_family    = LabelEncoder().fit(df["차종_계열"])
    y_full       = le_full.transform(df["추천차종"])
    y_family     = le_family.transform(df["차종_계열"])

    (X_tr, X_te,
     yf_tr, yf_te,
     yfam_tr, yfam_te) = train_test_split(
        X_base, y_full, y_family,
        test_size=0.2, random_state=42, stratify=y_full,
    )

    # ══════════════════════════════════════
    #  STAGE 1: 계열 (14) — XGB + LGBM 앙상블
    # ══════════════════════════════════════
    print("=" * 55)
    print("  STAGE 1: 차종 계열 (14 classes) - XGB+LGBM 앙상블")
    print("=" * 55)
    X_tr_s1, yfam_sm = smote_safe(X_tr, yfam_tr, "S1")

    print("  [S1-XGB] 튜닝 중...")
    s1_xgb = tune(XGBClassifier(random_state=42, eval_metric="mlogloss", verbosity=0),
                  XGB_PARAMS, X_tr_s1, yfam_sm, n_iter=25, tag="S1-XGB")

    print("  [S1-LGBM] 튜닝 중...")
    s1_lgbm = tune(LGBMClassifier(random_state=42, verbosity=-1),
                   LGBM_PARAMS, X_tr_s1, yfam_sm, n_iter=25, tag="S1-LGBM")

    s1_voting = VotingClassifier(
        estimators=[("xgb", s1_xgb), ("lgbm", s1_lgbm)], voting="soft"
    )
    s1_voting.fit(X_tr_s1, yfam_sm)
    s1_acc = accuracy_score(yfam_te, s1_voting.predict(X_te))
    print(f"  Stage1 앙상블 정확도: {s1_acc*100:.2f}%\n")

    print("  Stage1 cross-val 예측 생성...")
    s1_cv = cross_val_predict(s1_voting, X_tr, yfam_tr, cv=3, n_jobs=-1)

    # ══════════════════════════════════════
    #  STAGE 2: 전체 25 — XGB + LGBM 앙상블
    # ══════════════════════════════════════
    print("=" * 55)
    print("  STAGE 2: 최종 차종 (25 classes) - XGB+LGBM 앙상블")
    print("=" * 55)
    X_tr_s2 = X_tr.copy(); X_tr_s2["예측_계열"] = s1_cv
    X_te_s2 = X_te.copy(); X_te_s2["예측_계열"] = s1_voting.predict(X_te)
    col_s2  = X_tr_s2.columns.tolist()

    X_tr_s2_sm, yf_sm = smote_safe(X_tr_s2, yf_tr, "S2")

    print("  [S2-XGB] 튜닝 중...")
    s2_xgb = tune(XGBClassifier(random_state=42, eval_metric="mlogloss", verbosity=0),
                  XGB_PARAMS, X_tr_s2_sm, yf_sm, n_iter=25, tag="S2-XGB")

    print("  [S2-LGBM] 튜닝 중...")
    s2_lgbm = tune(LGBMClassifier(random_state=42, verbosity=-1),
                   LGBM_PARAMS, X_tr_s2_sm, yf_sm, n_iter=25, tag="S2-LGBM")

    s2_voting = VotingClassifier(
        estimators=[("xgb", s2_xgb), ("lgbm", s2_lgbm)], voting="soft"
    )
    s2_voting.fit(X_tr_s2_sm, yf_sm)

    preds_raw = s2_voting.predict(X_te_s2)
    s2_acc    = accuracy_score(yf_te, preds_raw)
    print(f"  Stage2 정확도 (후처리 전): {s2_acc*100:.2f}%\n")

    pred_labels = le_full.inverse_transform(preds_raw)
    pred_labels = postprocess(list(pred_labels), X_te_s2)
    y_pred      = le_full.transform(pred_labels)
    final_acc   = accuracy_score(yf_te, y_pred)
    print(f"  Stage2 정확도 (후처리 후): {final_acc*100:.2f}%\n")

    # ── 결과 ──────────────────────────────
    report = classification_report(
        yf_te, y_pred, target_names=le_full.classes_, output_dict=True
    )
    df_rep = (pd.DataFrame(report).T.iloc[:-3]
                [["precision","recall","f1-score","support"]])
    df_rep["support"] = df_rep["support"].astype(int)

    print("=" * 55)
    print(f"  Stage1 앙상블 : {s1_acc*100:.2f}%")
    print(f"  Stage2 최종   : {final_acc*100:.2f}%")
    print(f"  이전 기록     : 78.21% → {final_acc*100:.2f}% ({final_acc*100-78.21:+.2f}%p)")
    print("=" * 55)
    print("\n[ Precision / Recall / F1 상세 ]")
    print(df_rep.sort_values("f1-score").to_string(float_format="{:.3f}".format))

    weak = df_rep[df_rep["f1-score"] < 0.70].index.tolist()
    if weak:
        print(f"\n⚠  F1 < 0.70 라벨 ({len(weak)}개): {weak}")
    else:
        print("\n✓  F1 < 0.70 라벨 없음!")

    # ── 저장 ─────────────────────────────
    MODEL_DIR.mkdir(exist_ok=True)
    saves = {
        "stage1_model.pkl":         s1_voting,
        "stage2_model.pkl":         s2_voting,
        "encoder.pkl":              enc,
        "label_encoder.pkl":        le_full,
        "label_encoder_family.pkl": le_family,
        "col_order_base.pkl":       col_base,
        "col_order_s2.pkl":         col_s2,
    }
    for fname, obj in saves.items():
        with open(MODEL_DIR / fname, "wb") as f:
            pickle.dump(obj, f)
    print(f"\n저장 완료 → model/ ({len(saves)}개 파일)")
    return s1_voting, s2_voting, enc, le_full


if __name__ == "__main__":
    train()
