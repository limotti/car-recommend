import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
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


def add_features(df):
    """파생 피처 추가 - 가솔린/하이브리드 구분력 향상"""
    df = df.copy()
    df["자금여력비"] = df["초기자금"] / (df["월소득"] + 1)   # 초기자금/월소득 비율
    df["월소득_구간"] = pd.cut(df["월소득"],
                              bins=[0, 250, 400, 550, 999],
                              labels=[0, 1, 2, 3]).astype(int)
    df["자금_구간"] = pd.cut(df["초기자금"],
                            bins=[0, 500, 1500, 3000, 9999],
                            labels=[0, 1, 2, 3]).astype(int)
    return df


def preprocess(df):
    df = add_features(df)

    ENG_COLS = ["자금여력비", "월소득_구간", "자금_구간"]
    all_num = NUMERIC_COLS + ENG_COLS

    enc = OrdinalEncoder(categories=[FEATURE_ORDERS[c] for c in CAT_COLS],
                         handle_unknown="use_encoded_value", unknown_value=-1)
    df_cat = pd.DataFrame(enc.fit_transform(df[CAT_COLS]), columns=CAT_COLS)
    df_num = df[all_num].reset_index(drop=True)
    X = pd.concat([df_num, df_cat], axis=1)

    le = LabelEncoder()
    y = le.fit_transform(df["추천차종"])
    return X, y, enc, le, X.columns.tolist()


def train():
    df = pd.read_csv(DATA_PATH)
    print(f"데이터 로드: {len(df)}행, {df['추천차종'].nunique()}개 라벨\n")

    X, y, enc, le, col_order = preprocess(df)

    # SMOTE
    min_count = pd.Series(y).value_counts().min()
    k = min(5, min_count - 1)
    print(f"SMOTE 적용 중 (k={k})...")
    X_res, y_res = SMOTE(random_state=42, k_neighbors=k).fit_resample(X, y)
    print(f"SMOTE 후: {len(X_res)}행\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
    )

    # ── 1. Random Forest ──────────────────────────────
    print("[1/3] Random Forest 학습...")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"      정확도: {rf_acc*100:.2f}%\n")

    # ── 2. XGBoost (RandomizedSearchCV 튜닝) ──────────
    print("[2/3] XGBoost 하이퍼파라미터 튜닝 중 (이 작업은 1~2분 소요)...")
    xgb_params = {
        "n_estimators":      [200, 300, 500],
        "max_depth":         [4, 6, 8],
        "learning_rate":     [0.05, 0.1, 0.15],
        "subsample":         [0.7, 0.8, 0.9],
        "colsample_bytree":  [0.7, 0.8, 0.9],
        "min_child_weight":  [1, 3, 5],
        "gamma":             [0, 0.1, 0.2],
    }
    xgb_base = XGBClassifier(
        random_state=42, eval_metric="mlogloss", verbosity=0
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        xgb_base, xgb_params, n_iter=20, cv=cv,
        scoring="accuracy", random_state=42, n_jobs=-1, verbose=0
    )
    search.fit(X_train, y_train)
    best_xgb = search.best_estimator_
    xgb_acc  = accuracy_score(y_test, best_xgb.predict(X_test))
    print(f"      최적 파라미터: {search.best_params_}")
    print(f"      정확도: {xgb_acc*100:.2f}%\n")

    # ── 3. Voting Ensemble ────────────────────────────
    print("[3/3] Voting Ensemble (RF + XGBoost)...")
    voting = VotingClassifier(
        estimators=[("rf", rf), ("xgb", best_xgb)],
        voting="soft"
    )
    voting.fit(X_train, y_train)
    vote_acc = accuracy_score(y_test, voting.predict(X_test))
    print(f"      정확도: {vote_acc*100:.2f}%\n")

    # ── 최고 모델 선택 ────────────────────────────────
    results = {
        "RandomForest": (rf,       rf_acc),
        "XGBoost":      (best_xgb, xgb_acc),
        "Ensemble":     (voting,   vote_acc),
    }
    best_name = max(results, key=lambda k: results[k][1])
    best_model, best_acc = results[best_name]

    print("=" * 55)
    print(f"  RF:       {rf_acc*100:.2f}%")
    print(f"  XGBoost:  {xgb_acc*100:.2f}%")
    print(f"  Ensemble: {vote_acc*100:.2f}%")
    print(f"  >> 최고 모델: {best_name} ({best_acc*100:.2f}%)")
    print("=" * 55)

    # ── Classification Report ─────────────────────────
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=le.classes_,
                                   output_dict=True)
    df_report = pd.DataFrame(report).T.iloc[:-3]   # support 행 제외
    df_report = df_report[["precision","recall","f1-score","support"]]
    df_report["support"] = df_report["support"].astype(int)

    print("\n[ Precision / Recall / F1 상세 ]")
    print(df_report.sort_values("f1-score").to_string(float_format="{:.3f}".format))

    weak = df_report[df_report["f1-score"] < 0.7].index.tolist()
    if weak:
        print(f"\n⚠ F1 < 0.70 라벨: {weak}")

    # ── 변수 중요도 ───────────────────────────────────
    if hasattr(best_model, "feature_importances_"):
        imp = pd.Series(best_model.feature_importances_, index=col_order)
        print("\n[ 변수 중요도 ]")
        print(imp.sort_values(ascending=False).to_string(float_format="{:.4f}".format))

    # ── 저장 ─────────────────────────────────────────
    MODEL_DIR.mkdir(exist_ok=True)
    for name, obj in [
        ("model.pkl",         best_model),
        ("encoder.pkl",       enc),
        ("label_encoder.pkl", le),
        ("col_order.pkl",     col_order),
    ]:
        with open(MODEL_DIR / name, "wb") as f:
            pickle.dump(obj, f)

    print(f"\n저장 완료 → model/ (best: {best_name})")
    return best_model, enc, le


if __name__ == "__main__":
    train()
