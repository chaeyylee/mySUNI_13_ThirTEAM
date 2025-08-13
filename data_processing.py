import pandas as pd
from pathlib import Path
from thefuzz import process, fuzz
from tqdm import tqdm
from kiwipiepy import Kiwi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import random
import os
from dotenv import load_dotenv
import openai

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


VALID_NAMES = ['김민수', '박지훈', '이지우', '정해인', '최유리']

def ask_chatgpt(name: str) -> str:
    if not hasattr(openai, 'api_key') or not openai.api_key:
        return name
    try:
        prompt = (
            f"'{name}'이라는 이름에 오탈자가 있을 수 있습니다. "
            f"다음 목록 {VALID_NAMES} 중에서 가장 유사한 이름을 찾아주세요. "
            f"유사한 이름이 없다면 '{name}'을 그대로 반환해주세요. "
            f"답변은 이름 하나만 반환해야 합니다."
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return name

def correct_operator(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return name
    name = name.strip()
    match, score = process.extractOne(name, VALID_NAMES, scorer=fuzz.ratio)
    if score >= 60:
        return match
    elif score >= 40:
        return ask_chatgpt(name)
    else:
        return name

def refine_status(row):
    remark = str(row.get("remark", ""))
    if any(x in remark for x in ["점검이 필요", "전기 계통의 부하가 높아지는 추세임", "정밀 점검이 필요함"]):
        return "추후 정밀 점검 필요"
    elif any(x in remark for x in ["공정 품질에 영향", "즉각적인 유지보수가 요구됨"]):
        return "이상(교체필요)"
    else:
        return "이상없음"

def assign_inspector_shift(inspector):
    inspector_shift_map = {
        "서지우": "A", "이준서": "B", "정재훈": "C", "홍지민": "A"
    }
    if pd.isna(inspector):
        return random.choice(["A", "B", "C"])
    return inspector_shift_map.get(inspector, random.choice(["A", "B", "C"]))

def safe_merge(df1, df2, on, how="left", name=""):
    keys = [k for k in on if k in df1.columns and k in df2.columns]
    if not keys:
        print(f"{name} 병합 건너뜀: 공통 키 없음 → {on}")
        return df1
    try:
        return df1.merge(df2, on=keys, how=how)
    except Exception as e:
        print(f"병합 실패 ({name}):", e)
        return df1

def extract_keywords_tfidf(texts, top_k=4):
    from kiwipiepy import Kiwi
    from sklearn.feature_extraction.text import TfidfVectorizer

    kiwi = Kiwi()
    stopwords = {
        "의", "이", "가", "은", "는", "들", "좀", "잘", "걍", "과", "도", "를", "으로", "에", "하고", "뿐", "등",
        "있으며", "되어", "수", "있다", "있음", "및", "대한", "때문에", "것", "있고", "있어"
    }

    def extract_nouns(text):
        return [
            word for word, tag, _, _ in kiwi.analyze(text)[0][0]
            if tag.startswith("NN") and word not in stopwords
        ]

    noun_texts = [" ".join(extract_nouns(text)) for text in texts]
    vectorizer = TfidfVectorizer(max_features=300)
    X = vectorizer.fit_transform(noun_texts)
    
    keywords_list = []
    for row in X:
        row_data = row.toarray().flatten()
        indices = row_data.argsort()[-top_k:][::-1]
        keywords = [vectorizer.get_feature_names_out()[i] for i in indices if row_data[i] > 0]
        keywords_list.append(", ".join(keywords))
    return keywords_list


def run_preprocessing(base_path: Path, openai_key: str = None) -> pd.DataFrame:
    tqdm.pandas()
    if openai_key:
        openai.api_key = openai_key

    production = pd.read_csv(base_path / "production_log.csv")
    product_master = pd.read_csv(base_path / "product_master.csv")
    shift_schedule = pd.read_csv(base_path / "shift_schedule.csv")
    energy_usage = pd.read_csv(base_path / "energy_usage.csv")
    inspection = pd.read_csv(base_path / "inspection_result.csv")
    equipment_check = pd.read_csv(base_path / "equipment_check.csv")

    production = production.rename(columns={"production_date": "date"})
    shift_schedule = shift_schedule.rename(columns={"work_date": "date"})
    inspection = inspection.rename(columns={"inspection_date": "date"})
    equipment_check = equipment_check.rename(columns={"check_date": "date"})

    for df in [production, shift_schedule, inspection, equipment_check, energy_usage]:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.dropna(subset=["date"], inplace=True)

    production["produced_qty"] = pd.to_numeric(production["produced_qty"].replace("없음", pd.NA), errors="coerce")
    production["defect_qty"] = pd.to_numeric(production["defect_qty"], errors="coerce")
    production["operator"] = production["operator"].progress_apply(correct_operator)
    shift_schedule["operator"] = shift_schedule["operator"].progress_apply(correct_operator)
    equipment_check["status"] = equipment_check.apply(refine_status, axis=1)
    inspection["inspector_shift"] = inspection["inspector"].apply(assign_inspector_shift)
    inspection["result"] = inspection["result"].astype(str).replace({"불합격": "불합격(재검필요)", "재검": "불합격(재검필요)"})

    print("\n[INFO] 병합 전 데이터프레임별 중복 제거 수행...")
    shift_schedule = shift_schedule.sort_values("date").drop_duplicates(["factory_id", "line_id", "date", "operator"], keep='last')
    inspection = inspection.sort_values("date").drop_duplicates(["product_code", "date"], keep='last')
    equipment_check = equipment_check.sort_values("date").drop_duplicates(["factory_id", "line_id", "date"], keep='last')
    energy_usage = energy_usage.sort_values("date").drop_duplicates(["factory_id", "line_id", "date"], keep='last')
    print("[SUCCESS] 중복 제거 완료.\n")

    merged = production.copy()
    merged = safe_merge(merged, product_master, on=["product_code"], name="product_master")
    merged = safe_merge(merged, shift_schedule, on=["factory_id", "line_id", "date", "operator"], name="shift_schedule")
    merged = safe_merge(merged, energy_usage, on=["factory_id", "line_id", "date"], name="energy_usage")
    merged = safe_merge(merged, inspection, on=["product_code", "date"], name="inspection")
    merged = safe_merge(merged, equipment_check, on=["factory_id", "line_id", "date"], name="equipment_check")

    duplicate_keys = ["factory_id", "line_id", "product_code", "date", "operator"]
    duplicates = merged[merged.duplicated(subset=duplicate_keys, keep=False)]
    if not duplicates.empty:
        dup_log_path = base_path / "duplicates_log_A.csv"
        duplicates.to_csv(dup_log_path, index=False, encoding="utf-8-sig")
        print(f"\n병합 후 중복 {len(duplicates)}건 발견 → '{dup_log_path.name}' 저장")
        merged = merged.drop_duplicates(subset=duplicate_keys, keep="first").reset_index(drop=True)

    merged = merged.rename(columns={"result": "inspection_result", "status": "equipment_check"})
    merged["equipment_id"] = merged["equipment_id"].fillna("장비 점검 미실시")
    merged["shift"] = merged.groupby(["date", "operator"])["shift"].transform(lambda x: x.ffill().bfill())
    merged["produced_qty"] = merged["produced_qty"].fillna(0).astype(int)
    merged["defect_qty"] = merged["defect_qty"].fillna(0).astype(int)

    final_cols = [
        "factory_id","line_id", "equipment_id", "product_code","date","operator","shift",
        "produced_qty","defect_qty", "product_name","category","spec_weight",
        "electricity_kwh","gas_nm3", "inspector","inspector_shift","inspection_result","equipment_check","remark"
    ]
    for col in final_cols:
        if col not in merged.columns:
            merged[col] = pd.NA
    merged = merged.sort_values("date").reset_index(drop=True)

    if "remark" in merged.columns:
        print("remark_keywords 추출 중...")
        merged["remark"] = merged["remark"].fillna("")
        merged["remark_keywords"] = extract_keywords_tfidf(merged["remark"].tolist())

    return merged[final_cols + ["remark_keywords"]]



if __name__ == '__main__':
    BASE = Path(r"C:\Users\choij\OneDrive\문서\카카오톡 받은 파일\data_remark\data_remark")
    BASE.mkdir(parents=True, exist_ok=True)
    
    try:
        print("데이터 전처리 및 병합을 시작합니다.")
        # API 키는 필요 시 전달
        final_df = run_preprocessing(BASE) 
        
        # 저장
        out_fp = BASE / "result_accurate.csv"
        final_df.to_csv(out_fp, index=False, encoding="utf-8-sig")
        print(f"\n✅ 모든 처리 완료 → '{out_fp.name}' 저장")
        print(f" 최종 데이터 형태: {final_df.shape}")

    except FileNotFoundError as e:
        print(f"파일 경로 오류: {e}. 'BASE' 경로를 확인해주세요.")
    except Exception as e:
        print(f"처리 중 오류 발생: {e}")