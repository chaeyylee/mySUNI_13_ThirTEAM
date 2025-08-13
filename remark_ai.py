# remark_ai.py

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter, defaultdict
from kiwipiepy import Kiwi
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import os, base64, glob
from openai import OpenAI
import matplotlib.font_manager as fm
import platform

# 폰트 경로 지정
font_path = "fonts/NanumGothic.ttf"  # 실제 경로에 맞게 조정
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams["axes.unicode_minus"] = False


def run_remark_analysis(input_csv_path="uploads/result.csv",
                        output_csv_path="result_merged.csv",
                        output_dir="figs"):

    # 📌 OpenAI 설정
    MODEL_NAME = "gpt-4o"
    MAX_OUTPUT_TOKENS = 500
    QUESTION = """
    Please analyze the keyword trend graphs provided for each piece of equipment,
    and predict the expected timing and type of anomalies. Output should be in table format only.

    [Objective]
    - Analyze keyword time series patterns for each equipment
    - Calculate statistical indicators such as recent variation rate, moving average, standard deviation, and spike/drop thresholds (mean ± 2σ) to detect anomalies
    - If necessary, use short-term linear regression or exponential smoothing to forecast trends
    - Estimate the future time window (date range) when anomalies are likely to occur
    - Infer the most likely type of anomaly
    - Provide key supporting keywords and evidence
    - Suggest actionable countermeasures

    [Output Format]
    Only output the table below without any additional explanation.

    | Equipment ID | Predicted Date | Expected Anomaly Type | Supporting Keywords | Recommended Action |

    Conditions:
    - Dates must follow the format YYYY-MM-DD ± N days
    - Supporting keywords and values must be derived from actual graph observations and calculations (e.g., variation rate %, standard deviation, trendline, etc.)
    - Recommended actions should be specific and executable
    - Do NOT include any text outside of the table

    
    """

    # 경로 설정
    BASE_DIR = Path(__file__).parent.resolve()
    in_fp = BASE_DIR / input_csv_path
    out_fp = BASE_DIR / output_csv_path
    out_dir = BASE_DIR / output_dir
    out_dir.mkdir(exist_ok=True)

    # 데이터 불러오기
    merged = pd.read_csv(in_fp, encoding="utf-8-sig")
    kiwi = Kiwi()
    stopwords = {
        "의", "이", "가", "은", "는", "들", "좀", "잘", "걍", "과", "도", "를", "으로", "에", "하고", "뿐", "등",
        "있으며", "되어", "수", "있다", "있음", "및", "대한", "때문에", "것", "있고", "있어"
    }

    # 명사 추출
    def extract_nouns(text):
        return [
            word for word, tag, _, _ in kiwi.analyze(text)[0][0]
            if tag.startswith("NN") and word not in stopwords
        ]

    # 키워드 추출
    def extract_keywords_tfidf(texts, top_k=4):
        noun_texts = [" ".join(extract_nouns(text)) for text in texts]
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(noun_texts)

        keywords_list = []
        for row in X:
            row_data = row.toarray().flatten()
            indices = row_data.argsort()[-top_k:][::-1]
            keywords = [vectorizer.get_feature_names_out()[i] for i in indices if row_data[i] > 0]
            keywords_list.append(", ".join(keywords))
        return keywords_list

    # 키워드 분석
    merged = merged.dropna(subset=["remark", "date"]).copy()
    merged["remark_keywords"] = extract_keywords_tfidf(merged["remark"].fillna(""), top_k=4)

    # 저장
    merged.to_csv(out_fp, index=False, encoding="utf-8-sig")
    print(f"✅ remark_keywords 추가 후 저장 완료 → {out_fp.name}")

    # 장비별 트렌드 그래프 생성
    grouped = merged.groupby("equipment_id")
    for equip_id, group in grouped:
        group = group.copy()
        group["date"] = pd.to_datetime(group["date"], errors="coerce")
        group = group.dropna(subset=["date"])
        remarks = group["remark"].tolist()
        if not remarks:
            print(f"[{equip_id}] remark 없음 → 건너뜀")
            continue

        keywords = extract_keywords_tfidf(remarks, top_k=4)
        all_keywords = [kw.strip() for line in keywords for kw in line.split(",") if kw.strip()]
        counter = Counter(all_keywords)
        top_keywords = [kw for kw, _ in counter.most_common(5)]

        if not top_keywords:
            print(f"[{equip_id}] 상위 키워드 없음 → 건너뜀")
            continue

        tmp = group[["date", "remark_keywords"]].copy()
        tmp["kw_list"] = tmp["remark_keywords"].fillna("").apply(lambda s: [w.strip() for w in s.split(",") if w.strip()])
        tmp = tmp.explode("kw_list")
        tmp = tmp[tmp["kw_list"].isin(top_keywords)]
        if tmp.empty:
            print(f"[{equip_id}] 집계 결과 없음 → 건너뜀")
            continue

        trend_df = tmp.groupby(["date", "kw_list"]).size().unstack("kw_list").fillna(0).sort_index()
        if trend_df.to_numpy().sum() == 0:
            print(f"[{equip_id}] 모든 빈도 0 → 건너뜀")
            continue

        plt.figure(figsize=(14, 6))
        for col in trend_df.columns:
            line, = plt.plot(trend_df.index, trend_df[col], marker="o", label=col)
            x_pos = trend_df.index[-1]
            y_pos = trend_df[col].iloc[-1]
            plt.text(x_pos, y_pos, f"{col}", fontsize=10, color=line.get_color(), va="center", ha="left")
        plt.title(f"Keyword Trend for Equipment {equip_id}")
        plt.xlabel("Date")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45)
        plt.legend(title="Top Keywords", loc="upper left", bbox_to_anchor=(1.01, 1.0))
        plt.tight_layout()
        save_path = out_dir / f"trend_{equip_id}.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"[{equip_id}] 그래프 저장 완료 → {save_path}")

    # GPT Vision 요청 함수 정의
    import os

    def ask_openai_about_images(image_paths, question, model=MODEL_NAME, max_output_tokens=MAX_OUTPUT_TOKENS):
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            print("❗ OpenAI API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")
            return None

        client = OpenAI(api_key=api_key)
        contents = [{"type": "input_text", "text": question}]
        for path in image_paths:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            contents.append({
                "type": "input_image",
                "image_url": f"data:image/png;base64,{b64}"
            })

        response = client.responses.create(
            model=model,
            input=[{"role": "user", "content": contents}],
            max_output_tokens=max_output_tokens,
        )
        return getattr(response, "output_text", None) or str(response)

    # GPT Vision 분석 요청
    img_paths = sorted(glob.glob(str(out_dir / "trend_*.png")))
    if img_paths:
        print("\n[OpenAI] GPT 그래프 분석 요청 중...")
        gpt_answer = ask_openai_about_images(img_paths, QUESTION)
        if gpt_answer:
            print("\n===== GPT 응답 =====\n")
            print(gpt_answer)
            with open(out_dir / "openai_summary.txt", "w", encoding="utf-8") as f:
                f.write(gpt_answer.strip())
            print("📄 GPT 분석 결과 저장 완료 → openai_summary.txt")
        else:
            print("❌ GPT 응답 실패 또는 응답 없음")
    else:
        print("❌ 트렌드 그래프 이미지가 없어 GPT 분석 요청을 건너뜁니다.")





