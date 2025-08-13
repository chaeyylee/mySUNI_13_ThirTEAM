# summary를 위한 코드 analysis.py

import openai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

openai.api_key = api_key


def generate_summary_narrative(
    df,
    anova_results=None,
    top_keywords=[],
    line_trend_summary=None,
    equipment_keywords=None,
    cluster_summary=None,
    spc_outliers=None
):
    try:
        # 1. 기본 수치 계산
        total_production = df['production_qty'].sum() if 'production_qty' in df.columns else None
        avg_defect_rate = (df['defect_qty'].sum() / (total_production + 1e-5)) if total_production else None
        elec_sum = df['electricity_kwh'].sum() if 'electricity_kwh' in df.columns else None
        gas_sum = df['gas_nm3'].sum() if 'gas_nm3' in df.columns else None

        # 2. 최근 이슈 요약
        remark_text = "\n".join([
            f"- {item['date'].strftime('%Y-%m-%d')}: {item['remark']}" for item in top_keywords
        ]) if top_keywords else "없음"

        # 3. 클러스터 요약 (★ 표 형식 강조)
        if cluster_summary:
            cluster_text = "\n".join([
                f"- 클러스터 {cid}: \"{text}\"" for cid, text in cluster_summary.items()
            ])
        else:
            cluster_text = "클러스터 요약 없음"

        # 4. 장비 키워드
        eq_text = "\n".join([
            f"{eq}: {', '.join(kw)}" for eq, kw in equipment_keywords.items()
        ]) if equipment_keywords else "없음"

        # 5. 라인 생산 추이
        line_summary_text = (
            ", ".join([f"{line}은 생산량이 {trend}" for line, trend in line_trend_summary.items()])
            if line_trend_summary else "없음"
        )

        # 6. ANOVA 해석 요약
        anova_text = ""
        if anova_results:
            for k, v in anova_results.items():
                p = v.get("p_value")
                sig = v.get("is_significant")
                anova_text += f"{k.upper()} → p-value={p:.4f}, 유의미함: {sig}\n"
        else:
            anova_text = "없음"

        # 7. SPC 이상 탐지 요약
        if isinstance(spc_outliers, list) and len(spc_outliers) > 0:
            total_outliers = len(spc_outliers)
            lines = [item['line'] for item in spc_outliers if 'line' in item]
            from collections import Counter
            line_counts = Counter(lines)
            spc_text = f"총 {total_outliers}건의 이상값이 감지되었습니다.\n"
            spc_text += "📍 라인별 이상 감지 건수:\n"
            for line, cnt in line_counts.items():
                spc_text += f"- {line}: {cnt}건\n"
            spc_text += "🔎 상세 목록:\n"
            spc_text += "\n".join([
                f"  • 날짜: {item['date']}, 라인: {item['line']}, 설비: {item['equipment_id']}, 값: {item['value']:.2f}"
                for item in spc_outliers
            ])
        else:
            spc_text = "이상값 없음"

        # ✅ GPT 프롬프트
        prompt = f"""
당신은 공장 운영 데이터를 분석해 사람에게 발표 가능한 수준의 요약 보고서를 작성하는 전문가입니다.

아래 항목들을 기반으로 **서술형 보고서 형태의 자연어 요약**을 작성해주세요.

모든 항목을 반드시 포함하고, 자연스럽게 연결된 문장으로 구성하며, 총 15~20줄 분량이어야 합니다.

1. 📦 총 생산량 및 불량률
- 총 생산량: {total_production:,}개
- 평균 불량률: {avg_defect_rate:.2%}

2. ⚡ 에너지 사용량
- 전기: {elec_sum:,.1f} kWh
- 가스: {gas_sum:,.1f} Nm³

3. 📈 라인별 생산 추이
{line_summary_text}

4. 🔍 SPC 이상 탐지 결과
{spc_text}

5. 📊 ANOVA 분석 결과
{anova_text.strip()}

6. 🛠️ 장비별 자주 언급된 키워드
{eq_text}

7. 🧠 클러스터별 대표 이슈 문장
{cluster_text}

8. 📝 최근 주요 이슈 (Remark)
{remark_text}

✍️ 위 내용을 바탕으로 자연스럽고 명확한 리포트를 작성해주세요.
- 수치, 장비명, 키워드, 날짜를 명확하게 언급
- 단순 나열이 아닌 문장 흐름 유지
- 마치 발표자가 말하는 듯한 스타일

📌 리포트 마지막 문단에는, 분석자의 관점에서 전체 결과를 종합한 시사점 또는 느낀 점을 **2~3줄 분량의 자연스러운 문장**으로 꼭 덧붙여 주세요.
예: "이러한 결과를 종합해볼 때, ○○ 개선이 향후 핵심이 될 수 있습니다."
"""

        # GPT 호출
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=1200
        )

        return response['choices'][0]['message']['content']

    except Exception as e:
        return f"❌ GPT 요약 생성 중 오류: {e}"

