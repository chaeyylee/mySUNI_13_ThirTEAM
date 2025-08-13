from flask import Flask, render_template, request, redirect, url_for, flash
from pathlib import Path
import pandas as pd
import os
import plotly.express as px
from plotly.offline import plot
import plotly.io as pio
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import base64
from io import BytesIO
import statsmodels.api as sm
import statsmodels.formula.api as smf
from collections import Counter, defaultdict
import openai
from forecast_utils import poly_forecast
import plotly.graph_objects as go
import platform
import matplotlib.font_manager as fm

# 한글 폰트 설정
font_path = os.path.join("fonts", "NanumGothic.ttf")  # 또는 malgun.ttf
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rc("font", family=font_name)
plt.rcParams["axes.unicode_minus"] = False

# Flask 설정
app = Flask(__name__)
app.secret_key = "my-dev-secret-key"
UPLOAD_FOLDER = "uploads"
RESULT_PATH = os.path.join(UPLOAD_FOLDER, "result.csv")
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)

from data_processing import run_preprocessing

# KPI 계산 함수
def calculate_kpis(df):
    df['produced_qty'] = pd.to_numeric(df['produced_qty'], errors='coerce')
    df['defect_qty'] = pd.to_numeric(df['defect_qty'], errors='coerce')
    df['electricity_kwh'] = pd.to_numeric(df['electricity_kwh'], errors='coerce')
    df['gas_nm3'] = pd.to_numeric(df['gas_nm3'], errors='coerce')

    total_production = df['produced_qty'].sum()
    total_defect = df['defect_qty'].sum()
    defect_rate = total_defect / (total_production + 1e-5)

    # 평균 생산량과 평균 전력 사용량 (행 개수 기준)
    avg_production = df['produced_qty'].mean()
    avg_electricity = df['electricity_kwh'].mean()

    return {
        'defect_rate': f"{defect_rate * 100:.1f}%",
        'average_qty': f"{avg_production:.0f}",
        'average_electricity': f"{avg_electricity:.0f} kWh"
    }

# 그래프 함수 (생략 없이 그대로 유지)
def get_production_trend(df):
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["produced_qty"] = pd.to_numeric(df["produced_qty"], errors="coerce").fillna(0)
    daily_avg = df.groupby(["date", "line_id"])["produced_qty"].mean().reset_index()
    fig = px.line(daily_avg, x="date", y="produced_qty", color="line_id",
                  title="Production Trend (Daily Avg)")
    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

def get_defect_rate_distribution(df):
    df["produced_qty"] = pd.to_numeric(df["produced_qty"], errors="coerce").fillna(0)
    df["defect_qty"] = pd.to_numeric(df["defect_qty"], errors="coerce").fillna(0)
    df = df[df["produced_qty"] > 0].copy()
    df["defect_rate"] = df["defect_qty"] / df["produced_qty"]
    df = df[df["defect_rate"] > 0]
    fig = px.box(
        df,
        x="line_id",
        y="defect_rate",
        title="Defect Rate Distribution (Non-zero Only)",
        category_orders={"line_id": sorted(df["line_id"].dropna().unique())},
        points="outliers"
    )
    fig.update_layout(yaxis_range=[0, 0.03])
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)

def pareto_defect_lines_by_line(df):
    import plotly.graph_objects as go

    df = df.copy()
    df["defect_qty"] = pd.to_numeric(df["defect_qty"], errors="coerce").fillna(0)
    df = df.dropna(subset=["line_id"])

    line_order = ["L1", "L2", "L3"]
    defect_sum = df.groupby("line_id")["defect_qty"].sum().reindex(line_order).fillna(0)

    fig = go.Figure()

    # 막대그래프: 불량 수량
    fig.add_trace(go.Bar(
        x=defect_sum.index,
        y=defect_sum.values,
        name="Number of Defects",
        marker_color="indianred",
        width=[0.6] * len(defect_sum)
    ))

    # 레이아웃 설정
    fig.update_layout(
        title="Defects by Line",
        xaxis_title="line_id",
        yaxis=dict(title="Number of Defects"),
        legend=dict(x=0.5, y=1.1, orientation="h"),
        bargap=0.3
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)

def pareto_defect_equipment(df):
    df = df.copy()
    df["defect_qty"] = pd.to_numeric(df["defect_qty"], errors="coerce").fillna(0)
    df = df.dropna(subset=["equipment_id"])

    # 공정(장비)별 불량량 합계
    defect_sum = df.groupby("equipment_id")["defect_qty"].sum().sort_values(ascending=False)
    cumulative = defect_sum.cumsum() / defect_sum.sum() * 100

    import plotly.graph_objects as go
    fig = go.Figure()

    # 막대그래프: 불량량
    fig.add_trace(go.Bar(
        x=defect_sum.index,
        y=defect_sum.values,
        name="Number of Defects",
        marker=dict(color="green")
    ))

    fig.update_layout(
        xaxis=dict(
            title="equipment_id",
            categoryorder='array',
            categoryarray=["EQ101", "EQ203", "EQ305", "EQ400", "장비 점검 미실시"]  # 원하는 순서
        ),
        yaxis=dict(title="Number of Defects"),
        title="Defects by Equipment",
        bargap=0.3
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)

# SPC X bar chart 출력 (Matplotlib)
def spc_chart(series, name):
    mean = series.mean()
    std = series.std()
    ucl = mean + 3 * std
    lcl = mean - 3 * std
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(series.index, series.values, label=name)
    ax.axhline(mean, color='green', linestyle='--', label='Mean')
    ax.axhline(ucl, color='red', linestyle='--', label='UCL')
    ax.axhline(lcl, color='red', linestyle='--', label='LCL')
    ax.set_title(f'SPC Chart - {name}')
    ax.legend()
    ax.grid(True)
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# SPC X bar chart 출력 (Plotly)
def spc_chart_by_line(df, y_col='electricity_kwh'):
    df_summary = df.groupby(['date', 'line_id'])[y_col].mean().reset_index()
    stats = df_summary.groupby('line_id')[y_col].agg(['mean', 'std']).reset_index()
    stats['ucl'] = stats['mean'] + 3 * stats['std']
    stats['lcl'] = stats['mean'] - 3 * stats['std']

    fig = go.Figure()
    for line_name in df_summary['line_id'].unique():
        df_line = df_summary[df_summary['line_id'] == line_name]
        stat_line = stats[stats['line_id'] == line_name].iloc[0]
        fig.add_trace(go.Scatter(x=df_line['date'], y=df_line[y_col],
                                 mode='lines+markers', name=f'{line_name}'))
        fig.add_trace(go.Scatter(x=df_line['date'], y=[stat_line['mean']] * len(df_line),
                                 mode='lines', name=f'{line_name} Mean',
                                 line=dict(dash='dash', color='green'), showlegend=False))
        fig.add_trace(go.Scatter(x=df_line['date'], y=[stat_line['ucl']] * len(df_line),
                                 mode='lines', name=f'{line_name} UCL',
                                 line=dict(dash='dot', color='red'), showlegend=False))
        fig.add_trace(go.Scatter(x=df_line['date'], y=[stat_line['lcl']] * len(df_line),
                                 mode='lines', name=f'{line_name} LCL',
                                 line=dict(dash='dot', color='red'), showlegend=False))
    fig.update_layout(
        title=f"SPC Chart - {y_col} (by line)",
        xaxis_title="Date",
        yaxis_title=y_col,
        height=500
    )
    return fig.to_html(full_html=False)

# ANOVA를 이용한 분석
def factory_energy_anova(df):
    results = {}
    anova_df = df[['factory_id', 'electricity_kwh', 'gas_nm3']].dropna()

    model_elec = smf.ols('electricity_kwh ~ C(factory_id)', data=anova_df).fit()
    anova_elec = sm.stats.anova_lm(model_elec, typ=2)
    results["elec"] = {
        "anova_table": anova_elec.to_html(classes="table table-bordered"),
        "p_value": anova_elec.loc['C(factory_id)', 'PR(>F)'],
        "is_significant": anova_elec.loc['C(factory_id)', 'PR(>F)'] < 0.05
    }

    model_gas = smf.ols('gas_nm3 ~ C(factory_id)', data=anova_df).fit()
    anova_gas = sm.stats.anova_lm(model_gas, typ=2)
    results["gas"] = {
        "anova_table": anova_gas.to_html(classes="table table-bordered"),
        "p_value": anova_gas.loc['C(factory_id)', 'PR(>F)'],
        "is_significant": anova_gas.loc['C(factory_id)', 'PR(>F)'] < 0.05
    }
    return results

# Remark analysis
def get_top_remark_issues(df, top_n=5):
    df = df.copy()
    df['remark'] = df['remark'].fillna('')
    # 'has_issue' 열은 더 이상 사용하지 않으므로 간소화
    top_rows = (
        df[df['remark'] != '']
        .drop_duplicates(subset=['date', 'remark'])
        .sort_values('date', ascending=False)
        .head(top_n)[['date', 'remark']]
        .to_dict(orient='records')
    )
    return top_rows

# 날짜 별 에너지 소비 트렌드
def generate_energy_trend_split_charts(df):
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    daily_avg = df.groupby('date')[['electricity_kwh', 'gas_nm3']].mean()

    fig_elec = px.line(daily_avg, x=daily_avg.index, y='electricity_kwh', title='Daily Electricity Consumption Trend')
    energy_trend_elec = plot(fig_elec, output_type='div', include_plotlyjs=False)

    fig_gas = px.line(daily_avg, x=daily_avg.index, y='gas_nm3', title='Daily Gas Consumption Trend')
    energy_trend_gas = plot(fig_gas, output_type='div', include_plotlyjs=False)

    return energy_trend_elec, energy_trend_gas

# 공장/라인 별 생산량/불량률
def generate_factory_line_bar_charts_plotly(df):
    summary = df.groupby(['factory_id', 'line_id'])[['produced_qty', 'defect_qty']].mean().reset_index()

    fig_produced = px.bar(summary, x="factory_id", y="produced_qty", color="line_id", barmode="group", title="Produced Quantity per Factory & Line")
    produced_chart_html = fig_produced.to_html(full_html=False)

    defect_fig = px.bar(summary, x="factory_id", y="defect_qty", color="line_id", barmode="group", title="Defect Quantity per Factory & Line")
    defect_chart_html = defect_fig.to_html(full_html=False)

    return produced_chart_html, defect_chart_html

import plotly.graph_objects as go

def generate_remark_keyword_trend_plotly(df):

    print("✅ [remark 트렌드 - Plotly] 분석 시작...")
    if 'remark_keywords' not in df.columns or df['remark_keywords'].dropna().empty:
        print("⚠️ 'remark_keywords' 열이 없거나 비어있어 트렌드 분석을 건너뜁니다.")
        return {}

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    analysis_df = df.dropna(subset=['equipment_id', 'date', 'remark_keywords']).copy()

    result_html = {} 
    grouped = analysis_df.groupby('equipment_id')

    for equip_id, group in grouped:
        print(f"  - 장비 ID: {equip_id} 분석 중...")

        all_keywords_flat = ','.join(group['remark_keywords'].dropna()).split(',')
        all_keywords = [kw.strip() for kw in all_keywords_flat if kw.strip()]
        if not all_keywords: continue
        
        counter = Counter(all_keywords)
        top_keywords = [kw for kw, _ in counter.most_common(5)]
        if not top_keywords: continue
        
        group = group.sort_values("date")
        daily_keywords_grouped = (
            group.groupby("date")["remark_keywords"]
            .apply(lambda x: ", ".join(x))
            .apply(lambda x: [kw.strip() for kw in x.split(",") if kw.strip()])
        )
        trend_data = defaultdict(lambda: defaultdict(int))
        for date, keywords_in_day in daily_keywords_grouped.items():
            for kw in keywords_in_day:
                if kw in top_keywords:
                    trend_data[kw][date] += 1
        
        if not trend_data: continue
        trend_df = pd.DataFrame(trend_data).fillna(0).sort_index()
        if trend_df.empty: continue

        fig = go.Figure()

        for keyword in trend_df.columns:
            fig.add_trace(go.Scatter(
                x=trend_df.index, 
                y=trend_df[keyword],
                mode='lines+markers',
                name=keyword
            ))

        fig.update_layout(
            title=dict(
                text=f"<b>[{equip_id}]</b>",
                font=dict(size=14)
            ),
            xaxis_title="date",
            yaxis_title="Mention Count",
            legend_title="Top 5 Keywords",
            height=400,
            margin=dict(l=40, r=40, t=50, b=40)
        )

        plot_html = fig.to_html(full_html=False, include_plotlyjs=False)
        result_html[equip_id] = plot_html
        
    print("✅ [remark 트렌드 - Plotly] 전체 완료")
    return result_html


def load_openai_summary_table_html():
    path = os.path.join("figs", "openai_summary.txt")
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if len(lines) < 3:
        return None

    # 첫 줄: 헤더
    headers = [cell.strip() for cell in lines[0].split("|") if cell.strip()]

    # 3번째 줄부터 데이터 행 추출 (중간의 구분선은 무시)
    rows = []
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.split("|") if cell.strip()]
        if len(cells) == len(headers):
            rows.append(cells)

    if not rows:
        return None

    # HTML 테이블 생성
    html = '<table class="gpt-table">\n  <thead><tr>'
    html += ''.join(f"<th>{h}</th>" for h in headers)
    html += '</tr></thead>\n  <tbody>\n'
    for row in rows:
        html += '    <tr>' + ''.join(f"<td>{cell}</td>" for cell in row) + '</tr>\n'
    html += '  </tbody>\n</table>'
    return html


# summary 요약
def generate_summary_input(df):
    import pandas as pd
    from collections import Counter
    import re

    input_lines = []

    # 1. 총 생산량 및 평균 불량률
    try:
        df['produced_qty'] = pd.to_numeric(df['produced_qty'], errors='coerce')
        df['defect_qty'] = pd.to_numeric(df['defect_qty'], errors='coerce')
        total_production = df['produced_qty'].sum()
        avg_defect_rate = (df['defect_qty'].sum() / (total_production + 1e-5)) * 100
        input_lines.append(f"📈 총 생산량은 {int(total_production)}개이고, 평균 불량률은 {avg_defect_rate:.2f}%입니다.")
    except Exception as e:
        print("❗ 불량률 분석 오류:", e)
        input_lines.append("❗ 불량률 분석을 위한 데이터가 부족합니다.")

    # 2. 에너지 사용량
    try:
        elec = pd.to_numeric(df.get("electricity_kwh", df.get("electricity_usage")), errors='coerce').sum()
        gas = pd.to_numeric(df.get("gas_nm3", df.get("gas_usage")), errors='coerce').sum()
        input_lines.append(f"⚡ 총 전력 사용량은 {elec:,.1f} kWh, 가스 사용량은 {gas:,.1f} Nm³로 확인됩니다.")
    except:
        input_lines.append("❗ 에너지 사용량 정보를 불러오지 못했습니다.")

    # 3. ANOVA 분석 결과
    try:
        anova_results = factory_energy_anova(df)
        print("✅ ANOVA 결과:", anova_results)
        input_lines.append("📊 공장 간 에너지 사용량 차이 분석 결과:")

        for k, v in anova_results.items():
            p = v["p_value"]
            if v["is_significant"]:
                input_lines.append(f"🔹 {k.upper()} 사용량에 대해 공장 간 유의한 차이가 있습니다 (p-value={p:.4f})")
            else:
                input_lines.append(f"🔹 {k.upper()} 사용량에 대해 공장 간 유의한 차이는 없습니다 (p-value={p:.4f})")

    except Exception as e:
        print("❗ ANOVA 분석 실패:", e)
        input_lines.append("❗ ANOVA 분석에 실패했습니다.")

    # 4. 장비별 키워드
    try:
        valid_df = df.dropna(subset=["equipment_id", "remark_keywords"])
        result = {}
        for eq_id, group in valid_df.groupby("equipment_id"):
            keywords = []
            for kw_list in group["remark_keywords"]:
                if isinstance(kw_list, str) and kw_list.strip():
                    keywords.extend(re.findall(r"\w+", kw_list))
            if keywords:
                top = Counter(keywords).most_common(3)
                result[eq_id] = [kw for kw, _ in top]
        for eq_id, keywords in result.items():
            input_lines.append(f"🔍 장비 '{eq_id}'에서 '{', '.join(keywords)}' 키워드가 자주 언급되었습니다.")
    except:
        input_lines.append("❗ 장비별 키워드 분석 실패")

    # 5. SPC 이상 감지
    try:
        if "spc_outlier" in df.columns:
            outliers = int(df["spc_outlier"].sum())
            if outliers > 0:
                input_lines.append(f"⚠️ SPC 분석 결과, {outliers}건의 이상값이 탐지되었습니다.")
            else:
                input_lines.append("✅ SPC 분석 결과, 이상값은 탐지되지 않았습니다.")
    except:
        input_lines.append("❗ SPC 이상 탐지 분석 실패")

    # 5-1. 라인별 SPC 이상 감지 결과 요약 (Plotly 기준)
    try:
        spc_input = df.groupby(["date", "line_id"])["electricity_kwh"].mean().reset_index()
        outlier_lines = []
        for line in spc_input["line_id"].unique():
            values = spc_input[spc_input["line_id"] == line]["electricity_kwh"]
            mean = values.mean()
            std = values.std()
            ucl = mean + 3 * std
            lcl = mean - 3 * std
            outliers = values[(values > ucl) | (values < lcl)]
            if not outliers.empty:
                outlier_lines.append(line)
        input_lines.append("📊 SPC 이상 탐지 (라인별):")
        if outlier_lines:
            input_lines.append(f"🔹 다음 라인에서 이상 전력 사용 패턴이 감지됨: {', '.join(outlier_lines)}")
        else:
            input_lines.append("🔹 모든 라인의 전력 사용이 SPC 기준 내에 있음.")
    except Exception as e:
        print("❗ SPC Plotly 분석 실패:", e)
        input_lines.append("❗ SPC Plotly 분석 실패")

    # 5-2. 라인별 생산량 추이 분석
    try:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["produced_qty"] = pd.to_numeric(df["produced_qty"], errors="coerce").fillna(0)
        line_trend = df.groupby("line_id")["produced_qty"].mean()
        rising_lines = line_trend[line_trend > line_trend.mean()].index.tolist()
        falling_lines = line_trend[line_trend < line_trend.mean()].index.tolist()

        input_lines.append("📈 라인별 생산량 추이 분석:")
        if rising_lines:
            input_lines.append(f"🔹 평균 이상 생산량을 보인 라인: {', '.join(rising_lines)}")
        if falling_lines:
            input_lines.append(f"🔹 평균 이하 생산량을 보인 라인: {', '.join(falling_lines)}")
    except Exception as e:
        print("❗ 라인별 추이 분석 실패:", e)
        input_lines.append("❗ 라인별 추이 분석 실패")

    # 6. 클러스터별 대표 문장
    try:
        if "remark_cluster" in df.columns and "remark" in df.columns:
            cluster_summary = (
                df.groupby("remark_cluster")["remark"]
                .apply(lambda x: x.mode().iloc[0] if not x.mode().empty else "")
                .dropna().to_dict()
            )
            if cluster_summary:
                input_lines.append("📌 클러스터별 대표 문장:")
                for cluster_id, rep in cluster_summary.items():
                    input_lines.append(f"- 클러스터 {cluster_id}: {rep}")
    except:
        input_lines.append("❗ 클러스터 대표 문장 요약 실패")

    # 7. 최근 이슈 5건
    try:
        if "remark" in df.columns and "date" in df.columns:
            top_remarks = (
                df.dropna(subset=["remark", "date"])
                .sort_values("date", ascending=False)
                .head(5)
            )
            input_lines.append("📝 최근 주요 이슈 요약:")
            for _, row in top_remarks.iterrows():
                date_str = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
                input_lines.append(f"- {date_str}: {row['remark']}")
    except:
        input_lines.append("❗ 최근 이슈 요약 실패")

    return "\n".join(input_lines)

# openai 부르기 (summary)
from analysis import generate_summary_narrative

from flask import Flask
from dotenv import load_dotenv
import os
from openai import OpenAI
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def call_openai_summary(summary_input):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "너는 공장 데이터를 분석해 사람이 발표할 수 있는 자연어 요약 보고서를 작성하는 전문가야."},
                {"role": "user", "content": summary_input}
            ],
            temperature=0.5,
        )
        raw = response.choices[0].message.content
        cleaned = raw.lstrip("\n\r\t ")
        return cleaned
    except Exception as e:
        return f"❌ GPT 응답 실패: {e}"


@app.route("/", methods=["GET"])
def index():
    global result_df, processing_done, gpt_summary

    print("✅ index() 진입")

    gpt_summary = ""
    df = None
    kpis = {"defect_rate": "-", "average_qty": "-", "average_electricity": "-"}
    pareto_chart_line = None
    pareto_chart_html = None
    production_html, defect_html, energy_html = None, None, None
    spc_elec_img, spc_gas_img = None, None
    spc_by_line_html, spc_by_line_gas_html = None, None
    anova_results = None
    remark_top5 = []
    energy_trend_elec, energy_trend_gas = "", ""
    produced_chart_html, defect_chart_html = "", ""
    keyword_trend_html = {}
    gpt_summary_html = None
    forecast_elec_html = None
    forecast_gas_html = None

    if os.path.exists(RESULT_PATH):
        try:
            print("📂 result.csv 존재 확인됨")
            df = pd.read_csv(RESULT_PATH)
            print("✅ CSV 로딩 성공")
            
            df['date'] = pd.to_datetime(df['date'], errors="coerce")

            # --- electricity/gas 예측 코드 추가 ---
            from forecast_utils import poly_forecast
            import plotly.graph_objects as go
            import plotly.io as pio

            daily = (df.groupby("date")[["electricity_kwh", "gas_nm3"]]
                     .sum()
                     .sort_index()
                     .reset_index())

            electricity = daily.set_index("date")["electricity_kwh"]
            gas = daily.set_index("date")["gas_nm3"]

            elec_poly = poly_forecast(electricity, degree=2, n_train=45, horizon=5)
            gas_poly = poly_forecast(gas, degree=2, n_train=45, horizon=5)

            def forecast_plot(df_orig, df_forecast, title):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_orig.index, y=df_orig.values,
                                         mode='lines', name='Observed'))
                fig.add_trace(go.Scatter(x=df_forecast["date"], y=df_forecast["forecast"],
                                         mode='lines+markers', name='Forecast'))
                fig.add_trace(go.Scatter(x=df_forecast["date"], y=df_forecast["lower_approx"],
                                         mode='lines', name='Lower Band', line=dict(dash='dot')))
                fig.add_trace(go.Scatter(x=df_forecast["date"], y=df_forecast["upper_approx"],
                                         mode='lines', name='Upper Band', line=dict(dash='dot')))
                fig.update_layout(title=title, xaxis_title="Date", yaxis_title=title.split()[0])
                return pio.to_html(fig, full_html=False)

            forecast_elec_html = forecast_plot(electricity, elec_poly, "Electricity Forecast")
            forecast_gas_html = forecast_plot(gas, gas_poly, "Gas Forecast")
            # --- 예측 코드 끝 ---

            print("🔍 장비 및 키워드 Null 값 수:\n", df[['equipment_id', 'remark_keywords']].isnull().sum())
            print("🔍 장비 및 키워드 샘플:\n", df[['equipment_id', 'remark_keywords']].dropna().head())

            kpis = calculate_kpis(df)
            production_html = get_production_trend(df)
            defect_html = get_defect_rate_distribution(df)
            pareto_chart_line = pareto_defect_lines_by_line(df)
            pareto_chart_html = pareto_defect_equipment(df)

            elec_series = df.set_index("date")["electricity_kwh"].dropna()
            gas_series = df.set_index("date")["gas_nm3"].dropna()
            if not elec_series.empty:
                spc_elec_img = spc_chart(elec_series, "electricity_kwh")
            if not gas_series.empty:
                spc_gas_img = spc_chart(gas_series, "gas_nm3")

            spc_by_line_html = spc_chart_by_line(df, y_col='electricity_kwh')
            spc_by_line_gas_html = spc_chart_by_line(df, y_col='gas_nm3')
            anova_results = factory_energy_anova(df)
            remark_top5 = get_top_remark_issues(df)
            energy_trend_elec, energy_trend_gas = generate_energy_trend_split_charts(df)
            produced_chart_html, defect_chart_html = generate_factory_line_bar_charts_plotly(df)

            if "remark_keywords" in df.columns:
                keyword_trend_html = generate_remark_keyword_trend_plotly(df)
                print(f"📊 생성된 키워드 트렌드 그래프 수: {len(keyword_trend_html)}")

            from analysis import generate_summary_narrative
            summary_input = generate_summary_input(df)
            gpt_summary = call_openai_summary(summary_input)
            print("🧠 GPT 요약:\n", gpt_summary)
            gpt_summary_html = load_openai_summary_table_html()

        except Exception as e:
            print(f"❌ CSV 로드 또는 그래프 생성 오류: {e}")
            flash(f"❌ CSV 불러오기 또는 그래프 생성 중 오류가 발생했습니다: {e}")
            gpt_summary = f"❌ 요약 생성 실패: {e}"

    return render_template(
        "index.html",
        table=df.head().to_html(index=False, classes='table table-striped') if df is not None else None,
        kpis=kpis,
        production_chart=production_html,
        defect_chart=defect_html,
        pareto_chart_line=pareto_chart_line,
        pareto_chart=pareto_chart_html,
        spc_elec=spc_elec_img,
        spc_gas=spc_gas_img,
        spc_by_line=spc_by_line_html,
        spc_by_line_gas=spc_by_line_gas_html,
        anova_results=anova_results,
        remark_top5=remark_top5,
        energy_trend_elec=energy_trend_elec,
        energy_trend_gas=energy_trend_gas,
        produced_chart=produced_chart_html,
        defect_bar_chart=defect_chart_html,
        keyword_trend_html=keyword_trend_html,
        keyword_graphs=bool(keyword_trend_html),
        summary=gpt_summary,
        gpt_summary_html=gpt_summary_html,
        forecast_elec=forecast_elec_html,
        forecast_gas=forecast_gas_html
    )

@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("files")
    if not files or files[0].filename == "":
        flash("❌ 파일을 업로드해주세요.")
        return redirect(url_for("index"))

    # 기존 파일 삭제
    for f_name in os.listdir(UPLOAD_FOLDER):
        os.remove(os.path.join(UPLOAD_FOLDER, f_name))
        print(f"🗑️ 기존 파일 삭제: {f_name}")

    for f in files:
        if f and f.filename.endswith(".csv"):
            save_path = os.path.join(UPLOAD_FOLDER, f.filename)
            f.save(save_path)
            print(f"✔️ 저장됨: {f.filename}")
            
    try:
        # data_processing.py의 전처리 함수 실행
        df = run_preprocessing(Path(UPLOAD_FOLDER))
        df.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")
        print("✅ result.csv 저장 완료")
        flash("✔️ 파일 업로드 및 데이터 전처리가 완료되었습니다.")

        # remark_ai.py
        from remark_ai import run_remark_analysis
        run_remark_analysis(
            input_csv_path=os.path.abspath(RESULT_PATH),
            output_csv_path=os.path.abspath("result_merged.csv"),
            output_dir=os.path.abspath("figs")
        )
        print("📊 remark_ai 자동 실행 완료")

    except Exception as e:
        print(f"❌ 전처리 오류: {e}")
        flash(f"❌ 전처리 중 오류가 발생했습니다: {str(e)}")
        return redirect(url_for("index"))
        
    return redirect(url_for("index"))

@app.route("/summary")
def summary():
    df = pd.read_csv(RESULT_PATH)
    summary_input = generate_summary_input(df)
    print("🔍 summary_input 확인:\n", summary_input)

    # GPT 요약 요청
    try:
        gpt_summary = call_openai_summary(summary_input)
        print("📋 GPT 결과:\n", gpt_summary)
    except Exception as e:
        gpt_summary = f"❌ GPT 요약 실패: {str(e)}"
        print(gpt_summary)

    return render_template("index.html", summary=gpt_summary)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)

