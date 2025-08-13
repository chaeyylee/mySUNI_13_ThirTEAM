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

# 한글 폰트 설정
matplotlib.rc("font", family="Malgun Gothic")
plt.rcParams["axes.unicode_minus"] = False

# Flask 설정
app = Flask(__name__)
app.secret_key = 'your-secret-key'

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
    total_energy = df['electricity_kwh'].sum() + df['gas_nm3'].sum()
    defect_rate = total_defect / (total_production + 1e-5)

    return {
        'defect_rate': f"{defect_rate * 100:.1f}%",
        'production_qty': int(total_production),
        'energy_usage': f"{total_energy:.0f} kWh"
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

def get_energy_usage_chart(df):
    df["electricity_kwh"] = pd.to_numeric(df["electricity_kwh"], errors="coerce").fillna(0)
    df["gas_nm3"] = pd.to_numeric(df["gas_nm3"], errors="coerce").fillna(0)
    energy_avg = df.groupby("line_id")[["electricity_kwh", "gas_nm3"]].mean().reset_index()
    fig = px.bar(energy_avg, x="line_id", y=["electricity_kwh", "gas_nm3"],
                 barmode="group", title="Energy Usage by Line")
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)

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
            title=f"<b>장비 [{equip_id}] - 주요 키워드 트렌드</b>",
            xaxis_title="날짜",
            yaxis_title="언급 빈도",
            legend_title="상위 5개 키워드",
            height=400, # 그래프 높이 조절
            margin=dict(l=40, r=40, t=50, b=40)
        )
        
        plot_html = fig.to_html(full_html=False, include_plotlyjs=False)
        result_html[equip_id] = plot_html
        
    print("✅ [remark 트렌드 - Plotly] 전체 완료")
    return result_html


@app.route("/", methods=["GET"])
@app.route("/", methods=["GET"])
def index():
    print("✅ index() 진입")

    df = None
    kpis = {"defect_rate": "-", "production_qty": "-", "energy_usage": "-"}
    production_html, defect_html, energy_html = None, None, None
    spc_elec_img, spc_gas_img = None, None
    spc_by_line_html, spc_by_line_gas_html = None, None
    anova_results = None
    remark_top5 = []
    energy_trend_elec, energy_trend_gas = "", ""
    produced_chart_html, defect_chart_html = "", ""
    keyword_trend_html = {}

    if os.path.exists(RESULT_PATH):
        try:
            print("📂 result.csv 존재 확인됨")
            df = pd.read_csv(RESULT_PATH)
            print("✅ CSV 로딩 성공")
            
            df['date'] = pd.to_datetime(df['date'], errors="coerce")
            
            kpis = calculate_kpis(df)
            production_html = get_production_trend(df)
            defect_html = get_defect_rate_distribution(df)
            energy_html = get_energy_usage_chart(df)
            
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

        except Exception as e:
            print(f"❌ CSV 로드 또는 그래프 생성 오류: {e}")
            flash(f"❌ CSV 불러오기 또는 그래프 생성 중 오류가 발생했습니다: {e}")

    return render_template(
        "index.html",
        table=df.head().to_html(index=False, classes='table table-striped') if df is not None else None,
        kpis=kpis,
        production_chart=production_html,
        defect_chart=defect_html,
        energy_chart=energy_html,
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
        keyword_graphs=bool(keyword_trend_html)
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
    except Exception as e:
        print(f"❌ 전처리 오류: {e}")
        flash(f"❌ 전처리 중 오류가 발생했습니다: {str(e)}")
        return redirect(url_for("index"))
        
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)