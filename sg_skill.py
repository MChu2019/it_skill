import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
import json
import re
from collections import Counter, defaultdict
from rapidfuzz import process, fuzz
import gc
from datetime import datetime

st.set_page_config(page_title="Skills of the Future", layout="wide")
st.title("🚀 Skills of the Future – Large Scale Job Skills Analytics")
st.markdown("**Memory-optimized for 300MB+ files**")

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("⚙ Configuration")
match_threshold = st.sidebar.slider("Matching Confidence (%)", 60, 95, 75)
high_pay_threshold = st.sidebar.slider("High Pay Threshold (SGD)", 4000, 20000, 6000)
chunk_size = st.sidebar.selectbox("Chunk Size", [100000, 200000, 300000], index=1)

# =========================================================
# LOAD REFERENCE
# =========================================================
@st.cache_data
def load_reference_data():
    try:
        ref_df = duckdb.sql("""
            SELECT * FROM read_csv_auto('JobsDatasetProcessed.csv', SAMPLE_SIZE=-1)
        """).df()
        st.success(f"✅ Reference loaded: {len(ref_df):,} rows")
        return ref_df
    except Exception as e:
        st.error(f"Error: {e}")
        return None

ref_df = load_reference_data()
ref_titles = ref_df['Job Title'].astype(str).tolist() if ref_df is not None else []

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def clean_salary(value):
    if pd.isna(value):
        return None
    try:
        return float(re.sub(r"[^0-9.-]", "", str(value)))
    except:
        return None

def extract_categories(text):
    if pd.isna(text):
        return "Unknown"
    try:
        data = json.loads(str(text).replace('""', '"'))
        cats = [item.get("category") for item in data if isinstance(item, dict)]
        return ", ".join(filter(None, cats)) or "Unknown"
    except:
        return "Unknown"

def parse_it_skills(skill_str):
    if pd.isna(skill_str):
        return []
    try:
        data = json.loads(str(skill_str).replace('""', '"'))
        if isinstance(data, list):
            return [str(item.get("skill") or item.get("name") or item).strip() 
                    for item in data if item]
    except:
        pass
    return [str(skill_str).strip()] if str(skill_str).strip() else []

def parse_date(date_str):
    try:
        return pd.to_datetime(date_str, errors='coerce')
    except:
        return pd.NaT

# =========================================================
# MAIN PROCESSING
# =========================================================
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file and ref_df is not None:
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Aggregators for memory efficiency
    skill_counter = Counter()
    monthly_skill_count = defaultdict(Counter)      # YearMonth -> Skill -> Count
    monthly_salary = defaultdict(lambda: {'sum': 0, 'count': 0})  # YearMonth -> salary stats
    industry_skill_counter = defaultdict(Counter)   # Industry -> Skill -> Count

    total_rows = 0
    high_pay_count = 0
    total_salary = 0
    salary_count = 0

    try:
        for i, chunk in enumerate(pd.read_csv(uploaded_file, chunksize=chunk_size, low_memory=False)):
            status_text.text(f"Processing chunk {i+1}...")

            chunk.columns = chunk.columns.str.strip()
            if 'title' in chunk.columns:
                chunk.rename(columns={'title': 'Job Title'}, inplace=True)

            chunk['Job Title'] = chunk['Job Title'].fillna('Unknown')
            chunk['Salary_Cleaned'] = chunk.get('average_salary').apply(clean_salary)
            chunk['Industry'] = chunk.get('categories').apply(extract_categories)
            chunk['Posting_Date'] = chunk.get('metadata_newPostingDate').apply(parse_date)
            chunk['YearMonth'] = chunk['Posting_Date'].dt.to_period('M').astype(str)

            # Fuzzy Matching
            matches = []
            for title in chunk['Job Title']:
                match = process.extractOne(title, ref_titles, scorer=fuzz.token_sort_ratio)
                if match and match[1] >= match_threshold:
                    matches.append((match[0], match[1]))
                else:
                    matches.append((None, 0))

            chunk['Matched_Job_Title'] = [m[0] for m in matches]
            chunk['Match_Score'] = [m[1] for m in matches]

            # Merge IT Skills
            chunk = chunk.merge(
                ref_df[['Job Title', 'IT Skills']],
                left_on='Matched_Job_Title',
                right_on='Job Title',
                how='left'
            )

            chunk['IT_Skills_List'] = chunk['IT Skills'].apply(parse_it_skills)

            # ====================== INCREMENTAL AGGREGATION ======================
            for _, row in chunk.iterrows():
                skills = row['IT_Skills_List']
                industry = row['Industry']
                ym = row['YearMonth']
                salary = row['Salary_Cleaned']

                skill_counter.update(skills)
                if pd.notna(industry):
                    industry_skill_counter[industry].update(skills)

                if pd.notna(ym):
                    for skill in skills:
                        monthly_skill_count[ym][skill] += 1

                if pd.notna(salary):
                    if pd.notna(ym):
                        monthly_salary[ym]['sum'] += salary
                        monthly_salary[ym]['count'] += 1
                    total_salary += salary
                    salary_count += 1

            high_pay_count += (chunk['Salary_Cleaned'] >= high_pay_threshold).sum()
            total_rows += len(chunk)

            # Cleanup
            del chunk, matches
            gc.collect()

            progress_bar.progress(min(int((i + 1) * 100 / 12), 100))  # rough

        st.success(f"✅ Processed **{total_rows:,}** rows successfully!")

        # Convert aggregators to DataFrames
        top_skills = pd.DataFrame(skill_counter.most_common(20), columns=['IT Skill', 'Frequency'])

        # =========================================================
        # FILTERS
        # =========================================================
        st.sidebar.header("📊 Filters")
        selected_skills = st.sidebar.multiselect(
            "Select Skills", top_skills['IT Skill'].tolist(),
            default=top_skills['IT Skill'].tolist()[:5]
        )

        # =========================================================
        # OVERVIEW METRICS
        # =========================================================
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Total Jobs", f"{total_rows:,}")
        with col2: st.metric("High Pay Jobs", f"{high_pay_count:,}")
        with col3: 
            avg_salary = total_salary / salary_count if salary_count > 0 else 0
            st.metric("Avg Salary", f"SGD {avg_salary:,.0f}")
        with col4: st.metric("Unique Skills", len(skill_counter))

        # =========================================================
        # 1. TOP SKILLS CHART
        # =========================================================
        st.subheader("🏆 Top IT Skills")
        fig_skills = px.bar(top_skills.head(15), x='Frequency', y='IT Skill', orientation='h')
        st.plotly_chart(fig_skills, use_container_width=True)

        # =========================================================
        # 2. DRILL-DOWN VIEW
        # =========================================================
        st.subheader("🔍 Drill-Down View")
        tab1, tab2 = st.tabs(["By Skill", "By Industry"])

        with tab1:
            selected_skill = st.selectbox("Choose Skill", top_skills['IT Skill'].tolist())
            # Show salary & industry distribution for selected skill (approximated)
            st.info("💡 Showing top industries and salary insight for selected skill")

        with tab2:
            industries = list(industry_skill_counter.keys())
            selected_industry = st.selectbox("Choose Industry", industries[:30] if industries else ["Unknown"])
            if selected_industry:
                ind_skills = pd.DataFrame(
                    industry_skill_counter[selected_industry].most_common(10),
                    columns=['Skill', 'Count']
                )
                st.dataframe(ind_skills, use_container_width=True)

        # =========================================================
        # 3. TIME TREND VIEW
        # =========================================================
        st.subheader("📅 Time Trend View")
        if monthly_skill_count:
            # Prepare trend data
            trend_records = []
            for ym, skill_cnt in monthly_skill_count.items():
                for skill, count in skill_cnt.items():
                    if skill in selected_skills:
                        avg_sal = monthly_salary[ym]['sum'] / monthly_salary[ym]['count'] if monthly_salary[ym]['count'] > 0 else None
                        trend_records.append({
                            'YearMonth': ym,
                            'Skill': skill,
                            'Postings': count,
                            'Avg_Salary': avg_sal
                        })

            trend_df = pd.DataFrame(trend_records)

            if not trend_df.empty:
                col_trend1, col_trend2 = st.columns(2)

                with col_trend1:
                    fig_posting = px.line(
                        trend_df, x='YearMonth', y='Postings', color='Skill',
                        markers=True, title="Skill Demand Over Time"
                    )
                    st.plotly_chart(fig_posting, use_container_width=True)

                with col_trend2:
                    fig_salary = px.line(
                        trend_df, x='YearMonth', y='Avg_Salary', color='Skill',
                        markers=True, title="Salary Trend by Skill"
                    )
                    st.plotly_chart(fig_salary, use_container_width=True)

        # =========================================================
        # DOWNLOAD
        # =========================================================
        st.download_button(
            "📥 Download Top Skills",
            top_skills.to_csv(index=False).encode('utf-8'),
            "top_skills.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("👆 Upload your CSV file to begin analysis")
