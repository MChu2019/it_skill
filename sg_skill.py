import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
import json
import re
from collections import Counter
from rapidfuzz import process, fuzz
from datetime import datetime

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Skills of the Future", layout="wide")

st.title("🚀 Skills of the Future – Large Scale Job Skills Analytics")
st.markdown("Supports 1M+ rows using DuckDB + chunk processing")

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("⚙ Configuration")
match_threshold = st.sidebar.slider("Matching Confidence (%)", 60, 95, 75)
high_pay_threshold = st.sidebar.slider("High Pay Threshold (SGD)", 4000, 20000, 6000)
chunk_size = st.sidebar.selectbox("Chunk Size", [50000, 100000, 200000], index=1)

# =========================================================
# LOAD REFERENCE DATASET
# =========================================================
@st.cache_data
def load_reference_data():
    try:
        ref_path = "JobsDatasetProcessed.csv"

        query = f"""
        SELECT *
        FROM read_csv_auto('{ref_path}', SAMPLE_SIZE=-1)
        """

        conn = duckdb.connect()

        ref_df = conn.execute(query).fetchdf()

        st.success(f"✅ Loaded Reference Dataset: {len(ref_df):,} rows")

        return ref_df

    except Exception as e:
        st.error(f"❌ Error loading reference dataset: {e}")
        return None

ref_df = load_reference_data()

# =========================================================
# FILE UPLOAD
# =========================================================
uploaded_file = st.file_uploader(
    "Upload CSV file (supports 1M+ rows)",
    type=["csv", "txt"]
)

# =========================================================
# CLEANING FUNCTIONS
# =========================================================
def clean_salary(value):
    """Convert salary strings to numeric"""

    if pd.isna(value):
        return None

    value = str(value)

    # Remove currency and commas
    value = re.sub(r"[^0-9.-]", "", value)

    try:
        return float(value)
    except:
        return None


def standardize_category(text):
    """Standardize text categories"""

    if pd.isna(text):
        return "Unknown"

    return str(text).strip().title()


def parse_date(date_value):
    """Convert date to datetime"""

    try:
        return pd.to_datetime(date_value, errors='coerce')
    except:
        return pd.NaT


def parse_it_skills(skill_str):
    """Parse skills from JSON or comma-separated text"""

    if pd.isna(skill_str):
        return []

    try:
        if isinstance(skill_str, str):
            skills = json.loads(skill_str) if skill_str.strip().startswith('[') else skill_str.split(',')
            return [s.strip() for s in skills if s.strip()]
    except:
        return [skill_str.strip()] if isinstance(skill_str, str) else []

    return []


# =========================================================
# FUZZY MATCHING
# =========================================================
def find_best_match(title, ref_titles):

    if not isinstance(title, str):
        return None, 0

    '''
    match = process.extractOne(
        title,
        ref_titles,
        scorer=fuzz.token_sort_ratio
    )
    '''
    
    matches = process.cdist(
    chunk['Job Title'].astype(str).tolist(),
    ref_titles,
    scorer=fuzz.token_sort_ratio
    
    if match and match[1] >= match_threshold:
        return match[0], match[1]

    return None, 0

def extract_categories(text):
    if pd.isna(text):
        return None

    try:
        data = json.loads(text)   # convert string → Python list

        categories = [item.get("category") for item in data if "category" in item]

        return ", ".join(categories)  # join multiple categories

    except:
        return None
# =========================================================
# MAIN PROCESSING
# =========================================================
if uploaded_file and ref_df is not None:

    st.subheader("📂 Processing Large Dataset")

    # =====================================================
    # READ LARGE CSV IN CHUNKS
    # =====================================================
    chunks = []
    total_rows = 0

    progress = st.progress(0)

    try:
        for i, chunk in enumerate(pd.read_csv(uploaded_file,
                                              chunksize=chunk_size,
                                              low_memory=False)):

            # ================================================
            # STANDARDIZE COLUMN NAMES
            # ================================================
            chunk.columns = chunk.columns.str.strip()

            if 'title' in chunk.columns and 'Job Title' not in chunk.columns:
                chunk.rename(columns={'title': 'Job Title'}, inplace=True)

            # ================================================
            # HANDLE MISSING VALUES
            # ================================================
            chunk['Job Title'] = chunk['Job Title'].fillna('Unknown')

            # ================================================
            # CLEAN SALARY
            # ================================================
            if 'average_salary' in chunk.columns:
                chunk['Salary_Cleaned'] = chunk['average_salary'].apply(clean_salary)
            else:
                chunk['Salary_Cleaned'] = None

            # ================================================
            # STANDARDIZE INDUSTRY
            # ================================================
            if 'categories' in chunk.columns:
                chunk['Industry'] = chunk['categories'].apply(extract_categories)
            else:
                chunk['Industry'] = 'Unknown'

            # ================================================
            # PARSE DATES
            # ================================================
            if 'Posting Date' in chunk.columns:
                chunk['Posting_Date_Parsed'] = chunk['Posting Date'].apply(parse_date)
            else:
                chunk['Posting_Date_Parsed'] = pd.NaT

            # ================================================
            # FUZZY MATCHING
            # ================================================
            ref_titles = ref_df['Job Title'].astype(str).tolist()

            matches = []

            for title in chunk['Job Title']:
                best_match, score = find_best_match(title, ref_titles)
                matches.append((best_match, score))

            chunk['Matched_Job_Title'] = [m[0] for m in matches]
            chunk['Match_Score'] = [m[1] for m in matches]

            # ================================================
            # MERGE REFERENCE DATA
            # ================================================
            #merged_chunk = chunk.merge(
            #    ref_df[['Job Title', 'IT Skills']],
            #    left_on='Matched_Job_Title',
            #    right_on='Job Title',
            #    how='left'
            #)

            merged_chunk = chunk.merge(
            ref_df[['Job Title', 'IT Skills']],
            left_on='Matched_Job_Title',
            right_on='Job Title',
            how='left',
            suffixes=('', '_ref')
            )

            # ================================================
            # SKILLS PARSING
            # ================================================
            merged_chunk['IT_Skills_List'] = merged_chunk['IT Skills'].apply(parse_it_skills)

            chunks.append(merged_chunk)
            total_rows += len(chunk)

            progress.progress(min((i + 1) * 5, 100))

        # Combine all chunks
        merged = pd.concat(chunks, ignore_index=True)

        st.success(f"✅ Processed {total_rows:,} rows successfully")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
        st.stop()

    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================
    merged['High_Pay'] = merged['Salary_Cleaned'] >= high_pay_threshold

    merged['YearMonth'] = merged['Posting_Date_Parsed'].dt.to_period('M').astype(str)

    # =====================================================
    # SKILLS ANALYSIS
    # =====================================================
    all_skills = []

    for skills in merged['IT_Skills_List']:
        all_skills.extend(skills)

    skill_counts = Counter(all_skills)

    top_skills = pd.DataFrame(
        skill_counts.most_common(20),
        columns=['IT Skill', 'Frequency']
    )

    # =====================================================
    # SIDEBAR FILTERS
    # =====================================================
    st.sidebar.header("📊 Dashboard Filters")

    selected_industry = st.sidebar.multiselect(
        "Select Industry",
        merged['Industry'].dropna().unique(),
        default=merged['Industry'].dropna().unique()[:5]
    )

    selected_skill = st.sidebar.multiselect(
        "Select Skills",
        top_skills['IT Skill'].tolist(),
        default=top_skills['IT Skill'].tolist()[:5]
    )

    # Filter dataset
    filtered = merged[merged['Industry'].isin(selected_industry)]

    # =====================================================
    # OVERVIEW METRICS
    # =====================================================
    st.subheader("📈 Overview Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Jobs", f"{len(filtered):,}")

    with col2:
        st.metric("Matched Jobs", f"{filtered['IT Skills'].notna().sum():,}")

    with col3:
        st.metric(
            "Average Salary",
            f"SGD {filtered['Salary_Cleaned'].mean():,.0f}"
            if filtered['Salary_Cleaned'].notna().sum() > 0 else "N/A"
        )

    with col4:
        st.metric(
            "High Pay Jobs",
            f"{filtered['High_Pay'].sum():,}"
        )

    # =====================================================
    # TOP SALARY JOBS
    # =====================================================
    st.subheader("💰 Top Salary Job Postings")

    top_salary = filtered.sort_values(
        by='Salary_Cleaned',
        ascending=False
    ).head(20)

    st.dataframe(
        top_salary[[
            'Job Title',
            'Industry',
            'Salary_Cleaned',
            'Match_Score'
        ]],
        use_container_width=True
    )

    # =====================================================
    # TOP SKILLS CHART
    # =====================================================
    st.subheader("🏆 Top IT Skills")

    fig_skills = px.bar(
        top_skills.head(15),
        x='Frequency',
        y='IT Skill',
        orientation='h',
        title='Top 15 IT Skills',
        hover_data=['Frequency']
    )

    st.plotly_chart(fig_skills, use_container_width=True)

    # =====================================================
    # DRILL-DOWN VIEW BY SKILLS
    # =====================================================
    st.subheader("🔍 Drill-Down by Skills")

    drill_data = []

    for _, row in filtered.iterrows():

        for skill in row['IT_Skills_List']:

            drill_data.append({
                'Skill': skill,
                'Salary': row['Salary_Cleaned'],
                'Industry': row['Industry'],
                'Job Title': row['Job Title']
            })

    drill_df = pd.DataFrame(drill_data)

    if not drill_df.empty:

        selected_drill_skill = st.selectbox(
            "Select Skill",
            sorted(drill_df['Skill'].dropna().unique())
        )

        filtered_drill = drill_df[
            drill_df['Skill'] == selected_drill_skill
        ]

        fig_drill = px.box(
            filtered_drill,
            x='Industry',
            y='Salary',
            title=f'Salary Distribution for {selected_drill_skill}',
            hover_data=['Job Title']
        )

        st.plotly_chart(fig_drill, use_container_width=True)

        st.dataframe(filtered_drill.head(50), use_container_width=True)

    # =====================================================
    # TIME TREND VIEW
    # =====================================================
    st.subheader("📅 Time Trend by Skills")

    trend_data = []

    for _, row in filtered.iterrows():

        for skill in row['IT_Skills_List']:

            trend_data.append({
                'Skill': skill,
                'YearMonth': row['YearMonth']
            })

    trend_df = pd.DataFrame(trend_data)

    if not trend_df.empty:

        trend_summary = trend_df.groupby([
            'YearMonth',
            'Skill'
        ]).size().reset_index(name='Count')

        trend_summary = trend_summary[
            trend_summary['Skill'].isin(selected_skill)
        ]

        fig_trend = px.line(
            trend_summary,
            x='YearMonth',
            y='Count',
            color='Skill',
            markers=True,
            title='Skills Demand Over Time'
        )

        st.plotly_chart(fig_trend, use_container_width=True)

    # =====================================================
    # DOWNLOAD RESULTS
    # =====================================================
    st.subheader("📥 Download Results")

    csv = merged.to_csv(index=False).encode('utf-8')

    st.download_button(
        "Download Full Results",
        csv,
        "matched_jobs_with_skills.csv",
        "text/csv"
    )

else:

    st.info("👆 Upload a CSV file to begin analysis")






