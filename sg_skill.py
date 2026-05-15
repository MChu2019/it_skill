import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
import json
import re
from collections import Counter
from rapidfuzz import process, fuzz

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Skills of the Future", layout="wide")

st.title("🚀 Skills of the Future – High Performance Analytics")
st.markdown("Optimized for 1M+ rows using Unique Title Mapping and Vectorization.")

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("⚙ Configuration")
match_threshold = st.sidebar.slider("Matching Confidence (%)", 60, 95, 75)
high_pay_threshold = st.sidebar.slider("High Pay Threshold (SGD)", 4000, 20000, 6000)
chunk_size = st.sidebar.selectbox("Chunk Size", [100000, 200000, 500000], index=1)

# =========================================================
# HELPER FUNCTIONS
# =========================================================
@st.cache_data
def load_reference_data():
    try:
        ref_path = "JobsDatasetProcessed.csv"
        conn = duckdb.connect()
        # Use DuckDB to quickly load reference data
        ref_df = conn.execute(f"SELECT * FROM read_csv_auto('{ref_path}')").fetchdf()
        return ref_df
    except Exception as e:
        st.error(f"❌ Error loading reference dataset: {e}")
        return None

def clean_salary(series):
    """Vectorized salary cleaning"""
    return pd.to_numeric(series.astype(str).str.replace(r"[^0-9.-]", "", regex=True), errors='coerce')

def find_best_match(title, ref_titles, threshold):
    """Finds best match for a single unique title"""
    if not isinstance(title, str) or title == "Unknown":
        return None, 0
    res = process.extractOne(title, ref_titles, scorer=fuzz.token_sort_ratio)
    if res and res[1] >= threshold:
        return res[0], res[1]
    return None, 0

def parse_it_skills(skill_str):
    if pd.isna(skill_str): return []
    try:
        if skill_str.startswith('['):
            return [s.strip() for s in json.loads(skill_str)]
        return [s.strip() for s in skill_str.split(',') if s.strip()]
    except:
        return [skill_str]

# =========================================================
# CORE LOGIC
# =========================================================
ref_df = load_reference_data()
uploaded_file = st.file_uploader("Upload CSV file (1M+ rows)", type=["csv"])

if uploaded_file and ref_df is not None:
    st.subheader("📂 Processing Data")
    progress_bar = st.progress(0)
    
    # 1. READ & INITIAL CLEANING
    chunks = []
    # Get reference titles as a list once
    ref_titles_list = ref_df['Job Title'].astype(str).unique().tolist()

    reader = pd.read_csv(uploaded_file, chunksize=chunk_size, low_memory=False)
    
    for i, chunk in enumerate(reader):
        chunk.columns = chunk.columns.str.strip()
        # Standardize Columns
        if 'title' in chunk.columns: chunk.rename(columns={'title': 'Job Title'}, inplace=True)
        chunk['Job Title'] = chunk['Job Title'].fillna('Unknown')
        
        # Vectorized cleaning inside chunk
        if 'average_salary' in chunk.columns:
            chunk['Salary_Cleaned'] = clean_salary(chunk['average_salary'])
        
        if 'Posting Date' in chunk.columns:
            chunk['Posting_Date_Parsed'] = pd.to_datetime(chunk['Posting Date'], errors='coerce')
        
        chunks.append(chunk)
        progress_bar.progress(min((i + 1) * 5, 30))

    full_df = pd.concat(chunks, ignore_index=True)
    del chunks # Critical: Free up memory

    # 2. OPTIMIZED FUZZY MATCHING (Unique Titles Only)
    st.info("🔍 Matching unique job titles (this saves 99% of processing time)...")
    unique_titles = full_df['Job Title'].unique()
    
    # Map unique titles to matches
    match_map = {}
    for ut in unique_titles:
        match_title, score = find_best_match(ut, ref_titles_list, match_threshold)
        match_map[ut] = (match_title, score)
    
    # Fast vectorized mapping
    full_df['Matched_Job_Title'] = full_df['Job Title'].map(lambda x: match_map[x][0])
    full_df['Match_Score'] = full_df['Job Title'].map(lambda x: match_map[x][1])
    progress_bar.progress(60)

    # 3. MERGE SKILLS & PARSE
    merged = full_df.merge(ref_df[['Job Title', 'IT Skills']], 
                           left_on='Matched_Job_Title', 
                           right_on='Job Title', 
                           how='left', suffixes=('', '_ref'))
    
    merged['IT_Skills_List'] = merged['IT Skills'].apply(parse_it_skills)
    merged['High_Pay'] = merged['Salary_Cleaned'] >= high_pay_threshold
    merged['YearMonth'] = merged['Posting_Date_Parsed'].dt.to_period('M').astype(str)
    
    progress_bar.progress(100)
    st.success(f"✅ Processed {len(merged):,} rows")

    # =========================================================
    # DASHBOARD SECTION
    # =========================================================
    # Explode skills for fast analytics (one row per skill)
    exploded = merged.explode('IT_Skills_List')
    exploded = exploded[exploded['IT_Skills_List'].notna()]

    # Filter Sidebar
    st.sidebar.header("📊 Filters")
    all_industries = merged['Industry'].unique() if 'Industry' in merged.columns else ["Unknown"]
    selected_industry = st.sidebar.multiselect("Industry", all_industries, default=all_industries[:3])
    
    # Filtered View
    if 'Industry' in merged.columns:
        filtered_df = merged[merged['Industry'].isin(selected_industry)]
        filtered_exp = exploded[exploded['Industry'].isin(selected_industry)]
    else:
        filtered_df = merged
        filtered_exp = exploded

    # Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Jobs", f"{len(filtered_df):,}")
    m2.metric("Avg Salary", f"SGD {filtered_df['Salary_Cleaned'].mean():,.0f}")
    m3.metric("High Pay Jobs", f"{filtered_df['High_Pay'].sum():,}")

    # Top Skills Chart
    st.subheader("🏆 Top IT Skills in Demand")
    top_skills = filtered_exp['IT_Skills_List'].value_counts().head(15).reset_index()
    top_skills.columns = ['Skill', 'Count']
    fig_skills = px.bar(top_skills, x='Count', y='Skill', orientation='h', title="Skill Frequency")
    st.plotly_chart(fig_skills, use_container_width=True)

    # Drill Down
    st.subheader("🔍 Salary Distribution per Skill")
    skill_list = sorted(top_skills['Skill'].unique())
    selected_skill = st.selectbox("Pick a skill to analyze", skill_list)
    
    skill_data = filtered_exp[filtered_exp['IT_Skills_List'] == selected_skill]
    fig_box = px.box(skill_data, x='Matched_Job_Title', y='Salary_Cleaned', title=f"Pay for {selected_skill}")
    st.plotly_chart(fig_box, use_container_width=True)

    # Download
    st.download_button("📥 Download Matched Data", merged.to_csv(index=False), "results.csv", "text/csv")

else:
    st.info("👈 Please upload your Job Postings CSV to start.")



