import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
import json
import re
from collections import Counter
from rapidfuzz import process, fuzz
import gc

st.set_page_config(page_title="Skills of the Future", layout="wide")
st.title("🚀 Skills of the Future – Large Scale Job Skills Analytics")
st.markdown("**Optimized for 300MB+ files using DuckDB + streaming**")

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("⚙ Configuration")
match_threshold = st.sidebar.slider("Matching Confidence (%)", 60, 95, 75)
high_pay_threshold = st.sidebar.slider("High Pay Threshold (SGD)", 4000, 20000, 6000)
chunk_size = st.sidebar.selectbox("Chunk Size", [100000, 200000, 300000], index=1)

# =========================================================
# LOAD REFERENCE (once)
# =========================================================
@st.cache_data
def load_reference_data():
    try:
        ref_df = duckdb.sql(f"""
            SELECT * FROM read_csv_auto('JobsDatasetProcessed.csv', SAMPLE_SIZE=-1)
        """).df()
        st.success(f"✅ Reference loaded: {len(ref_df):,} rows")
        return ref_df
    except Exception as e:
        st.error(f"Error loading reference: {e}")
        return None

ref_df = load_reference_data()
ref_titles = ref_df['Job Title'].astype(str).tolist() if ref_df is not None else []

# =========================================================
# HELPER FUNCTIONS (Optimized)
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
        data = json.loads(text.replace('""', '"'))
        cats = [item.get("category") for item in data if isinstance(item, dict) and "category" in item]
        return ", ".join(cats) if cats else "Unknown"
    except:
        return "Unknown"

def parse_it_skills(skill_str):
    if pd.isna(skill_str):
        return []
    try:
        if isinstance(skill_str, str):
            data = json.loads(skill_str.replace('""', '"'))
            if isinstance(data, list):
                return [item.get("skill") or item.get("name") or str(item) 
                       for item in data if item]
    except:
        pass
    return [str(skill_str).strip()] if str(skill_str).strip() else []

# =========================================================
# MAIN PROCESSING
# =========================================================
uploaded_file = st.file_uploader("Upload CSV file (supports large files)", type=["csv"])

if uploaded_file and ref_df is not None:
    st.subheader("📂 Processing Large File...")

    progress_bar = st.progress(0)
    status_text = st.empty()

    conn = duckdb.connect(database=":memory:")

    all_skill_counter = Counter()
    total_rows = 0
    high_pay_count = 0
    salary_sum = 0
    salary_count = 0

    try:
        for i, chunk in enumerate(pd.read_csv(uploaded_file, chunksize=chunk_size, low_memory=False)):
            status_text.text(f"Processing chunk {i+1}...")

            # Basic cleaning
            chunk.columns = chunk.columns.str.strip()
            if 'title' in chunk.columns:
                chunk = chunk.rename(columns={'title': 'Job Title'})

            chunk['Job Title'] = chunk['Job Title'].fillna('Unknown')
            chunk['Salary_Cleaned'] = chunk['average_salary'].apply(clean_salary)
            chunk['Industry'] = chunk['categories'].apply(extract_categories)

            # Fuzzy Matching (still the heaviest part)
            matches = []
            for title in chunk['Job Title']:
                match = process.extractOne(title, ref_titles, scorer=fuzz.token_sort_ratio)
                if match and match[1] >= match_threshold:
                    matches.append((match[0], match[1]))
                else:
                    matches.append((None, 0))

            chunk['Matched_Job_Title'] = [m[0] for m in matches]
            chunk['Match_Score'] = [m[1] for m in matches]

            # Merge with reference (IT Skills)
            chunk = chunk.merge(
                ref_df[['Job Title', 'IT Skills']],
                left_on='Matched_Job_Title',
                right_on='Job Title',
                how='left',
                suffixes=('', '_ref')
            )

            chunk['IT_Skills_List'] = chunk['IT Skills'].apply(parse_it_skills)

            # === Incremental Aggregation ===
            for skills in chunk['IT_Skills_List']:
                all_skill_counter.update(skills)

            high_pay_count += (chunk['Salary_Cleaned'] >= high_pay_threshold).sum()
            valid_salaries = chunk['Salary_Cleaned'].dropna()
            salary_sum += valid_salaries.sum()
            salary_count += len(valid_salaries)
            total_rows += len(chunk)

            # Free memory
            del chunk, matches
            gc.collect()

            progress_bar.progress(min((i + 1) * 8, 100))   # rough estimate

        st.success(f"✅ Processed {total_rows:,} rows successfully!")

        # =========================================================
        # TOP SKILLS
        # =========================================================
        top_skills = pd.DataFrame(
            all_skill_counter.most_common(20),
            columns=['IT Skill', 'Frequency']
        )

        # =========================================================
        # DASHBOARD FILTERS
        # =========================================================
        st.sidebar.header("📊 Filters")
        selected_skills = st.sidebar.multiselect(
            "Select Skills", 
            top_skills['IT Skill'].tolist(),
            default=top_skills['IT Skill'].tolist()[:5]
        )

        # =========================================================
        # METRICS
        # =========================================================
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Jobs", f"{total_rows:,}")
        with col2:
            st.metric("High Pay Jobs", f"{high_pay_count:,}")
        with col3:
            avg_salary = salary_sum / salary_count if salary_count > 0 else 0
            st.metric("Avg Salary", f"SGD {avg_salary:,.0f}")
        with col4:
            st.metric("Unique Skills", len(all_skill_counter))

        # Top Skills Chart
        st.subheader("🏆 Top IT Skills")
        fig = px.bar(top_skills.head(15), x='Frequency', y='IT Skill', orientation='h')
        st.plotly_chart(fig, use_container_width=True)

        # =========================================================
        # DOWNLOAD (Only summary + top skills)
        # =========================================================
        summary_df = pd.DataFrame({
            'IT Skill': top_skills['IT Skill'],
            'Frequency': top_skills['Frequency']
        })

        st.download_button(
            "📥 Download Top Skills",
            summary_df.to_csv(index=False).encode('utf-8'),
            "top_skills.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error: {e}")

else:
    st.info("👆 Upload your CSV file to start analysis")
