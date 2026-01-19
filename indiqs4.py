import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Student Assessment Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- 1. Robust Data Loading Function ---
@st.cache_data
def process_excel_file(uploaded_file):
    try:
        xls = pd.ExcelFile(uploaded_file)
        
        # Helper: Find sheet by keyword
        def get_sheet(keywords):
            for sheet in xls.sheet_names:
                if any(k.lower() in sheet.lower() for k in keywords):
                    return sheet
            return None

        bl_sheet = get_sheet(["baseline", "bl"])
        el_sheet = get_sheet(["endline", "el"])
        key_sheet = get_sheet(["answer", "key"])

        if not all([bl_sheet, el_sheet, key_sheet]):
            return None, "Missing sheets. Please ensure Baseline, Endline, and Answer Key sheets exist."

        # Read Data
        bl_df = pd.read_excel(uploaded_file, sheet_name=bl_sheet)
        el_df = pd.read_excel(uploaded_file, sheet_name=el_sheet)
        key_df = pd.read_excel(uploaded_file, sheet_name=key_sheet)

        # Force Assessment Labels
        bl_df['Assessment Type'] = 'Baseline'
        el_df['Assessment Type'] = 'Endline'

    except Exception as e:
        return None, f"Error reading file: {str(e)}"

    # Normalize Columns
    def normalize_cols(df):
        df.columns = [c.strip() for c in df.columns]
        col_map = {
            'grade': 'Grade', 'Class': 'Grade', 
            'Student ID': 'Student ID', 'student id': 'Student ID', 'ID': 'Student ID',
            'Center': 'Center', 'center': 'Center'
        }
        df.rename(columns=col_map, inplace=True)
        return df

    bl_df = normalize_cols(bl_df)
    el_df = normalize_cols(el_df)
    key_df = normalize_cols(key_df)

    # Parse JSON/Values
    def extract_value(cell_value):
        if pd.isna(cell_value): return None
        try:
            if isinstance(cell_value, (int, float)): return int(cell_value)
            s_val = str(cell_value).strip()
            if s_val.isdigit(): return int(s_val)
            if "{" in s_val:
                data = json.loads(s_val)
                return int(data.get('value'))
        except: return None
        return None

    # Process Q Columns
    def process_q_cols(df):
        q_cols = [c for c in df.columns if c.upper().startswith('Q') and c[1:].isdigit()]
        for col in q_cols:
            df[col] = df[col].apply(extract_value)
        return df, q_cols

    bl_df, bl_q_cols = process_q_cols(bl_df)
    el_df, el_q_cols = process_q_cols(el_df)

    # Melt to Long Format
    base_id_vars = ['Grade', 'Student ID', 'Assessment Type', 'Center', 'State']
    
    def robust_melt(df, q_cols):
        ids = [c for c in base_id_vars if c in df.columns]
        return df.melt(id_vars=ids, value_vars=q_cols, 
                       var_name='Question Label', value_name='Student Response')

    bl_long = robust_melt(bl_df, bl_q_cols)
    el_long = robust_melt(el_df, el_q_cols)
    
    combined_df = pd.concat([bl_long, el_long], ignore_index=True)
    combined_df['Question #'] = combined_df['Question Label'].str.extract(r'(\d+)').astype(int)

    # Clean Answer Key
    if 'Grade' in key_df.columns and key_df['Grade'].dtype == 'O':
        key_df['Grade'] = key_df['Grade'].str.replace('G', '').astype(int)

    # Merge
    analysis_df = pd.merge(combined_df, key_df, 
                           left_on=['Assessment Type', 'Grade', 'Question #'], 
                           right_on=['Assessment', 'Grade', 'Question #'], 
                           how='left')

    # Calculate Accuracy
    analysis_df['Is Correct'] = (analysis_df['Student Response'] == analysis_df['Correct Value']).astype(int)
    analysis_df.loc[analysis_df['Student Response'].isna(), 'Is Correct'] = 0
    
    # --- Calculate Total Scores for Discrimination Index ---
    # We need total score per student per assessment to correlate with item score
    total_scores = analysis_df.groupby(['Grade', 'Assessment Type', 'Student ID'])['Is Correct'].sum().reset_index(name='Total Score')
    analysis_df = pd.merge(analysis_df, total_scores, on=['Grade', 'Assessment Type', 'Student ID'], how='left')

    return analysis_df, key_df

# --- 2. Main Dashboard ---
def main():
    st.title("📊 Student Assessment Analysis")
    st.write("Upload your Excel workbook to unlock insights.")

    uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx'])
    
    if uploaded_file:
        with st.spinner('Processing data...'):
            full_df, key_or_error = process_excel_file(uploaded_file)

        if full_df is None:
            st.error(f"❌ {key_or_error}")
            return

        key_df = key_or_error
        
        # Sidebar
        st.sidebar.header("Filters")
        if 'Grade' not in full_df.columns:
            st.error("Grade column missing.")
            return

        grades = sorted(full_df['Grade'].dropna().astype(int).unique())
        selected_grade = st.sidebar.selectbox("Select Grade", grades)
        
        grade_df = full_df[full_df['Grade'] == selected_grade].copy()

        # KPI Row
        st.markdown(f"### Grade {selected_grade} Overview")
        m1, m2, m3, m4 = st.columns(4)
        
        scores = grade_df.groupby('Assessment Type')['Is Correct'].mean() * 100
        bl, el = scores.get('Baseline', 0), scores.get('Endline', 0)
        
        m1.metric("Students", grade_df['Student ID'].nunique())
        m2.metric("Baseline Avg", f"{bl:.1f}%")
        m3.metric("Endline Avg", f"{el:.1f}%", delta=f"{el-bl:.1f}%")
        
        el_df = grade_df[grade_df['Assessment Type']=='Endline']
        if not el_df.empty:
            hardest = el_df.groupby('Question #')['Is Correct'].mean().idxmin()
            m4.metric("Hardest Q (Endline)", f"Q{hardest}")
        else:
            m4.metric("Hardest Q", "N/A")

        # --- TABS ---
        tab1, tab2, tab3 = st.tabs([
            "📈 Strategic View", 
            "🧠 Misconception Analysis", 
            "🧐 Single Question Deep Dive"
        ])

        # === TAB 1: Comparison ===
        with tab1:
            st.subheader("Accuracy Comparison (Baseline vs Endline)")
            q_stats = grade_df.groupby(['Assessment Type', 'Question #'])['Is Correct'].mean().reset_index()
            q_stats['Accuracy %'] = (q_stats['Is Correct'] * 100).round(1)
            
            fig = px.bar(q_stats, x='Question #', y='Accuracy %', color='Assessment Type', 
                         barmode='group', text='Accuracy %',
                         color_discrete_map={'Baseline': '#FFA07A', 'Endline': '#20B2AA'})
            fig.update_layout(xaxis=dict(tickmode='linear', dtick=1))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("") # Added relevant diagram tag

        # === TAB 2: Misconception Analysis ===
        with tab2:
            st.subheader("Why are students getting it wrong?")
            col_sel, _ = st.columns([1, 4])
            with col_sel:
                assess_type = st.radio("Select Assessment", grade_df['Assessment Type'].unique(), horizontal=True, index=0 if 'Endline' in grade_df['Assessment Type'].unique() else 0)
            
            dist_df = grade_df[grade_df['Assessment Type'] == assess_type]
            
            if not dist_df.empty:
                misconception_data = []
                all_qs = sorted(dist_df['Question #'].unique())
                
                for q_num in all_qs:
                    q_subset = dist_df[dist_df['Question #'] == q_num]
                    total_count = len(q_subset)
                    if total_count == 0: continue
                    
                    q_key = key_df[(key_df['Grade'] == selected_grade) & 
                                   (key_df['Question #'] == q_num) & 
                                   (key_df['Assessment'] == assess_type)]
                    
                    if q_key.empty: continue
                    correct_val = int(q_key['Correct Value'].values[0])

                    counts = q_subset['Student Response'].value_counts()
                    correct_count = counts.get(correct_val, 0)
                    correct_pct = (correct_count / total_count) * 100
                    
                    distractors = counts.drop(correct_val, errors='ignore')
                    if not distractors.empty:
                        top_dist_val = distractors.idxmax()
                        top_dist_count = distractors.max()
                        top_dist_pct = (top_dist_count / total_count) * 100
                        
                        lbl_col = f"Value{int(top_dist_val)}"
                        if lbl_col in q_key.columns:
                            dist_label = f"{q_key[lbl_col].values[0]} ({int(top_dist_val)})"
                        else:
                            dist_label = f"Option {int(top_dist_val)}"
                    else:
                        top_dist_pct = 0
                        dist_label = "None"

                    misconception_data.append({'Question': f"Q{q_num}", 'Type': 'Correct Answer', 'Percentage': correct_pct, 'Label': 'Correct'})
                    misconception_data.append({'Question': f"Q{q_num}", 'Type': 'Most Common Error', 'Percentage': top_dist_pct, 'Label': dist_label})

                misc_df = pd.DataFrame(misconception_data)
                fig_misc = px.bar(misc_df, x='Percentage', y='Question', color='Type', orientation='h', barmode='group',
                                  text='Percentage', hover_data=['Label'], height=600,
                                  color_discrete_map={'Correct Answer': '#2E8B57', 'Most Common Error': '#CD5C5C'})
                fig_misc.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig_misc.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_title="Percentage of Students")
                st.plotly_chart(fig_misc, use_container_width=True)
                st.caption("") # Added relevant diagram tag

        # === TAB 3: Single Question Deep Dive (Updated with Metrics) ===
        with tab3:
            st.subheader("Deep Dive: Statistics & Student Shift")
            
            c_sel, _ = st.columns([1, 3])
            with c_sel:
                all_qs = sorted(grade_df['Question #'].unique())
                selected_q = st.selectbox("Select Question Number:", all_qs)

            # 1. Question Details
            q_key_row = key_df[(key_df['Grade'] == selected_grade) & (key_df['Question #'] == selected_q)]
            
            if not q_key_row.empty:
                # Prefer Endline text if available, else first found
                if 'Endline' in q_key_row['Assessment'].values:
                    q_row = q_key_row[q_key_row['Assessment'] == 'Endline'].iloc[0]
                else:
                    q_row = q_key_row.iloc[0]
                    
                q_text = q_row['Question Text']
                c_val = int(q_row['Correct Value'])
                c_txt = q_row[f"Value{c_val}"] if f"Value{c_val}" in q_row else f"Option {c_val}"
                
                st.markdown(f"**Q:** {q_text}")
                st.success(f"**Correct Answer:** {c_txt}")
            else:
                st.warning("Question text not found in Answer Key.")

            st.markdown("---")

            # 2. Psychometric Parameters (Difficulty, SD, Discrimination)
            st.markdown("#### 📊 Item Analysis Parameters")
            st.caption("") # Added relevant diagram tag
            
            def get_metrics(df, assess_name):
                subset = df[(df['Question #'] == selected_q) & (df['Assessment Type'] == assess_name)]
                if subset.empty: return None
                
                # Difficulty (p-value)
                p = subset['Is Correct'].mean()
                
                # SD
                sd = subset['Is Correct'].std()
                
                # Discrimination (Point Biserial Correlation)
                # Correlation between item score (0/1) and Total Score
                if subset['Is Correct'].std() == 0 or subset['Total Score'].std() == 0:
                    di = 0.0
                else:
                    di = subset['Is Correct'].corr(subset['Total Score'])
                
                return {
                    "Assessment": assess_name,
                    "Difficulty (%)": f"{p*100:.1f}%",
                    "SD (Variability)": f"{sd:.2f}",
                    "Discrimination Index": f"{di:.2f}"
                }

            m_bl = get_metrics(grade_df, "Baseline")
            m_el = get_metrics(grade_df, "Endline")
            
            metrics = [m for m in [m_bl, m_el] if m]
            
            if metrics:
                st.table(pd.DataFrame(metrics).set_index("Assessment"))
                c_inf1, c_inf2 = st.columns(2)
                with c_inf1:
                    st.caption("**Discrimination Index (DI):** >0.3 is good. It means high-scoring students got this right, and low-scoring students got it wrong.")
                with c_inf2:
                    st.caption("**Standard Deviation (SD):** Measure of spread. 0 means everyone got the same result (all right or all wrong).")

            st.markdown("---")

            # 3. Student Flow Analysis (Donut Chart)
            st.markdown("#### 🔄 Learning Shift (Matched Students Only)")
            
            # Filter for specific Question
            q_df = grade_df[grade_df['Question #'] == selected_q]
            
            # Pivot to get Baseline and Endline side-by-side per student
            pivot = q_df.pivot_table(index='Student ID', columns='Assessment Type', values='Is Correct', aggfunc='max')
            
            if 'Baseline' in pivot.columns and 'Endline' in pivot.columns:
                matched = pivot.dropna()
                total_n = len(matched)
                
                if total_n > 0:
                    # Categories
                    stagnant = len(matched[(matched['Baseline']==0) & (matched['Endline']==0)]) # Wrong -> Wrong
                    improved = len(matched[(matched['Baseline']==0) & (matched['Endline']==1)]) # Wrong -> Right
                    retained = len(matched[(matched['Baseline']==1) & (matched['Endline']==1)]) # Right -> Right
                    regressed = len(matched[(matched['Baseline']==1) & (matched['Endline']==0)]) # Right -> Wrong
                    
                    # Chart
                    labels = ['Stagnant (Wrong->Wrong)', 'Improved (Wrong->Right)', 'Retained (Right->Right)', 'Regressed (Right->Wrong)']
                    values = [stagnant, improved, retained, regressed]
                    colors = ['#EF553B', '#00CC96', '#636EFA', '#AB63FA']
                    
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4, marker_colors=colors)])
                        fig_pie.update_layout(title_text=f"Student Progress (n={total_n})", showlegend=False)
                        st.plotly_chart(fig_pie, use_container_width=True)
                        st.caption("") # Added relevant diagram tag
                    
                    with c2:
                        st.markdown(f"""
                        * **🚀 Improved:** {improved} students ({improved/total_n:.1%})
                        * **✅ Retained:** {retained} students ({retained/total_n:.1%})
                        * **⚠️ Regressed:** {regressed} students ({regressed/total_n:.1%})
                        * **❌ Stagnant:** {stagnant} students ({stagnant/total_n:.1%})
                        """)
                else:
                    st.info("No students took both tests for this question.")
            else:
                st.warning("Need both Baseline and Endline data to show progress flow.")

if __name__ == "__main__":
    main()
