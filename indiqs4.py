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


# --- 1. Data Loading & Processing ---
@st.cache_data
def process_excel_file(uploaded_file):
    try:
        xls = pd.ExcelFile(uploaded_file)

        def get_sheet(keywords):
            for sheet in xls.sheet_names:
                if any(k.lower() in sheet.lower() for k in keywords):
                    return sheet
            return None

        bl_sheet = get_sheet(["baseline", "bl"])
        el_sheet = get_sheet(["endline", "el"])
        key_sheet = get_sheet(["answer", "key"])

        if not all([bl_sheet, el_sheet, key_sheet]):
            return None, "Missing sheets. Ensure Baseline, Endline, and Answer Key sheets exist."

        bl_df = pd.read_excel(uploaded_file, sheet_name=bl_sheet)
        el_df = pd.read_excel(uploaded_file, sheet_name=el_sheet)
        key_df = pd.read_excel(uploaded_file, sheet_name=key_sheet)

        bl_df['Assessment Type'] = 'Baseline'
        el_df['Assessment Type'] = 'Endline'

    except Exception as e:
        return None, f"Error reading file: {str(e)}"

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

    def extract_value(cell_value):
        if pd.isna(cell_value): return None
        try:
            if isinstance(cell_value, (int, float)): return int(cell_value)
            s_val = str(cell_value).strip()
            if s_val.isdigit(): return int(s_val)
            if "{" in s_val:
                data = json.loads(s_val)
                return int(data.get('value'))
        except:
            return None
        return None

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

    if 'Grade' in key_df.columns and key_df['Grade'].dtype == 'O':
        key_df['Grade'] = key_df['Grade'].str.replace('G', '').astype(int)

    analysis_df = pd.merge(combined_df, key_df,
                           left_on=['Assessment Type', 'Grade', 'Question #'],
                           right_on=['Assessment', 'Grade', 'Question #'],
                           how='left')

    analysis_df['Is Correct'] = (analysis_df['Student Response'] == analysis_df['Correct Value']).astype(int)
    analysis_df.loc[analysis_df['Student Response'].isna(), 'Is Correct'] = 0

    # --- New: Calculate Total Score per Student for Discrimination Index ---
    # Sum 'Is Correct' for each student/assessment
    total_scores = analysis_df.groupby(['Grade', 'Assessment Type', 'Student ID'])['Is Correct'].sum().reset_index(
        name='Total Score')
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

        # KPI Overview
        st.markdown(f"### Grade {selected_grade} Overview")
        m1, m2, m3, m4 = st.columns(4)

        scores = grade_df.groupby('Assessment Type')['Is Correct'].mean() * 100
        bl, el = scores.get('Baseline', 0), scores.get('Endline', 0)

        m1.metric("Students", grade_df['Student ID'].nunique())
        m2.metric("Baseline Avg", f"{bl:.1f}%")
        m3.metric("Endline Avg", f"{el:.1f}%", delta=f"{el - bl:.1f}%")

        el_df_full = grade_df[grade_df['Assessment Type'] == 'Endline']
        if not el_df_full.empty:
            hardest = el_df_full.groupby('Question #')['Is Correct'].mean().idxmin()
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

        # === TAB 2: Misconception Analysis ===
        with tab2:
            st.subheader("Why are students getting it wrong?")
            assess_type = st.radio("Select Assessment", grade_df['Assessment Type'].unique(), horizontal=True,
                                   index=0 if 'Endline' in grade_df['Assessment Type'].unique() else 0)

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
                        dist_label = q_key[lbl_col].values[
                            0] if lbl_col in q_key.columns else f"Option {int(top_dist_val)}"
                    else:
                        top_dist_pct = 0
                        dist_label = "None"

                    misconception_data.append(
                        {'Question': f"Q{q_num}", 'Type': 'Correct Answer', 'Percentage': correct_pct,
                         'Label': 'Correct'})
                    misconception_data.append(
                        {'Question': f"Q{q_num}", 'Type': 'Most Common Error', 'Percentage': top_dist_pct,
                         'Label': dist_label})

                misc_df = pd.DataFrame(misconception_data)
                fig_misc = px.bar(misc_df, x='Percentage', y='Question', color='Type', orientation='h', barmode='group',
                                  text='Percentage', hover_data=['Label'], height=600,
                                  color_discrete_map={'Correct Answer': '#2E8B57', 'Most Common Error': '#CD5C5C'})
                fig_misc.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig_misc.update_layout(yaxis={'categoryorder': 'total ascending'}, xaxis_title="Percentage of Students")
                st.plotly_chart(fig_misc, use_container_width=True)

        # === TAB 3: Single Question Deep Dive (Updated with SD & Shift) ===
        with tab3:
            st.subheader("Deep Dive: Statistics & Student Shift")

            c_sel, _ = st.columns([1, 3])
            with c_sel:
                all_qs = sorted(grade_df['Question #'].unique())
                selected_q = st.selectbox("Select Question Number:", all_qs)

            # --- 1. Question Text & Correct Answer Display ---
            # Try to get Endline key (most recent), fallback to Baseline
            q_key_row = key_df[(key_df['Grade'] == selected_grade) & (key_df['Question #'] == selected_q)]
            if not q_key_row.empty:
                # Prioritize Endline info
                if 'Endline' in q_key_row['Assessment'].values:
                    q_row = q_key_row[q_key_row['Assessment'] == 'Endline'].iloc[0]
                else:
                    q_row = q_key_row.iloc[0]

                st.markdown(f"**Question Text:** {q_row['Question Text']}")

                # Show Correct Answer text
                c_val = int(q_row['Correct Value'])
                c_txt = q_row[f"Value{c_val}"] if f"Value{c_val}" in q_row else f"Option {c_val}"
                st.success(f"**Correct Answer:** {c_txt}")
            else:
                st.warning("Question text not found in Answer Key.")

            st.markdown("---")

            # --- 2. Psychometric Parameters Table ---
            st.markdown("#### 📊 Question Parameters (Item Analysis)")

            # Helper to calculate metrics
            def calculate_metrics(df, assessment_name):
                subset = df[(df['Question #'] == selected_q) & (df['Assessment Type'] == assessment_name)]
                if subset.empty: return None

                # 1. Difficulty (p-value): % Correct
                p_val = subset['Is Correct'].mean()

                # 2. Standard Deviation (SD): Variability of the item score
                sd_val = subset['Is Correct'].std()

                # 3. Discrimination (Point Biserial Correlation): Corr between Item Score and Total Score
                # Note: Total score should ideally exclude the item itself to avoid auto-correlation,
                # but standard easy computation often just uses total score.
                if subset['Is Correct'].std() == 0 or subset['Total Score'].std() == 0:
                    disc_val = 0  # No variance, no correlation
                else:
                    disc_val = subset['Is Correct'].corr(subset['Total Score'])

                return {
                    "Assessment": assessment_name,
                    "Difficulty (%)": f"{p_val * 100:.1f}%",
                    "SD (Variability)": f"{sd_val:.2f}",
                    "Discrimination Index": f"{disc_val:.2f}"
                }

            bl_metrics = calculate_metrics(grade_df, "Baseline")
            el_metrics = calculate_metrics(grade_df, "Endline")

            metrics_data = []
            if bl_metrics: metrics_data.append(bl_metrics)
            if el_metrics: metrics_data.append(el_metrics)

            if metrics_data:
                st.table(pd.DataFrame(metrics_data).set_index("Assessment"))
                st.caption(
                    "* **Discrimination Index:** How well this question differentiates high performers from low performers. (>0.3 is good).")
                st.caption(
                    "* **SD:** Standard deviation of student responses (0=Wrong, 1=Right). High SD means mixed results.")

            st.markdown("---")

            # --- 3. Student Shift Analysis (Sankey-like Logic) ---
            st.markdown("#### 🔄 Learning Shift: Same Students (Baseline ➔ Endline)")

            # Filter for students present in BOTH Baseline and Endline for this specific question
            q_df = grade_df[grade_df['Question #'] == selected_q]

            pivot_q = q_df.pivot_table(index='Student ID', columns='Assessment Type', values='Is Correct',
                                       aggfunc='max')

            if 'Baseline' in pivot_q.columns and 'Endline' in pivot_q.columns:
                # Drop students who missed one of the exams
                matched = pivot_q.dropna()
                total_matched = len(matched)

                if total_matched > 0:
                    # Categories
                    # 0->0: Stagnant (Incorrect)
                    # 0->1: Improved (Learned)
                    # 1->1: Retained (Mastered)
                    # 1->0: Regressed (Forgot/Confused)

                    stagnant = len(matched[(matched['Baseline'] == 0) & (matched['Endline'] == 0)])
                    improved = len(matched[(matched['Baseline'] == 0) & (matched['Endline'] == 1)])
                    retained = len(matched[(matched['Baseline'] == 1) & (matched['Endline'] == 1)])
                    regressed = len(matched[(matched['Baseline'] == 1) & (matched['Endline'] == 0)])

                    # Create Donut Chart
                    labels = ['Remained Incorrect', 'Improved (Learned)', 'Retained Correct', 'Regressed']
                    values = [stagnant, improved, retained, regressed]
                    colors = ['#EF553B', '#00CC96', '#636EFA', '#AB63FA']  # Red, Green, Blue, Purple

                    c1, c2 = st.columns([1, 1])

                    with c1:
                        fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4, marker_colors=colors)])
                        fig_pie.update_layout(title_text=f"Student Progress (n={total_matched})", showlegend=False)
                        st.plotly_chart(fig_pie, use_container_width=True)

                    with c2:
                        st.markdown(f"""
                        * **🚀 {improved} students ({improved / total_matched:.1%})** learned the concept (Wrong ➔ Right).
                        * **✅ {retained} students ({retained / total_matched:.1%})** already knew it (Right ➔ Right).
                        * **⚠️ {regressed} students ({regressed / total_matched:.1%})** got confused (Right ➔ Wrong).
                        * **❌ {stagnant} students ({stagnant / total_matched:.1%})** still don't know it (Wrong ➔ Wrong).
                        """)
                else:
                    st.info("No common students found between Baseline and Endline for this question.")
            else:
                st.warning("Need both Baseline and Endline data to show progress.")


if __name__ == "__main__":
    main()