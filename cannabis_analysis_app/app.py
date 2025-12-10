import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.graph_objs import Figure as PlotlyFigure

from analysis import *
import numpy as np

import itertools

# --------------------------------------------------
# GLOBAL COUNTER (must be ABOVE safe_plot)
# --------------------------------------------------
global_counter = itertools.count()

# --------------------------------------------------
# SAFE PLOT FUNCTION
# --------------------------------------------------
def safe_plot(fig, prefix="plot"):
    try:
        st.plotly_chart(
            fig,
            key=f"{prefix}_{next(global_counter)}",
            width="stretch"  # NEW parameter replacing use_container_width
        )
    except Exception as e:
        st.error(f"Chart could not be displayed: {e}")

# Global counter for unique Streamlit keys
PLOT_COUNTER = 0

import streamlit as st

# # -------- SIMPLE PASSWORD PROTECTION -------- #
# def check_password():
#     def password_entered():
#         if st.session_state["password"] == st.secrets["app_password"]:
#             st.session_state["password_correct"] = True
#             del st.session_state["password"]
#         else:
#             st.session_state["password_correct"] = False

#     if "password_correct" not in st.session_state:
#         st.text_input(
#             "Enter password:", type="password",
#             on_change=password_entered, key="password"
#         )
#         return False
#     elif not st.session_state["password_correct"]:
#         st.text_input(
#             "Enter password:", type="password",
#             on_change=password_entered, key="password"
#         )
#         st.error("❌ Wrong password")
#         return False
#     else:
#         return True

# if not check_password():
#     st.stop()


PLOT_COUNTER = 0

# --------------------------------------------------------------------
# PAGE CONFIG + AESTHETIC CSS THEME
# --------------------------------------------------------------------
st.set_page_config(
    page_title="🚀 Smart Data Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Premium UI Styling
st.markdown("""
    <style>

    /* Background */
    .main { background-color: #f7f9fc; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1f2937 !important;
    }

    [data-testid="stSidebar"] * {
        color: #e5e7eb !important;
        font-size: 15px !important;
    }

    h1, h2, h3, h4 {
        font-family: 'Segoe UI', sans-serif;
    }

    /* Nice shadow cards for sections */
    .section-card {
        padding: 25px;
        border-radius: 12px;
        background-color: white;
        box-shadow: 0px 4px 16px rgba(0,0,0,0.06);
        margin-bottom: 30px;
    }

    /* Upload section highlight */
    .upload-box {
        padding: 18px;
        background-color: #eef2ff;
        border-radius: 10px;
        border: 1px dashed #6366f1;
        margin-bottom: 20px;
    }

    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------
# HEADER BANNER
# --------------------------------------------------------------------
st.markdown("""
<div style="padding: 25px; background: linear-gradient(90deg, #4f46e5, #3b82f6);
            border-radius: 12px; margin-bottom: 25px;">
    <h1 style="color: white; text-align: center; font-size: 38px; margin-bottom: 5px;">
        🚀 Smart Data & Notebook Analysis Platform
    </h1>
    <p style="color: #dbeafe; text-align: center; font-size: 17px;">
        Upload your files or Jupyter notebook and instantly explore insights, charts, statistics — no technical knowledge required.
    </p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------
# SIDEBAR (Rewritten for Non-Technical Users)
# --------------------------------------------------------------------
st.sidebar.markdown("## 📂 Upload Your Files")

st.sidebar.markdown("""
### 🧠 What can I upload?

This tool accepts:
- **Excel files** (.xlsx, .xls)  
- **CSV files** (.csv)  
- **JSON files** (.json)  
- **ZIP files** (containing multiple datasets)  
- **Jupyter Notebooks** (.ipynb)

Once uploaded, the system will **automatically read your data, clean it, and generate charts and summaries** for you.
""")

# Notebook Upload
st.sidebar.markdown("### 📘 Upload a Notebook (Optional)")
st.sidebar.markdown(
    "If you have a Jupyter Notebook with analysis code, upload it here and the app will run it for you."
)
notebook_file = st.sidebar.file_uploader(
    "Upload Notebook (.ipynb)",
    type=["ipynb"]
)

# Supporting Data Files
st.sidebar.markdown("### 📁 Upload Supporting Data")
st.sidebar.markdown(
    "If your notebook uses extra datasets, upload them here (CSV, Excel, JSON, or ZIP)."
)
support_files = st.sidebar.file_uploader(
    "Add datasets your notebook needs",
    type=["csv", "xlsx", "xls", "json", "zip"],
    accept_multiple_files=True
)

# Data for EDA
st.sidebar.markdown("### 📊 Upload Data for Automatic Analysis")
st.sidebar.markdown(
    "Upload any dataset you want to explore. The app will create charts, summaries, and insights for you automatically."
)
data_files = st.sidebar.file_uploader(
    "Upload CSV / Excel / JSON",
    type=["csv", "xlsx", "xls", "json"],
    accept_multiple_files=True
)

st.sidebar.markdown("---")
st.sidebar.markdown("🔒 Your files stay private and secure.")

# --------------------------------------------------------------------
# Helper for safe chart output (prevents duplicate element ID errors)
# --------------------------------------------------------------------
def safe_plot(fig, prefix="plot"):
    try:
        st.plotly_chart(fig, width="stretch", key=f"{prefix}_{next(global_counter)}")
    except Exception as e:
        st.error(f"Chart could not be displayed: {e}")




# --------------------------------------------------------------------
# NOTEBOOK EXECUTION
# --------------------------------------------------------------------
if notebook_file:
    st.subheader("📘 Notebook Execution")

    mode = st.radio(
        "Choose Execution Mode",
        ["Cell-by-Cell (Jupyter Style)", "Unified Output (Standard Execution)"]
    )

    # Prepare notebook environment
    tmp_dir, nb_path = prepare_notebook_environment(
        notebook_file, support_files or []
    )

    script_text = convert_notebook_to_script(nb_path)

    # -----------------------------------------------------------
    # CELL-BY-CELL MODE
    # -----------------------------------------------------------
    if mode == "Cell-by-Cell (Jupyter Style)":
        st.info("Running notebook cell-by-cell…")

        results = execute_notebook_cells(nb_path, tmp_dir)

        for cell in results:
            st.markdown(f"### 📘 Cell {cell['cell_number']}")

            if cell["code"]:
                st.code(cell["code"], language="python")
            else:
                # Markdown cell
                st.markdown(cell["outputs"][0])
                continue

            # Outputs
            if not cell["outputs"]:
                st.write("⬜ No output")
            else:
                for out in cell["outputs"]:
                    if isinstance(out, pd.DataFrame):
                        st.dataframe(out)

                    elif isinstance(out, PlotlyFigure):
                        safe_plot(out, prefix=f"cell_{cell['cell_number']}")

                    else:
                        st.write(out)

        st.success("Notebook executed successfully.")
        st.stop()

    # -----------------------------------------------------------
    # UNIFIED MODE
    # -----------------------------------------------------------
    else:
        st.info("Executing notebook as full script…")

        error, env = execute_script_in_dir(script_text, tmp_dir)

        if error:
            st.error(f"❌ Notebook Error:\n\n{error}")
            st.stop()

        st.subheader("📘 Final DataFrames")
        for k, v in env.items():
            
            if isinstance(v, pd.DataFrame):
                st.write(f"### 🔹 {k}")
                st.dataframe(v)

        st.subheader("📊 Final Plots (plot_* or final_*) Only")
        for k, v in env.items():

            if not isinstance(v, PlotlyFigure):
                continue

            key_l = k.lower()
            st.subheader("📊 All Notebook Charts")
            for k, v in env.items():
              if isinstance(v, PlotlyFigure):
               st.write(f"### {k}")
               safe_plot(v, prefix=f"unified_{k}")


            # st.write(f"### {k}")
            # safe_plot(v, prefix=f"unified_{k}")

        st.success("Notebook executed successfully.")
        st.stop()


# --------------------------------------------------------------------
# EDA MODE (DATA FILES)
# --------------------------------------------------------------------
if not data_files:
    st.info("Upload a CSV / Excel / JSON file for EDA.")
    st.stop()

file = data_files[0]
st.header(f"📁 Loaded File: {file.name}")

# Excel sheet selection
if file.name.lower().endswith((".xlsx", ".xls")):
    sheets = get_excel_sheets(file)
    sheet = st.selectbox("Choose Excel Sheet", sheets)
    df = load_file(file, sheet)
else:
    df = load_file(file)

# Clean + detect types
df_clean = clean_data(df)
num_cols, cat_cols = detect_types(df_clean)


# --------------------------------------------------------------------
# TABS for EDA
# --------------------------------------------------------------------
tabs = st.tabs([
    "Preview", "Summary", "Auto EDA",
    "Charts", "Correlations", "Categories"
])


# -------------------------------------------------------
# PREVIEW TAB
# -------------------------------------------------------
with tabs[0]:
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())


# -------------------------------------------------------
# SUMMARY TAB
# -------------------------------------------------------
with tabs[1]:
    st.subheader("📊 Summary")
    summary = generate_summary(df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", summary["Rows"])
    col2.metric("Columns", summary["Columns"])
    col3.metric("Missing", summary["Missing Values"])
    col4.metric("Duplicates", summary["Duplicates"])

    st.write("### Full Describe()")
    st.dataframe(df.describe(include="all"))


# -------------------------------------------------------
# AUTO EDA TAB
# -------------------------------------------------------
with tabs[2]:
    st.subheader("🤖 Automated EDA Insights")

    # Missing Values
    st.markdown("### 🔻 Missing Value Analysis")
    safe_plot(eda_missing_values(df), prefix="eda_missing")

    st.markdown("---")

    # Unique Values
    st.markdown("### 🧩 Unique Value Distribution")
    safe_plot(eda_unique_values(df), prefix="eda_unique")

    st.markdown("---")

    # Data Types
    st.markdown("### 🧬 Data Type Summary")
    safe_plot(eda_datatypes(df), prefix="eda_types")



# -------------------------------------------------------
# CHART BUILDER TAB
# -------------------------------------------------------
with tabs[3]:
    st.subheader("📈 Chart Builder")

    chart_type = st.selectbox(
        "Chart Type",
        ["Histogram", "Boxplot", "Bar", "Line", "Scatter", "Pie"]
    )

    chart_color = st.color_picker("🎨 Pick a Chart Color", "#1f77b4")
    st.markdown("---")

    # ---------------------------------------------------
    # AI-style description helper (SAFE CORRELATION)
    # ---------------------------------------------------
    def describe_chart(df, chart_type, x=None, y=None):

        text = f"### 🧠 Chart Summary\n"

        if chart_type == "Histogram":
            text += f"- Shows how values of **{x}** are distributed.\n"
            text += f"- Mean: `{df[x].mean():.2f}`, Median: `{df[x].median():.2f}`.\n"
            text += f"- Skewness: `{df[x].skew():.2f}`.\n"

        elif chart_type == "Boxplot":
            text += f"- Summarizes the spread of **{x}**.\n"
            text += f"- Q1: `{df[x].quantile(0.25):.2f}`, Q3: `{df[x].quantile(0.75):.2f}`.\n"

        elif chart_type in ["Bar", "Line", "Scatter"]:
            text += f"- Relationship between **{x}** and **{y}**.\n"

            try:
                safe_df = df[[x, y]].dropna()
            except:
                text += "- Unable to compute correlation.\n"
                return text

            if safe_df.shape[0] < 2:
                text += "- Not enough data to compute correlation.\n"
                return text

            try:
                corr_matrix = safe_df.corr()
                if corr_matrix.shape == (2, 2):
                    corr = corr_matrix.iloc[0, 1]
                    text += f"- Correlation: `{corr:.2f}`.\n"
                else:
                    text += "- Correlation not meaningful.\n"
            except:
                text += "- Correlation error.\n"

        elif chart_type == "Pie":
            text += f"- Distribution of categories in **{x}**.\n"
            text += f"- `{df[x].nunique()}` unique categories.\n"

        return text

    # ---------------------------------------------------
    # HISTOGRAM
    # ---------------------------------------------------
    if chart_type == "Histogram":
        col = st.selectbox("Select Numeric Column", num_cols)

        st.markdown(f"### 📊 Histogram of **{col}**")

        fig = plot_hist(df_clean, col)
        fig.update_traces(
            marker_color=chart_color,
            hovertemplate="<b>Value:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>"
        )

        safe_plot(fig, prefix="hist")
        st.markdown(describe_chart(df_clean, "Histogram", x=col))

    # ---------------------------------------------------
    # BOXPLOT
    # ---------------------------------------------------
    elif chart_type == "Boxplot":
        col = st.selectbox("Select Numeric Column", num_cols)

        st.markdown(f"### 📦 Boxplot for **{col}**")

        fig = plot_box(df_clean, col)
        fig.update_traces(
            marker_color=chart_color,
            hovertemplate="<b>Value:</b> %{y}<extra></extra>"
        )

        safe_plot(fig, prefix="box")
        st.markdown(describe_chart(df_clean, "Boxplot", x=col))

    # ---------------------------------------------------
    # BAR / LINE / SCATTER
    # ---------------------------------------------------
    elif chart_type in ["Bar", "Line", "Scatter"]:

        st.markdown("### 🎛 Chart Settings")

        x = st.selectbox("Choose X-axis", df_clean.columns)
        y = st.selectbox("Choose Y-axis (numeric)", num_cols)

        st.markdown(f"### 📊 {chart_type} Chart — **{y} vs {x}**")

        df_plot = df_clean.dropna(subset=[x, y]).copy()

        try:
            df_plot[x] = df_plot[x].astype(str)
        except:
            pass

        # AUTO AGGREGATE when repeating X
        if df_plot[x].nunique() < df_plot.shape[0]:
            df_plot = df_plot.groupby(x, as_index=False)[y].sum()

        # ----------------- BAR -----------------
        if chart_type == "Bar":
            fig = plot_bar(df_plot, x, y)
            fig.update_traces(
                marker_color=chart_color,
                hovertemplate="<b>%{x}</b><br><b>Value:</b> %{y}<extra></extra>"
            )

        # ----------------- LINE -----------------
        elif chart_type == "Line":
            df_plot = df_plot.sort_values(by=x)
            fig = plot_line(df_plot, x, y)
            fig.update_traces(
                marker_color=chart_color,
                hovertemplate="<b>%{x}</b><br><b>Value:</b> %{y}<extra></extra>"
            )

        # ----------------- SCATTER -----------------
        elif chart_type == "Scatter":
            fig = plot_scatter(df_plot, x, y)
            fig.update_traces(
                marker_color=chart_color,
                hovertemplate="<b>%{x}</b><br><b>Value:</b> %{y}<extra></extra>"
            )

        safe_plot(fig, prefix=chart_type.lower())
        st.markdown(describe_chart(df_plot, chart_type, x=x, y=y))

    # ---------------------------------------------------
    # PIE CHART
    # ---------------------------------------------------
    elif chart_type == "Pie":

        st.markdown("### 🥧 Pie Chart Settings")

        col = st.selectbox("Select Category Column", cat_cols)

        st.markdown(f"### 🥧 Pie Chart of **{col}**")

        if df_clean[col].nunique() < 2:
            st.warning("⚠ Pie chart requires at least 2 categories.")
            st.stop()

        fig = plot_pie(df_clean, col)

        fig.update_traces(
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
            marker=dict(colors=[chart_color])
        )

        safe_plot(fig, prefix="pie")
        st.markdown(describe_chart(df_clean, "Pie", x=col))

# -------------------------------------------------------
# CORRELATIONS TAB
# -------------------------------------------------------
with tabs[4]:
    st.subheader("🔥 Correlation Heatmap")
    safe_plot(correlation_heatmap(df_clean), prefix="corr_heatmap")

    if len(num_cols) >= 3:
        st.subheader("📊 Pairplot")
        safe_plot(pairplot(df_clean, num_cols), prefix="pairplot")

# -------------------------------------------------------
# CATEGORIES TAB
# -------------------------------------------------------
with tabs[5]:
    st.subheader("📦 Category Distribution")

    if cat_cols:
        col = st.selectbox("Choose Category Column", cat_cols)
        summary_df = group_summary(df_clean, col)

        st.dataframe(summary_df)
        safe_plot(
            px.bar(summary_df, x=col, y="Count", title=f"Category Count: {col}"),
            prefix="cat_bar"
        )
    else:
        st.info("No categorical columns detected.")
        
        
