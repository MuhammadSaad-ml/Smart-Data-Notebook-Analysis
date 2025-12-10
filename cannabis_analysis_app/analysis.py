import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs import Figure as PlotlyFigure
import nbformat
from nbconvert import PythonExporter
import tempfile
import os
import sys
import io
import traceback
import zipfile
import plotly.io as pio


# -----------------------------------------------------------------------------
# PREPARE NOTEBOOK ENVIRONMENT (temp folder)
# -----------------------------------------------------------------------------
def prepare_notebook_environment(notebook_file, support_files):
    temp_dir = tempfile.mkdtemp()

    # Save notebook
    nb_path = os.path.join(temp_dir, notebook_file.name)
    with open(nb_path, "wb") as f:
        f.write(notebook_file.read())

    # Save supporting files (CSV/Excel/ZIP)
    for file in support_files:
        save_path = os.path.join(temp_dir, file.name)
        with open(save_path, "wb") as f:
            f.write(file.read())

        # Extract ZIP files
        if file.name.lower().endswith(".zip"):
            with zipfile.ZipFile(save_path, "r") as z:
                z.extractall(temp_dir)

    return temp_dir, nb_path


# -----------------------------------------------------------------------------
# Convert Notebook → Python Script
# -----------------------------------------------------------------------------
def convert_notebook_to_script(nb_path):
    nb = nbformat.read(nb_path, as_version=4)
    exporter = PythonExporter()
    script, _ = exporter.from_notebook_node(nb)
    return script


# -----------------------------------------------------------------------------
# Execute Notebook Cell-by-Cell (Jupyter-like)
# -----------------------------------------------------------------------------
def execute_notebook_cells(nb_path, work_dir):
    nb = nbformat.read(nb_path, as_version=4)
    cells = nb.cells

    current_dir = os.getcwd()
    os.chdir(work_dir)

    env = {"__name__": "__main__"}

    captured_fig = None

    # Patch Plotly to catch .show()
    def custom_show(fig):
        nonlocal captured_fig
        captured_fig = fig

    pio.show = custom_show
    env["show"] = custom_show

    cell_results = []

    for idx, cell in enumerate(cells):
        # Markdown cell
        if cell["cell_type"] == "markdown":
            cell_results.append({
                "cell_number": idx + 1,
                "code": None,
                "outputs": [cell["source"]]
            })
            continue

        if cell["cell_type"] != "code":
            continue

        code = cell["source"]
        outputs = []

        # Capture print output
        stdout_buffer = io.StringIO()
        sys.stdout = stdout_buffer

        last_value = None

        try:
            exec(compile(code, "<cell>", "exec"), env)

            # Try capturing last expression
            try:
                lines = [line for line in code.split("\n") if line.strip()]
                last_line = lines[-1]

                safe_start = (
                    "print", "for ", "while ", "if ", "def ",
                    "class ", "import ", "from ", "#"
                )

                if not last_line.startswith(safe_start):
                    last_value = eval(last_line, env)
            except:
                last_value = None

        except Exception:
            error_msg = traceback.format_exc()
            outputs.append(f"ERROR:\n{error_msg}")

        finally:
            sys.stdout = sys.__stdout__

        # Collect print output
        printed = stdout_buffer.getvalue()
        if printed.strip():
            outputs.append(printed)

        # Plotly show() image
        if captured_fig is not None:
            outputs.append(captured_fig)
            captured_fig = None

        # Last expression
        if isinstance(last_value, pd.DataFrame):
            outputs.append(last_value)

        elif isinstance(last_value, PlotlyFigure):
            outputs.append(last_value)

        elif last_value is not None:
            outputs.append(last_value)
        
        
        
        cell_results.append({
            "cell_number": idx + 1,
            "code": code,
            "outputs": outputs
        })

    os.chdir(current_dir)
    return cell_results


# -----------------------------------------------------------------------------
# Execute Notebook as Full Script (Unified)
# -----------------------------------------------------------------------------
def execute_script_in_dir(script, work_dir):
    # NEW: capture text + markdown output
    outputs = []

    env = {
        "__name__": "__main__",
        "display": lambda x: None,
        "print": lambda *args: outputs.append({
            "type": "text",
            "value": " ".join(str(a) for a in args)
        })
    }

    # Patch markdown cells 
    def md(text):
        outputs.append({"type": "markdown", "value": text})
    env["md"] = md

    current_dir = os.getcwd()
    os.chdir(work_dir)

    error = None
    try:
        exec(script, env)
    except Exception as e:
        error = str(e)

    os.chdir(current_dir)

    # Store outputs inside env so Streamlit can display them
    env["unified_outputs"] = outputs

    return error, env


# -----------------------------------------------------------------------------
# Load File (CSV/Excel/JSON)
# -----------------------------------------------------------------------------
def load_file(file, sheet=None):
    name = file.name.lower()

    if name.endswith(".csv"):
        return pd.read_csv(file)

    if name.endswith((".xlsx", ".xls")):
        if sheet:
            return pd.read_excel(file, sheet_name=sheet)
        return pd.ExcelFile(file)

    if name.endswith(".json"):
        return pd.read_json(file)

    raise ValueError("Unsupported file format")


def get_excel_sheets(file):
    return pd.ExcelFile(file).sheet_names


# -----------------------------------------------------------------------------
# Summary / Cleaning / Types
# -----------------------------------------------------------------------------
def generate_summary(df):
    return {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Missing Values": df.isna().sum().sum(),
        "Duplicates": df.duplicated().sum(),
    }


def clean_data(df):
    df = df.drop_duplicates()
    df = df.fillna(df.mean(numeric_only=True))
    return df


def detect_types(df):
    return (
        df.select_dtypes(include="number").columns.tolist(),
        df.select_dtypes(include="object").columns.tolist()
    )


# -----------------------------------------------------------------------------
# AUTO EDA + CHARTS
# -----------------------------------------------------------------------------
def eda_missing_values(df):
    mv = df.isna().sum().reset_index()
    mv.columns = ["Column", "Missing Values"]
    return px.bar(mv, x="Column", y="Missing Values", text_auto=True)


def eda_unique_values(df):
    uv = pd.DataFrame({"Column": df.columns, "Unique Values": df.nunique()})
    return px.bar(uv, x="Column", y="Unique Values", text_auto=True)


def eda_datatypes(df):
    dt = pd.DataFrame({"Column": df.columns, "Dtype": df.dtypes.astype(str)})
    return px.bar(dt, x="Column", y="Dtype")


def correlation_heatmap(df):
    return px.imshow(df.corr(numeric_only=True), text_auto=True)


def plot_hist(df, col):
    return px.histogram(df, x=col, nbins=40, text_auto=True)


def plot_box(df, col):
    return px.box(df, y=col, points="all")


def plot_bar(df, x, y):
    fig = px.bar(df, x=x, y=y, text=y)
    fig.update_traces(textposition='outside')
    return fig


def plot_line(df, x, y):
    fig = px.line(df, x=x, y=y, markers=True, text=df[y])
    return fig


def plot_scatter(df, x, y):
    fig = px.scatter(df, x=x, y=y, text=y, trendline="ols")
    fig.update_traces(textposition="top center")
    return fig


def plot_pie(df, names):
    fig = px.pie(df, names=names)
    fig.update_traces(
        textinfo="percent+label",
        textposition="inside"
    )
    return fig



def pairplot(df, cols):
    return px.scatter_matrix(df, dimensions=cols[:5])


def group_summary(df, col):
    tmp = df[col].value_counts().reset_index()
    tmp.columns = [col, "Count"]
    return tmp

# Summary Dashboard Helpers
def dataset_overview(df):
    return {
        "Rows": len(df),
        "Columns": df.shape[1],
        "Missing (%)": round(df.isna().mean().mean() * 100, 2),
        "Duplicate Rows": df.duplicated().sum(),
        "Numeric Columns": df.select_dtypes(include="number").shape[1],
        "Categorical Columns": df.select_dtypes(include="object").shape[1],
        "Memory Usage (MB)": round(df.memory_usage().sum() / (1024 * 1024), 2)
    }


def missing_heatmap(df):
    import seaborn as sns
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.isna(), cbar=False, yticklabels=False, ax=ax)
    ax.set_title("Missing Value Heatmap")
    return fig

## Column-Level EDA

def profile_column(df, col):
    series = df[col]
    summary = {}

    summary["Type"] = str(series.dtype)
    summary["Missing"] = series.isna().sum()
    summary["Unique"] = series.nunique()

    if pd.api.types.is_numeric_dtype(series):
        summary["Mean"] = series.mean()
        summary["Median"] = series.median()
        summary["Std"] = series.std()
        summary["Min"] = series.min()
        summary["Max"] = series.max()
        summary["Skew"] = series.skew()
        summary["Kurtosis"] = series.kurtosis()

    return summary

# ## Outlier Detection (Z-score + IQR)

# def detect_outliers_iqr(series):
#     Q1 = series.quantile(0.25)
#     Q3 = series.quantile(0.75)
#     IQR = Q3 - Q1
#     return series[(series < Q1 - 1.5*IQR) | (series > Q3 + 1.5*IQR)]


# def detect_outliers_zscore(series, threshold=3):
#     from scipy.stats import zscore
#     zs = zscore(series.dropna())
#     return series[(abs(zs) > threshold)]

# ## Relationship Detection

# def strong_correlations(df, threshold=0.5):
#     corr = df.corr(numeric_only=True)
#     strong = (
#         corr.where(abs(corr) >= threshold)
#             .stack()
#             .reset_index()
#     )
#     strong.columns = ["Var1", "Var2", "Corr"]
#     strong = strong[strong["Var1"] != strong["Var2"]]
#     return strong

# ## Target-Aware EDA

# def target_analysis(df, target):
#     numeric_cols = df.select_dtypes(include="number").columns.tolist()
#     numeric_cols.remove(target)
#     corr = df[numeric_cols + [target]].corr()[target].sort_values(ascending=False)
#     return corr

# ## Time-Series EDA

# def detect_datetime_cols(df):
#     return [
#         col for col in df.columns
#         if pd.api.types.is_datetime64_any_dtype(df[col])
#     ]


# def ts_line(df, date_col, y_col):
#     return px.line(df.sort_values(date_col), x=date_col, y=y_col, title=f"{y_col} over Time")

# ## Data Cleaning Tools

# def drop_missing_rows(df):
#     return df.dropna()


# def drop_missing_cols(df, threshold=0.5):
#     return df.loc[:, df.isna().mean() < threshold]


# def fill_missing(df, method="mean"):
#     df = df.copy()
#     for col in df.columns:
#         if df[col].isna().sum() == 0:
#             continue
#         if method == "mean" and pd.api.types.is_numeric_dtype(df[col]):
#             df[col] = df[col].fillna(df[col].mean())
#         elif method == "median" and pd.api.types.is_numeric_dtype(df[col]):
#             df[col] = df[col].fillna(df[col].median())
#         else:
#             df[col] = df[col].fillna(df[col].mode()[0])
#     return df

# ## Auto isnights text

# def auto_insights(df):
#     insights = []
#     if df.isna().sum().sum() > 0:
#         insights.append("Dataset contains missing values.")

#     corr = df.corr(numeric_only=True)
#     if (abs(corr) > 0.7).sum().sum() > len(corr):
#         insights.append("Strong correlations found — feature redundancy possible.")

#     for col in df.select_dtypes(include="number").columns:
#         if df[col].skew() > 1:
#             insights.append(f"Column '{col}' is highly skewed.")

#     return insights
