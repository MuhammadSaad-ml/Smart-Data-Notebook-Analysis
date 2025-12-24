<!-- ================= BADGES ================= -->

<!-- ================= BADGES ================= -->

<!-- Core -->
<img src="https://img.shields.io/badge/Data_Science-4CAF50"> <img src="https://img.shields.io/badge/Python-3776AB"> <img src="https://img.shields.io/badge/Streamlit-FF4B4B"> <img src="https://img.shields.io/badge/Web_App-673AB7"> <img src="https://img.shields.io/badge/Interactive_UI-9C27B0">

<!-- EDA & Notebooks -->
<img src="https://img.shields.io/badge/EDA-Automated-E91E63"> <img src="https://img.shields.io/badge/Jupyter-Notebook-F57C00"> <img src="https://img.shields.io/badge/Notebook_Execution-Enabled-FF9800"> <img src="https://img.shields.io/badge/Markdown-Supported-795548">

<!-- File Handling -->
<img src="https://img.shields.io/badge/File_Upload-Secure-009688"> <img src="https://img.shields.io/badge/CSV-Excel-JSON-3F51B5"> <img src="https://img.shields.io/badge/ZIP_Multi--File-Supported-3F51B5"> <img src="https://img.shields.io/badge/Large_Files-200MB-2196F3">

<!-- Libraries -->
<img src="https://img.shields.io/badge/Pandas-Data_Analysis-795548"> <img src="https://img.shields.io/badge/NumPy-Scientific-2196F3"> <img src="https://img.shields.io/badge/Plotly-Interactive_Charts-607D8B"> <img src="https://img.shields.io/badge/Stats-Insights-607D8B">

<!-- App Quality -->
<img src="https://img.shields.io/badge/Open_Source-Yes-8BC34A"> <img src="https://img.shields.io/badge/End--to--End-Data_App-FFC107"> <img src="https://img.shields.io/badge/Beginner_Friendly-Yes-03A9F4"> <img src="https://img.shields.io/badge/No_Code_Required-True-4CAF50">


---

# 🚀 Smart Data & Notebook Analysis Web App (Streamlit)

This repository contains an **interactive Smart Data & Notebook Analysis Web Application** built using **Streamlit**.

The app provides **three powerful analysis views**:
1. **Jupyter Notebook (Cell-by-Cell Execution)**
2. **Unified Python Data View**
3. **Automated Exploratory Data Analysis (EDA)**

Users can upload datasets or notebooks and instantly explore insights, charts, statistics, and relationships — **no technical setup required**.

---

## ▶ Click On Image To Open App
[![Smart Data & Notebook Analysis App](https://github.com/MuhammadSaad-ml/Smart-Data-Notebook-Analysis/raw/main/cannabis_analysis_app/App.png)](https://uxpsrcrrejhyxsnsg6disb.streamlit.app/)
🔗 Live App: https://uxpsrcrrejhyxsnsg6disb.streamlit.app/


---

## ✨ Application Views & Features

---

## 🧠 View 1: Jupyter Notebook Interface (Cell-by-Cell)

This view allows users to upload a **Jupyter Notebook (`.ipynb`)** and run it **cell by cell**, exactly like Jupyter Lab.

### 🔹 What this view supports:
- Proper **Markdown rendering** inside notebook cells  
- Python code execution cell-by-cell  
- Automatic display of:
  - Charts (Matplotlib / Plotly)
  - Tables
  - Printed outputs
- Clean separation between **Markdown, code, and output**

### 📘 Upload Notebook
```text
Upload Notebook (.ipynb)
```

> Ideal for users who already have analysis notebooks and want to run them without setting up Python environments.

---

## 📁 View 2: Unified Python Data View (Supporting Data)

This view is designed to support **datasets required by the notebook or Python logic**.

### 🔹 Key capabilities:
- Upload supporting files:
  - CSV
  - Excel
  - JSON
  - ZIP (multiple datasets)
- Automatically loads data into memory
- Displays **important DataFrames created from the main dataset**
- Focuses on **core cleaned / transformed data**, not every intermediate step

### 📂 Supported uploads:
```text
CSV, XLSX, XLS, JSON, ZIP
(Max 200 MB per file)
```

> This view acts as the **backend data layer** for notebooks and Python logic.

---

## 📊 View 3: Automated Exploratory Data Analysis (EDA)

This is the **fully automated EDA interface**, designed for deep data understanding with **zero coding**.

Once a dataset is uploaded, the app generates structured analysis sections like:

### 📁 Loaded File
```text
📁 Loaded File: Example. CSV, XLSX, XLS, JSON, ZIP
```

---

### 🔍 Preview
- First rows of the dataset
- Column names & data types
- Shape of the dataset

---

### 📘 Summary
- Descriptive statistics
- Missing values overview
- Numerical & categorical insights

---

### 🤖 Auto EDA
- Automated insights
- Distribution analysis
- Data quality checks

---

### 📈 Charts
- Histograms
- Bar charts
- Scatter plots
- Interactive Plotly visuals

---

### 🔗 Correlations
- Feature correlation matrix
- Heatmaps
- Relationship analysis between numerical variables

---

### 🧩 Categories
- Categorical feature analysis
- Value counts
- Frequency distributions

> This section is ideal for **non-technical users**, analysts, and rapid exploration.

---

## 📁 Project Structure & Files Included

```text
cannabis_analysis_app/
│
├── app.py              # Main Streamlit UI
├── analysis.py         # Core EDA & data logic
├── config.py           # App configuration
├── requirements.txt    # Dependencies
│
├── Smart Data & Notebook Analysis/               # Testing & experiments
└── __pycache__/        # Compiled Python files
```

---

## 🔄 Data Flow Pipeline

1. Upload dataset / notebook  
2. Notebook execution (optional)  
3. Supporting data loading  
4. Unified DataFrame processing  
5. Automated EDA generation  
6. Charts & insights display  

---

## 🎯 Use Cases

- Exploratory Data Analysis (EDA)
- Notebook execution without setup
- Rapid dataset understanding
- Teaching & learning data analysis
- Portfolio & demo projects

---

## 🔒 Privacy & Security
- Files remain private
- No permanent storage
- Session-based processing only

---

## 👤 Author

**Muhammad Saad**  
Data Analyst & Data Scientist 🌱
