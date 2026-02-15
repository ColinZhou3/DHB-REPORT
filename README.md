# Streamlit Sales & Price Trend App (Fast / Cloud-friendly)

Main idea: **filter first, then melt only selected date columns**.
This avoids long processing and memory crash on Streamlit Cloud.

## Deploy
Upload these to GitHub repo root:
- sales_analysis_app.py
- requirements.txt
- README.md

Streamlit Cloud main file: `sales_analysis_app.py`

## How it works
- Upload xlsx/xls/csv
- Keyword search products, pick SKUs
- Pick banner/store/date range
- Click **Run analysis**
- App only reshapes the rows + date columns you selected
