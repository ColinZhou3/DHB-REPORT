# Streamlit Sales & Price Trend App (Fast v3)

Fixes:
- Keyword search supports comma **OR**:
  - `grape seedless` = AND
  - `grape, seedless` = OR
  - `grape -raisin` = exclude
- Sidebar shows matched count + example names when zero.

Deploy:
- Upload `sales_analysis_app.py`, `requirements.txt`, `README.md` to GitHub repo root
- Streamlit Cloud main file: `sales_analysis_app.py`
