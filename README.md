# Streamlit Sales & Price Trend App (Keyword Select)

This app is made for your common raw-data format:
- Many date columns ending with `_SALES` and `_VOLUME`
- Base columns like `PRODUCT_NAME`, `PRODUCT_CODE`, `FORMAT_NAME`, `STORE_NAME`...

You can:
- Upload xlsx/xls/csv
- Type keyword (example: `grape seedless` or `grape -raisin`)
- Select products (multi-select)
- Filter banner / store / date range
- Compare Sales / Avg Price / Volume trends
- Download aggregated CSV

## Run locally

```bash
pip install -r requirements.txt
streamlit run sales_analysis_app.py
```

## Deploy on Streamlit Community Cloud (GitHub)

1. Create a GitHub repo
2. Upload these 3 files to the repo root:
   - `sales_analysis_app.py`
   - `requirements.txt`
   - `README.md`
3. Go to Streamlit Cloud and create a new app
4. Select your repo + branch
5. Set **Main file path** to: `sales_analysis_app.py`
6. Deploy

## Notes

- Date parsing is flexible (`dayfirst=True`). Your column names can be like:
  - `06/01/2026_SALES`
  - `2026-01-06_SALES`
  - `06-01-2026_SALES`
- For very big files: first run may take longer because it reshapes wide → long.
