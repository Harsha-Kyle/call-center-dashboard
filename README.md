# Call Center Sentiment Analytics Dashboard

Interactive call-centre dashboard built with Python, Dash, Pandas, Plotly and SQLite.

## Features
- Sentiment classification: rating 4+ Positive, 3 Neutral, below 3 Negative.
- Filters for sentiment, date range, resolution, and duration.
- KPI cards, five charts, empty-result handling, and reset filters.
- Save, view, and delete saved filter snapshots.
- Render-ready deployment using Gunicorn.

## Run locally
1. Put `CallCenterDataset.xlsx` in this folder. It must contain worksheet `Sheet1` and columns `Satisfaction rating`, `Date`, `Resolved`, and `AvgTalkDuration`.
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python app.py`
4. Open `http://127.0.0.1:8050`.

## Deploy on Render
1. Push the project files and the workbook to GitHub. Only publish the workbook if you have permission to share its contents.
2. In Render choose **New → Blueprint**, connect the repository, and deploy.
3. Open the generated URL and test the filters and charts.

## Notes
- The workbook is loaded at startup. A missing workbook or required columns will produce a clear startup error.
- SQLite history may be lost on hosts with ephemeral filesystems. Use a persistent disk or managed database if saved views must survive restarts.
- The negative-sentiment chart is not a predictive churn model.
- Do not publish personal or confidential customer data.
