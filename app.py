import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State, ctx, no_update
import dash_bootstrap_components as dbc

APP_TITLE = "Call Center Sentiment Analytics"
DATA_FILE = Path(os.getenv("DATA_FILE", "CallCenterDataset.xlsx"))
SHEET_NAME = os.getenv("SHEET_NAME", "Sheet1")
DB_PATH = os.getenv("DATABASE_PATH", "call_center_data.db")
REQUIRED = {"Satisfaction rating", "Date", "Resolved", "AvgTalkDuration"}
COLORS = {"Positive": "#16a34a", "Neutral": "#f59e0b", "Negative": "#ef4444"}


def classify_sentiment(value):
    rating = pd.to_numeric(value, errors="coerce")
    if pd.isna(rating):
        return None
    return "Positive" if rating >= 4 else "Neutral" if rating == 3 else "Negative"


def parse_duration(series):
    numeric = pd.to_numeric(series, errors="coerce")
    missing = numeric.isna()
    if missing.any():
        numeric.loc[missing] = pd.to_timedelta(
            series.loc[missing].astype(str), errors="coerce"
        ).dt.total_seconds()
    return numeric


def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}. Add the workbook to the repository "
            "or set the DATA_FILE environment variable."
        )
    data = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)
    data.columns = [str(c).strip() for c in data.columns]
    missing = REQUIRED - set(data.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))

    data = data.copy()
    data["Satisfaction rating"] = pd.to_numeric(data["Satisfaction rating"], errors="coerce")
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce", utc=True).dt.tz_convert(None)
    data["AvgTalkDuration"] = parse_duration(data["AvgTalkDuration"])
    data["Resolved"] = data["Resolved"].astype(str).str.strip().str.upper().replace({
        "YES": "Y", "NO": "N", "TRUE": "Y", "FALSE": "N", "1": "Y", "0": "N"
    })
    data.loc[~data["Resolved"].isin(["Y", "N"]), "Resolved"] = None
    data["Sentiment"] = data["Satisfaction rating"].apply(classify_sentiment)
    data = data.dropna(subset=["Date", "AvgTalkDuration", "Sentiment", "Resolved"])
    data = data[data["AvgTalkDuration"] >= 0].reset_index(drop=True)
    if data.empty:
        raise ValueError("No valid rows remain after cleaning the dataset.")
    return data


df = load_data()
MIN_DATE, MAX_DATE = df["Date"].min().date(), df["Date"].max().date()
MIN_DURATION, MAX_DURATION = float(df["AvgTalkDuration"].min()), float(df["AvgTalkDuration"].max())


def connect_db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with closing(connect_db()) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS past_filters (
            id INTEGER PRIMARY KEY AUTOINCREMENT, sentiment TEXT, start_date TEXT,
            end_date TEXT, resolved TEXT, duration_min REAL, duration_max REAL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
        conn.commit()


def empty_figure(title):
    fig = go.Figure()
    fig.add_annotation(text="No records match these filters.", x=.5, y=.5,
                       xref="paper", yref="paper", showarrow=False)
    fig.update_layout(title=title, template="plotly_white", height=340)
    return fig


def style(fig):
    fig.update_layout(template="plotly_white", height=340,
                      margin=dict(l=35, r=20, t=60, b=40), legend_title_text="")
    return fig


def stat_card(label, value, note):
    return dbc.Card(dbc.CardBody([
        html.Div(label, className="stat-label"),
        html.Div(value, className="stat-value"),
        html.Div(note, className="stat-note"),
    ]), className="shadow-sm border-0 h-100")


init_db()
app = dash.Dash(
    __name__, external_stylesheets=[dbc.themes.FLATLY], title=APP_TITLE,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server

app.index_string = """<!DOCTYPE html><html><head>
{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
body{background:#f3f6fb;color:#172033;font-family:Inter,"Segoe UI",sans-serif}
.page{max-width:1450px;margin:auto;padding:24px 18px 40px}
.hero{background:linear-gradient(120deg,#102a56,#2563eb);color:white;border-radius:20px;padding:26px}
.hero h1{font-weight:800}.hero p{color:#dbeafe;margin:0}
.credit{color:#dbeafe;font-size:.85rem;text-align:right}
.stat-label{font-size:.72rem;font-weight:700;color:#64748b;letter-spacing:.6px}
.stat-value{font-size:1.75rem;font-weight:800;color:#1d4ed8;margin-top:5px}
.stat-note{font-size:.78rem;color:#64748b}
.section-title{font-weight:750}
@media(max-width:767px){.page{padding:12px}.hero{padding:20px}.credit{text-align:left;margin-top:10px}}
</style></head><body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>"""

app.layout = dbc.Container(fluid=True, className="page", children=[
    dbc.Row([
        dbc.Col([html.Div("CALL CENTER ANALYTICS", className="small fw-bold mb-2"),
                 html.H1("Customer Sentiment Dashboard"),
                 html.P("Explore satisfaction, resolution outcomes, and call duration.")], md=8),
        dbc.Col(html.Div("Done by: Harsha Kyle", className="credit"), md=4,
                className="d-flex align-items-end justify-content-md-end"),
    ], className="hero mb-4 g-2"),
    dbc.Row([
        dbc.Col(stat_card("TOTAL CALLS", f"{len(df):,}", "Valid records"), xs=6, lg=3, className="mb-3"),
        dbc.Col(stat_card("POSITIVE SENTIMENT", f"{(df.Sentiment == 'Positive').mean():.1%}", "Of valid calls"), xs=6, lg=3, className="mb-3"),
        dbc.Col(stat_card("RESOLUTION RATE", f"{(df.Resolved == 'Y').mean():.1%}", "Calls marked resolved"), xs=6, lg=3, className="mb-3"),
        dbc.Col(stat_card("AVG. DURATION", f"{df.AvgTalkDuration.mean():.0f}s", "Seconds per call"), xs=6, lg=3, className="mb-3"),
    ]),
    dbc.Card(dbc.CardBody([
        html.H5("Filter your analysis", className="section-title"),
        dbc.Row([
            dbc.Col([html.Label("Sentiment"), dcc.Dropdown(id="sentiment-filter",
                options=[{"label": x, "value": x} for x in COLORS], value=[], multi=True,
                placeholder="All sentiments")], md=3, className="mb-3"),
            dbc.Col([html.Label("Date range"), dcc.DatePickerRange(id="date-picker",
                min_date_allowed=MIN_DATE, max_date_allowed=MAX_DATE, start_date=MIN_DATE,
                end_date=MAX_DATE, display_format="YYYY-MM-DD")], md=4, className="mb-3"),
            dbc.Col([html.Label("Resolution"), dcc.Dropdown(id="resolved-filter",
                options=[{"label": "Resolved", "value": "Y"}, {"label": "Unresolved", "value": "N"}],
                value=None, placeholder="All calls", clearable=True)], md=2, className="mb-3"),
            dbc.Col([html.Label("Duration (seconds)"), dcc.RangeSlider(id="duration-slider",
                min=MIN_DURATION, max=max(MAX_DURATION, MIN_DURATION + 1), step=1,
                value=[MIN_DURATION, MAX_DURATION],
                marks={int(MIN_DURATION): str(int(MIN_DURATION)), int(MAX_DURATION): str(int(MAX_DURATION))},
                tooltip={"placement": "bottom", "always_visible": False})], md=3, className="mb-3"),
        ]),
        dbc.Button("Reset filters", id="reset-filters", color="secondary", outline=True, size="sm"),
    ]), className="mb-4 border-0 shadow-sm"),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="sentiment-pie", config={"displaylogo": False})), className="border-0 shadow-sm"), lg=6, className="mb-3"),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="sentiment-trend", config={"displaylogo": False})), className="border-0 shadow-sm"), lg=6, className="mb-3"),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="resolution-bar", config={"displaylogo": False})), className="border-0 shadow-sm"), lg=6, className="mb-3"),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="duration-box", config={"displaylogo": False})), className="border-0 shadow-sm"), lg=6, className="mb-3"),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="negative-sentiment", config={"displaylogo": False})), className="border-0 shadow-sm"), width=12, className="mb-3"),
    ]),
    dbc.Card(dbc.CardBody([
        html.H5("Saved filter views", className="section-title"),
        html.P("Save or review filter settings. Saved history may be lost on ephemeral hosting.", className="text-secondary small"),
        dbc.Button("Save current view", id="save-view-btn", color="primary", className="me-2 mb-2"),
        dbc.Button("View saved data", id="view-past-btn", color="secondary", outline=True, className="me-2 mb-2"),
        dbc.Button("Delete all saved views", id="delete-all-btn", color="danger", outline=True, className="mb-2"),
        html.Div(id="past-data-output", className="mt-3"),
    ]), className="border-0 shadow-sm"),
    html.Div("Built with Python, Dash, Pandas, Plotly and SQLite", className="text-center text-secondary small mt-4"),
])


@app.callback(
    Output("sentiment-filter", "value"), Output("date-picker", "start_date"),
    Output("date-picker", "end_date"), Output("resolved-filter", "value"),
    Output("duration-slider", "value"), Input("reset-filters", "n_clicks"),
    prevent_initial_call=True,
)
def reset_filters(_):
    return [], MIN_DATE, MAX_DATE, None, [MIN_DURATION, MAX_DURATION]


@app.callback(
    Output("sentiment-pie", "figure"), Output("sentiment-trend", "figure"),
    Output("resolution-bar", "figure"), Output("duration-box", "figure"),
    Output("negative-sentiment", "figure"),
    Input("sentiment-filter", "value"), Input("date-picker", "start_date"),
    Input("date-picker", "end_date"), Input("resolved-filter", "value"),
    Input("duration-slider", "value"),
)
def update_graphs(sentiments, start_date, end_date, resolved, duration_range):
    data = df.copy()
    if sentiments:
        data = data[data["Sentiment"].isin(sentiments)]
    if start_date:
        data = data[data["Date"] >= pd.Timestamp(start_date)]
    if end_date:
        data = data[data["Date"] < pd.Timestamp(end_date) + pd.Timedelta(days=1)]
    if resolved:
        data = data[data["Resolved"] == resolved]
    if duration_range and len(duration_range) == 2:
        data = data[data["AvgTalkDuration"].between(*map(float, duration_range))]

    titles = ["Sentiment distribution", "Sentiment trends over time",
              "Resolution and sentiment", "Call duration by sentiment",
              "Negative-sentiment calls by resolution"]
    if data.empty:
        return tuple(empty_figure(t) for t in titles)

    pie = px.pie(data, names="Sentiment", title=titles[0], hole=.45,
                 color="Sentiment", color_discrete_map=COLORS)
    pie.update_traces(textposition="inside", textinfo="percent+label")
    monthly = data.assign(Month=data["Date"].dt.to_period("M").astype(str)).groupby(
        ["Month", "Sentiment"]).size().reset_index(name="Calls")
    trend = px.line(monthly, x="Month", y="Calls", color="Sentiment", markers=True,
                    title=titles[1], color_discrete_map=COLORS)
    resolution_data = data.groupby(["Resolved", "Sentiment"]).size().reset_index(name="Calls")
    resolution_data["Resolution"] = resolution_data["Resolved"].map({"Y": "Resolved", "N": "Unresolved"})
    resolution = px.bar(resolution_data, x="Resolution", y="Calls", color="Sentiment",
                        barmode="group", title=titles[2], color_discrete_map=COLORS)
    duration = px.box(data, x="Sentiment", y="AvgTalkDuration", color="Sentiment",
                      title=titles[3], color_discrete_map=COLORS, points="outliers")
    duration.update_yaxes(title="Duration (seconds)")
    neg = data[data["Sentiment"] == "Negative"].groupby("Resolved").size().reindex(
        ["Y", "N"], fill_value=0).rename("Calls").reset_index()
    neg["Resolution"] = neg["Resolved"].map({"Y": "Resolved", "N": "Unresolved"})
    negative = px.bar(neg, x="Resolution", y="Calls", color="Resolved", title=titles[4],
                      color_discrete_map={"Y": "#16a34a", "N": "#ef4444"})
    negative.update_layout(showlegend=False)
    return tuple(style(fig) for fig in [pie, trend, resolution, duration, negative])


@app.callback(
    Output("past-data-output", "children"),
    Input("save-view-btn", "n_clicks"), Input("view-past-btn", "n_clicks"),
    Input("delete-all-btn", "n_clicks"),
    State("sentiment-filter", "value"), State("date-picker", "start_date"),
    State("date-picker", "end_date"), State("resolved-filter", "value"),
    State("duration-slider", "value"), prevent_initial_call=True,
)
def manage_saved_views(_, __, ___, sentiments, start, end, resolved, duration):
    action = ctx.triggered_id
    message = None
    with closing(connect_db()) as conn:
        if action == "save-view-btn":
            conn.execute(
                "INSERT INTO past_filters(sentiment,start_date,end_date,resolved,duration_min,duration_max) VALUES(?,?,?,?,?,?)",
                (",".join(sentiments or []), str(start or ""), str(end or ""), resolved or "",
                 float(duration[0]) if duration else None, float(duration[1]) if duration else None),
            )
            conn.commit()
            message = dbc.Alert("Current filter view saved.", color="success", dismissable=True)
        elif action == "delete-all-btn":
            conn.execute("DELETE FROM past_filters")
            conn.commit()
            return dbc.Alert("Saved filter history cleared.", color="info", dismissable=True)
        rows = conn.execute(
            "SELECT id, sentiment, start_date, end_date, resolved, duration_min, duration_max, created_at FROM past_filters ORDER BY id DESC LIMIT 100"
        ).fetchall()

    if not rows:
        return html.Div([message, html.P("No saved views yet.", className="text-secondary")])
    table_df = pd.DataFrame([dict(row) for row in rows]).rename(columns={
        "id": "ID", "sentiment": "Sentiment", "start_date": "Start date",
        "end_date": "End date", "resolved": "Resolution",
        "duration_min": "Min duration (s)", "duration_max": "Max duration (s)",
        "created_at": "Saved at",
    })
    return html.Div([message, dbc.Table.from_dataframe(
        table_df, striped=True, hover=True, responsive=True, size="sm"
    )])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8050")), debug=False)
