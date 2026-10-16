# Call Center Sentiment Analytics — Stable Dashboard

import os
import sqlite3
from contextlib import closing
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import (
    dcc,
    html,
    Input,
    Output,
    State,
    ctx,
    dash_table,
    ALL,
    no_update,
)
import dash_bootstrap_components as dbc


# ============================================================
# CONFIGURATION
# ============================================================

APP_TITLE = "Call Center Intelligence"
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = Path(os.getenv("DATA_FILE", str(BASE_DIR / "CallCenterDataset.xlsx")))
SHEET_NAME = os.getenv("SHEET_NAME", "Sheet1")
DB_PATH = Path(os.getenv("DATABASE_PATH", str(BASE_DIR / "call_center_data.db")))
PORT = int(os.getenv("PORT", "8050"))

NAVY = "#17233f"
BLUE = "#4f6bed"
MUTED = "#758198"

COLORS = {
    "Positive": "#16a34a",
    "Neutral": "#f59e0b",
    "Negative": "#ef4444",
}

REQUIRED_COLUMNS = {
    "Satisfaction rating",
    "Date",
    "Resolved",
    "AvgTalkDuration",
}


# ============================================================
# DATA PROCESSING
# ============================================================

def classify_sentiment(value):
    rating = pd.to_numeric(value, errors="coerce")

    if pd.isna(rating):
        return None

    if rating >= 4:
        return "Positive"
    if rating == 3:
        return "Neutral"
    return "Negative"


def parse_duration(series):
    """Convert numeric durations or time strings into seconds."""
    original = series.copy()
    result = pd.to_numeric(original, errors="coerce")

    # Excel stores time-only durations as fractions of a day.
    excel_fraction = result.between(0, 1, inclusive="neither")
    result.loc[excel_fraction] *= 86400

    missing = result.isna()

    if missing.any():
        values = original.loc[missing].astype(str).str.strip()

        # Try parsing standard durations such as 00:02:30.
        parsed = pd.to_timedelta(values, errors="coerce")
        result.loc[missing] = parsed.dt.total_seconds()

        # Fallback for MM:SS strings.
        remaining = result.loc[missing].isna()

        for index, value in values.loc[remaining].items():
            parts = value.split(":")
            try:
                if len(parts) == 2:
                    result.loc[index] = float(parts[0]) * 60 + float(parts[1])
                elif len(parts) == 3:
                    result.loc[index] = (
                        float(parts[0]) * 3600
                        + float(parts[1]) * 60
                        + float(parts[2])
                    )
            except (ValueError, TypeError):
                continue

    return result


def normalize_resolution(series):
    mapping = {
        "Y": "Y",
        "YES": "Y",
        "TRUE": "Y",
        "1": "Y",
        "RESOLVED": "Y",
        "COMPLETED": "Y",
        "N": "N",
        "NO": "N",
        "FALSE": "N",
        "0": "N",
        "UNRESOLVED": "N",
        "PENDING": "N",
    }

    return (
        series.astype("string")
        .str.strip()
        .str.upper()
        .map(mapping)
    )


def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_FILE.resolve()}\n\n"
            "Place CallCenterDataset.xlsx beside app.py, "
            "or set the DATA_FILE environment variable."
        )

    try:
        data = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)
    except ValueError as exc:
        raise ValueError(
            f"Sheet '{SHEET_NAME}' was not found in {DATA_FILE.name}. "
            "Update SHEET_NAME to match your workbook."
        ) from exc

    data.columns = [str(column).strip() for column in data.columns]

    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(sorted(missing))
        )

    data = data.copy()

    data["Satisfaction rating"] = pd.to_numeric(
        data["Satisfaction rating"], errors="coerce"
    )

    # Parse dates without forcing timezone conversion.
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

    try:
        if data["Date"].dt.tz is not None:
            data["Date"] = data["Date"].dt.tz_localize(None)
    except (AttributeError, TypeError):
        pass

    data["AvgTalkDuration"] = parse_duration(data["AvgTalkDuration"])
    data["Resolved"] = normalize_resolution(data["Resolved"])
    data["Sentiment"] = data["Satisfaction rating"].apply(classify_sentiment)

    data = data.dropna(
        subset=[
            "Date",
            "Satisfaction rating",
            "AvgTalkDuration",
            "Resolved",
            "Sentiment",
        ]
    )

    data = data[data["AvgTalkDuration"] >= 0].copy()

    if data.empty:
        raise ValueError(
            "No valid rows remain after cleaning. Check the date, "
            "satisfaction, resolution and duration columns."
        )

    data["Resolution"] = data["Resolved"].map({
        "Y": "Resolved",
        "N": "Unresolved",
    })

    data.reset_index(drop=True, inplace=True)
    return data


df = load_data()

MIN_DATE = df["Date"].min().date()
MAX_DATE = df["Date"].max().date()
MIN_DURATION = float(df["AvgTalkDuration"].min())
MAX_DURATION = float(df["AvgTalkDuration"].max())

SLIDER_MIN = min(0.0, MIN_DURATION)
SLIDER_MAX = max(MAX_DURATION, SLIDER_MIN + 1.0)

SLIDER_STEP = max((SLIDER_MAX - SLIDER_MIN) / 500, 0.1)

DEFAULT_DURATION = [MIN_DURATION, MAX_DURATION]


# ============================================================
# SQLITE DATABASE AND MIGRATIONS
# ============================================================

def connect_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(DB_PATH), timeout=15)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with closing(connect_db()) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS past_filters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL DEFAULT 'Saved view',
                sentiment TEXT NOT NULL DEFAULT '',
                start_date TEXT NOT NULL DEFAULT '',
                end_date TEXT NOT NULL DEFAULT '',
                resolved TEXT NOT NULL DEFAULT '',
                duration_min REAL,
                duration_max REAL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(past_filters)"
            ).fetchall()
        }

        migrations = {
            "name": "TEXT NOT NULL DEFAULT 'Saved view'",
            "sentiment": "TEXT NOT NULL DEFAULT ''",
            "start_date": "TEXT NOT NULL DEFAULT ''",
            "end_date": "TEXT NOT NULL DEFAULT ''",
            "resolved": "TEXT NOT NULL DEFAULT ''",
            "duration_min": "REAL",
            "duration_max": "REAL",
            "created_at": "TEXT NOT NULL DEFAULT ''",
        }

        for column, definition in migrations.items():
            if column not in columns:
                connection.execute(
                    f"ALTER TABLE past_filters ADD COLUMN {column} {definition}"
                )

        connection.commit()


init_db()


# ============================================================
# SHARED HELPERS
# ============================================================

def seconds_label(value):
    if pd.isna(value):
        return "—"

    value = max(0.0, float(value))

    if value >= 3600:
        return f"{int(value // 3600)}h {int((value % 3600) // 60)}m"

    if value >= 60:
        return f"{int(value // 60)}m {int(value % 60)}s"

    if value.is_integer():
        return f"{int(value)}s"

    return f"{value:.1f}s"


def filter_data(sentiments, start_date, end_date, resolved, duration):
    data = df

    if sentiments:
        data = data[data["Sentiment"].isin(sentiments)]

    if start_date:
        data = data[data["Date"] >= pd.Timestamp(start_date)]

    if end_date:
        data = data[
            data["Date"] < pd.Timestamp(end_date) + pd.Timedelta(days=1)
        ]

    if resolved in ("Y", "N"):
        data = data[data["Resolved"] == resolved]

    if duration and len(duration) == 2:
        low, high = sorted([float(duration[0]), float(duration[1])])
        data = data[data["AvgTalkDuration"].between(low, high)]

    return data.copy()


def empty_figure(message="No records match these filters."):
    figure = go.Figure()

    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 14, "color": MUTED},
    )

    figure.update_layout(
        template="plotly_white",
        height=310,
        margin=dict(l=15, r=15, t=20, b=20),
        xaxis={"visible": False},
        yaxis={"visible": False},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return figure


def style_figure(figure):
    figure.update_layout(
        template="plotly_white",
        height=310,
        margin=dict(l=25, r=15, t=20, b=35),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={
            "family": "Inter, Segoe UI, sans-serif",
            "color": NAVY,
            "size": 12,
        },
        legend_title_text="",
        hoverlabel={"bgcolor": NAVY, "font_color": "white"},
    )

    figure.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor="#e9edf5",
    )

    figure.update_yaxes(
        gridcolor="#edf1f7",
        zeroline=False,
    )

    return figure


def metric_card(title, value_id, note_id, icon_name, color):
    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.Div(title, className="metric-label"),
                html.Div(
                    html.I(className=f"bi {icon_name}"),
                    className="metric-icon",
                    style={
                        "color": color,
                        "backgroundColor": f"{color}15",
                    },
                ),
            ], className="d-flex justify-content-between align-items-center"),
            html.Div(id=value_id, className="metric-value"),
            html.Div(id=note_id, className="metric-note"),
        ]),
        className="metric-card h-100",
    )


def chart_card(title, subtitle, graph_id):
    return dbc.Card(
        dbc.CardBody([
            html.H5(title, className="chart-title"),
            html.P(subtitle, className="chart-subtitle"),
            dcc.Graph(
                id=graph_id,
                figure=empty_figure(),
                config={
                    "displaylogo": False,
                    "responsive": True,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                },
                style={"height": "330px"},
            ),
        ]),
        className="panel-card h-100",
    )


def make_insights(data):
    if data.empty:
        return dbc.Alert(
            "No insights available for the current filters.",
            color="light",
        )

    negative_rate = (data["Sentiment"] == "Negative").mean()
    positive_rate = (data["Sentiment"] == "Positive").mean()
    resolution_rate = (data["Resolved"] == "Y").mean()

    unresolved_negative = int(
        (
            (data["Sentiment"] == "Negative")
            & (data["Resolved"] == "N")
        ).sum()
    )

    cards = [
        (
            "warning" if negative_rate >= 0.25 else "success",
            "Negative sentiment",
            f"{negative_rate:.1%} of filtered calls are negative.",
        ),
        (
            "success" if resolution_rate >= 0.75 else "warning",
            "Resolution performance",
            f"{resolution_rate:.1%} of filtered calls are resolved.",
        ),
        (
            "danger" if unresolved_negative else "success",
            "Unresolved negative calls",
            f"{unresolved_negative:,} calls are negative and unresolved.",
        ),
        (
            "info",
            "Positive customer experience",
            f"{positive_rate:.1%} of filtered calls are positive.",
        ),
    ]

    return dbc.Row([
        dbc.Col(
            dbc.Alert([
                html.Div(title, className="fw-bold mb-1"),
                html.Div(message, className="small"),
            ], color=color, className="insight-alert"),
            xs=12,
            md=6,
            xl=3,
            className="mb-2",
        )
        for color, title, message in cards
    ])


# ============================================================
# DASH APPLICATION
# ============================================================

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.BOOTSTRAP,
    ],
    title=APP_TITLE,
    meta_tags=[{
        "name": "viewport",
        "content": "width=device-width, initial-scale=1",
    }],
)

server = app.server

app.index_string = """<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root{--background:#f4f6fb;--surface:#fff;--ink:#17233f;--muted:#758198;--border:#e9edf5;--primary:#4f6bed}
*{box-sizing:border-box}
html{scroll-behavior:smooth;scroll-padding-top:20px}
body{background:var(--background);color:var(--ink);font-family:'DM Sans',sans-serif}
.page-shell{max-width:1700px;margin:auto;padding:24px}
.sidebar{background:linear-gradient(180deg,#111a31,#203457);color:white;border-radius:22px;padding:23px 15px;min-height:calc(100vh - 48px)}
.brand-mark{height:42px;width:42px;flex-shrink:0;border-radius:13px;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#8995ff,#4f6bed);font-size:18px;font-weight:800;color:white}
.brand-name{font-family:Manrope,sans-serif;font-weight:800;font-size:17px}
.sidebar-caption{font-size:10px;font-weight:800;letter-spacing:1.6px;color:#aab7d0;margin:28px 10px 10px}
.sidebar .nav-link{color:#bdc8dc;border-radius:11px;padding:11px 12px;margin:3px 0;font-weight:600;font-size:13px}
.sidebar .nav-link:hover,.sidebar .nav-link.active{color:white;background:#ffffff19}
.sidebar-foot{font-size:12px;color:#aab7d0;line-height:1.7;overflow-wrap:anywhere}
.content-area{padding-left:8px}
.eyebrow{font-size:11px;font-weight:800;letter-spacing:1.5px;color:var(--primary)}
.page-title{font-family:Manrope,sans-serif;font-size:clamp(26px,3vw,36px);font-weight:800;letter-spacing:-1px;margin:5px 0}
.page-subtitle{color:var(--muted);font-size:14px;margin:0}
.topline{display:flex;justify-content:space-between;align-items:center;gap:16px;margin-bottom:24px}
.status-pill{display:inline-flex;align-items:center;gap:8px;white-space:nowrap;background:#e8f8ef;color:#15803d;border-radius:50px;padding:9px 13px;font-size:12px;font-weight:700}
.status-dot{width:7px;height:7px;border-radius:50%;background:#22c55e}
.metric-card,.panel-card,.filter-card{background:var(--surface);border:1px solid var(--border);border-radius:17px;box-shadow:0 5px 22px #26395d08}
.metric-card{transition:transform .18s ease,box-shadow .18s ease}
.metric-card:hover{transform:translateY(-3px);box-shadow:0 12px 26px #26395d12}
.metric-label{color:var(--muted);font-size:11px;font-weight:800;letter-spacing:.7px}
.metric-icon{width:38px;height:38px;flex-shrink:0;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:18px}
.metric-value{font-family:Manrope,sans-serif;font-weight:800;font-size:clamp(22px,2vw,30px);letter-spacing:-1px;margin-top:14px;line-height:1.2;overflow-wrap:anywhere}
.metric-note{font-size:12px;color:var(--muted);margin-top:7px}
.section-heading{font-family:Manrope,sans-serif;font-weight:800;font-size:19px;letter-spacing:-.4px}
.section-description{font-size:13px;color:var(--muted);margin-top:4px}
.filter-card{padding:20px}
.form-label{font-size:12px;font-weight:700;color:#45536b;margin-bottom:7px}
.form-control,.btn{border-radius:10px!important}
.btn{font-weight:700}
.insight-alert{border:0;border-radius:13px;min-height:88px}
.dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th{background:#f7f8fc!important;color:#66728a!important;font-size:11px!important;text-transform:uppercase;letter-spacing:.5px}
.dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner td{font-size:13px!important}
.footer{color:#98a2b3;font-size:12px;padding:24px 0 5px;text-align:center}
.saved-view{border:1px solid var(--border);border-radius:13px;background:#fff;padding:14px}
@media(max-width:991px){.page-shell{padding:14px}.sidebar{min-height:auto;margin-bottom:16px}.sidebar-caption,.sidebar-foot{display:none}.sidebar .nav{flex-direction:row!important;flex-wrap:wrap}.sidebar .nav-link{padding:8px 10px}.content-area{padding-left:0}}
@media(max-width:575px){.page-shell{padding:9px}.sidebar{padding:15px}.filter-card{padding:14px}.topline{align-items:flex-start;flex-direction:column}.status-pill{white-space:normal}}
</style>
</head>
<body>
{%app_entry%}
<footer>
{%config%}
{%scripts%}
{%renderer%}
</footer>
</body>
</html>"""


# ============================================================
# PAGE LAYOUT
# ============================================================

app.layout = dbc.Container(
    fluid=True,
    className="page-shell",
    children=[
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div("CC", className="brand-mark"),
                    html.Div([
                        html.Div("CallCenter", className="brand-name"),
                        html.Div(
                            "INTELLIGENCE",
                            style={
                                "fontSize": "9px",
                                "letterSpacing": "1.8px",
                                "color": "#aab7d0",
                            },
                        ),
                    ]),
                ], className="d-flex align-items-center gap-2 mb-4 px-2"),

                html.Div("WORKSPACE", className="sidebar-caption"),
                dbc.Nav([
                    dbc.NavLink(
                        [html.I(className="bi bi-grid-1x2 me-2"), "Overview"],
                        href="#overview",
                    ),
                    dbc.NavLink(
                        [html.I(className="bi bi-emoji-smile me-2"), "Sentiment"],
                        href="#sentiment",
                    ),
                    dbc.NavLink(
                        [html.I(className="bi bi-telephone me-2"), "Call explorer"],
                        href="#calls",
                    ),
                    dbc.NavLink(
                        [html.I(className="bi bi-bookmark-star me-2"), "Saved views"],
                        href="#saved",
                    ),
                ], vertical=True, pills=True),

                html.Div("DATASET", className="sidebar-caption"),
                html.Div([
                    html.Div([
                        html.I(
                            className="bi bi-circle-fill me-2",
                            style={"fontSize": "8px", "color": "#4ade80"},
                        ),
                        "Dataset loaded",
                    ]),
                    html.Div(f"{len(df):,} valid records", className="mt-2"),
                    html.Div(
                        "Done by Harsha Kyle, Mann and Manoj Kumar",
                        className="mt-3",
                    ),
                ], className="sidebar-foot px-2"),
            ], lg=2, md=12),

            dbc.Col([
                html.Div([
                    html.Div([
                        html.Div("CUSTOMER EXPERIENCE / ANALYTICS", className="eyebrow"),
                        html.H1("Customer Sentiment", className="page-title"),
                        html.P(
                            "Understand satisfaction, resolution outcomes, and call patterns.",
                            className="page-subtitle",
                        ),
                    ]),
                    html.Div([
                        html.Span(className="status-dot"),
                        " Dataset connected",
                    ], className="status-pill"),
                ], className="topline"),

                # Filters
                html.Div([
                    html.Div("FILTERS", className="eyebrow mb-1"),
                    html.H4("Refine your analysis", className="section-heading mb-1"),
                    html.P(
                        "All charts, metrics, insights and records update with your selection.",
                        className="section-description mb-3",
                    ),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Sentiment", className="form-label"),
                            dcc.Dropdown(
                                id="sentiment-filter",
                                options=[{"label": item, "value": item} for item in COLORS],
                                value=[],
                                multi=True,
                                placeholder="All sentiments",
                                clearable=True,
                            ),
                        ], xs=12, md=6, xl=3, className="mb-3"),

                        dbc.Col([
                            html.Label("Date range", className="form-label"),
                            dcc.DatePickerRange(
                                id="date-picker",
                                min_date_allowed=MIN_DATE,
                                max_date_allowed=MAX_DATE,
                                start_date=MIN_DATE,
                                end_date=MAX_DATE,
                                display_format="DD MMM YYYY",
                                minimum_nights=0,
                                clearable=True,
                            ),
                        ], xs=12, md=6, xl=4, className="mb-3"),

                        dbc.Col([
                            html.Label("Resolution", className="form-label"),
                            dcc.Dropdown(
                                id="resolved-filter",
                                options=[
                                    {"label": "Resolved", "value": "Y"},
                                    {"label": "Unresolved", "value": "N"},
                                ],
                                value=None,
                                placeholder="All calls",
                                clearable=True,
                            ),
                        ], xs=12, md=6, xl=2, className="mb-3"),

                        dbc.Col([
                            html.Label("Duration (seconds)", className="form-label"),
                            dcc.RangeSlider(
                                id="duration-slider",
                                min=SLIDER_MIN,
                                max=SLIDER_MAX,
                                step=SLIDER_STEP,
                                value=DEFAULT_DURATION,
                                marks={
                                    round(SLIDER_MIN): str(round(SLIDER_MIN)),
                                    round(SLIDER_MAX): str(round(SLIDER_MAX)),
                                },
                                tooltip={"placement": "bottom", "always_visible": False},
                                allowCross=False,
                            ),
                        ], xs=12, md=6, xl=3, className="mb-3"),
                    ]),
                    html.Div([
                        dbc.Button(
                            [html.I(className="bi bi-arrow-counterclockwise me-2"), "Reset filters"],
                            id="reset-filters",
                            color="light",
                            className="me-2",
                        ),
                        dbc.Button(
                            [html.I(className="bi bi-download me-2"), "Export CSV"],
                            id="export-btn",
                            color="primary",
                        ),
                        dcc.Download(id="download-csv"),
                        html.Span(
                            id="filter-count",
                            className="ms-auto text-secondary small align-self-center",
                        ),
                    ], className="d-flex flex-wrap gap-2 align-items-center"),
                ], className="filter-card mb-4"),

                # KPIs
                dbc.Row([
                    dbc.Col(metric_card("TOTAL CALLS", "kpi-total", "kpi-total-note", "bi-telephone", BLUE),
                            xs=6, xl=3, className="mb-3"),
                    dbc.Col(metric_card("POSITIVE SENTIMENT", "kpi-positive", "kpi-positive-note", "bi-emoji-smile", "#16a34a"),
                            xs=6, xl=3, className="mb-3"),
                    dbc.Col(metric_card("RESOLUTION RATE", "kpi-resolution", "kpi-resolution-note", "bi-check2-circle", "#0ea5e9"),
                            xs=6, xl=3, className="mb-3"),
                    dbc.Col(metric_card("AVG. CALL DURATION", "kpi-duration", "kpi-duration-note", "bi-clock-history", "#8b5cf6"),
                            xs=6, xl=3, className="mb-3"),
                ]),

                html.Div(id="insights-area", className="mb-3"),

                html.Div(id="overview", className="section-heading mb-1", children="Performance overview"),
                html.P(
                    "A visual snapshot of customer experience and operational outcomes.",
                    className="section-description mb-3",
                ),

                dbc.Row([
                    dbc.Col(chart_card("Sentiment mix", "Share of positive, neutral and negative calls", "sentiment-pie"),
                            xs=12, lg=5, className="mb-3"),
                    dbc.Col(chart_card("Sentiment over time", "Monthly call volume by sentiment", "sentiment-trend"),
                            xs=12, lg=7, className="mb-3"),
                    dbc.Col(chart_card("Resolution outcomes", "Sentiment distribution by resolution status", "resolution-bar"),
                            xs=12, lg=6, className="mb-3"),
                    dbc.Col(chart_card("Call duration", "Duration distribution by sentiment", "duration-box"),
                            xs=12, lg=6, className="mb-3"),
                    dbc.Col(chart_card("Negative calls by outcome", "Find negative calls that remain unresolved", "negative-sentiment"),
                            xs=12, className="mb-3"),
                ]),

                html.Div(id="sentiment", className="section-heading mt-3 mb-1", children="Customer satisfaction analysis"),
                html.P(
                    "Explore rating patterns and average call handling time.",
                    className="section-description mb-3",
                ),

                dbc.Row([
                    dbc.Col(chart_card("Satisfaction ratings", "Number of calls at each rating level", "rating-distribution"),
                            xs=12, md=6, className="mb-3"),
                    dbc.Col(chart_card("Duration by resolution", "Average handling time for each outcome", "duration-resolution"),
                            xs=12, md=6, className="mb-3"),
                ]),

                html.Div(id="calls", className="section-heading mt-3 mb-1", children="Call explorer"),
                html.P(
                    "Search, sort, and inspect records from the filtered dataset.",
                    className="section-description mb-3",
                ),

                dbc.Card(
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Input(
                                    id="table-search",
                                    placeholder="Search date, rating, sentiment or status...",
                                    type="text",
                                    debounce=True,
                                ),
                            ], xs=12, md=7),
                            dbc.Col([
                                html.Div(
                                    id="table-summary",
                                    className="text-secondary small text-md-end pt-2",
                                ),
                            ], xs=12, md=5),
                        ], className="mb-3"),

                        dash_table.DataTable(
                            id="calls-table",
                            columns=[
                                {"name": "Date", "id": "Date"},
                                {"name": "Satisfaction", "id": "Satisfaction rating", "type": "numeric"},
                                {"name": "Sentiment", "id": "Sentiment"},
                                {"name": "Resolution", "id": "Resolution"},
                                {"name": "Duration (s)", "id": "AvgTalkDuration", "type": "numeric"},
                            ],
                            data=[],
                            page_size=10,
                            page_action="native",
                            sort_action="native",
                            filter_action="native",
                            sort_mode="multi",
                            export_format="csv",
                            export_headers="display",
                            style_table={"overflowX": "auto"},
                            style_cell={
                                "padding": "12px",
                                "fontFamily": "DM Sans, sans-serif",
                                "fontSize": "13px",
                                "textAlign": "left",
                                "minWidth": "100px",
                                "maxWidth": "240px",
                                "whiteSpace": "normal",
                                "height": "auto",
                                "border": "none",
                            },
                            style_header={
                                "backgroundColor": "#f7f8fc",
                                "fontWeight": "800",
                                "color": "#66728a",
                                "border": "none",
                            },
                            style_data={
                                "borderBottom": "1px solid #edf1f7",
                                "color": NAVY,
                            },
                            style_data_conditional=[
                                {
                                    "if": {
                                        "filter_query": '{Sentiment} = "Positive"',
                                        "column_id": "Sentiment",
                                    },
                                    "color": "#15803d",
                                    "fontWeight": "700",
                                },
                                {
                                    "if": {
                                        "filter_query": '{Sentiment} = "Neutral"',
                                        "column_id": "Sentiment",
                                    },
                                    "color": "#b77905",
                                    "fontWeight": "700",
                                },
                                {
                                    "if": {
                                        "filter_query": '{Sentiment} = "Negative"',
                                        "column_id": "Sentiment",
                                    },
                                    "color": "#dc2626",
                                    "fontWeight": "700",
                                },
                            ],
                        ),
                    ]),
                    className="panel-card mb-4",
                ),

                # Saved views
                html.Div(id="saved", className="section-heading mt-3 mb-1", children="Saved filter views"),
                html.P(
                    "Save, apply, refresh or delete filter setups stored in SQLite.",
                    className="section-description mb-3",
                ),

                dbc.Card(
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Input(
                                    id="view-name",
                                    placeholder="Optional name for this view",
                                    type="text",
                                    maxLength=80,
                                ),
                            ], xs=12, md=5, className="mb-2"),
                            dbc.Col([
                                dbc.Button(
                                    [html.I(className="bi bi-bookmark-plus me-2"), "Save current view"],
                                    id="save-view-btn",
                                    color="primary",
                                    className="w-100",
                                ),
                            ], xs=12, md=3, className="mb-2"),
                            dbc.Col([
                                dbc.Button(
                                    [html.I(className="bi bi-arrow-clockwise me-2"), "Refresh"],
                                    id="view-past-btn",
                                    color="light",
                                    className="w-100",
                                ),
                            ], xs=6, md=2, className="mb-2"),
                            dbc.Col([
                                dbc.Button(
                                    [html.I(className="bi bi-trash me-2"), "Delete all"],
                                    id="delete-all-btn",
                                    color="danger",
                                    outline=True,
                                    className="w-100",
                                ),
                            ], xs=6, md=2, className="mb-2"),
                        ]),
                        html.Div(id="saved-feedback", className="mb-2"),
                        html.Div(id="past-data-output"),
                    ]),
                    className="panel-card mb-3",
                ),

                html.Div(
                    "Built with Python · Dash · Pandas · Plotly · SQLite | "
                    "Done by Harsha Kyle, Mann and Manoj Kumar",
                    className="footer",
                ),
            ], lg=10, md=12, className="content-area"),
        ], className="g-3"),
    ],
)


# ============================================================
# RESET FILTERS
# ============================================================

@app.callback(
    Output("sentiment-filter", "value"),
    Output("date-picker", "start_date"),
    Output("date-picker", "end_date"),
    Output("resolved-filter", "value"),
    Output("duration-slider", "value"),
    Output("table-search", "value"),
    Input("reset-filters", "n_clicks"),
    prevent_initial_call=True,
)
def reset_filters(_):
    return [], MIN_DATE, MAX_DATE, None, DEFAULT_DURATION, ""


# ============================================================
# DASHBOARD CALLBACK
# ============================================================

@app.callback(
    Output("kpi-total", "children"),
    Output("kpi-total-note", "children"),
    Output("kpi-positive", "children"),
    Output("kpi-positive-note", "children"),
    Output("kpi-resolution", "children"),
    Output("kpi-resolution-note", "children"),
    Output("kpi-duration", "children"),
    Output("kpi-duration-note", "children"),
    Output("filter-count", "children"),
    Output("insights-area", "children"),
    Output("sentiment-pie", "figure"),
    Output("sentiment-trend", "figure"),
    Output("resolution-bar", "figure"),
    Output("duration-box", "figure"),
    Output("negative-sentiment", "figure"),
    Output("rating-distribution", "figure"),
    Output("duration-resolution", "figure"),
    Output("calls-table", "data"),
    Output("table-summary", "children"),
    Input("sentiment-filter", "value"),
    Input("date-picker", "start_date"),
    Input("date-picker", "end_date"),
    Input("resolved-filter", "value"),
    Input("duration-slider", "value"),
    Input("table-search", "value"),
)
def update_dashboard(sentiments, start_date, end_date, resolved, duration_range, search):
    data = filter_data(sentiments, start_date, end_date, resolved, duration_range)
    count = len(data)

    if data.empty:
        empty = empty_figure()
        return (
            "0", "No matching records",
            "—", "No matching records",
            "—", "No matching records",
            "—", "No matching records",
            "0 calls match these filters",
            make_insights(data),
            empty, empty, empty, empty, empty, empty, empty,
            [],
            "No records to display",
        )

    positive_rate = (data["Sentiment"] == "Positive").mean()
    resolution_rate = (data["Resolved"] == "Y").mean()
    average_duration = data["AvgTalkDuration"].mean()

    # Sentiment donut
    sentiment_counts = (
        data["Sentiment"]
        .value_counts()
        .reindex(["Positive", "Neutral", "Negative"], fill_value=0)
        .rename_axis("Sentiment")
        .reset_index(name="Calls")
    )
    sentiment_counts = sentiment_counts[sentiment_counts["Calls"] > 0]

    pie = px.pie(
        sentiment_counts,
        names="Sentiment",
        values="Calls",
        hole=0.64,
        color="Sentiment",
        color_discrete_map=COLORS,
        category_orders={"Sentiment": ["Positive", "Neutral", "Negative"]},
    )
    pie.update_traces(
        textposition="outside",
        textinfo="percent",
        hovertemplate="%{label}: %{value} calls (%{percent})<extra></extra>",
        marker={"line": {"color": "white", "width": 3}},
    )
    pie.add_annotation(
        text=f"<b>{count:,}</b><br><span style='font-size:11px'>calls</span>",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 22, "color": NAVY},
    )

    # Monthly sentiment trend
    monthly = (
        data.assign(Month=data["Date"].dt.to_period("M").astype(str))
        .groupby(["Month", "Sentiment"])
        .size()
        .reset_index(name="Calls")
    )
    trend = px.line(
        monthly,
        x="Month",
        y="Calls",
        color="Sentiment",
        markers=True,
        color_discrete_map=COLORS,
        category_orders={"Sentiment": ["Positive", "Neutral", "Negative"]},
    )
    trend.update_traces(line={"width": 3}, marker={"size": 7})

    # Resolution outcomes
    resolution_data = (
        data.groupby(["Resolution", "Sentiment"])
        .size()
        .reset_index(name="Calls")
    )
    resolution = px.bar(
        resolution_data,
        x="Resolution",
        y="Calls",
        color="Sentiment",
        barmode="stack",
        color_discrete_map=COLORS,
        category_orders={
            "Resolution": ["Resolved", "Unresolved"],
            "Sentiment": ["Positive", "Neutral", "Negative"],
        },
    )

    # Call duration distribution
    duration = px.box(
        data,
        x="Sentiment",
        y="AvgTalkDuration",
        color="Sentiment",
        color_discrete_map=COLORS,
        points="outliers",
        category_orders={"Sentiment": ["Positive", "Neutral", "Negative"]},
    )
    duration.update_yaxes(title="Duration (seconds)")

    # Negative calls by resolution
    negative_counts = (
        data[data["Sentiment"] == "Negative"]
        .groupby("Resolution")
        .size()
        .reindex(["Resolved", "Unresolved"], fill_value=0)
        .rename("Calls")
        .reset_index()
    )
    negative = px.bar(
        negative_counts,
        x="Resolution",
        y="Calls",
        color="Resolution",
        color_discrete_map={"Resolved": "#16a34a", "Unresolved": "#ef4444"},
        category_orders={"Resolution": ["Resolved", "Unresolved"]},
    )
    negative.update_layout(showlegend=False)

    # Satisfaction rating distribution
    rating_data = (
        data.groupby("Satisfaction rating")
        .size()
        .reset_index(name="Calls")
        .sort_values("Satisfaction rating")
    )
    rating_fig = px.bar(
        rating_data,
        x="Satisfaction rating",
        y="Calls",
        color="Satisfaction rating",
        color_continuous_scale=["#ef4444", "#f59e0b", "#16a34a"],
    )
    rating_fig.update_layout(coloraxis_showscale=False)
    rating_fig.update_xaxes(title="Satisfaction rating", dtick=1)

    # Average duration by resolution
    duration_resolution_data = (
        data.groupby("Resolution")["AvgTalkDuration"]
        .mean()
        .reindex(["Resolved", "Unresolved"])
        .dropna()
        .reset_index()
    )
    duration_resolution_fig = px.bar(
        duration_resolution_data,
        x="Resolution",
        y="AvgTalkDuration",
        color="Resolution",
        color_discrete_map={"Resolved": "#4f6bed", "Unresolved": "#8b5cf6"},
        category_orders={"Resolution": ["Resolved", "Unresolved"]},
    )
    duration_resolution_fig.update_layout(showlegend=False)
    duration_resolution_fig.update_yaxes(title="Average duration (seconds)")

    figures = [
        style_figure(figure)
        for figure in [
            pie,
            trend,
            resolution,
            duration,
            negative,
            rating_fig,
            duration_resolution_fig,
        ]
    ]

    # Prepare table data
    table = data[
        [
            "Date",
            "Satisfaction rating",
            "Sentiment",
            "Resolution",
            "AvgTalkDuration",
        ]
    ].copy()

    table["Date"] = table["Date"].dt.strftime("%Y-%m-%d")
    table["Satisfaction rating"] = table["Satisfaction rating"].round(1)
    table["AvgTalkDuration"] = table["AvgTalkDuration"].round(1)

    if search and str(search).strip():
        term = str(search).strip().lower()
        mask = table.astype(str).apply(
            lambda column: column.str.lower().str.contains(
                term, regex=False, na=False
            )
        ).any(axis=1)
        table = table[mask]

    table = table.sort_values("Date", ascending=False).to_dict("records")

    return (
        f"{count:,}",
        "Calls in current selection",
        f"{positive_rate:.1%}",
        "Rated 4 or 5",
        f"{resolution_rate:.1%}",
        "Marked as resolved",
        seconds_label(average_duration),
        "Mean talk duration",
        f"{count:,} calls match these filters",
        make_insights(data),
        *figures,
        table,
        f"Showing {len(table):,} rows after search. Use column headers to sort or filter.",
    )


# ============================================================
# EXPORT CSV
# ============================================================

@app.callback(
    Output("download-csv", "data"),
    Input("export-btn", "n_clicks"),
    State("sentiment-filter", "value"),
    State("date-picker", "start_date"),
    State("date-picker", "end_date"),
    State("resolved-filter", "value"),
    State("duration-slider", "value"),
    prevent_initial_call=True,
)
def export_csv(_, sentiments, start_date, end_date, resolved, duration_range):
    data = filter_data(sentiments, start_date, end_date, resolved, duration_range)

    return dcc.send_data_frame(
        data.to_csv,
        "filtered_call_center_data.csv",
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# SAVED FILTER VIEWS
# ============================================================

def fetch_saved_views():
    with closing(connect_db()) as connection:
        return connection.execute("""
            SELECT id, name, sentiment, start_date, end_date, resolved,
                   duration_min, duration_max, created_at
            FROM past_filters
            ORDER BY id DESC
            LIMIT 100
        """).fetchall()


def render_saved_views(rows):
    if not rows:
        return html.P(
            "No saved views yet. Save a filter setup to see it here.",
            className="text-secondary small mt-2",
        )

    cards = []

    for row in rows:
        row = dict(row)

        sentiments = row["sentiment"] or "All"
        resolution = {
            "Y": "Resolved",
            "N": "Unresolved",
            "": "All",
        }.get(row["resolved"], "All")

        low = row["duration_min"]
        high = row["duration_max"]
        duration_text = (
            f"{low:g}–{high:g} sec"
            if low is not None and high is not None
            else "All durations"
        )

        description = (
            f"Sentiment: {sentiments} · "
            f"Dates: {row['start_date'] or 'Any'} to {row['end_date'] or 'Any'} · "
            f"Resolution: {resolution} · Duration: {duration_text}"
        )

        cards.append(
            dbc.Card(
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(
                                row["name"] or f"Saved view #{row['id']}",
                                className="fw-bold",
                            ),
                            html.Div(description, className="small text-secondary mt-1"),
                            html.Div(
                                f"Saved: {row['created_at']}",
                                className="small text-secondary mt-1",
                            ),
                        ], xs=12, md=8),
                        dbc.Col([
                            dbc.ButtonGroup([
                                dbc.Button(
                                    "Apply",
                                    id={"type": "apply-view", "index": row["id"]},
                                    color="primary",
                                    size="sm",
                                    n_clicks=0,
                                ),
                                dbc.Button(
                                    "Delete",
                                    id={"type": "delete-view", "index": row["id"]},
                                    color="danger",
                                    outline=True,
                                    size="sm",
                                    n_clicks=0,
                                ),
                            ], className="mt-2 mt-md-0"),
                        ], xs=12, md=4, className="d-flex justify-content-md-end align-items-center"),
                    ], className="align-items-center"),
                ]),
                className="mb-2",
            )
        )

    return html.Div(cards)


@app.callback(
    Output("saved-feedback", "children"),
    Output("past-data-output", "children"),
    Output("sentiment-filter", "value", allow_duplicate=True),
    Output("date-picker", "start_date", allow_duplicate=True),
    Output("date-picker", "end_date", allow_duplicate=True),
    Output("resolved-filter", "value", allow_duplicate=True),
    Output("duration-slider", "value", allow_duplicate=True),
    Input("save-view-btn", "n_clicks"),
    Input("view-past-btn", "n_clicks"),
    Input("delete-all-btn", "n_clicks"),
    Input({"type": "apply-view", "index": ALL}, "n_clicks"),
    Input({"type": "delete-view", "index": ALL}, "n_clicks"),
    State("view-name", "value"),
    State("sentiment-filter", "value"),
    State("date-picker", "start_date"),
    State("date-picker", "end_date"),
    State("resolved-filter", "value"),
    State("duration-slider", "value"),
    prevent_initial_call=True,
)
def manage_saved_views(
    _save_clicks,
    _refresh_clicks,
    _delete_all_clicks,
    _apply_clicks,
    _delete_clicks,
    view_name,
    sentiments,
    start_date,
    end_date,
    resolved,
    duration,
):
    action = ctx.triggered_id
    feedback = no_update
    apply_values = [no_update] * 5

    try:
        with closing(connect_db()) as connection:
            if action == "save-view-btn":
                name = (view_name or "").strip() or (
                    "Saved view " + datetime.now().strftime("%Y-%m-%d %H:%M")
                )

                connection.execute(
                    """
                    INSERT INTO past_filters
                    (name, sentiment, start_date, end_date, resolved,
                     duration_min, duration_max)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        name,
                        ",".join(sentiments or []),
                        str(start_date or ""),
                        str(end_date or ""),
                        resolved or "",
                        float(duration[0]) if duration else None,
                        float(duration[1]) if duration else None,
                    ),
                )
                connection.commit()
                feedback = dbc.Alert(
                    "Current filter view saved successfully.",
                    color="success",
                    dismissable=True,
                )

            elif action == "delete-all-btn":
                connection.execute("DELETE FROM past_filters")
                connection.commit()
                feedback = dbc.Alert(
                    "All saved views have been deleted.",
                    color="info",
                    dismissable=True,
                )

            elif isinstance(action, dict) and action.get("type") == "delete-view":
                connection.execute(
                    "DELETE FROM past_filters WHERE id = ?",
                    (int(action["index"]),),
                )
                connection.commit()
                feedback = dbc.Alert(
                    "Saved view deleted.",
                    color="info",
                    dismissable=True,
                )

            elif isinstance(action, dict) and action.get("type") == "apply-view":
                row = connection.execute(
                    "SELECT * FROM past_filters WHERE id = ?",
                    (int(action["index"]),),
                ).fetchone()

                if row:
                    saved_sentiments = [
                        item for item in (row["sentiment"] or "").split(",") if item
                    ]

                    saved_start = row["start_date"] or None
                    saved_end = row["end_date"] or None
                    saved_resolved = row["resolved"] or None

                    low = row["duration_min"]
                    high = row["duration_max"]

                    if low is None:
                        low = MIN_DURATION
                    if high is None:
                        high = MAX_DURATION

                    low = max(SLIDER_MIN, min(float(low), SLIDER_MAX))
                    high = max(SLIDER_MIN, min(float(high), SLIDER_MAX))

                    if low > high:
                        low, high = high, low

                    apply_values = [
                        saved_sentiments,
                        saved_start,
                        saved_end,
                        saved_resolved,
                        [low, high],
                    ]

                    feedback = dbc.Alert(
                        "Saved filter view applied.",
                        color="success",
                        dismissable=True,
                    )

            rows = connection.execute("""
                SELECT id, name, sentiment, start_date, end_date, resolved,
                       duration_min, duration_max, created_at
                FROM past_filters
                ORDER BY id DESC
                LIMIT 100
            """).fetchall()

        return feedback, render_saved_views(rows), *apply_values

    except sqlite3.Error as exc:
        return (
            dbc.Alert(
                f"Database error: {exc}",
                color="danger",
                dismissable=True,
            ),
            no_update,
            *([no_update] * 5),
        )


# ============================================================
# INITIAL SAVED VIEWS
# ============================================================

app.layout.children[0].children[1].children.append(
    html.Div(id="initial-load-placeholder", style={"display": "none"})
)

# The saved views are populated on page load and after every saved-view action.
@app.callback(
    Output("past-data-output", "children", allow_duplicate=True),
    Input("view-past-btn", "n_clicks"),
    prevent_initial_call="initial_duplicate",
)
def refresh_saved_views(_):
    return render_saved_views(fetch_saved_views())


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=PORT,
        debug=False,
    )
