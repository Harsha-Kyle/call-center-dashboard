# ============================================================
# CALL CENTER INTELLIGENCE
# ============================================================

import os
import re
import json
import sqlite3
import threading
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import dash
from dash import (
    dcc,
    html,
    dash_table,
    Input,
    Output,
    State,
    ALL,
    ctx,
    no_update,
)
import dash_bootstrap_components as dbc


# ============================================================
# CONFIGURATION
# ============================================================

APP_TITLE = "Call Center Intelligence"

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

DATA_FILE = Path(
    os.getenv("DATA_FILE", str(BASE_DIR / "CallCenterDataset.xlsx"))
)
SHEET_NAME = os.getenv("SHEET_NAME", "Sheet1")
DATABASE_FILE = Path(
    os.getenv("DATABASE_FILE", str(BASE_DIR / "call_center_dashboard.db"))
)

PORT = int(os.getenv("PORT", "8050"))
DEBUG = os.getenv("DASH_DEBUG", "false").lower() == "true"

REQUIRED_COLUMNS = {
    "Satisfaction rating",
    "Date",
    "Resolved",
    "AvgTalkDuration",
}

C = {
    "nav": "#202A3B",
    "nav_text": "#C7D0DD",
    "page": "#F4F6F9",
    "card": "#FFFFFF",
    "text": "#202B3C",
    "muted": "#748094",
    "border": "#E3E8EF",
    "blue": "#3569C8",
    "green": "#258765",
    "orange": "#C78326",
    "red": "#C95454",
    "purple": "#7865B7",
}

SENTIMENT_COLORS = {
    "Positive": C["green"],
    "Neutral": C["orange"],
    "Negative": C["red"],
}

_DB_LOCK = threading.Lock()
DATABASE_FILE.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATABASE
# ============================================================

def db_connection():
    connection = sqlite3.connect(str(DATABASE_FILE), timeout=20)
    connection.row_factory = sqlite3.Row
    return connection


def init_database():
    with _DB_LOCK:
        with db_connection() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS saved_views (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    filters_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            con.commit()


init_database()


def get_saved_views():
    with _DB_LOCK:
        with db_connection() as con:
            rows = con.execute(
                """
                SELECT id, name, filters_json, created_at
                FROM saved_views
                ORDER BY id DESC
                """
            ).fetchall()
    return [dict(row) for row in rows]


def make_saved_view_options(views):
    return [
        {
            "label": f"{row['name']}  ·  {row['created_at']}",
            "value": row["id"],
        }
        for row in views
    ]


# ============================================================
# DATA CLEANING
# ============================================================

def parse_duration(value):
    if pd.isna(value):
        return None

    if isinstance(value, pd.Timedelta):
        return max(value.total_seconds(), 0)

    if hasattr(value, "total_seconds") and not isinstance(value, str):
        try:
            return max(float(value.total_seconds()), 0)
        except Exception:
            pass

    if isinstance(value, (int, float)):
        number = float(value)
        if number < 0:
            return None
        # Excel time values are often fractions of one day.
        if 0 < number < 1:
            return number * 86400
        return number

    value = str(value).strip()

    if not value or value.lower() in {"nan", "none", "nat"}:
        return None

    try:
        number = float(value)
        if number < 0:
            return None
        if 0 < number < 1:
            return number * 86400
        return number
    except ValueError:
        pass

    if re.fullmatch(r"\d{1,3}:\d{1,2}(:\d{1,2})?", value):
        parts = [int(x) for x in value.split(":")]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        return parts[0] * 60 + parts[1]

    text = value.lower()
    h = re.search(r"(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour|hours)\b", text)
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|min|mins|minute|minutes)\b", text)
    s = re.search(r"(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)\b", text)

    if h or m or s:
        total = 0
        if h:
            total += float(h.group(1)) * 3600
        if m:
            total += float(m.group(1)) * 60
        if s:
            total += float(s.group(1))
        return total

    try:
        return max(pd.to_timedelta(value).total_seconds(), 0)
    except Exception:
        return None


def normalise_resolution(value):
    if pd.isna(value):
        return "Unknown"

    if isinstance(value, bool):
        return "Resolved" if value else "Unresolved"

    text = str(value).strip().lower()

    if text in {
        "1", "true", "yes", "y", "resolved", "complete",
        "completed", "closed", "success", "successful",
    }:
        return "Resolved"

    if text in {
        "0", "false", "no", "n", "unresolved", "not resolved",
        "open", "pending", "incomplete", "failed",
    }:
        return "Unresolved"

    return str(value).strip().title() or "Unknown"


def format_duration(seconds):
    if seconds is None or pd.isna(seconds):
        return "N/A"

    seconds = max(int(round(float(seconds))), 0)
    hours, remaining = divmod(seconds, 3600)
    minutes, seconds = divmod(remaining, 60)

    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Place CallCenterDataset.xlsx beside app.py, "
            "or set the DATA_FILE environment variable."
        )

    df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)
    df.columns = [str(col).strip() for col in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "Missing required Excel columns: "
            + ", ".join(sorted(missing))
            + "\nFound columns: "
            + ", ".join(df.columns)
        )

    df = df.copy()
    df["Satisfaction rating"] = pd.to_numeric(
        df["Satisfaction rating"], errors="coerce"
    )
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Date"] = df["Date"].dt.tz_localize(None)
    df["DurationSeconds"] = df["AvgTalkDuration"].apply(parse_duration)
    df["ResolvedStatus"] = df["Resolved"].apply(normalise_resolution)

    df = df.dropna(
        subset=["Satisfaction rating", "Date", "DurationSeconds"]
    ).copy()
    df = df[df["DurationSeconds"] >= 0].copy()

    if df.empty:
        raise ValueError(
            "No valid rows remain after cleaning. Check your Date, "
            "Satisfaction rating, and AvgTalkDuration columns."
        )

    df["Sentiment"] = df["Satisfaction rating"].apply(
        lambda rating: (
            "Positive" if rating >= 4
            else "Neutral" if rating >= 3
            else "Negative"
        )
    )
    df["DateOnly"] = df["Date"].dt.date
    df["CallDate"] = df["Date"].dt.strftime("%Y-%m-%d")
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    df["DurationMinutes"] = df["DurationSeconds"] / 60
    df["DurationLabel"] = df["DurationSeconds"].apply(format_duration)
    df = df.sort_values("Date", ascending=False).reset_index(drop=True)
    df["CallID"] = [f"CALL-{i + 1:05d}" for i in range(len(df))]

    return df


DF = load_data()

MIN_DATE = DF["DateOnly"].min()
MAX_DATE = DF["DateOnly"].max()
MIN_DURATION = float(DF["DurationSeconds"].min())
MAX_DURATION = float(DF["DurationSeconds"].max())

SLIDER_MIN = min(MIN_DURATION, 0)
SLIDER_MAX = max(MAX_DURATION, SLIDER_MIN + 1)

if SLIDER_MIN == SLIDER_MAX:
    SLIDER_MAX = SLIDER_MIN + 1

DEFAULT_DURATION = [SLIDER_MIN, SLIDER_MAX]


# ============================================================
# FILTERS AND CHART HELPERS
# ============================================================

def filtered_data(start_date, end_date, sentiment, resolution, duration_range):
    df = DF.copy()

    if start_date:
        df = df[df["DateOnly"] >= pd.to_datetime(start_date).date()]
    if end_date:
        df = df[df["DateOnly"] <= pd.to_datetime(end_date).date()]
    if sentiment and sentiment != "All":
        df = df[df["Sentiment"] == sentiment]
    if resolution and resolution != "All":
        df = df[df["ResolvedStatus"] == resolution]

    if duration_range and len(duration_range) == 2:
        low, high = sorted(map(float, duration_range))
        df = df[df["DurationSeconds"].between(low, high, inclusive="both")]

    return df


def empty_figure(message="No data matches these filters"):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5, y=0.5,
        xref="paper", yref="paper",
        showarrow=False,
        font={"size": 13, "color": C["muted"]},
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=300,
        margin=dict(l=20, r=20, t=15, b=15),
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig


def style_fig(fig, height=300):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=12, color=C["text"]),
        height=height,
        margin=dict(l=18, r=18, t=22, b=25),
        hoverlabel=dict(bgcolor="white", font_size=12),
        dragmode=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=10),
        ),
    )
    fig.update_xaxes(showgrid=False, linecolor=C["border"])
    fig.update_yaxes(gridcolor="#EDF0F5", zeroline=False)
    return fig


def kpi_card(label, value_id, note_id, icon, accent):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.Div(
                            html.I(className=f"bi {icon}"),
                            className="kpi-icon",
                            style={"color": accent, "background": f"{accent}16"},
                        ),
                        html.Div(
                            [
                                html.Div(label, className="kpi-label"),
                                html.Div(id=value_id, className="kpi-value"),
                            ],
                            className="flex-grow-1",
                        ),
                    ],
                    className="d-flex align-items-center gap-3",
                ),
                html.Div(id=note_id, className="kpi-note"),
            ]
        ),
        className="dashboard-card kpi-card h-100",
    )



def chart_card(title, subtitle, graph_id):
    return dbc.Card(
        dbc.CardBody(
            [
                html.H6(title, className="chart-title"),
                html.Div(subtitle, className="chart-subtitle"),
                dcc.Graph(
                    id=graph_id,
                    config={
                        "displayModeBar": False,
                        "responsive": True,
                        "scrollZoom": False,
                        "staticPlot": True,
                    },
                    style={"width": "100%"},
                ),
            ]
        ),
        className="dashboard-card chart-card h-100",
    )


def insights_for(df):
    if df.empty:
        return [
            html.Div(
                "Change the filters to generate insights.",
                className="empty-insight",
            )
        ]

    total = len(df)
    positive = (df["Sentiment"] == "Positive").sum()
    negative = (df["Sentiment"] == "Negative").sum()
    resolved = (df["ResolvedStatus"] == "Resolved").sum()

    items = [
        (
            "Positive customer experience",
            f"{positive / total * 100:.1f}% of selected calls have positive sentiment.",
            "bi-emoji-smile",
            C["green"],
        ),
        (
            "Resolution performance",
            f"{resolved / total * 100:.1f}% of selected calls are marked resolved.",
            "bi-check2-circle",
            C["blue"],
        ),
        (
            "Customer satisfaction",
            f"Average satisfaction is {df['Satisfaction rating'].mean():.2f} out of 5.",
            "bi-star",
            C["orange"],
        ),
        (
            "Average call duration",
            f"The average call lasts {format_duration(df['DurationSeconds'].mean())}.",
            "bi-clock",
            C["purple"],
        ),
    ]

    negative_rate = negative / total * 100
    if negative_rate >= 25:
        items.append(
            (
                "Review negative calls",
                f"{negative_rate:.1f}% of selected calls have negative sentiment.",
                "bi-exclamation-circle",
                C["red"],
            )
        )

    return [
        html.Div(
            [
                html.Div(
                    html.I(className=f"bi {icon}"),
                    className="insight-icon",
                    style={"color": color, "background": f"{color}16"},
                ),
                html.Div(
                    [
                        html.Div(title, className="insight-title"),
                        html.Div(text, className="insight-text"),
                    ]
                ),
            ],
            className="insight-item",
        )
        for title, text, icon, color in items
    ]


def sentiment_figure(df):
    if df.empty:
        return empty_figure()

    counts = df["Sentiment"].value_counts().reindex(
        ["Positive", "Neutral", "Negative"], fill_value=0
    )

    fig = go.Figure(
        go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=0.67,
            sort=False,
            marker=dict(
                colors=[SENTIMENT_COLORS[x] for x in counts.index],
                line=dict(color="white", width=3),
            ),
            textinfo="percent",
            hovertemplate="%{label}: %{value} calls (%{percent})<extra></extra>",
        )
    )
    fig.update_layout(
        annotations=[
            dict(
                text=f"<b>{len(df):,}</b><br><span style='font-size:10px'>Calls</span>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color=C["text"]),
            )
        ],
        showlegend=True,
        legend=dict(orientation="h", y=-0.12, x=0.02),
    )
    return style_fig(fig)


def monthly_figure(df):
    if df.empty:
        return empty_figure()

    monthly = df.groupby(["Month", "Sentiment"]).size().reset_index(name="Calls")
    fig = px.line(
        monthly,
        x="Month",
        y="Calls",
        color="Sentiment",
        markers=True,
        color_discrete_map=SENTIMENT_COLORS,
        category_orders={"Sentiment": ["Positive", "Neutral", "Negative"]},
    )
    fig.update_traces(line_width=2.5, marker_size=6)
    fig.update_layout(xaxis_title="Month", yaxis_title="Calls", hovermode="x unified")
    return style_fig(fig)


def resolution_figure(df):
    if df.empty:
        return empty_figure()

    grouped = df.groupby(["ResolvedStatus", "Sentiment"]).size().reset_index(name="Calls")
    fig = px.bar(
        grouped,
        x="ResolvedStatus",
        y="Calls",
        color="Sentiment",
        barmode="stack",
        color_discrete_map=SENTIMENT_COLORS,
    )
    fig.update_layout(xaxis_title="Resolution status", yaxis_title="Calls")
    return style_fig(fig)



def rating_figure(df):
    if df.empty:
        return empty_figure()

    grouped = (
        df["Satisfaction rating"]
        .round()
        .value_counts()
        .sort_index()
        .rename_axis("Rating")
        .reset_index(name="Calls")
    )

    # Use discrete colors rather than mapping a numeric Rating
    # column to a continuous color scale.
    rating_colors = {
        1: "#C95454",
        2: "#D97757",
        3: "#C78326",
        4: "#72A66A",
        5: "#258765",
    }

    grouped["RatingLabel"] = grouped["Rating"].astype(str)
    grouped["BarColor"] = grouped["Rating"].map(
        lambda rating: rating_colors.get(int(rating), "#3569C8")
    )

    fig = px.bar(
        grouped,
        x="RatingLabel",
        y="Calls",
        text="Calls",
        custom_data=["Rating"],
    )

    fig.update_traces(
        marker_color=grouped["BarColor"].tolist(),
        textposition="outside",
        hovertemplate="Rating: %{customdata[0]}<br>Calls: %{y}<extra></extra>",
    )

    fig.update_layout(
        showlegend=False,
        xaxis_title="Satisfaction rating",
        yaxis_title="Calls",
        xaxis=dict(
            type="category",
            categoryorder="array",
            categoryarray=["1", "2", "3", "4", "5"],
        ),
        yaxis=dict(rangemode="tozero"),
        margin=dict(l=18, r=18, t=30, b=25),
    )

    return style_fig(fig)


def duration_figure(df):
    if df.empty:
        return empty_figure()

    fig = px.box(
        df,
        x="Sentiment",
        y="DurationMinutes",
        color="Sentiment",
        points="outliers",
        color_discrete_map=SENTIMENT_COLORS,
        category_orders={"Sentiment": ["Positive", "Neutral", "Negative"]},
    )
    fig.update_layout(
        xaxis_title="Sentiment",
        yaxis_title="Duration (minutes)",
        showlegend=False,
    )
    return style_fig(fig)


def negative_figure(df):
    negative = df[df["Sentiment"] == "Negative"]
    if negative.empty:
        return empty_figure("No negative calls in this selection")

    grouped = (
        negative["ResolvedStatus"].value_counts()
        .rename_axis("Resolution status").reset_index(name="Calls")
    )
    fig = px.bar(
        grouped,
        x="Resolution status",
        y="Calls",
        text="Calls",
        color="Resolution status",
        color_discrete_sequence=[C["red"], C["orange"], C["purple"]],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        showlegend=False,
        xaxis_title="Resolution status",
        yaxis_title="Negative calls",
    )
    return style_fig(fig)


def duration_by_resolution_figure(df):
    if df.empty:
        return empty_figure()

    grouped = (
        df.groupby("ResolvedStatus")["DurationMinutes"]
        .mean().reset_index(name="AverageMinutes")
    )
    fig = px.bar(
        grouped,
        x="ResolvedStatus",
        y="AverageMinutes",
        text="AverageMinutes",
        color="ResolvedStatus",
        color_discrete_sequence=[C["blue"], C["purple"], C["orange"]],
    )
    fig.update_traces(texttemplate="%{text:.1f} min", textposition="outside")
    fig.update_layout(
        showlegend=False,
        xaxis_title="Resolution status",
        yaxis_title="Average duration (minutes)",
    )
    return style_fig(fig)


# ============================================================
# UI STYLE
# ============================================================

CSS = r"""
@import url('https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css');

:root {
  --page: #F4F6F9;
  --card: #FFFFFF;
  --text: #202B3C;
  --muted: #748094;
  --border: #E3E8EF;
  --nav: #202A3B;
  --blue: #3569C8;
}

* { box-sizing: border-box; }

html { scroll-behavior: smooth; scroll-padding-top: 20px; }

body {
  margin: 0;
  background: var(--page);
  color: var(--text);
  font-family: "Segoe UI", Arial, sans-serif;
}

.app-shell { min-height: 100vh; }

.sidebar {
  width: 232px;
  min-width: 232px;
  min-height: 100vh;
  background: var(--nav);
  color: white;
  padding: 24px 15px;
  align-self: stretch;
}

.brand {
  display: flex;
  align-items: center;
  gap: 11px;
  padding: 3px 8px 28px;
}

.brand-icon {
  height: 39px;
  width: 39px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #34445C;
  color: #EAF0F8;
  border-radius: 10px;
  font-size: 19px;
}

.brand-name { font-size: 15px; font-weight: 750; letter-spacing: -.2px; }
.brand-subtitle { font-size: 9px; color: #AAB7C9; letter-spacing: 1.4px; margin-top: 3px; }
.nav-heading { font-size: 10px; color: #94A3B8; letter-spacing: 1.4px; padding: 0 12px 8px; }

.sidebar-nav { display: flex; flex-direction: column; gap: 4px; }

.sidebar-link {
  display: block;
  padding: 11px 12px;
  border-radius: 8px;
  color: #C7D0DD;
  font-size: 12px;
  font-weight: 600;
  text-decoration: none;
}

.sidebar-link i { margin-right: 10px; font-size: 14px; }

.sidebar-link:hover, .sidebar-link:focus {
  background: #34445C;
  color: white;
  text-decoration: none;
}

.sidebar-note {
  margin-top: 35px;
  padding: 13px;
  border: 1px solid #3A465A;
  border-radius: 10px;
  color: #B7C3D4;
  font-size: 11px;
  line-height: 1.6;
}

.main-content { min-width: 0; flex: 1; padding: 27px 30px 36px; }

.topbar {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 15px;
  margin-bottom: 23px;
}

.page-title { font-size: 27px; font-weight: 750; letter-spacing: -.7px; margin: 0; }
.page-subtitle { font-size: 12px; color: var(--muted); margin: 7px 0 0; }

.dataset-badge {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 8px 11px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: white;
  color: #506078;
  font-size: 10px;
  white-space: nowrap;
}

.dashboard-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  box-shadow: 0 2px 7px rgba(28, 43, 67, .025);
}

.kpi-card { min-height: 132px; }
.kpi-card .card-body { padding: 17px; }

.kpi-icon {
  height: 40px;
  width: 40px;
  min-width: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 10px;
  font-size: 17px;
}

.kpi-label { color: var(--muted); font-size: 11px; font-weight: 650; }
.kpi-value { font-size: 25px; font-weight: 750; letter-spacing: -.5px; margin-top: 4px; overflow-wrap: anywhere; }
.kpi-note { font-size: 10px; color: var(--muted); margin-top: 13px; }

.section-header { margin: 25px 0 13px; scroll-margin-top: 20px; }
.section-title { font-size: 15px; font-weight: 750; margin: 0; }
.section-subtitle { color: var(--muted); font-size: 11px; margin: 5px 0 0; }

.filter-card .card-body { padding: 18px; }
.filter-label { color: #536177; font-size: 10px; font-weight: 750; letter-spacing: .45px; margin-bottom: 7px; }

.form-control, .form-select {
  min-height: 38px;
  border-color: #DDE3EC;
  border-radius: 7px;
  font-size: 12px;
}

.form-control:focus, .form-select:focus {
  border-color: #7D9DDD;
  box-shadow: 0 0 0 2px #3569C81A;
}

.btn { border-radius: 7px; font-size: 11px; font-weight: 650; padding: 8px 12px; }
.btn-primary { background: var(--blue); border-color: var(--blue); }
.btn-primary:hover { background: #2C58A7; border-color: #2C58A7; }

.filter-count { font-size: 11px; color: var(--muted); padding-top: 8px; }

.chart-card .card-body { padding: 17px 17px 8px; }
.chart-title { font-size: 12px; font-weight: 750; margin: 0; }
.chart-subtitle { font-size: 10px; color: var(--muted); margin-top: 5px; }

.insights-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
.insight-item { display: flex; gap: 10px; padding: 13px; border: 1px solid var(--border); border-radius: 10px; background: white; }
.insight-icon { height: 32px; width: 32px; min-width: 32px; display: flex; align-items: center; justify-content: center; border-radius: 8px; }
.insight-title { font-size: 11px; font-weight: 750; margin: 1px 0 5px; }
.insight-text { color: var(--muted); font-size: 10px; line-height: 1.55; }
.empty-insight { font-size: 11px; color: var(--muted); padding: 10px; }

.table-card .card-body { padding: 17px; }
.table-summary, .saved-feedback { color: var(--muted); font-size: 11px; margin-top: 10px; }

.dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th {
  background: #F5F7FA !important;
  color: #536177 !important;
  font-size: 10px !important;
  font-weight: 750 !important;
  border: none !important;
}
.dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner td {
  font-size: 11px !important;
  color: #354258 !important;
  border: none !important;
  border-bottom: 1px solid #EDF0F5 !important;
}

.calendar-presets { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.calendar-presets .btn { padding: 6px 10px; font-size: 10px; }

.DateRangePicker, .DateRangePickerInput { border-radius: 7px; }
.DateInput { width: 105px; }
.DateInput_input { font-size: 11px; padding: 8px 6px; border-radius: 7px; }
.DateRangePickerInput { border: 1px solid #DDE3EC; }
.DateRangePickerInput_arrow { color: #8A95A6; }
.CalendarDay__selected, .CalendarDay__selected:active, .CalendarDay__selected:hover {
  background: #3569C8; border-color: #3569C8; color: white;
}
.CalendarDay__selected_span { background: #E1EAF9; border-color: #D2DFF4; color: #253D66; }
.CalendarDay__hovered_span, .CalendarDay__hovered_span:hover { background: #EDF2FB; border-color: #DDE6F6; color: #253D66; }
.DayPicker, .DayPicker__withBorder { border-radius: 10px; box-shadow: 0 8px 24px #1C2B431A; }

.footer-note { text-align: center; color: #8C97A8; font-size: 10px; margin-top: 27px; }

@media (max-width: 1000px) {
  .sidebar { width: 195px; min-width: 195px; }
  .main-content { padding: 22px 18px; }
  .kpi-value { font-size: 22px; }
}

@media (max-width: 760px) {
  .app-shell { display: block !important; }
  .sidebar { width: 100%; min-width: 100%; min-height: auto; padding: 13px; }
  .brand { padding: 2px 6px 12px; }
  .sidebar-nav { flex-direction: row; flex-wrap: wrap; }
  .sidebar-link { padding: 8px 9px; font-size: 10px; }
  .sidebar-note, .nav-heading { display: none; }
  .main-content { padding: 18px 12px 26px; }
  .page-title { font-size: 22px; }
  .dataset-badge { font-size: 0; }
  .insights-grid { grid-template-columns: 1fr; }
}
"""


# ============================================================
# APP SETUP
# ============================================================

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css",
    ],
    title=APP_TITLE,
)
server = app.server

app.index_string = f"""
<!DOCTYPE html>
<html>
<head>
    {{%metas%}}
    <title>{{%title%}}</title>
    {{%favicon%}}
    {{%css%}}
    <style>{CSS}</style>
</head>
<body>
    {{%app_entry%}}
    <footer>
        {{%config%}}
        {{%scripts%}}
        {{%renderer%}}
    </footer>
</body>
</html>
"""


# ============================================================
# SIDEBAR AND FILTER CONTROLS
# ============================================================

sidebar = html.Aside(
    [
        html.Div(
            [
                html.Div(
                    html.I(className="bi bi-headset"),
                    className="brand-icon",
                ),
                html.Div(
                    [
                        html.Div("Call Center", className="brand-name"),
                        html.Div("PERFORMANCE ANALYTICS", className="brand-subtitle"),
                    ]
                ),
            ],
            className="brand",
        ),
        html.Div("WORKSPACE", className="nav-heading"),
        html.Nav(
            [
                html.A(
                    [html.I(className="bi bi-grid-1x2"), "Overview"],
                    href="#overview",
                    className="sidebar-link",
                ),
                html.A(
                    [html.I(className="bi bi-pie-chart"), "Sentiment"],
                    href="#sentiment",
                    className="sidebar-link",
                ),
                html.A(
                    [html.I(className="bi bi-table"), "Call explorer"],
                    href="#call-explorer",
                    className="sidebar-link",
                ),
                html.A(
                    [html.I(className="bi bi-bookmark"), "Saved views"],
                    href="#saved-views",
                    className="sidebar-link",
                ),
            ],
            className="sidebar-nav",
        ),
        html.Div(
            [
                html.Div(
                    [html.I(className="bi bi-info-circle me-2"), "About this dashboard"],
                    style={"fontWeight": "700", "marginBottom": "5px"},
                ),
                "Use the filters to explore customer satisfaction, "
                "call outcomes and handling duration.",
            ],
            className="sidebar-note",
        ),
    ],
    className="sidebar",
)

date_picker = dcc.DatePickerRange(
    id="date-filter",
    min_date_allowed=MIN_DATE,
    max_date_allowed=MAX_DATE,
    start_date=MIN_DATE,
    end_date=MAX_DATE,
    display_format="DD MMM YYYY",
    minimum_nights=0,
    number_of_months_shown=2,
    updatemode="singledate",
    clearable=False,
    stay_open_on_select=False,
)

sentiment_dropdown = dcc.Dropdown(
    id="sentiment-filter",
    options=[
        {"label": "All sentiments", "value": "All"},
        {"label": "Positive", "value": "Positive"},
        {"label": "Neutral", "value": "Neutral"},
        {"label": "Negative", "value": "Negative"},
    ],
    value="All",
    clearable=False,
)

resolution_values = sorted(DF["ResolvedStatus"].dropna().unique().tolist())
resolution_dropdown = dcc.Dropdown(
    id="resolution-filter",
    options=[
        {"label": "All outcomes", "value": "All"},
        *[{"label": value, "value": value} for value in resolution_values],
    ],
    value="All",
    clearable=False,
)

duration_slider = dcc.RangeSlider(
    id="duration-filter",
    min=SLIDER_MIN,
    max=SLIDER_MAX,
    step=max((SLIDER_MAX - SLIDER_MIN) / 100, 0.1),
    value=DEFAULT_DURATION,
    marks={
        int(SLIDER_MIN): format_duration(SLIDER_MIN),
        int(SLIDER_MAX): format_duration(SLIDER_MAX),
    },
    tooltip={"placement": "bottom", "always_visible": False},
    allowCross=False,
)


# ============================================================
# FILTER PANEL
# ============================================================

filters_panel = dbc.Card(
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div("DATE RANGE", className="filter-label"),
                            date_picker,
                            html.Div(
                                [
                                    dbc.Button("Last 7 days", id="preset-7", color="light", size="sm", n_clicks=0),
                                    dbc.Button("Last 30 days", id="preset-30", color="light", size="sm", n_clicks=0),
                                    dbc.Button("All dates", id="preset-all", color="light", size="sm", n_clicks=0),
                                ],
                                className="calendar-presets",
                            ),
                        ],
                        xs=12, xl=4,
                    ),
                    dbc.Col(
                        [
                            html.Div("SENTIMENT", className="filter-label"),
                            sentiment_dropdown,
                            html.Div("RESOLUTION", className="filter-label mt-3"),
                            resolution_dropdown,
                        ],
                        xs=12, md=6, xl=4,
                    ),
                    dbc.Col(
                        [
                            html.Div("CALL DURATION", className="filter-label"),
                            html.Div(
                                "Drag either handle to set the duration range.",
                                style={"fontSize": "10px", "color": C["muted"], "marginBottom": "15px"},
                            ),
                            html.Div(duration_slider, style={"padding": "5px 8px 12px"}),
                            html.Div(
                                [
                                    dbc.Button(
                                        [html.I(className="bi bi-arrow-counterclockwise me-1"), "Reset"],
                                        id="reset-filters-btn",
                                        color="light",
                                        className="me-2 mb-2",
                                        n_clicks=0,
                                    ),
                                    dbc.Button(
                                        [html.I(className="bi bi-download me-1"), "Export CSV"],
                                        id="export-csv-btn",
                                        color="primary",
                                        className="mb-2",
                                        n_clicks=0,
                                    ),
                                ],
                                className="mt-3",
                            ),
                        ],
                        xs=12, md=6, xl=4,
                    ),
                ],
                className="g-3 align-items-start",
            ),
            html.Hr(style={"borderColor": C["border"], "margin": "18px 0 8px"}),
            html.Div(id="filter-count", className="filter-count"),
        ]
    ),
    className="dashboard-card filter-card",
)


# ============================================================
# PAGE SECTIONS
# ============================================================

overview_section = html.Section(
    [
        html.Div(
            [
                html.H5("Performance overview", className="section-title"),
                html.P(
                    "Key indicators for the selected calls.",
                    className="section-subtitle",
                ),
            ],
            className="section-header",
        ),
        dbc.Row(
            [
                dbc.Col(
                    kpi_card("Total calls", "kpi-total", "kpi-total-note", "bi-telephone", C["blue"]),
                    xs=12, sm=6, xl=3, className="mb-3",
                ),
                dbc.Col(
                    kpi_card("Positive sentiment", "kpi-positive", "kpi-positive-note", "bi-emoji-smile", C["green"]),
                    xs=12, sm=6, xl=3, className="mb-3",
                ),
                dbc.Col(
                    kpi_card("Resolution rate", "kpi-resolution", "kpi-resolution-note", "bi-check2-circle", C["purple"]),
                    xs=12, sm=6, xl=3, className="mb-3",
                ),
                dbc.Col(
                    kpi_card("Average satisfaction", "kpi-rating", "kpi-rating-note", "bi-star", C["orange"]),
                    xs=12, sm=6, xl=3, className="mb-3",
                ),
            ],
            className="g-3",
        ),
        html.Div(
            [
                html.H5("Highlights", className="section-title"),
                html.P(
                    "Automatically generated observations based on the current filters.",
                    className="section-subtitle",
                ),
            ],
            className="section-header",
        ),
        html.Div(id="insights-container", className="insights-grid"),
    ],
    id="overview",
)

sentiment_section = html.Section(
    [
        html.Div(
            [
                html.H5("Sentiment and trends", className="section-title"),
                html.P(
                    "Explore customer sentiment, satisfaction and resolution patterns.",
                    className="section-subtitle",
                ),
            ],
            className="section-header",
        ),
        dbc.Row(
            [
                dbc.Col(
                    chart_card("Sentiment distribution", "Share of calls by sentiment.", "sentiment-chart"),
                    xs=12, lg=5, className="mb-3",
                ),
                dbc.Col(
                    chart_card("Monthly sentiment trend", "Call volumes by sentiment over time.", "monthly-chart"),
                    xs=12, lg=7, className="mb-3",
                ),
            ],
            className="g-3",
        ),
        dbc.Row(
            [
                dbc.Col(
                    chart_card("Resolution outcomes", "Resolution status split by sentiment.", "resolution-chart"),
                    xs=12, lg=6, className="mb-3",
                ),
                dbc.Col(
                    chart_card("Satisfaction rating", "Frequency of each rating.", "rating-chart"),
                    xs=12, lg=6, className="mb-3",
                ),
            ],
            className="g-3",
        ),
        dbc.Row(
            [
                dbc.Col(
                    chart_card("Negative calls by outcome", "Resolution status of negative-sentiment calls.", "negative-chart"),
                    xs=12, lg=6, className="mb-3",
                ),
                dbc.Col(
                    chart_card("Average duration by outcome", "Average handling time for each resolution status.", "duration-resolution-chart"),
                    xs=12, lg=6, className="mb-3",
                ),
            ],
            className="g-3",
        ),
        dbc.Row(
            [
                dbc.Col(
                    chart_card("Call duration by sentiment", "Compare duration distributions across sentiment groups.", "duration-chart"),
                    xs=12, className="mb-3",
                ),
            ],
            className="g-3",
        ),
    ],
    id="sentiment",
)

call_explorer_section = html.Section(
    [
        html.Div(
            [
                html.H5("Call explorer", className="section-title"),
                html.P(
                    "Search records, sort columns and use the table's column filters.",
                    className="section-subtitle",
                ),
            ],
            className="section-header",
        ),
        dbc.Card(
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Input(
                                    id="table-search",
                                    placeholder="Search call ID, date, sentiment, resolution or duration...",
                                    type="text",
                                    debounce=True,
                                ),
                                xs=12, md=9,
                            ),
                            dbc.Col(
                                dbc.Button(
                                    [html.I(className="bi bi-x-circle me-1"), "Clear search"],
                                    id="clear-search-btn",
                                    color="light",
                                    className="w-100",
                                    n_clicks=0,
                                ),
                                xs=12, md=3,
                            ),
                        ],
                        className="g-2 mb-3",
                    ),
                    dash_table.DataTable(
                        id="calls-table",
                        columns=[
                            {"name": "Call ID", "id": "CallID"},
                            {"name": "Date", "id": "CallDate"},
                            {"name": "Rating", "id": "Satisfaction rating", "type": "numeric"},
                            {"name": "Sentiment", "id": "Sentiment"},
                            {"name": "Resolution", "id": "ResolvedStatus"},
                            {"name": "Duration", "id": "DurationLabel"},
                            {"name": "Minutes", "id": "DurationMinutes", "type": "numeric"},
                        ],
                        data=[],
                        page_action="native",
                        page_current=0,
                        page_size=10,
                        sort_action="native",
                        sort_mode="multi",
                        filter_action="native",
                        fixed_rows={"headers": True},
                        style_table={"overflowX": "auto", "minWidth": "100%"},
                        style_cell={
                            "padding": "11px 12px",
                            "fontFamily": "Segoe UI, Arial, sans-serif",
                            "fontSize": "11px",
                            "textAlign": "left",
                            "minWidth": "85px",
                            "maxWidth": "230px",
                            "whiteSpace": "normal",
                            "height": "auto",
                        },
                        style_header={
                            "backgroundColor": "#F5F7FA",
                            "fontWeight": "700",
                            "border": "none",
                        },
                        style_data={"backgroundColor": "white", "border": "none"},
                        style_data_conditional=[
                            {
                                "if": {"filter_query": '{Sentiment} = "Positive"', "column_id": "Sentiment"},
                                "color": C["green"], "fontWeight": "700",
                            },
                            {
                                "if": {"filter_query": '{Sentiment} = "Neutral"', "column_id": "Sentiment"},
                                "color": C["orange"], "fontWeight": "700",
                            },
                            {
                                "if": {"filter_query": '{Sentiment} = "Negative"', "column_id": "Sentiment"},
                                "color": C["red"], "fontWeight": "700",
                            },
                        ],
                    ),
                    html.Div(id="table-summary", className="table-summary"),
                ]
            ),
            className="dashboard-card table-card",
        ),
    ],
    id="call-explorer",
)

initial_views = get_saved_views()

saved_views_section = html.Section(
    [
        html.Div(
            [
                html.H5("Saved views", className="section-title"),
                html.P(
                    "Save the current filters and load them again later.",
                    className="section-subtitle",
                ),
            ],
            className="section-header",
        ),
        dbc.Card(
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Div("VIEW NAME", className="filter-label"),
                                    dbc.Input(
                                        id="saved-view-name",
                                        placeholder="e.g. Negative calls in the last month",
                                        maxLength=80,
                                        type="text",
                                    ),
                                ],
                                xs=12, md=5,
                            ),
                            dbc.Col(
                                [
                                    html.Div("CHOOSE A SAVED VIEW", className="filter-label"),
                                    dcc.Dropdown(
                                        id="saved-view-select",
                                        options=make_saved_view_options(initial_views),
                                        value=None,
                                        placeholder="Select a saved view...",
                                        clearable=True,
                                    ),
                                ],
                                xs=12, md=5,
                            ),
                            dbc.Col(
                                [
                                    html.Div("ACTIONS", className="filter-label"),
                                    html.Div(
                                        [
                                            dbc.Button(
                                                "Save current",
                                                id="save-view-btn",
                                                color="primary",
                                                className="me-2 mb-2",
                                                n_clicks=0,
                                            ),
                                            dbc.Button(
                                                "Load view",
                                                id="load-view-btn",
                                                color="light",
                                                className="mb-2",
                                                n_clicks=0,
                                            ),
                                        ],
                                        className="d-flex flex-wrap",
                                    ),
                                ],
                                xs=12, md=2,
                            ),
                        ],
                        className="g-3 align-items-end",
                    ),
                    html.Div(id="saved-feedback", className="saved-feedback"),
                    html.Hr(style={"borderColor": C["border"], "margin": "18px 0"}),
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Button(
                                    [html.I(className="bi bi-arrow-clockwise me-1"), "Refresh list"],
                                    id="refresh-views-btn",
                                    color="light",
                                    n_clicks=0,
                                ),
                                xs=12, md="auto",
                            ),
                            dbc.Col(
                                dbc.Button(
                                    [html.I(className="bi bi-trash3 me-1"), "Delete all views"],
                                    id="delete-all-views-btn",
                                    color="danger",
                                    outline=True,
                                    n_clicks=0,
                                ),
                                xs=12, md="auto",
                            ),
                        ],
                        className="g-2",
                    ),
                    html.Div(id="saved-views-table", className="mt-3"),
                ]
            ),
            className="dashboard-card",
        ),
    ],
    id="saved-views",
)


# ============================================================
# FULL LAYOUT
# ============================================================

app.layout = html.Div(
    [
        dcc.Store(id="saved-views-store", data=initial_views),
        dcc.Download(id="download-csv"),
        html.Div(
            [
                sidebar,
                html.Main(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H1(APP_TITLE, className="page-title"),
                                        html.P(
                                            "Customer experience and operational performance",
                                            className="page-subtitle",
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.I(className="bi bi-file-earmark-spreadsheet"),
                                        f"  {DATA_FILE.name}  ·  {len(DF):,} records",
                                    ],
                                    className="dataset-badge",
                                ),
                            ],
                            className="topbar",
                        ),
                        filters_panel,
                        overview_section,
                        sentiment_section,
                        call_explorer_section,
                        saved_views_section,
                        html.Div(
                            "Call Center Intelligence  ·  Customer experience analytics",
                            className="footer-note",
                        ),
                    ],
                    className="main-content",
                ),
            ],
            className="app-shell d-flex",
        ),
    ]
)


# ============================================================
# MAIN DASHBOARD CALLBACK
# ============================================================

@app.callback(
    Output("kpi-total", "children"),
    Output("kpi-total-note", "children"),
    Output("kpi-positive", "children"),
    Output("kpi-positive-note", "children"),
    Output("kpi-resolution", "children"),
    Output("kpi-resolution-note", "children"),
    Output("kpi-rating", "children"),
    Output("kpi-rating-note", "children"),
    Output("filter-count", "children"),
    Output("insights-container", "children"),
    Output("sentiment-chart", "figure"),
    Output("monthly-chart", "figure"),
    Output("resolution-chart", "figure"),
    Output("rating-chart", "figure"),
    Output("negative-chart", "figure"),
    Output("duration-resolution-chart", "figure"),
    Output("duration-chart", "figure"),
    Output("calls-table", "data"),
    Output("table-summary", "children"),
    Input("date-filter", "start_date"),
    Input("date-filter", "end_date"),
    Input("sentiment-filter", "value"),
    Input("resolution-filter", "value"),
    Input("duration-filter", "value"),
    Input("table-search", "value"),
)
def update_dashboard(start_date, end_date, sentiment, resolution, duration_range, search):
    df = filtered_data(start_date, end_date, sentiment, resolution, duration_range)

    if df.empty:
        empty = empty_figure()
        return (
            "0", "No calls match the selected filters",
            "0%", "Positive calls",
            "0%", "Resolved calls",
            "—", "No satisfaction data",
            "Showing 0 calls",
            insights_for(df),
            empty, empty, empty, empty, empty, empty, empty,
            [],
            "No records found. Try adjusting the filters.",
        )

    total = len(df)
    positive = int((df["Sentiment"] == "Positive").sum())
    negative = int((df["Sentiment"] == "Negative").sum())
    resolved = int((df["ResolvedStatus"] == "Resolved").sum())
    positive_pct = positive / total * 100
    negative_pct = negative / total * 100
    resolved_pct = resolved / total * 100
    average_rating = df["Satisfaction rating"].mean()
    average_duration = df["DurationSeconds"].mean()

    table_df = df.copy()

    if search and search.strip():
        term = search.strip().lower()
        columns = [
            "CallID", "CallDate", "Sentiment", "ResolvedStatus",
            "DurationLabel", "Satisfaction rating",
        ]
        mask = pd.Series(False, index=table_df.index)
        for column in columns:
            mask |= table_df[column].astype(str).str.lower().str.contains(
                re.escape(term), na=False
            )
        table_df = table_df[mask]

    table_columns = [
        "CallID", "CallDate", "Satisfaction rating", "Sentiment",
        "ResolvedStatus", "DurationLabel", "DurationMinutes",
    ]
    table_data = table_df[table_columns].copy()
    table_data["DurationMinutes"] = table_data["DurationMinutes"].round(2)

    return (
        f"{total:,}",
        f"{negative:,} negative calls · {negative_pct:.1f}%",
        f"{positive_pct:.1f}%",
        f"{positive:,} positive calls",
        f"{resolved_pct:.1f}%",
        f"{resolved:,} resolved calls",
        f"{average_rating:.2f}/5",
        f"Average call: {format_duration(average_duration)}",
        f"Showing {total:,} of {len(DF):,} calls",
        insights_for(df),
        sentiment_figure(df),
        monthly_figure(df),
        resolution_figure(df),
        rating_figure(df),
        negative_figure(df),
        duration_by_resolution_figure(df),
        duration_figure(df),
        table_data.to_dict("records"),
        f"{len(table_data):,} matching records before table pagination.",
    )


# ============================================================
# DATE PRESETS
# ============================================================

@app.callback(
    Output("date-filter", "start_date"),
    Output("date-filter", "end_date"),
    Input("preset-7", "n_clicks"),
    Input("preset-30", "n_clicks"),
    Input("preset-all", "n_clicks"),
    prevent_initial_call=True,
)
def set_date_preset(click_7, click_30, click_all):
    triggered = ctx.triggered_id

    if triggered == "preset-all":
        return MIN_DATE, MAX_DATE

    if triggered == "preset-7":
        days = 7
    elif triggered == "preset-30":
        days = 30
    else:
        return no_update, no_update

    end = MAX_DATE
    start = max(MIN_DATE, end - timedelta(days=days - 1))
    return start, end


# ============================================================
# RESET FILTERS
# ============================================================

@app.callback(
    Output("date-filter", "start_date", allow_duplicate=True),
    Output("date-filter", "end_date", allow_duplicate=True),
    Output("sentiment-filter", "value"),
    Output("resolution-filter", "value"),
    Output("duration-filter", "value"),
    Input("reset-filters-btn", "n_clicks"),
    prevent_initial_call=True,
)
def reset_filters(n_clicks):
    if not n_clicks:
        return (no_update,) * 5
    return MIN_DATE, MAX_DATE, "All", "All", DEFAULT_DURATION


# ============================================================
# CLEAR SEARCH
# ============================================================

@app.callback(
    Output("table-search", "value"),
    Output("calls-table", "filter_query"),
    Input("clear-search-btn", "n_clicks"),
    prevent_initial_call=True,
)
def clear_search(n_clicks):
    if not n_clicks:
        return no_update, no_update
    return "", ""


# ============================================================
# EXPORT CSV
# ============================================================

@app.callback(
    Output("download-csv", "data"),
    Input("export-csv-btn", "n_clicks"),
    State("date-filter", "start_date"),
    State("date-filter", "end_date"),
    State("sentiment-filter", "value"),
    State("resolution-filter", "value"),
    State("duration-filter", "value"),
    prevent_initial_call=True,
)
def export_csv(n_clicks, start_date, end_date, sentiment, resolution, duration_range):
    if not n_clicks:
        return no_update

    df = filtered_data(start_date, end_date, sentiment, resolution, duration_range)

    columns = [
        "CallID", "Date", "Satisfaction rating", "Sentiment",
        "Resolved", "ResolvedStatus", "AvgTalkDuration",
        "DurationSeconds", "DurationMinutes",
    ]
    columns = [column for column in columns if column in df.columns]

    return dcc.send_data_frame(
        df[columns].to_csv,
        filename=f"call_center_report_{datetime.now():%Y%m%d_%H%M%S}.csv",
        index=False,
    )


# ============================================================
# SAVED VIEW TABLE
# ============================================================

def saved_views_table(views):
    if not views:
        return html.Div(
            "No saved views yet. Apply filters and save your first view.",
            className="saved-feedback",
        )

    return dbc.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("View name"),
                        html.Th("Created"),
                    ]
                )
            ),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(view["name"], style={"fontWeight": "650"}),
                            html.Td(view["created_at"]),
                        ]
                    )
                    for view in views
                ]
            ),
        ],
        hover=True,
        responsive=True,
        class_name="align-middle",
    )


# ============================================================
# MANAGE SAVED VIEWS
# ============================================================

@app.callback(
    Output("saved-feedback", "children"),
    Output("saved-views-table", "children"),
    Output("saved-view-select", "options"),
    Output("saved-views-store", "data"),
    Input("save-view-btn", "n_clicks"),
    Input("refresh-views-btn", "n_clicks"),
    Input("delete-all-views-btn", "n_clicks"),
    State("saved-view-name", "value"),
    State("date-filter", "start_date"),
    State("date-filter", "end_date"),
    State("sentiment-filter", "value"),
    State("resolution-filter", "value"),
    State("duration-filter", "value"),
    prevent_initial_call=True,
)
def manage_saved_views(
    save_clicks,
    refresh_clicks,
    delete_clicks,
    name,
    start_date,
    end_date,
    sentiment,
    resolution,
    duration_range,
):
    triggered = ctx.triggered_id
    message = ""

    if triggered == "save-view-btn":
        name = (name or "").strip()

        if not name:
            views = get_saved_views()
            return (
                "Enter a view name before saving.",
                saved_views_table(views),
                make_saved_view_options(views),
                views,
            )

        filters = {
            "start_date": str(start_date) if start_date else None,
            "end_date": str(end_date) if end_date else None,
            "sentiment": sentiment or "All",
            "resolution": resolution or "All",
            "duration_range": duration_range or DEFAULT_DURATION,
        }

        with _DB_LOCK:
            with db_connection() as con:
                con.execute(
                    """
                    INSERT INTO saved_views (name, filters_json, created_at)
                    VALUES (?, ?, ?)
                    """,
                    (
                        name,
                        json.dumps(filters),
                        datetime.now().strftime("%Y-%m-%d %H:%M"),
                    ),
                )
                con.commit()

        message = f"Saved view: {name}"

    elif triggered == "delete-all-views-btn":
        with _DB_LOCK:
            with db_connection() as con:
                con.execute("DELETE FROM saved_views")
                con.commit()
        message = "All saved views deleted."

    elif triggered == "refresh-views-btn":
        message = "Saved views refreshed."

    views = get_saved_views()
    return (
        message,
        saved_views_table(views),
        make_saved_view_options(views),
        views,
    )


# ============================================================
# LOAD SAVED VIEW
# ============================================================

@app.callback(
    Output("date-filter", "start_date", allow_duplicate=True),
    Output("date-filter", "end_date", allow_duplicate=True),
    Output("sentiment-filter", "value", allow_duplicate=True),
    Output("resolution-filter", "value", allow_duplicate=True),
    Output("duration-filter", "value", allow_duplicate=True),
    Output("saved-feedback", "children", allow_duplicate=True),
    Input("load-view-btn", "n_clicks"),
    State("saved-view-select", "value"),
    prevent_initial_call=True,
)
def load_saved_view(n_clicks, selected_id):
    if not n_clicks:
        return (no_update,) * 6

    if selected_id is None:
        return no_update, no_update, no_update, no_update, no_update, "Choose a saved view first."

    with _DB_LOCK:
        with db_connection() as con:
            row = con.execute(
                "SELECT name, filters_json FROM saved_views WHERE id = ?",
                (selected_id,),
            ).fetchone()

    if row is None:
        return no_update, no_update, no_update, no_update, no_update, "Saved view not found. Refresh the list."

    try:
        filters = json.loads(row["filters_json"])
    except (ValueError, TypeError):
        return no_update, no_update, no_update, no_update, no_update, "Could not read the saved filters."

    start = filters.get("start_date") or MIN_DATE
    end = filters.get("end_date") or MAX_DATE
    duration = filters.get("duration_range") or DEFAULT_DURATION

    try:
        duration = [
            max(SLIDER_MIN, min(SLIDER_MAX, float(duration[0]))),
            max(SLIDER_MIN, min(SLIDER_MAX, float(duration[1]))),
        ]
        duration.sort()
    except (TypeError, ValueError, IndexError):
        duration = DEFAULT_DURATION

    return (
        start,
        end,
        filters.get("sentiment", "All"),
        filters.get("resolution", "All"),
        duration,
        f"Loaded view: {row['name']}",
    )


# ============================================================
# INITIAL SAVED VIEWS DISPLAY
# ============================================================

@app.callback(
    Output("saved-views-table", "children", allow_duplicate=True),
    Input("saved-views-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def render_initial_saved_views(views):
    return saved_views_table(views or [])


# ============================================================
# RUN: JUPYTER OR APP.PY
# ============================================================

if __name__ == "__main__":
    try:
        from IPython import get_ipython
        shell = get_ipython()
        in_jupyter = shell is not None and "IPKernelApp" in shell.config
    except Exception:
        in_jupyter = False

    if in_jupyter:
        app.run(
            host="127.0.0.1",
            port=PORT,
            debug=False,
            jupyter_mode="external",
        )
    else:
        app.run(
            host="0.0.0.0",
            port=PORT,
            debug=DEBUG,
        )
