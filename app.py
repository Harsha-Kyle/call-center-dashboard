import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import dash_bootstrap_components as dbc
import sqlite3

# Load dataset
file_path = "CallCenterDataset.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Ensure required columns exist
required_columns = {"Satisfaction rating", "Date", "Resolved", "AvgTalkDuration"}
missing_columns = required_columns - set(df.columns)
if missing_columns:
    raise ValueError(f"Missing columns in dataset: {missing_columns}")

# Convert Satisfaction Rating to Sentiment
def classify_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

df["Sentiment"] = df["Satisfaction rating"].apply(classify_sentiment)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["AvgTalkDuration"] = pd.to_timedelta(df["AvgTalkDuration"].astype(str), errors="coerce").dt.total_seconds()
df.dropna(subset=["Date", "AvgTalkDuration", "Sentiment", "Resolved"], inplace=True)

min_duration, max_duration = df["AvgTalkDuration"].min(), df["AvgTalkDuration"].max()

# Initialize Dash App with Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# SQLite Database Setup
conn = sqlite3.connect("call_center_data.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS past_filters (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sentiment TEXT,
        start_date TEXT,
        end_date TEXT,
        resolved TEXT,
        duration_range TEXT
    )
""")
conn.commit()

# Layout
app.layout = dbc.Container([
    
    dbc.Row([
        dbc.Col(html.H1("Call Center Sentiment Analysis Dashboard", className="text-primary"), md=9),
        dbc.Col(html.H5("Done by: Harsha Kyle and Mann", className="text-end fw-bold text-secondary"), md=3)
    ], className="mb-4"),
    # Filters Section
    dbc.Row([
        dbc.Col([
            html.Label("Select Sentiment:", className="fw-bold"),
            dcc.Dropdown(
                id="sentiment-filter",
                options=[{"label": s, "value": s} for s in df["Sentiment"].unique()],
                value=None,
                multi=True,
                placeholder="Filter by sentiment...",
                className="form-select"
            ),
        ], md=4),
        dbc.Col([
            html.Label("Select Date Range:", className="fw-bold"),
            dcc.DatePickerRange(
                id="date-picker",
                start_date=df["Date"].min(),
                end_date=df["Date"].max(),
                display_format="YYYY-MM-DD",
                className="form-control"
            ),
        ], md=4),
        dbc.Col([
            html.Label("Resolved Calls:", className="fw-bold"),
            dcc.Dropdown(
                id="resolved-filter",
                options=[{"label": "Yes", "value": "Y"}, {"label": "No", "value": "N"}],
                value=None,
                multi=False,
                placeholder="Filter by resolution...",
                className="form-select"
            ),
        ], md=4),
    ], className="mb-3"),

    # Additional Filters
    dbc.Row([
        dbc.Col([
            html.Label("Call Duration (seconds):", className="fw-bold"),
            dcc.RangeSlider(
                id="duration-slider",
                min=min_duration,
                max=max_duration,
                step=10,
                value=[min_duration, max_duration],
                marks={int(min_duration): str(int(min_duration)), int(max_duration): str(int(max_duration))},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ], md=12),
    ], className="mb-4"),

    # Graphs
    dbc.Row([
        dbc.Col(dcc.Graph(id="sentiment-pie"), md=6),
        dbc.Col(dcc.Graph(id="sentiment-trend"), md=6),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="resolution-bar"), md=6),
        dbc.Col(dcc.Graph(id="duration-box"), md=6),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="churn-indicator"), md=12),
    ]),

    # Buttons for Saving, Viewing & Deleting Data
    html.Div(className="text-center mt-4", children=[
        html.Button("Save View", id="save-view-btn", className="btn btn-primary", n_clicks=0),
        html.Button("View Past Data", id="view-past-btn", className="btn btn-secondary", n_clicks=0),
        html.Button("Delete All", id="delete-all-btn", className="btn btn-danger", n_clicks=0),
    ]),

    html.Div(id="past-data-output", className="mt-4")
], fluid=True)

# Callback for updating graphs
@app.callback(
    [
        Output("sentiment-pie", "figure"),
        Output("sentiment-trend", "figure"),
        Output("resolution-bar", "figure"),
        Output("duration-box", "figure"),
        Output("churn-indicator", "figure"),
    ],
    [
        Input("sentiment-filter", "value"),
        Input("date-picker", "start_date"),
        Input("date-picker", "end_date"),
        Input("resolved-filter", "value"),
        Input("duration-slider", "value"),
    ]
)
def update_graphs(selected_sentiments, start_date, end_date, resolved_status, duration_range):
    filtered_df = df.copy()
    
    if selected_sentiments:
        filtered_df = filtered_df[filtered_df["Sentiment"].isin(selected_sentiments)]
    
    filtered_df = filtered_df[(filtered_df["Date"] >= start_date) & (filtered_df["Date"] <= end_date)]
    
    if resolved_status:
        filtered_df = filtered_df[filtered_df["Resolved"] == resolved_status]

    filtered_df = filtered_df[
        (filtered_df["AvgTalkDuration"] >= duration_range[0]) & 
        (filtered_df["AvgTalkDuration"] <= duration_range[1])
    ]

    pie_fig = px.pie(
        filtered_df, 
        names="Sentiment", 
        title="Sentiment Distribution", 
        color="Sentiment",
        color_discrete_map={"Positive": "green", "Neutral": "orange", "Negative": "red"}
    )

    sentiment_trend = (
        filtered_df.groupby(filtered_df["Date"].dt.strftime("%Y-%m"))["Sentiment"]
        .value_counts()
        .unstack()
        .fillna(0)
    )
    trend_fig = px.line(sentiment_trend, title="Sentiment Trends Over Time")

    resolution_counts = filtered_df.groupby(["Resolved", "Sentiment"]).size().reset_index(name="Count")
    resolution_fig = px.bar(
        resolution_counts,
        x="Resolved",
        y="Count",
        color="Sentiment",
        title="Impact of Resolution on Sentiment",
        barmode="group",
        color_discrete_map={"Positive": "green", "Neutral": "orange", "Negative": "red"}
    )

    duration_fig = px.box(
        filtered_df, 
        x="Sentiment", 
        y="AvgTalkDuration", 
        color="Sentiment",
        title="Call Duration vs. Sentiment"
    )

    churn_df = (
        filtered_df[filtered_df["Sentiment"] == "Negative"]
        .groupby("Resolved")
        .size()
        .reset_index(name="Count")
    )
    churn_fig = px.bar(
        churn_df,
        x="Resolved",
        y="Count",
        title="Churn Indicator: Resolution vs. Negative Sentiment",
        color="Resolved",
        color_discrete_map={"Y": "green", "N": "red"}
    )

    return pie_fig, trend_fig, resolution_fig, duration_fig, churn_fig

# Callback for Saving, Viewing & Deleting Past Data
@app.callback(
    Output("past-data-output", "children"),
    [Input("save-view-btn", "n_clicks"),
     Input("view-past-btn", "n_clicks"),
     Input("delete-all-btn", "n_clicks")],
    [State("sentiment-filter", "value"),
     State("date-picker", "start_date"),
     State("date-picker", "end_date"),
     State("resolved-filter", "value"),
     State("duration-slider", "value")]
)
def handle_past_data(save_click, view_click, delete_click, sentiment, start_date, end_date, resolved, duration_range):
    ctx = dash.callback_context
    if not ctx.triggered:
        return ""

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "save-view-btn":
        cursor.execute("INSERT INTO past_filters (sentiment, start_date, end_date, resolved, duration_range) VALUES (?, ?, ?, ?, ?)",
                       (str(sentiment), start_date, end_date, resolved, str(duration_range)))
        conn.commit()

    elif button_id == "delete-all-btn":
        cursor.execute("DELETE FROM past_filters")
        conn.commit()

    past_data = pd.read_sql("SELECT * FROM past_filters", conn)
    return html.Pre(past_data.to_string())

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=10000)

