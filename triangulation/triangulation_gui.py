from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "triangulation"
DEFAULT_FILE = "path_b_calibrated_results_guarded_trials.csv"

REQUIRED_COLS = {
    "category",
    "distance_cm",
    "angle_deg",
    "ok",
    "x_true_m",
    "y_true_m",
    "x_est_m",
    "y_est_m",
    "xy_err_cm",
    "dist_err_cm",
    "angle_err_deg",
}


def _available_result_files() -> list[Path]:
    if not OUTPUT_DIR.exists():
        return []
    return sorted(OUTPUT_DIR.glob("path_b_calibrated_results*.csv"))


def _load_result(file_name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / file_name
    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")

    df = pd.read_csv(path)
    missing = sorted(REQUIRED_COLS.difference(df.columns))
    if missing:
        raise ValueError(f"{file_name} missing required columns: {missing}")

    good = df[df["ok"] == True].copy()
    if good.empty:
        return good

    good["distance_cm"] = good["distance_cm"].astype(float)
    good["angle_deg"] = good["angle_deg"].astype(float)
    good["category"] = good["category"].astype(str)
    good["xy_err_cm"] = good["xy_err_cm"].astype(float)
    good["dist_err_cm"] = good["dist_err_cm"].astype(float)
    good["angle_err_deg"] = good["angle_err_deg"].astype(float)
    return good


def _filter_df(df: pd.DataFrame, category: str, distance: float, angle: float) -> pd.DataFrame:
    out = df
    if category != "ALL":
        out = out[out["category"] == category]
    if distance is not None:
        out = out[np.isclose(out["distance_cm"], float(distance))]
    if angle is not None:
        out = out[np.isclose(out["angle_deg"], float(angle))]
    return out


def _summary_text(df: pd.DataFrame, file_name: str) -> str:
    if df.empty:
        return f"File: {file_name}\nNo rows after filtering."

    xy = df["xy_err_cm"]
    de = df["dist_err_cm"].abs()
    ae = df["angle_err_deg"].abs()
    lines = [
        f"File: {file_name}",
        f"Rows: {len(df)}",
        f"XY error median / p90: {xy.median():.2f} / {xy.quantile(0.9):.2f} cm",
        f"|Distance error| median / p90: {de.median():.2f} / {de.quantile(0.9):.2f} cm",
        f"|Angle error| median / p90: {ae.median():.2f} / {ae.quantile(0.9):.2f} deg",
        f"Pass rates: XY<=20cm {((xy<=20).mean()*100):.1f}% | |dist_err|<=5cm {((de<=5).mean()*100):.1f}% | |angle_err|<=10deg {((ae<=10).mean()*100):.1f}%",
    ]
    return "\n".join(lines)


FILES = _available_result_files()
FILE_OPTIONS = [{"label": p.name, "value": p.name} for p in FILES]
DEFAULT_VALUE = DEFAULT_FILE if any(p.name == DEFAULT_FILE for p in FILES) else (FILES[0].name if FILES else None)

app = Dash(__name__)
app.title = "Triangulation Inspector"

app.layout = html.Div(
    [
        html.H2("Triangulation Inspector"),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Result file"),
                        dcc.Dropdown(id="file-name", options=FILE_OPTIONS, value=DEFAULT_VALUE, clearable=False),
                    ],
                    style={"width": "40%", "display": "inline-block", "verticalAlign": "top"},
                ),
                html.Div(
                    [
                        html.Label("Category"),
                        dcc.Dropdown(id="category", clearable=False),
                    ],
                    style={"width": "18%", "display": "inline-block", "paddingLeft": "12px"},
                ),
                html.Div(
                    [
                        html.Label("Distance (cm)"),
                        dcc.Dropdown(id="distance", clearable=True, placeholder="ALL"),
                    ],
                    style={"width": "18%", "display": "inline-block", "paddingLeft": "12px"},
                ),
                html.Div(
                    [
                        html.Label("Angle (deg)"),
                        dcc.Dropdown(id="angle", clearable=True, placeholder="ALL"),
                    ],
                    style={"width": "18%", "display": "inline-block", "paddingLeft": "12px"},
                ),
            ]
        ),
        dcc.Graph(id="xy-map"),
        dcc.Graph(id="err-hist"),
        dcc.Graph(id="err-by-distance"),
        html.Pre(id="stats", style={"backgroundColor": "#f6f8fa", "padding": "10px", "borderRadius": "6px"}),
    ],
    style={"maxWidth": "1400px", "margin": "0 auto", "padding": "12px"},
)


@app.callback(
    Output("category", "options"),
    Output("category", "value"),
    Output("distance", "options"),
    Output("angle", "options"),
    Input("file-name", "value"),
)
def refresh_filter_options(file_name: str):
    if not file_name:
        return [], None, [], []
    df = _load_result(file_name)
    if df.empty:
        return [{"label": "ALL", "value": "ALL"}], "ALL", [], []

    cats = sorted(df["category"].dropna().unique().tolist())
    category_options = [{"label": "ALL", "value": "ALL"}] + [{"label": c, "value": c} for c in cats]

    distances = sorted(df["distance_cm"].dropna().unique().tolist())
    distance_options = [{"label": f"{d:.0f}", "value": float(d)} for d in distances]

    angles = sorted(df["angle_deg"].dropna().unique().tolist())
    angle_options = [{"label": f"{a:.0f}", "value": float(a)} for a in angles]

    return category_options, "ALL", distance_options, angle_options


@app.callback(
    Output("xy-map", "figure"),
    Output("err-hist", "figure"),
    Output("err-by-distance", "figure"),
    Output("stats", "children"),
    Input("file-name", "value"),
    Input("category", "value"),
    Input("distance", "value"),
    Input("angle", "value"),
)
def update_plots(file_name: str, category: str, distance: float | None, angle: float | None):
    if not file_name:
        empty = go.Figure()
        return empty, empty, empty, "No result file selected."

    df = _load_result(file_name)
    sub = _filter_df(df, category if category is not None else "ALL", distance, angle)

    fig_xy = go.Figure()
    if not sub.empty:
        fig_xy.add_trace(
            go.Scatter(
                x=sub["x_true_m"],
                y=sub["y_true_m"],
                mode="markers",
                name="True",
                marker=dict(size=8, color="#1f77b4", opacity=0.8),
            )
        )
        fig_xy.add_trace(
            go.Scatter(
                x=sub["x_est_m"],
                y=sub["y_est_m"],
                mode="markers",
                name="Estimated",
                marker=dict(size=7, color="#d62728", symbol="diamond", opacity=0.7),
            )
        )
        # Draw linking segments for the first N points to keep UI responsive.
        max_lines = min(250, len(sub))
        for _, r in sub.head(max_lines).iterrows():
            fig_xy.add_trace(
                go.Scatter(
                    x=[r["x_true_m"], r["x_est_m"]],
                    y=[r["y_true_m"], r["y_est_m"]],
                    mode="lines",
                    line=dict(color="rgba(100,100,100,0.25)", width=1),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    fig_xy.update_layout(
        title="True vs Estimated Position (x,y)",
        xaxis_title="x (m)",
        yaxis_title="y (m)",
        yaxis_scaleanchor="x",
        template="plotly_white",
        legend=dict(orientation="h"),
    )

    if sub.empty:
        fig_hist = go.Figure().update_layout(title="XY Error Histogram", template="plotly_white")
    else:
        fig_hist = px.histogram(sub, x="xy_err_cm", nbins=40, title="XY Error Histogram", template="plotly_white")
        fig_hist.update_xaxes(title="xy_err_cm")

    if sub.empty:
        fig_box = go.Figure().update_layout(title="XY Error by Distance", template="plotly_white")
    else:
        fig_box = px.box(
            sub,
            x="distance_cm",
            y="xy_err_cm",
            points="all",
            title="XY Error by Distance",
            template="plotly_white",
        )
        fig_box.update_xaxes(title="distance_cm")
        fig_box.update_yaxes(title="xy_err_cm")

    stats = _summary_text(sub, file_name)
    return fig_xy, fig_hist, fig_box, stats


if __name__ == "__main__":
    app.run(debug=True)

