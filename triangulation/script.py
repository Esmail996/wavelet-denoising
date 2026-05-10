from pathlib import Path
base = Path('output')
base.mkdir(exist_ok=True)
app = '''import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px

app = Dash(__name__)

# Replace these with your own files if needed
# expected columns:
# results.csv -> id, x, y, z, pred_x, pred_y, pred_z
# truth.csv   -> id, x, y, z
results = pd.read_csv("results.csv")
truth = pd.read_csv("truth.csv")

for col in ["id", "x", "y", "z"]:
    if col not in truth.columns:
        raise ValueError(f"truth.csv missing column: {col}")
for col in ["id", "pred_x", "pred_y", "pred_z"]:
    if col not in results.columns:
        raise ValueError(f"results.csv missing column: {col}")

merged = results.merge(truth[["id", "x", "y", "z"]], on="id", suffixes=("_pred", "_true"))
merged["err"] = np.sqrt((merged["pred_x"]-merged["x"])**2 + (merged["pred_y"]-merged["y"])**2 + (merged["pred_z"]-merged["z"])**2)

app.layout = html.Div([
    html.H2("Triangulation Inspector"),
    html.Div([
        html.Div([
            html.Label("Select point id"),
            dcc.Dropdown(id="point-id", options=[{"label": str(i), "value": i} for i in merged["id"].tolist()], value=merged["id"].iloc[0])
        ], style={"width":"30%","display":"inline-block","verticalAlign":"top"}),
        html.Div([
            html.Label("Compare axis"),
            dcc.RadioItems(id="axis", options=[{"label": a, "value": a} for a in ["x","y","z"]], value="x", inline=True)
        ], style={"width":"30%","display":"inline-block","paddingLeft":"20px"})
    ]),
    dcc.Graph(id="scatter"),
    dcc.Graph(id="error-bar"),
    html.Pre(id="stats")
])

@app.callback(
    Output("scatter", "figure"),
    Output("error-bar", "figure"),
    Output("stats", "children"),
    Input("point-id", "value"),
    Input("axis", "value")
)
def update(point_id, axis):
    row = merged.loc[merged["id"] == point_id].iloc[0]
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter3d(x=merged["x"], y=merged["y"], z=merged["z"], mode="markers", name="True", marker=dict(size=4)))
    fig1.add_trace(go.Scatter3d(x=merged["pred_x"], y=merged["pred_y"], z=merged["pred_z"], mode="markers", name="Predicted", marker=dict(size=4, symbol="diamond")))
    fig1.add_trace(go.Scatter3d(x=[row["x"], row["pred_x"]], y=[row["y"], row["pred_y"]], z=[row["z"], row["pred_z"]], mode="lines+markers", name=f"Selected {point_id}"))
    fig1.update_layout(scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"), margin=dict(l=0,r=0,b=0,t=30), title="3D triangulation vs truth")

    fig2 = px.bar(merged, x="id", y="err", title=f"Per-point Euclidean error")
    fig2.add_vline(x=point_id, line_width=2, line_dash="dash", line_color="red")
    stats = f"RMSE: {np.sqrt((merged['err']**2).mean()):.4f}\nMAE: {merged['err'].mean():.4f}\nSelected point {point_id}: true=({row['x']:.3f},{row['y']:.3f},{row['z']:.3f}), pred=({row['pred_x']:.3f},{row['pred_y']:.3f},{row['pred_z']:.3f}), err={row['err']:.4f}\nAxis view: {axis}"
    return fig1, fig2, stats

if __name__ == '__main__':
    app.run(debug=True)
'''
(base / 'triangulation_gui.py').write_text(app)
(base / 'requirements.txt').write_text('dash\npandas\nnumpy\nplotly\n')
(base / 'README.md').write_text('Run: python triangulation_gui.py\nPlace results.csv and truth.csv in the same folder.\n')
print((base / 'triangulation_gui.py').as_posix())