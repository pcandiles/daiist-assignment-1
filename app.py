"""
Assignment 1 — Gradio dashboard.

You're using a script for this stage. If you'd rather use a notebook,
write app.ipynb instead and delete this file — main.py refuses to run if
it finds both app.py and app.ipynb, so exactly one of them must exist.

Replace this docstring and everything below it with your own code. When run
via `uv run python main.py app`, this file must build and launch a Gradio
app (a `demo` that calls `.launch()`) using only what your training stage
already produced — it must never retrain anything itself. At minimum, the
dashboard must let you:

- Compare your three trained models via a prediction-vs-actual plot.
- See feature and/or target distributions.
- For classification: move a decision-threshold slider and watch the
  confusion matrix, and a business-cost number tied to your REPORT.md's
  framing, change with it.

How you structure the code beyond that — file layout, function
boundaries, naming — is your call to make and be able to defend.
"""

from pathlib import Path

import gradio as gr
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"

PRIMARY = "#ff5a5f"
PRIMARY_DARK = "#c13041"
PRIMARY_LIGHT = "#ffe4e4"
SECONDARY = "#00a699"
ACCENT = "#fc642d"
INK = "#484848"
INK_MUTED = "#767676"
GRID = "#ebebeb"
TEXT = "#222222"
BG = "#fff8f7"

MODEL_COLORS = {
    "Scikit-learn Ridge": PRIMARY,
    "Manual PyTorch": SECONDARY,
    "Standard PyTorch": ACCENT,
    "Naive baseline": INK_MUTED,
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "axes.titlecolor": TEXT, "axes.titleweight": "bold", "axes.titlesize": 12,
    "legend.frameon": False,
})


preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
sklearn_model = joblib.load(MODELS_DIR / "sklearn_ridge.joblib")
manual_params = torch.load(MODELS_DIR / "manual_pytorch.pt")
nn_state = torch.load(MODELS_DIR / "nn_pytorch.pt")
baseline = joblib.load(MODELS_DIR / "baseline.joblib")
test_set = joblib.load(MODELS_DIR / "test_set.joblib")

X_test_t = preprocessor.transform(test_set["X_test"]).astype("float32")
X_test_t = np.nan_to_num(X_test_t, nan=0.0)
X_test_tensor = torch.tensor(X_test_t, dtype=torch.float32)
y_true_eur = np.expm1(test_set["y_test_log"])

model_nn = nn.Linear(nn_state["weight"].shape[1], 1)
model_nn.load_state_dict(nn_state)
model_nn.eval()

with torch.no_grad():
    preds_log = {
        "Scikit-learn Ridge": sklearn_model.predict(X_test_t),
        "Manual PyTorch": (X_test_tensor @ manual_params["w"] + manual_params["b"]).squeeze().numpy(),
        "Standard PyTorch": model_nn(X_test_tensor).squeeze().numpy(),
        "Naive baseline": np.full(len(X_test_t), baseline["median_log_price"]),
    }
predictions = {name: np.expm1(p) for name, p in preds_log.items()}
model_names = list(predictions)
feature_names = preprocessor.get_feature_names_out()

listings = pd.read_csv(ROOT / "listings.csv")
listings["price"] = pd.to_numeric(listings["price"].str.replace(r"[$,]", "", regex=True), errors="coerce")
listings = listings.loc[listings["price"].notna() & (listings["number_of_reviews_ltm"] > 0)].copy()
numeric_cols = sorted(listings.select_dtypes(include="number").columns)


def euro_metrics(preds):
    err = y_true_eur - preds
    return {"RMSE": float(np.sqrt((err ** 2).mean())), "MAE": float(np.abs(err).mean())}


metrics = {name: euro_metrics(p) for name, p in predictions.items()}
best_model = min([n for n in model_names if n != "Naive baseline"], key=lambda n: metrics[n]["RMSE"])
price_cap = float(np.percentile(y_true_eur, 99))


def pred_vs_actual(model_name):
    shown = y_true_eur <= price_cap
    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.scatter(y_true_eur[shown], predictions[model_name][shown], alpha=0.3, s=6,
               color=MODEL_COLORS[model_name], linewidths=0)
    ax.plot([0, price_cap], [0, price_cap], color=TEXT, linestyle="--", linewidth=1,
            label="perfect prediction")
    ax.set_xlabel("Actual €/night")
    ax.set_ylabel("Predicted €/night")
    ax.set_title(model_name)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def residuals_plot(model_name):
    residuals = predictions[model_name] - y_true_eur
    shown = y_true_eur <= price_cap
    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.scatter(y_true_eur[shown], residuals[shown], alpha=0.3, s=6,
               color=MODEL_COLORS[model_name], linewidths=0)
    ax.axhline(0, color=TEXT, linewidth=1, linestyle="--")
    ax.set_xlabel("Actual €/night")
    ax.set_ylabel("Residual (predicted − actual, €)")
    ax.set_title(model_name)
    fig.tight_layout()
    return fig


def error_dist_plot(model_name):
    errors = predictions[model_name] - y_true_eur
    cap = float(np.percentile(np.abs(errors), 99))
    mean_error = float(errors.mean())
    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.hist(errors[np.abs(errors) <= cap], bins=60, color=MODEL_COLORS[model_name],
            edgecolor="white", linewidth=0.4)
    ax.axvline(0, color=TEXT, linewidth=1, linestyle="--", label="zero error")
    ax.axvline(mean_error, color=PRIMARY_DARK, linewidth=1.6, label=f"mean error €{mean_error:.1f}")
    ax.set_xlabel("Error (predicted − actual, €/night)")
    ax.set_ylabel("Listings")
    ax.set_title(model_name)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def top_features_plot(n=15):
    coef = sklearn_model.coef_
    top = np.argsort(np.abs(coef))[-n:]
    fig, ax = plt.subplots(figsize=(6, 6.2))
    ax.barh(feature_names[top], coef[top], color=[PRIMARY if c >= 0 else SECONDARY for c in coef[top]])
    ax.axvline(0, color=TEXT, linewidth=0.8)
    ax.set_xlabel("Effect on log-price  (red = raises, teal = lowers)")
    ax.set_title(f"Top {n} price drivers")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=8, labelcolor=INK)
    fig.tight_layout()
    return fig


def madrid_map():
    hoods = (
        listings.groupby("neighbourhood_cleansed")
        .agg(latitude=("latitude", "mean"), longitude=("longitude", "mean"),
             median_price=("price", "median"), n_listings=("price", "size"))
        .reset_index()
    )
    hoods = hoods[hoods["n_listings"] >= 10]
    fig = px.scatter_map(
        hoods, lat="latitude", lon="longitude",
        color="median_price", size="n_listings", size_max=38,
        hover_name="neighbourhood_cleansed",
        hover_data={"median_price": ":.0f", "n_listings": True, "latitude": False, "longitude": False},
        color_continuous_scale=[PRIMARY_LIGHT, PRIMARY, PRIMARY_DARK, "#5c0a12"],
        range_color=(hoods["median_price"].quantile(0.05), hoods["median_price"].quantile(0.95)),
        center={"lat": 40.4168, "lon": -3.7038}, zoom=10.6,
        map_style="carto-positron",
        labels={"median_price": "Median €/night", "n_listings": "Listings"},
    )
    fig.update_traces(marker={"opacity": 0.85})
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=600,
                      paper_bgcolor="white", font_color=INK)
    return fig


def show_distribution(column):
    values = listings[column].dropna()
    values = values[values <= values.quantile(0.99)]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.hist(values, bins=50, color=PRIMARY, edgecolor="white", linewidth=0.4)
    ax.set_xlabel(column)
    ax.set_ylabel("Listings")
    ax.set_title(f"{column}  (top 1% clipped)")
    fig.tight_layout()
    return fig


def card(body, title="", accent=PRIMARY, tinted=False):
    background = "#fff5f5" if tinted else "#ffffff"
    heading = f'<div class="card-title" style="color:{accent}">{title}</div>' if title else ""
    return (f'<div class="card" style="border-left:4px solid {accent};background:{background}">'
            f'{heading}<div class="card-body">{body}</div></div>')


def tab_header(title, body):
    return card(f"<p>{body}</p>", title=title)


def takeaway(title, body):
    return card(f"<p>{body}</p>", title=title, accent=PRIMARY_DARK, tinted=True)


def build_kpi_cards():
    tiles = ""
    for name in model_names:
        m = metrics[name]
        color = MODEL_COLORS[name]
        is_best = name == best_model
        badge = '<span class="best-badge">&#9733; Best model</span>' if is_best else ""
        tiles += f"""
        <div class="kpi{' kpi-best' if is_best else ''}" style="border-top:4px solid {color}">
          {badge}
          <div class="kpi-name" style="color:{color}">{name}</div>
          <div class="kpi-value">€{m['RMSE']:.1f}</div>
          <div class="kpi-label">RMSE per night</div>
          <div class="kpi-value-sm">€{m['MAE']:.1f}</div>
          <div class="kpi-label">MAE per night</div>
        </div>"""
    note = card(
        "<b>RMSE</b> punishes large misses more heavily, because errors are squared before averaging. "
        "<b>MAE</b> is the plain average mistake in €/night, the number a host would understand. "
        f"<b>{best_model}</b> has the lowest RMSE of the three learned models on the held-out test set.",
        accent=INK,
    )
    return f'<div class="kpi-row">{tiles}</div>{note}'


PAIR_TEXT = {
    frozenset({"Scikit-learn Ridge", "Manual PyTorch"}): (
        "Both fit the same linear model to log-price, but not quite the same objective. "
        f"scikit-learn's Ridge adds a small L2 penalty on the coefficients (alpha = {sklearn_model.alpha:g}, "
        "picked on the validation set) and solves it exactly. The manual PyTorch loop minimises plain "
        "mean squared error, with no penalty, by gradient descent with a fixed step size for 500 epochs. "
        "With a penalty this weak the two objectives have almost the same minimum, so a fully converged "
        "loop would land very close to Ridge. Most of the gap you see comes from the loop stopping "
        "before it reaches that minimum, not from a difference in the model."
    ),
    frozenset({"Scikit-learn Ridge", "Standard PyTorch"}): (
        "scikit-learn computes the Ridge solution exactly: no randomness, no iteration count to tune. "
        "Standard PyTorch minimises plain mean squared error, with no penalty term, using Adam, an "
        "adaptive optimizer that adjusts the effective learning rate per parameter based on recent "
        "gradients. The problem has a single minimum, so Adam isn't heading somewhere different; in "
        "500 epochs it simply doesn't get all the way there. A small gap is normal; a large one means "
        "Adam needed more epochs or a better learning rate."
    ),
    frozenset({"Manual PyTorch", "Standard PyTorch"}): (
        "Both are gradient descent, but they differ in how they take each step. The manual "
        "loop uses a single fixed learning rate for every parameter at every epoch — simple "
        "but sensitive to that one choice. Adam adapts: it tracks a running estimate of the "
        "gradient's mean and variance per parameter, scaling updates so that rarely-touched "
        "weights can still make progress. On tabular data with standardised features the "
        "advantage of Adam is modest, which is why these two should produce similar but not "
        "identical results. A larger gap usually means the manual loop's fixed learning rate "
        "was a poor match for this particular loss landscape."
    ),
    frozenset({"Naive baseline", "Scikit-learn Ridge"}): (
        "The naive baseline predicts the same price for every single listing: the training-set "
        f"median of log-price, about €{np.expm1(baseline['median_log_price']):.0f}/night. "
        "It ignores location, size, amenities and recency of reviews entirely. Ridge regression "
        "uses all of those signals. The gap between them is the raw value added by the feature "
        "engineering: every euro of RMSE improvement over the baseline is a euro that better "
        "features are responsible for. If this gap were small, the conclusion would be brutal — "
        "the features aren't adding much — but in practice the model should pull well ahead."
    ),
    frozenset({"Naive baseline", "Manual PyTorch"}): (
        "Comparing the naive median predictor to a gradient-descent linear model shows how much "
        "iterative optimisation over real features buys you. The naive baseline has zero "
        "parameters to learn — it just computes a constant. The manual loop fits a weight "
        "for each engineered feature. The delta between them is the payoff from all the "
        "preprocessing and feature work upstream of training."
    ),
    frozenset({"Naive baseline", "Standard PyTorch"}): (
        "The naive baseline is a sanity-check floor: any model that knows nothing about a "
        "listing except that it exists in Madrid should beat it easily. Standard PyTorch with "
        "Adam and engineered features should clear the baseline by a wide margin. If it "
        "doesn't, look upstream — the features, the preprocessing, or the train/test split "
        "are the likely culprits, not the optimizer."
    ),
}


def build_comparison(model_a, model_b):
    if model_a == model_b:
        return card("Pick two different models to see how they compare.", title="Model comparison")
    ma, mb = metrics[model_a], metrics[model_b]
    winner = model_a if ma["RMSE"] < mb["RMSE"] else model_b
    pct = abs(ma["RMSE"] - mb["RMSE"]) / max(ma["RMSE"], mb["RMSE"]) * 100

    def pill(name, m):
        return (f'<span class="pill" style="background:{MODEL_COLORS[name]}">{name}</span>'
                f'<span class="pill-stats">RMSE <b>€{m["RMSE"]:.1f}</b> · MAE <b>€{m["MAE"]:.1f}</b></span>')

    summary = (f'<div class="cmp-row">{pill(model_a, ma)}{pill(model_b, mb)}'
               f'<span class="cmp-verdict"><b>{winner}</b> is {pct:.1f}% better on RMSE</span></div>')
    return card(summary + f"<p>{PAIR_TEXT[frozenset({model_a, model_b})]}</p>", title="Model comparison")


RESIDUAL_INTERPRETATION = takeaway("How to read this chart", """
Each dot is a listing from the test set. The horizontal axis is the real price the host charged; the vertical axis shows how far the model was off — positive means it guessed too high, negative means too low. A perfect model would have every dot sitting exactly on the zero line.</p><p>
What you'll almost certainly see here is a <b>fan shape</b>: the cloud of dots gets taller as price increases. That's not a bug — it's the model telling you it's much more comfortable predicting a €60 apartment than a €400 penthouse. Expensive listings are rare in the training data and carry features (rooftop terrace, designer interiors, prime location) that don't translate cleanly into a linear combination of numbers. The model does its best, but it's essentially extrapolating.</p><p>
You may also notice the dots drift slightly below zero as price rises — the model systematically under-shoots for expensive listings. This is a classic artefact of linear regression: the high-end outliers pull the fitted line upward during training, but there aren't enough of them to overcome the gravitational pull of the bulk of listings clustered at lower prices.
""")

ERROR_INTERPRETATION = takeaway("How to read this chart", """
This histogram shows all the individual prediction mistakes the model made on the test set — one bar per range of error, in €/night. Positive errors mean the model thought a listing was worth more than it actually was; negative errors mean it undershot.</p><p>
Ideally you want this to look like a bell curve centred tightly on zero: most errors small, symmetrically distributed, tailing off quickly in both directions. The wider and more lopsided the shape, the less reliable the model's predictions are in practice.</p><p>
The <b>dark red line</b> is the average error. If it sits noticeably to the left of zero, the model has a habit of under-predicting — common when training data is dominated by mid-range listings and the model "plays it safe." A line to the right means it tends to over-shoot. Either way, a consistent bias in one direction is more worrying than random scatter, because it means the model is systematically wrong rather than just noisy.
""")


THEME_COLORS = {
    "body_background_fill": BG,
    "body_text_color": TEXT,
    "body_text_color_subdued": INK_MUTED,
    "background_fill_primary": "#ffffff",
    "background_fill_secondary": PRIMARY_LIGHT,
    "block_background_fill": "#ffffff",
    "block_border_color": GRID,
    "block_label_background_fill": "#ffffff",
    "block_label_text_color": INK,
    "block_title_text_color": INK,
    "border_color_primary": GRID,
    "input_background_fill": "#ffffff",
    "input_border_color": "#dddddd",
    "color_accent_soft": PRIMARY_LIGHT,
}
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.red,
    neutral_hue=gr.themes.colors.gray,
    font=[gr.themes.GoogleFont("Nunito"), "Helvetica Neue", "Arial", "sans-serif"],
    radius_size=gr.themes.sizes.radius_lg,
).set(
    **THEME_COLORS,
    **{f"{name}_dark": value for name, value in THEME_COLORS.items()},
    color_accent=PRIMARY,
    block_shadow="0 1px 4px rgba(0,0,0,.06)",
)

PIN_SVG = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120' "
    "viewBox='0 0 100 100'%3E%3Cpath fill='%23FF5A5F' fill-opacity='0.045' "
    "d='M50 5C30.67 5 15 20.67 15 40c0 12.21 5.86 23.04 14.9 29.9L50 97 "
    "70.1 69.9C79.14 63.04 85 52.21 85 40 85 20.67 69.33 5 50 5z"
    "m0 46c-8.84 0-16-7.16-16-16s7.16-16 16-16 16 7.16 16 16-7.16 16-16 16z"
    "'/%3E%3C/svg%3E"
)

CSS = f"""
body, .gradio-container {{
    background-color: {BG} !important;
    background-image: url("{PIN_SVG}") !important;
    background-size: 120px 120px !important;
}}
.gradio-container {{ max-width: 100% !important; width: 100% !important; }}

.hero {{
    background: linear-gradient(135deg, {PRIMARY_DARK} 0%, {PRIMARY} 100%);
    border-radius: 16px; padding: 28px 32px;
}}
.hero h1 {{ margin: 0 0 8px 0; font-size: 26px; font-weight: 800; color: #ffffff; }}
.hero p  {{ margin: 0; font-size: 14px; line-height: 1.6; color: #ffffff; }}

.tab-container {{ gap: 8px; height: auto !important; padding: 6px 0; }}
.tab-container::after {{ display: none !important; }}
.tab-container button {{
    background: #ffffff !important;
    color: {INK} !important;
    border: 1px solid #e5e5e5 !important;
    border-radius: 999px !important;
    padding: 8px 20px !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    height: auto !important;
}}
.tab-container button:hover:not(.selected) {{
    background: {PRIMARY_LIGHT} !important;
    color: {PRIMARY_DARK} !important;
    border-color: {PRIMARY} !important;
}}
.tab-container button.selected {{
    background: {PRIMARY} !important;
    color: #ffffff !important;
    border-color: {PRIMARY} !important;
}}
.tab-container button.selected::after {{ display: none !important; }}

.card {{
    border-radius: 12px; padding: 16px 20px; margin: 4px 0;
    border: 1px solid {GRID}; box-shadow: 0 1px 4px rgba(0,0,0,.06);
}}
.card-title {{ font-size: 15px; font-weight: 800; margin-bottom: 6px; }}
.card-body, .card-body p {{ font-size: 13.5px; line-height: 1.7; color: {TEXT}; margin: 0 0 8px 0; }}
.card-body p:last-child {{ margin-bottom: 0; }}
.card-body b {{ color: #111111; }}

.kpi-row {{ display: flex; flex-wrap: wrap; gap: 14px; margin-bottom: 10px; }}
.kpi {{
    flex: 1; min-width: 200px; position: relative;
    background: #ffffff; border-radius: 12px; padding: 18px 22px;
    border: 1px solid {GRID}; box-shadow: 0 1px 4px rgba(0,0,0,.06);
}}
.kpi-best {{ box-shadow: 0 0 0 2px {PRIMARY}, 0 6px 18px rgba(255,90,95,.22); }}
.best-badge {{
    position: absolute; top: 12px; right: 12px;
    background: {PRIMARY}; color: #ffffff; font-size: 11px; font-weight: 800;
    padding: 3px 10px; border-radius: 999px;
}}
.kpi-name  {{ font-size: 12px; font-weight: 800; text-transform: uppercase;
              letter-spacing: .06em; margin-bottom: 10px; }}
.kpi-value {{ font-size: 30px; font-weight: 800; color: {TEXT}; line-height: 1; }}
.kpi-value-sm {{ font-size: 20px; font-weight: 700; color: {TEXT}; line-height: 1; margin-top: 14px; }}
.kpi-label {{ font-size: 11px; font-weight: 700; color: {INK_MUTED}; text-transform: uppercase;
              letter-spacing: .05em; margin-top: 4px; }}

.cmp-row {{ display: flex; flex-wrap: wrap; align-items: center; gap: 10px 18px; margin-bottom: 10px; }}
.pill {{ color: #ffffff; font-size: 12px; font-weight: 800; padding: 4px 12px;
         border-radius: 999px; margin-right: 6px; }}
.pill-stats {{ font-size: 13px; color: {TEXT}; }}
.cmp-verdict {{ font-size: 13px; color: {TEXT}; background: {PRIMARY_LIGHT};
                padding: 4px 12px; border-radius: 999px; }}

.section-title {{ font-size: 12px; font-weight: 800; color: {INK}; text-transform: uppercase;
                  letter-spacing: .08em; margin: 6px 0 2px 2px; }}
"""

FORCE_LIGHT_JS = """
() => {
  const url = new URL(window.location.href);
  if (url.searchParams.get('__theme') !== 'light') {
    url.searchParams.set('__theme', 'light');
    window.location.replace(url.href);
  }
}
"""


def comparison_tab(plot_fn, default_a, default_b, header, footer_cards):
    gr.HTML(header)
    with gr.Row(equal_height=True):
        with gr.Column():
            dropdown_a = gr.Dropdown(model_names, value=default_a, label="Model A")
            plot_a = gr.Plot(show_label=False)
        with gr.Column():
            dropdown_b = gr.Dropdown(model_names, value=default_b, label="Model B")
            plot_b = gr.Plot(show_label=False)
    comparison_html = gr.HTML()
    for footer in footer_cards:
        gr.HTML(footer)

    def update(a, b):
        return plot_fn(a), plot_fn(b), build_comparison(a, b)

    outputs = [plot_a, plot_b, comparison_html]
    dropdown_a.change(update, [dropdown_a, dropdown_b], outputs)
    dropdown_b.change(update, [dropdown_a, dropdown_b], outputs)
    demo.load(update, [dropdown_a, dropdown_b], outputs)


with gr.Blocks(title="Madrid Airbnb Price Prediction", fill_width=True) as demo:
    gr.HTML("""
    <div class="hero">
      <h1>Madrid Airbnb &mdash; Price Prediction Dashboard</h1>
      <p>Hosts set ask prices. The market sets real prices.<br>
         We model only listings with a review in the last 12&nbsp;months &mdash; actual bookings,
         not wishful ask prices &mdash; as the honest signal of what guests are willing to pay.</p>
    </div>""")

    with gr.Tab("Dashboard"):
        gr.HTML('<div class="section-title">Test-set performance</div>')
        gr.HTML(build_kpi_cards())
        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                gr.HTML('<div class="section-title">What drives the price</div>')
                features_plot = gr.Plot(show_label=False)
            with gr.Column(scale=3):
                gr.HTML('<div class="section-title">Where prices are highest in Madrid</div>')
                map_plot = gr.Plot(show_label=False)
        demo.load(top_features_plot, outputs=features_plot)
        demo.load(madrid_map, outputs=map_plot)

    with gr.Tab("Predicted vs Actual"):
        comparison_tab(
            pred_vs_actual, "Scikit-learn Ridge", "Manual PyTorch",
            tab_header(
                "How close is each model to the real booking price?",
                "Each point is a listing from the held-out test set. The dashed diagonal is perfect "
                "accuracy: points above it mean the model overshot, points below mean it undershot. "
                "Pick one model per panel to compare them side by side.",
            ),
            [takeaway(
                "What to look for",
                "All three learned models should sit clearly closer to the diagonal than the naive baseline. "
                "If two models look nearly identical, that's a good sign: both converged to the same answer "
                "despite different training routes. A model that looks noticeably worse probably didn't "
                "converge fully, so check its learning rate.",
            )],
        )

    with gr.Tab("Residuals"):
        comparison_tab(
            residuals_plot, "Scikit-learn Ridge", "Standard PyTorch",
            tab_header(
                "Where does each model get it wrong, and do they fail in the same places?",
                "A residual is the prediction error for a single listing. Plotting it against the true "
                "price reveals systematic blind spots. Use the two panels to check whether different "
                "training approaches share the same failure modes.",
            ),
            [RESIDUAL_INTERPRETATION, takeaway(
                "Business implication",
                "A property manager pricing a €300/night listing with this model should expect more "
                "uncertainty than for an €80/night one. The model is most reliable, and most useful, in "
                "the mid-market segment where recently reviewed listings are dense.",
            )],
        )

    with gr.Tab("Error Distribution"):
        comparison_tab(
            error_dist_plot, "Scikit-learn Ridge", "Naive baseline",
            tab_header(
                "What does a typical mistake look like, in euros?",
                "This histogram translates model error into plain money. Compare two models to see "
                "whether one is more tightly centred on zero, or whether one systematically over- or "
                "under-predicts.",
            ),
            [ERROR_INTERPRETATION, takeaway(
                "Rule of thumb",
                "The MAE on the dashboard cards is the number to quote to a host. "
                "\"This model is typically off by €X per night\" is the sentence that turns a model "
                "evaluation into a business conversation.",
            )],
        )

    with gr.Tab("Market Data Explorer"):
        gr.HTML(tab_header(
            "Explore the raw signals behind the model",
            "Distributions of every numeric variable, restricted to listings with at least one review "
            "in the last 12 months. That filter removes phantom listings that sit on the platform but "
            "have never attracted a real guest.",
        ))
        column_dropdown = gr.Dropdown(numeric_cols, value="price", label="Variable")
        distribution_plot = gr.Plot(show_label=False)
        gr.HTML(takeaway(
            "Where to start",
            "Start with <b>price</b>: the right-skewed shape is why we train on log(price), which "
            "compresses the long tail and keeps the regression well-behaved. Then check "
            "<b>reviews_per_month</b> and <b>number_of_reviews_ltm</b>, the demand signals that anchor "
            "the business framing.",
        ))
        column_dropdown.change(show_distribution, column_dropdown, distribution_plot)
        demo.load(show_distribution, column_dropdown, distribution_plot)


if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, js=FORCE_LIGHT_JS)
