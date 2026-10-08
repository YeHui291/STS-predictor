# -*- coding: utf-8 -*-
"""Cemented Tailings Backfill Splitting Tensile Strength (STS) Intelligent Prediction System - Streamlit GUI

Driven by the optimal model POA-NGB (NGBoost + Pelican Optimization Algorithm):
  Home / Single-point prediction (with 95% CI and posterior) / Batch prediction (Excel) / Model Info
Run:  streamlit run app.py   or   python -m sts_predictor
"""
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Ensure the package is importable regardless of launch directory
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from sts_predictor.config import load_config
from sts_predictor.core import get_predictor, find_train_data, DEPLOY_ROOT

cfg = load_config()
APP = cfg["app"]
MODEL_CFG = cfg["model"]
FEATURES = cfg["features"]
FEAT_BY_KEY = {f["key"]: f for f in FEATURES}

# Times New Roman throughout (matplotlib + streamlit CSS)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["pdf.fonttype"] = 42

# ============================================================
# Page config + Times New Roman academic style
# ============================================================
st.set_page_config(
    page_title=APP["title"],
    page_icon=APP["page_icon"],
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    * { font-family: 'Times New Roman', 'Liberation Serif', serif; }
    .stApp { max-width: 1300px; margin: 0 auto; padding: 20px; }
    .stButton>button {
        background-color: #1e3a5f; color: white; border-radius: 6px;
        padding: 12px 28px; font-weight: 600; border: none; font-size: 15px;
        font-family: 'Times New Roman', serif;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2c5282; transform: translateY(-1px);
        box-shadow: 0 3px 8px rgba(30,58,95,.35);
    }
    .stMetric {
        background-color: white; border-radius: 8px; padding: 14px;
        box-shadow: 0 2px 4px rgba(0,0,0,.06); border-left: 4px solid #3182ce;
    }
    .card {
        background-color: white; border-radius: 8px; padding: 22px;
        box-shadow: 0 2px 6px rgba(0,0,0,.06); margin-bottom: 18px;
        border: 1px solid #e2e8f0;
    }
    .highlight {
        background-color: #ebf8ff; border-left: 4px solid #3182ce;
        padding: 14px 20px; border-radius: 0 6px 6px 0; margin: 14px 0;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a202c 0%, #23304a 100%);
    }
    section[data-testid="stSidebar"] * { color: #e2e8f0; }
    h1, h2, h3, h4, h5, .stMarkdown, p, span, div, label {
        font-family: 'Times New Roman', 'Liberation Serif', serif;
    }

    /* ---- 主标题定向样式: 仅作用于首页大标题 ---- */
    h1#cemented-tailings-backfill-splitting-tensile-strength-intelligent-prediction-system-v1-0-0 {
        display: grid;
    }
    h1#cemented-tailings-backfill-splitting-tensile-strength-intelligent-prediction-system-v1-0-0 span {
        border-top-left-radius: 50px !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Cached loading
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model():
    return get_predictor()


@st.cache_data(show_spinner=False)
def train_target():
    """加载训练目标用于强度分位; 部署包缺数据时返回空数组, UI 自动隐藏该指标"""
    path = find_train_data()
    if not path:
        return np.array([], dtype=float)
    df = pd.read_excel(path)
    return df[MODEL_CFG["target"]].dropna().to_numpy(dtype=float)


with st.spinner("Loading POA-NGB optimal model ..."):
    predictor = load_model()
y_train = train_target()

# ============================================================
# Sidebar navigation
# ============================================================
st.title(f'{APP["page_icon"]} {APP["title"]}')
st.markdown(f"**{APP['subtitle']}**")

with st.sidebar:
    st.title("Navigation")
    page = st.radio("Select function",
                    ["Home", "Single-point Prediction", "Batch Prediction", "Model Info"])
    st.markdown("---")
    st.markdown(f"**Model**: {MODEL_CFG['name']}")
    st.markdown(f"**Version**: {APP['version']}")
    st.markdown("---")
    st.markdown("**Usage Tips**")
    st.markdown("- Single-point: manually enter 13 mix/curing parameters")
    st.markdown("- Batch: upload an Excel (.xlsx) with the same headers")
    st.markdown("- Results include 95% CI and can be exported as CSV")


# ============================================================
# Home
# ============================================================
if page == "Home":
    st.header("System Overview")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Core Features")
        st.write("- Intelligent prediction of **Splitting Tensile Strength (STS)** of CTB")
        st.write("- **POA-NGBoost** probabilistic model with 95% confidence interval")
        st.write("- Single-point manual input and Excel batch prediction")
        st.write("- One-click CSV export of results")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Input Parameters (13)")
        st.write("- Gradation: D10, Cu, Cc")
        st.write("- Chemical composition: CaO, Al2O3, MgO")
        st.write("- Mix & geometry: CT, CTR, D, H, Cw")
        st.write("- Curing conditions: temperature TEMP, age T")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="highlight">', unsafe_allow_html=True)
    st.subheader("Quick Start")
    st.write("1. Select 'Single-point Prediction' on the left, enter mix parameters and click predict;")
    st.write("2. Or select 'Batch Prediction' to upload an Excel for multi-sample prediction;")
    st.write("3. 'Model Info' shows model accuracy on training / test / external validation sets.")
    st.markdown('</div>', unsafe_allow_html=True)

    m = MODEL_CFG["metrics"]
    st.subheader("Model Accuracy Overview")
    cols = st.columns(3)
    for col, (tag, key) in zip(cols,
                               [("Training n=%d" % MODEL_CFG["train_samples"], "train"),
                                ("Independent Test n=%d" % MODEL_CFG["test_samples"], "test"),
                                ("External Validation n=%d" % MODEL_CFG["external_samples"], "external")]):
        col.metric(tag, f"R2 = {m[key]['R2']:.3f}",
                   f"RMSE {m[key]['RMSE_kPa']:.1f} kPa")
    st.caption("POA-NGB achieved the best overall performance among six candidate models on the external validation set, "
               "hence selected as the prediction engine of this system.")


# ============================================================
# Single-point prediction
# ============================================================
elif page == "Single-point Prediction":
    st.header("Single-point Splitting Tensile Strength Prediction")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Mix / Curing Parameter Input")

    groups = [FEATURES[0:5], FEATURES[5:9], FEATURES[9:13]]
    values = {}
    for group, col in zip(groups, st.columns(3)):
        with col:
            for f in group:
                label = f"{f['label']} ({f['key']})" if f["unit"] == "" \
                    else f"{f['label']} ({f['unit']})"
                if f.get("options"):
                    values[f["key"]] = st.selectbox(
                        label, options=f["options"],
                        index=f["options"].index(f["default"])
                        if f["default"] in f["options"] else 0,
                        help=f["help"])
                else:
                    values[f["key"]] = st.number_input(
                        label, min_value=float(f["min"]),
                        max_value=float(f["max"]),
                        value=float(f["default"]),
                        step=float(f["step"]), help=f["help"],
                        format="%.3f")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Start Prediction"):
        try:
            with st.spinner("Model inference ..."):
                res, samples = predictor.predict_one(values)
            pred, lo, hi = (res["Predicted STS (MPa)"],
                            res["CI Low (MPa)"], res["CI High (MPa)"])
            pct = float((y_train <= pred).mean() * 100.0)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Prediction Results")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Predicted STS", f"{pred:.3f} MPa", f"{pred*1000:.1f} kPa")
            k2.metric("95% CI Lower", f"{lo:.3f} MPa")
            k3.metric("95% CI Upper", f"{hi:.3f} MPa")
            if y_train.size > 0:
                k4.metric("Training Percentile", f"{pct:.0f}%")
            else:
                k4.metric("Training Percentile", "N/A")
            st.caption("CI from NGBoost predictive distribution (log1p-space 95% interval inverse-transformed via expm1); "
                       "percentile is the relative position of this prediction among the 458 training STS samples.")

            # ---- Posterior distribution ----
            st.subheader("Predictive Probability Distribution")
            fig, ax = plt.subplots(figsize=(8.6, 3.4))
            ax.hist(samples, bins=70, density=True, color="#3182ce",
                    alpha=0.28, edgecolor="none", label="Posterior samples")
            xs = np.linspace(np.quantile(samples, 0.001),
                             np.quantile(samples, 0.999), 400)
            try:
                ax.plot(xs, gaussian_kde(samples)(xs),
                        color="#1e3a5f", lw=2.2, label="Density")
            except Exception:
                pass
            ax.axvspan(lo, hi, color="#3182ce", alpha=0.13, label="95% CI")
            ax.axvline(pred, color="#c53030", lw=2.2, label="Prediction")
            ax.set_xlabel("STS (MPa)")
            ax.set_ylabel("Probability Density")
            ax.legend(frameon=False, fontsize=9, loc="upper right")
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ============================================================
# Batch prediction
# ============================================================
elif page == "Batch Prediction":
    st.header("Batch Splitting Tensile Strength Prediction")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Upload Excel Data")
    up = st.file_uploader("Upload an .xlsx file with the 13 feature columns", type=["xlsx"])
    st.caption("Required headers: " + ", ".join(FEAT_BY_KEY.keys())
               + " (D10(um) alias for D10(um) is accepted)")
    st.markdown('</div>', unsafe_allow_html=True)

    if up is not None:
        try:
            raw = pd.read_excel(up)
        except Exception as e:
            st.error(f"Failed to read Excel: {e}")
            st.stop()

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Data Preview (editable before prediction)")
        edited = st.data_editor(raw, use_container_width=True, num_rows="dynamic")

        if st.button("Start Batch Prediction"):
            try:
                with st.spinner(f"Predicting {len(edited)} samples ..."):
                    pred_df = predictor.predict(edited)
                out = pd.concat(
                    [edited.reset_index(drop=True),
                     pred_df.reset_index(drop=True)], axis=1)

                st.success(f"Completed {len(out)} predictions!")
                st.subheader("Prediction Results")
                st.dataframe(out, use_container_width=True)

                c1, c2, c3, c4 = st.columns(4)
                p = pred_df["Predicted STS (MPa)"]
                c1.metric("Sample Count", f"{len(p)}")
                c2.metric("Mean STS", f"{p.mean():.3f} MPa")
                c3.metric("Max STS", f"{p.max():.3f} MPa")
                c4.metric("Min STS", f"{p.min():.3f} MPa")

                # ---- Bar chart with CI error bars ----
                st.subheader("Per-sample Predictions (error bars = 95% CI)")
                fig, ax = plt.subplots(figsize=(9.2, 3.8))
                idx = np.arange(len(p))
                ax.bar(idx, p, color="#3182ce", alpha=0.75, width=0.62,
                       yerr=[p - pred_df["CI Low (MPa)"],
                             pred_df["CI High (MPa)"] - p],
                       capsize=3, ecolor="#1a202c", error_kw={"lw": 1})
                ax.set_xlabel("Sample Index")
                ax.set_ylabel("Predicted STS (MPa)")
                ax.set_xticks(idx[::max(1, len(idx) // 20)])
                ax.spines[["top", "right"]].set_visible(False)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                # ---- Export ----
                csv = out.to_csv(index=False).encode("utf-8-sig")
                st.download_button("Export Prediction Results (CSV)", csv,
                                   "STS_predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")
        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# Model Info
# ============================================================
elif page == "Model Info":
    st.header("Model Information")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Model Overview")
    st.write(f"**Model**: {MODEL_CFG['full_name']}")
    st.write(f"**Pipeline**: mean imputation + standardization (preprocessing) -> NGBRegressor")
    st.write(f"**Target variable**: {MODEL_CFG['target']}")
    st.write(f"**Target transform**: {MODEL_CFG['target_transform']}")
    st.write(f"**Data split**: {MODEL_CFG['split']}")
    try:
        model_rel = os.path.relpath(predictor.pkl_path, DEPLOY_ROOT)
    except ValueError:
        model_rel = predictor.pkl_path
    st.write(f"**Model file**: `{model_rel}`")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Performance on Three Datasets")
    m = MODEL_CFG["metrics"]
    perf = pd.DataFrame({
        "Dataset": ["Training", "Independent Test", "External Validation"],
        "Samples": [MODEL_CFG["train_samples"], MODEL_CFG["test_samples"],
                  MODEL_CFG["external_samples"]],
        "R2": [m["train"]["R2"], m["test"]["R2"], m["external"]["R2"]],
        "RMSE (kPa)": [m["train"]["RMSE_kPa"], m["test"]["RMSE_kPa"],
                       m["external"]["RMSE_kPa"]],
        "MAE (kPa)": [m["train"]["MAE_kPa"], m["test"]["MAE_kPa"],
                      m["external"]["MAE_kPa"]],
    })
    st.dataframe(perf, use_container_width=True, hide_index=True)

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bars = ax.bar(perf["Dataset"], perf["R2"],
                  color=["#63b3ed", "#2c5282", "#1a365d"], width=0.55)
    for b, v in zip(bars, perf["R2"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                ha="center", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(r"$R^2$")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Input Feature Ranges (based on 458 modeling samples)")
    ftab = pd.DataFrame([{
        "Column": f["key"],
        "Description": f["label"],
        "Unit": f["unit"] if f["unit"] else "-",
        "Min": f["min"], "Default/Median": f["default"], "Max": f["max"],
    } for f in FEATURES])
    st.dataframe(ftab, use_container_width=True, hide_index=True)
    st.caption("Both manual input and batch prediction feed the model with the same 13 columns in the exact training order.")
    st.markdown('</div>', unsafe_allow_html=True)
