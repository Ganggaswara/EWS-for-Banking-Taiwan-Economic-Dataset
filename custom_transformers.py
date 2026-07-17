"""
EWS Bank Distress Prediction — Streamlit App
Input manual fitur finansial penting -> prediksi probabilitas kebangkrutan dari 3 model.

Folder yang dibutuhkan di direktori yang sama:
  model_deploy/
    ├── logreg_pipeline.pkl
    ├── rf_pipeline.pkl
    ├── xgb_pipeline.pkl
    ├── feature_stats.csv
    ├── thresholds.json
    └── selected_features.json

Semua artefak ini dihasilkan dari Bagian 10 notebook UAS_final.ipynb.
"""

import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# WAJIB diimpor sebelum joblib.load — pickle butuh class ini importable
# dari module yang sama persis dengan saat pipeline di-pickle di notebook.
from custom_transformers import MedianImputer, IQRCapper, CorrelationFilter, ImportanceSelector  # noqa: F401

# --------------------------------------------------------------------------
# Konfigurasi halaman
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="EWS Prediksi Kebangkrutan Bank",
    page_icon="🏦",
    layout="wide",
)

MODEL_DIR = "model_deploy"


@st.cache_resource
def load_artifacts():
    models = {
        "Logistic Regression": joblib.load(f"{MODEL_DIR}/logreg_pipeline.pkl"),
        "Random Forest": joblib.load(f"{MODEL_DIR}/rf_pipeline.pkl"),
        "XGBoost": joblib.load(f"{MODEL_DIR}/xgb_pipeline.pkl"),
    }
    feature_stats = pd.read_csv(f"{MODEL_DIR}/feature_stats.csv", index_col=0)
    with open(f"{MODEL_DIR}/thresholds.json") as f:
        thresholds = json.load(f)
    with open(f"{MODEL_DIR}/selected_features.json") as f:
        selected_features = json.load(f)
    return models, feature_stats, thresholds, selected_features


try:
    models, feature_stats, thresholds, selected_features = load_artifacts()
    artifacts_ok = True
except FileNotFoundError as e:
    artifacts_ok = False
    missing_error = str(e)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🏦 Early Warning System — Prediksi Kebangkrutan Bank")
st.caption(
    "Perbandingan Logistic Regression, Random Forest, dan XGBoost berdasarkan "
    f"{len(selected_features) if artifacts_ok else '11'} rasio keuangan hasil reduksi dimensi "
    "(permutation importance)."
)

if not artifacts_ok:
    st.error(
        "Artefak model belum ditemukan. Pastikan folder `model_deploy/` "
        "(hasil ekspor dari Bagian 10 notebook UAS_final.ipynb) ada di direktori yang sama "
        f"dengan `app.py`.\n\nDetail error: {missing_error}"
    )
    st.stop()

# --------------------------------------------------------------------------
# Sidebar — form input manual
# --------------------------------------------------------------------------
st.sidebar.header("📋 Input Rasio Keuangan")
st.sidebar.caption(
    "Nilai default diisi otomatis dengan median dari data training. "
    "Ubah sesuai laporan keuangan bank yang ingin diperiksa."
)

input_values = {}
with st.sidebar.form("input_form"):
    for feat in selected_features:
        stats = feature_stats.loc[feat]
        input_values[feat] = st.number_input(
            label=feat,
            min_value=float(stats["min"]),
            max_value=float(stats["max"]),
            value=float(stats["median"]),
            format="%.6f",
            help=f"Rentang training: {stats['min']:.4f} s.d. {stats['max']:.4f} "
                 f"(rata-rata: {stats['mean']:.4f})",
        )
    submitted = st.form_submit_button("🔍 Prediksi", use_container_width=True)

# --------------------------------------------------------------------------
# Prediksi
# --------------------------------------------------------------------------
if submitted:
    X_input = pd.DataFrame([input_values])[selected_features]

    st.subheader("Hasil Prediksi Ketiga Model")

    cols = st.columns(3)
    summary_rows = []

    for i, (name, pipe) in enumerate(models.items()):
        prob_bangkrut = pipe.predict_proba(X_input)[0, 1]
        threshold = thresholds[name]
        status = "⚠️ BERISIKO BANGKRUT" if prob_bangkrut >= threshold else "✅ SEHAT"
        color = "#d62728" if prob_bangkrut >= threshold else "#2ca02c"

        with cols[i]:
            st.markdown(f"#### {name}")
            st.metric("Probabilitas Bangkrut", f"{prob_bangkrut:.2%}")
            st.markdown(
                f"<span style='color:{color}; font-weight:bold;'>{status}</span>",
                unsafe_allow_html=True,
            )
            st.caption(f"Threshold keputusan (optimal F2-score): {threshold:.3f}")

        summary_rows.append(
            {"Model": name, "Probabilitas Bangkrut": prob_bangkrut,
             "Threshold": threshold, "Status": status}
        )

    st.divider()

    # Bar chart perbandingan probabilitas antar model
    summary_df = pd.DataFrame(summary_rows)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=summary_df["Model"],
        y=summary_df["Probabilitas Bangkrut"],
        marker_color=["#d62728" if p >= t else "#2ca02c"
                      for p, t in zip(summary_df["Probabilitas Bangkrut"], summary_df["Threshold"])],
        text=[f"{p:.1%}" for p in summary_df["Probabilitas Bangkrut"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Perbandingan Probabilitas Kebangkrutan Antar Model",
        yaxis_title="Probabilitas",
        yaxis_range=[0, 1],
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Lihat detail input yang dikirim ke model"):
        st.dataframe(X_input.T.rename(columns={0: "Nilai Input"}), use_container_width=True)

    st.info(
        "**Catatan interpretasi:** threshold keputusan tiap model bukan 0.5 default, "
        "melainkan hasil optimasi F2-score dari notebook riset (memprioritaskan Recall, "
        "karena melewatkan bank yang benar-benar akan bangkrut jauh lebih mahal daripada "
        "false alarm dalam konteks Early Warning System)."
    )
else:
    st.info("Isi form di sidebar kiri, lalu klik **Prediksi** untuk melihat hasil ketiga model.")

    with st.expander("ℹ️ Tentang 11 fitur yang digunakan"):
        st.write(
            "Fitur-fitur ini dipilih lewat *permutation importance* (bukan seleksi manual) "
            "di notebook riset, sebagai rasio keuangan yang secara kumulatif menyumbang 90% "
            "kepentingan prediksi terhadap kebangkrutan bank."
        )
        st.dataframe(feature_stats.loc[selected_features], use_container_width=True)

st.divider()
st.caption(
    "Model dilatih ulang di ruang fitur tereduksi (setelah CorrelationFilter + "
    "ImportanceSelector) agar konsisten dengan hasil ablation study pada notebook riset. "
    "Aplikasi ini adalah simulasi akademik untuk keperluan skripsi, bukan alat keputusan "
    "regulasi resmi."
)
