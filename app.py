from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import folium
import requests
from streamlit_folium import st_folium

APP_DIR = Path(__file__).resolve().parent
# Root proyek PSD (folder berisi outputs/, data/, dll.).
# Atur PSD_PROJECT_ROOT jika aplikasi diletakkan di folder terpisah.
ROOT = Path(os.environ.get("PSD_PROJECT_ROOT", APP_DIR)).expanduser().resolve()
OUT = ROOT / "outputs"
TEMP_DIR = OUT / "sentinel2_temporal_composite_v6"
MODEL_DIR = OUT / "modeling_v6_baseline"
SAMPLE_GEOJSON = OUT / "dataset_v6_review" / "dataset_6kelas_v6_kandidat_representatif_303.geojson"
RASTER_DIR = OUT / "classification_raster_v6_aoi40km_offshore"
RASTER_ASSETS = {
    "KNN": {
        "preview": RASTER_DIR / "classification_KNN_quality_screened_aoi40km_offshore_preview.png",
        "metadata": RASTER_DIR / "classification_KNN_quality_screened_aoi40km_offshore_metadata.json",
        "area_summary": RASTER_DIR / "summary_area_prediksi_KNN_quality_screened_aoi40km.csv",
    },
    "SVM_RBF": {
        "preview": RASTER_DIR / "classification_SVM_RBF_quality_screened_aoi40km_offshore_preview.png",
        "metadata": RASTER_DIR / "classification_SVM_RBF_quality_screened_aoi40km_offshore_metadata.json",
        "area_summary": RASTER_DIR / "summary_area_prediksi_SVM_RBF_quality_screened_aoi40km.csv",
    },
}
COMPOSITE_CSV = TEMP_DIR / "fitur_sentinel2a_l2a_v6_temporal_2026-09-01_2026-10-10.csv"
DATASETS = {
    "Semua sampel (303)": TEMP_DIR / "dataset_modeling_v6_all_reviewed.csv",
    "Tersaring kualitas (255)": TEMP_DIR / "dataset_modeling_v6_quality_screened.csv",
}
VARIANT_KEYS = {
    "Semua sampel (303)": "all_reviewed",
    "Tersaring kualitas (255)": "quality_screened",
}
CLASSES = ["sawah", "bangunan", "mangrove", "lahan_hijau", "laut", "danau"]
CLASS_LABELS = {
    "sawah": "Sawah",
    "bangunan": "Bangunan",
    "mangrove": "Mangrove",
    "lahan_hijau": "Lahan hijau",
    "laut": "Perairan terbuka / laut",
    "danau": "Danau / ranu",
}
COLORS = {
    "sawah": "#D6A23A",
    "bangunan": "#E76F51",
    "mangrove": "#287271",
    "lahan_hijau": "#6A994E",
    "laut": "#277DA1",
    "danau": "#7B61A8",
}
FEATURES = [
    "B01_mean", "B02_mean", "B03_mean", "B04_mean", "B05_mean", "B06_mean",
    "B07_mean", "B08_mean", "B8A_mean", "B09_mean", "B11_mean", "B12_mean",
    "NDVI_mean", "NDWI_mean", "MNDWI_mean", "NDBI_mean",
]
BANDS = [
    ("B01", "Coastal aerosol", "60 m", "Aerosol pesisir dan koreksi atmosfer"),
    ("B02", "Blue", "10 m", "Kanal biru; air, atmosfer, dan komposit warna"),
    ("B03", "Green", "10 m", "Kanal hijau; kontras vegetasi-air dan indeks NDWI/MNDWI"),
    ("B04", "Red", "10 m", "Kanal merah; penyerapan klorofil dan masukan NDVI"),
    ("B05", "Red Edge 1", "20 m", "Tepi merah; respons spektral vegetasi"),
    ("B06", "Red Edge 2", "20 m", "Tepi merah; perubahan karakteristik vegetasi"),
    ("B07", "Red Edge 3", "20 m", "Tepi merah; karakteristik kanopi/vegetasi"),
    ("B08", "Near Infrared (NIR)", "10 m", "Pantulan NIR; vegetasi, biomassa relatif, dan masukan indeks"),
    ("B8A", "Narrow NIR", "20 m", "Kanal NIR sempit; karakteristik vegetasi"),
    ("B09", "Water vapour", "60 m", "Informasi uap air atmosfer"),
    ("B10", "Cirrus", "60 m", "Deteksi awan tipis/cirrus; tidak termasuk fitur reflektansi permukaan L2A yang dipakai"),
    ("B11", "SWIR 1", "20 m", "Kelembapan permukaan/vegetasi, material, masukan MNDWI/NDBI"),
    ("B12", "SWIR 2", "20 m", "Karakteristik kelembapan dan material permukaan"),
]
FEATURE_DETAILS = [
    ("B01_mean", "Rata-rata reflektansi", "Rata-rata nilai reflektansi B01 pada piksel valid di dalam poligon."),
    ("B02_mean", "Rata-rata reflektansi", "Rata-rata nilai reflektansi B02 pada piksel valid di dalam poligon."),
    ("B03_mean", "Rata-rata reflektansi", "Rata-rata nilai reflektansi B03 pada piksel valid di dalam poligon."),
    ("B04_mean", "Rata-rata reflektansi", "Rata-rata nilai reflektansi B04 pada piksel valid di dalam poligon."),
    ("B05_mean", "Rata-rata reflektansi", "Rata-rata nilai reflektansi B05 pada piksel valid di dalam poligon."),
    ("B06_mean", "Rata-rata reflektansi", "Rata-rata nilai reflektansi B06 pada piksel valid di dalam poligon."),
    ("B07_mean", "Rata-rata reflektansi", "Rata-rata nilai reflektansi B07 pada piksel valid di dalam poligon."),
    ("B08_mean", "Rata-rata reflektansi", "Rata-rata nilai reflektansi B08 pada piksel valid di dalam poligon."),
    ("B8A_mean", "Rata-rata reflektansi", "Rata-rata nilai reflektansi B8A pada piksel valid di dalam poligon."),
    ("B09_mean", "Rata-rata reflektansi", "Rata-rata nilai reflektansi B09 pada piksel valid di dalam poligon."),
    ("B11_mean", "Rata-rata reflektansi", "Rata-rata nilai reflektansi B11 pada piksel valid di dalam poligon."),
    ("B12_mean", "Rata-rata reflektansi", "Rata-rata nilai reflektansi B12 pada piksel valid di dalam poligon."),
    ("NDVI_mean", "(B08 − B04) / (B08 + B04)", "Indeks kontras NIR-merah yang umum digunakan untuk membantu mengenali vegetasi."),
    ("NDWI_mean", "(B03 − B08) / (B03 + B08)", "Indeks hijau-NIR untuk membantu analisis air/vegetasi."),
    ("MNDWI_mean", "(B03 − B11) / (B03 + B11)", "Indeks hijau-SWIR untuk membantu membedakan badan air dari permukaan lain."),
    ("NDBI_mean", "(B11 − B08) / (B11 + B08)", "Indeks SWIR-NIR yang dapat membantu mengidentifikasi kawasan terbangun."),
]
MODEL_LABELS = {
    "DecisionTree": "Decision Tree",
    "RandomForest": "Random Forest",
    "SVM_RBF": "SVM RBF",
    "KNN": "K-Nearest Neighbors (KNN)",
    "LogisticRegression": "Logistic Regression",
}

st.set_page_config(page_title="SIG Tutupan Lahan Jawa Timur | v6", page_icon="🗺️", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.3rem; padding-bottom: 2.6rem; max-width: 1500px;}
      .hero {padding: 1.4rem 1.6rem; border-radius: 18px; background: linear-gradient(110deg,#102a43,#1e6091); color: white; margin-bottom: 1rem;}
      .hero h1 {margin:0; font-size:2.0rem; color:white;}
      .hero p {margin:.5rem 0 0 0; color:#e6f2ff; font-size:1rem;}
      .small-note {font-size:.86rem; color:#667085;}
      div[data-testid="stMetric"] {border:1px solid #e5e7eb; padding:.75rem; border-radius:12px;}
    </style>
    """,
    unsafe_allow_html=True,
)


def read_csv_safe(path: Path, **kwargs: Any) -> pd.DataFrame | None:
    try:
        if path.exists():
            return pd.read_csv(path, **kwargs)
    except Exception as exc:  # dashboard should show a useful error, not crash
        st.warning(f"Tidak dapat membaca {path.relative_to(ROOT)}: {exc}")
    return None


def load_geojson(path: Path) -> dict[str, Any] | None:
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if data.get("type") == "FeatureCollection":
                return data
    except Exception as exc:
        st.warning(f"GeoJSON tidak dapat dibaca: {exc}")
    return None


def label_for(value: str) -> str:
    return CLASS_LABELS.get(str(value).strip().lower(), str(value))


def fmt_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (ValueError, TypeError):
        return "—"


def choose_path_message(paths: list[Path]) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        with st.expander("Berkas proyek yang belum ditemukan", expanded=True):
            st.write("Pastikan aplikasi disimpan di folder root `PSD`, bersama folder `outputs`. Berkas yang belum ditemukan:")
            for p in missing:
                try:
                    st.code(str(p.relative_to(ROOT)))
                except ValueError:
                    st.code(str(p))


@st.cache_data(ttl=1800, show_spinner=False)
def google_satellite_config(center: list[float], zoom: int) -> dict[str, str] | None:
    """Return official Google Map Tiles satellite session config when an API key is provided."""
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()
    if not api_key:
        try:
            api_key = str(st.secrets.get("GOOGLE_MAPS_API_KEY", "")).strip()
        except Exception:
            api_key = ""
    if not api_key:
        return None
    try:
        session_resp = requests.post(
            "https://tile.googleapis.com/v1/createSession",
            params={"key": api_key},
            json={"mapType": "satellite", "language": "id", "region": "ID"},
            timeout=8,
        )
        session_resp.raise_for_status()
        session_token = session_resp.json().get("session")
        if not session_token:
            raise ValueError("Response createSession tidak berisi session token.")
        lat, lon = float(center[0]), float(center[1])
        pad_lat = 1.2 if zoom >= 8 else 3.0
        pad_lon = 1.8 if zoom >= 8 else 4.0
        viewport_resp = requests.get(
            "https://tile.googleapis.com/tile/v1/viewport",
            params={
                "session": session_token, "key": api_key, "zoom": zoom,
                "north": min(85.0, lat + pad_lat), "south": max(-85.0, lat - pad_lat),
                "east": min(180.0, lon + pad_lon), "west": max(-180.0, lon - pad_lon),
            },
            timeout=8,
        )
        viewport_resp.raise_for_status()
        attribution = str(viewport_resp.json().get("copyright", "Google Maps"))
        tile_url = f"https://tile.googleapis.com/v1/2dtiles/{{z}}/{{x}}/{{y}}?session={session_token}&key={api_key}"
        return {"tiles": tile_url, "attr": f"Google Maps | {attribution}"}
    except Exception as exc:
        st.warning(f"Google Maps Satellite tidak dapat diinisialisasi; memakai Esri sebagai fallback. Detail: {exc}")
        return None


st.markdown(
    "<div class='hero'><h1>SIG Klasifikasi Tutupan Lahan Jawa Timur</h1>"
    "<p>Dashboard data understanding, fitur Sentinel-2A, eksperimen model, dan sebaran sampel spasial — versi UTS v6.</p></div>",
    unsafe_allow_html=True,
)
st.caption("Periode komposit: 1 September–10 Oktober 2026 • 6 kelas • 12 band reflektansi + 4 indeks spektral")

with st.sidebar:
    st.header("Pengaturan tampilan")
    selected_variant_label = st.radio("Dataset eksperimen", list(DATASETS.keys()), index=1)
    selected_variant = VARIANT_KEYS[selected_variant_label]
    selected_model_label = st.selectbox("Model untuk detail", list(MODEL_LABELS.values()), index=3)
    selected_model = {v: k for k, v in MODEL_LABELS.items()}[selected_model_label]
    st.markdown("---")
    st.markdown("**Keterangan penting**")
    st.caption("Hasil yang ditampilkan adalah baseline dengan split acak terstratifikasi, bukan evaluasi spasial. Peta menampilkan poligon sampel dan, jika tersedia, raster prediksi pada AOI pesisir 40 × 40 km; bukan raster klasifikasi seluruh provinsi.")

selected_dataset_path = DATASETS[selected_variant_label]
model_out = MODEL_DIR / selected_variant
composite_df = read_csv_safe(COMPOSITE_CSV, dtype={"sample_id": str})
dataset_df = read_csv_safe(selected_dataset_path, dtype={"sample_id": str})
# Paket mandiri menyimpan hasil perbandingan per-varian di masing-masing subfolder.
# Baca tabel varian aktif secara langsung; file gabungan opsional tidak wajib ada.
comparison = read_csv_safe(model_out / "perbandingan_model.csv")
geojson = load_geojson(SAMPLE_GEOJSON)

# If a modeling dataset is present, it is the authoritative table for current experiment counts.
active_df = dataset_df if dataset_df is not None else composite_df
if active_df is not None and "class_name" in active_df.columns:
    active_df = active_df.copy()
    active_df["class_name"] = active_df["class_name"].astype(str).str.lower().str.strip()

# Tabs: dashboard, samples, models, map, methods/policy.
tab_overview, tab_data, tab_model, tab_map, tab_method = st.tabs(
    ["Ringkasan", "Data & Fitur", "Eksperimen Model", "Peta Sampel", "Metodologi & Kebijakan"]
)

with tab_overview:
    st.subheader("Ringkasan proyek")
    n_samples = int(len(active_df)) if active_df is not None else 0
    n_classes = int(active_df["class_name"].nunique()) if active_df is not None and "class_name" in active_df.columns else 0
    train_n = test_n = None
    split_df = read_csv_safe(model_out / "split_train_test.csv", dtype={"sample_id": str})
    if split_df is not None and "split" in split_df.columns:
        split_counts = split_df["split"].astype(str).str.lower().value_counts()
        train_n = int(split_counts.get("train", 0))
        test_n = int(split_counts.get("test", 0))
    if train_n is None:
        train_n, test_n = (242, 61) if selected_variant == "all_reviewed" else (204, 51)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah sampel", f"{n_samples:,}" if n_samples else "—")
    c2.metric("Jumlah kelas", str(n_classes) if n_classes else "6 (target)")
    c3.metric("Data training", f"{train_n:,}" if train_n is not None else "—")
    c4.metric("Data testing", f"{test_n:,}" if test_n is not None else "—")

    st.markdown("#### Distribusi sampel per kelas")
    if active_df is not None and "class_name" in active_df.columns:
        class_counts = active_df["class_name"].value_counts().reindex(CLASSES, fill_value=0).rename_axis("class_name").reset_index(name="Jumlah sampel")
        class_counts["Kelas"] = class_counts["class_name"].map(label_for)
        fig = px.bar(class_counts, x="Kelas", y="Jumlah sampel", color="class_name", color_discrete_map=COLORS,
                     category_orders={"Kelas": [label_for(x) for x in CLASSES]}, text="Jumlah sampel")
        fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=25, b=10), height=330, xaxis_title=None)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(class_counts[["Kelas", "Jumlah sampel"]], hide_index=True, width="stretch")
    else:
        st.info("Dataset model belum ditemukan. Letakkan folder `outputs` pada root proyek sesuai README.")

    st.markdown("#### Hasil baseline model")
    if comparison is not None and not comparison.empty:
        comp = comparison.copy()
        comp = comp[comp.get("dataset_variant", pd.Series(dtype=str)).astype(str).eq(selected_variant)] if "dataset_variant" in comp.columns else comp
        if "test_f1_macro" in comp.columns and "model" in comp.columns:
            comp["Model"] = comp["model"].map(MODEL_LABELS).fillna(comp["model"])
            comp["F1-macro CV"] = pd.to_numeric(comp.get("cv_f1_macro_mean"), errors="coerce")
            comp["F1-macro test"] = pd.to_numeric(comp.get("test_f1_macro"), errors="coerce")
            chart = comp.melt(id_vars=["Model"], value_vars=["F1-macro CV", "F1-macro test"], var_name="Evaluasi", value_name="F1-macro")
            fig = px.bar(chart, x="Model", y="F1-macro", color="Evaluasi", barmode="group", text_auto=".3f",
                         color_discrete_sequence=["#2A6F97", "#E09F3E"], range_y=[0, 1.05])
            fig.update_layout(height=370, margin=dict(l=10, r=10, t=25, b=10), legend_title=None)
            st.plotly_chart(fig, width="stretch")
            display_cols = [c for c in ["model", "n_samples", "n_train", "n_test", "cv_f1_macro_mean", "test_f1_macro", "test_balanced_accuracy", "test_accuracy"] if c in comp.columns]
            st.dataframe(comp[display_cols].rename(columns={"model":"Model", "n_samples":"Sampel", "n_train":"Train", "n_test":"Test", "cv_f1_macro_mean":"F1-macro CV", "test_f1_macro":"F1-macro test", "test_balanced_accuracy":"Balanced accuracy test", "test_accuracy":"Accuracy test"}), hide_index=True, width="stretch")
            if {"model", "test_accuracy", "test_f1_macro"}.issubset(comp.columns):
                rank = comp.copy()
                rank["test_accuracy"] = pd.to_numeric(rank["test_accuracy"], errors="coerce")
                rank["test_f1_macro"] = pd.to_numeric(rank["test_f1_macro"], errors="coerce")
                best_acc_row = rank.sort_values(["test_accuracy", "test_f1_macro"], ascending=False).iloc[0]
                best_f1_row = rank.sort_values(["test_f1_macro", "test_accuracy"], ascending=False).iloc[0]
                st.success(
                    f"Kinerja test terbaik pada varian ini: akurasi {MODEL_LABELS.get(str(best_acc_row['model']), str(best_acc_row['model']))} "
                    f"{float(best_acc_row['test_accuracy']):.2%}; F1-macro terbaik {MODEL_LABELS.get(str(best_f1_row['model']), str(best_f1_row['model']))} "
                    f"{float(best_f1_row['test_f1_macro']):.2%}."
                )
                st.caption("Skor berasal dari satu stratified random split; skor testing bukan akurasi setiap piksel raster dan belum mengukur generalisasi spasial.")
        else:
            st.warning("File perbandingan model ditemukan, tetapi kolom metrik yang diharapkan tidak lengkap.")
    else:
        st.info("Laporan model belum ditemukan. Jalankan `train_models_v6.py` atau salin folder output hasil training ke proyek.")

    st.info("Interpretasi: metrik ini berasal dari split acak terstratifikasi 80:20 dan CV 5-fold pada data latih. Evaluasi blok spasial belum berhasil karena sampel bangunan terkonsentrasi pada satu blok; jangan klaim skor ini mewakili seluruh Jawa Timur.")

with tab_data:
    st.subheader("Data understanding dan fitur")
    st.markdown("**Tujuan analisis:** membangun baseline klasifikasi enam kelas tutupan lahan/penutup permukaan di Jawa Timur dari fitur reflektansi Sentinel-2A Level-2A, membandingkan model, dan menyediakan antarmuka untuk meninjau sebaran sampel serta hasil eksperimen.")
    st.markdown("Tutupan lahan adalah kondisi biofisik yang terlihat pada permukaan. Penggunaan lahan merujuk pada fungsi atau pemanfaatannya. Citra multispektral terutama memberi bukti tentang tutupan; fungsi penggunaan lahan memerlukan konteks tambahan dan verifikasi.")

    if active_df is not None:
        st.markdown("#### Pengumpulan data (data collection)")
        st.markdown(
            "- **Sampel berlabel:** geometri poligon enam kelas dalam GeoJSON proyek; dataset kandidat awal 303 sampel dan varian tersaring kualitas 255 sampel. Label tetap mengikuti data sampel yang disediakan/dikurasi dan tidak otomatis menjadi ground truth independen.\n"
            "- **Batas referensi:** GeoJSON batas wilayah Jawa Timur yang disertakan pada proyek.\n"
            "- **Citra:** Sentinel-2 Level-2A yang diminta melalui Copernicus Data Space Ecosystem Process API untuk 1 September–10 Oktober 2026 (inklusif).\n"
            "- **Prakomposisi:** median temporal per piksel dengan `dataMask > 0` dan SCL kelas 4/5/6; sampel diringkas memakai piksel di dalam geometri poligon.\n"
            "- **Grid output:** 10 m menggunakan nearest-neighbour untuk menyelaraskan grid; penyelarasan grid tidak meningkatkan resolusi informasi band 20 m/60 m."
        )
        st.markdown("#### Sampel modeling")
        class_counts = active_df["class_name"].value_counts().reindex(CLASSES, fill_value=0) if "class_name" in active_df.columns else pd.Series(dtype=int)
        summary = pd.DataFrame({"class_name": CLASSES, "Kelas": [label_for(c) for c in CLASSES], "Jumlah sampel": [int(class_counts.get(c, 0)) for c in CLASSES]})
        st.dataframe(summary[["Kelas", "Jumlah sampel"]], hide_index=True, width="stretch")
        if "status" in active_df.columns:
            st.caption("Status ekstraksi pada dataset pilihan: " + ", ".join(f"{k}: {v}" for k, v in active_df["status"].value_counts().items()))
        class_split_path = model_out / "distribusi_kelas.csv"
        class_split = read_csv_safe(class_split_path)
        if class_split is not None and {"class_name", "all_count", "train_count", "test_count"}.issubset(class_split.columns):
            class_split["Kelas"] = class_split["class_name"].astype(str).str.lower().map(label_for)
            st.markdown("#### Rincian training dan testing per kelas")
            st.dataframe(class_split[["Kelas", "all_count", "train_count", "test_count"]].rename(columns={"all_count":"Total", "train_count":"Training", "test_count":"Testing"}), hide_index=True, width="stretch")

        st.markdown("#### Deskripsi 13 band Sentinel-2 MSI")
        band_df = pd.DataFrame(BANDS, columns=["Band", "Nama", "Resolusi asli", "Kegunaan umum"])
        st.dataframe(band_df, hide_index=True, width="stretch")
        st.caption("Sensor Sentinel-2 MSI memiliki 13 band. B10 (Cirrus, 60 m) dideskripsikan untuk memenuhi kebutuhan sensor, tetapi tidak menjadi fitur karena produk L2A reflektansi permukaan yang digunakan tidak menyertakan band B10. Model memakai 12 band reflektansi permukaan.")

        st.markdown("#### Fitur numerik model")
        st.write("12 fitur reflektansi rata-rata (B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12) ditambah empat indeks:")
        st.latex(r"NDVI = \frac{B08-B04}{B08+B04}")
        st.latex(r"NDWI = \frac{B03-B08}{B03+B08}")
        st.latex(r"MNDWI = \frac{B03-B11}{B03+B11}")
        st.latex(r"NDBI = \frac{B11-B08}{B11+B08}")
        st.caption("Indeks dihitung dari reflektansi komposit median temporal. Bila penyebut mendekati nol, nilai indeks diperlakukan sebagai tidak tersedia dan ditangani di pipeline.")
        feature_df = pd.DataFrame(FEATURE_DETAILS, columns=["Fitur", "Definisi / rumus", "Deskripsi"])
        st.markdown("#### Deskripsi seluruh 16 fitur model")
        st.dataframe(feature_df, hide_index=True, width="stretch")

        st.markdown("#### Pratinjau baris data")
        display_cols = [c for c in ["sample_id", "class_id", "class_name", "status", *FEATURES] if c in active_df.columns]
        search = st.text_input("Cari sample_id atau kelas", key="sample_search")
        preview = active_df[display_cols].copy()
        if search:
            mask = preview.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
            preview = preview[mask]
        st.dataframe(preview.head(100), hide_index=True, width="stretch")
    else:
        st.warning("File dataset modeling tidak ditemukan.")

with tab_model:
    st.subheader("Detail evaluasi model")
    report_path = model_out / f"classification_report_{selected_model}.csv"
    cm_path = model_out / f"confusion_matrix_{selected_model}.csv"
    pred_path = model_out / f"prediksi_test_{selected_model}.csv"
    report = read_csv_safe(report_path, index_col=0)
    cm = read_csv_safe(cm_path, index_col=0)
    predictions = read_csv_safe(pred_path, dtype={"sample_id": str})
    if report is not None:
        st.markdown(f"#### Classification report — {MODEL_LABELS[selected_model]}")
        report_show = report.copy()
        report_show.index.name = "Kelas / ringkasan"
        st.dataframe(report_show, width="stretch")
    else:
        st.info(f"Laporan belum ditemukan: {report_path.relative_to(ROOT)}")
    if cm is not None:
        st.markdown("#### Confusion matrix")
        cm.index = [label_for(x) for x in cm.index]
        cm.columns = [label_for(x) for x in cm.columns]
        st.dataframe(cm, width="stretch")
        try:
            z = cm.apply(pd.to_numeric, errors="coerce").fillna(0)
            fig = px.imshow(z, text_auto=True, aspect="auto", color_continuous_scale="Blues",
                            labels=dict(x="Prediksi", y="Label aktual", color="Jumlah"))
            fig.update_layout(height=440, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")
        except Exception as exc:
            st.caption(f"Grafik confusion matrix tidak tersedia: {exc}")
    if predictions is not None:
        st.markdown("#### Sampel data uji dan prediksi")
        if {"y_true", "y_pred"}.issubset(predictions.columns):
            predictions["hasil"] = np.where(predictions["y_true"].astype(str) == predictions["y_pred"].astype(str), "Benar", "Salah")
        st.dataframe(predictions, hide_index=True, width="stretch")

with tab_map:
    st.subheader("Peta interaktif sampel dan hasil prediksi")
    map_mode = st.radio(
        "Isi peta",
        ["Label sampel", "Prediksi sampel test"],
        horizontal=True,
        help="Prediksi sampel test mewarnai poligon hanya untuk sample_id pada berkas prediksi holdout; ini bukan klasifikasi per piksel seluruh provinsi.",
    )
    available_raster_models = [
        model_name for model_name in ("KNN", "SVM_RBF")
        if RASTER_ASSETS[model_name]["preview"].exists() and RASTER_ASSETS[model_name]["metadata"].exists()
    ]
    selected_raster_model = None
    if available_raster_models:
        default_raster_index = available_raster_models.index("KNN") if "KNN" in available_raster_models else 0
        selected_raster_model = st.selectbox(
            "Model raster yang ditampilkan", available_raster_models,
            index=default_raster_index,
            format_func=lambda value: MODEL_LABELS.get(value, value),
            help="KNN akan dipilih sebagai default apabila raster KNN sudah dibuat; raster SVM tetap tersedia sebagai pembanding.",
        )
        selected_raster_assets = RASTER_ASSETS[selected_raster_model]
        raster_preview = selected_raster_assets["preview"]
        raster_metadata = selected_raster_assets["metadata"]
        raster_area_summary = selected_raster_assets["area_summary"]
    else:
        selected_raster_assets = RASTER_ASSETS["SVM_RBF"]
        raster_preview = selected_raster_assets["preview"]
        raster_metadata = selected_raster_assets["metadata"]
        raster_area_summary = selected_raster_assets["area_summary"]
    raster_available = bool(available_raster_models)
    show_raster_overlay = st.checkbox(
        "Tampilkan raster prediksi per-piksel (AOI pesisir 40 × 40 km termasuk perairan)",
        value=bool(raster_available), disabled=not raster_available, key="show_aoi_raster_overlay",
        help="Overlay mencakup kotak AOI pesisir 40 × 40 km termasuk perairan sekitar; bukan seluruh provinsi.",
    )
    raster_meta = None
    if raster_available:
        try:
            raster_meta = json.loads(raster_metadata.read_text(encoding="utf-8"))
            if not raster_meta.get("complete_aoi", False):
                st.warning("Raster AOI belum lengkap: ada tile yang belum diproses. Jalankan ulang skrip klasifikasi tanpa --max-tiles.")
            st.caption(f"Model raster: {MODEL_LABELS.get(selected_raster_model, selected_raster_model)} • {raster_meta.get('processed_tiles', '?')}/{raster_meta.get('total_intersecting_tiles', '?')} tile. Cakupan hanya kotak AOI pesisir 40 × 40 km termasuk perairan sekitar, bukan seluruh Jawa Timur.")
        except Exception as exc:
            st.warning(f"Metadata raster tidak dapat dibaca: {exc}")
    else:
        st.info("Raster per-piksel offshore belum ditemukan pada paket aplikasi. Raster SVM sudah tersedia di paket proyek lokal; jalankan `python ./classify_raster_aoi_v6_offshore.py --model KNN` dari root PSD untuk membuat alternatif KNN, dengan kredensial Copernicus tersedia di `.env` lokal.")
    if map_mode == "Prediksi sampel test":
        pred_path_for_map = model_out / f"prediksi_test_{selected_model}.csv"
        preds_map = read_csv_safe(pred_path_for_map, dtype={"sample_id": str})
    else:
        preds_map = None

    if map_geojson := geojson:
        features_source = map_geojson.get("features", [])
        pred_lookup = {}
        if map_mode == "Prediksi sampel test":
            if preds_map is None or not {"sample_id", "y_true", "y_pred"}.issubset(preds_map.columns):
                st.error("Berkas prediksi test tidak ditemukan atau kolomnya tidak lengkap untuk model/varian ini.")
                features_source = []
            else:
                pred_lookup = {
                    str(r["sample_id"]): (str(r["y_true"]).strip().lower(), str(r["y_pred"]).strip().lower())
                    for _, r in preds_map.iterrows()
                }
                features_source = [f for f in features_source if str((f.get("properties") or {}).get("sample_id", "")) in pred_lookup]

        # Add display-only properties without modifying source GeoJSON.
        features_display = []
        for feature in features_source:
            clone_feature = dict(feature)
            props = dict(feature.get("properties") or {})
            sid = str(props.get("sample_id", ""))
            original_class = str(props.get("class_name", "")).strip().lower()
            if map_mode == "Prediksi sampel test":
                y_true, y_pred = pred_lookup[sid]
                props["actual_class_name"] = y_true
                props["predicted_class_name"] = y_pred
                props["display_class"] = y_pred
            else:
                props["actual_class_name"] = original_class
                props["predicted_class_name"] = ""
                props["display_class"] = original_class
            clone_feature["properties"] = props
            features_display.append(clone_feature)

        display_class_options = [c for c in CLASSES if any(str((f.get("properties") or {}).get("display_class", "")).lower() == c for f in features_display)]
        default_classes = display_class_options if display_class_options else CLASSES
        map_classes = st.multiselect("Tampilkan kelas", options=CLASSES, default=default_classes, format_func=label_for, key=f"map_class_filter_{map_mode}_{selected_variant}_{selected_model}")
        kept = [f for f in features_display if str((f.get("properties") or {}).get("display_class", "")).lower() in map_classes]

        if map_mode == "Prediksi sampel test":
            n_correct = sum(1 for f in features_display if (f.get("properties") or {}).get("actual_class_name") == (f.get("properties") or {}).get("predicted_class_name"))
            n_test_map = len(features_display)
            st.caption(f"Menampilkan poligon sampel pada holdout test untuk {MODEL_LABELS[selected_model]}: {n_correct}/{n_test_map} benar menurut label aktual pada berkas evaluasi. Hanya poligon data test yang ditampilkan.")
        else:
            st.caption("Menampilkan label kelas sampel GeoJSON. Status pada properti GeoJSON merupakan metadata lineage sumber dan tidak disajikan sebagai status validasi terkini.")

        if not kept:
            st.warning("Tidak ada sampel sesuai filter kelas.")
        else:
            points = []
            for f in kept:
                p = f.get("properties") or {}
                lon, lat = p.get("centroid_lon"), p.get("centroid_lat")
                try:
                    points.append((float(lat), float(lon)))
                except (ValueError, TypeError):
                    pass
            center = [float(np.mean([p[0] for p in points])), float(np.mean([p[1] for p in points]))] if points else [-7.5, 112.5]
            zoom_start = 8
            if show_raster_overlay and raster_meta and raster_meta.get("bounds_wgs84"):
                rb = raster_meta["bounds_wgs84"]
                center = [(float(rb["south"]) + float(rb["north"])) / 2, (float(rb["west"]) + float(rb["east"])) / 2]
                zoom_start = 9
            google_config = google_satellite_config(center, zoom_start)
            fmap = folium.Map(location=center, zoom_start=zoom_start, tiles=None, control_scale=True, prefer_canvas=True)
            if google_config:
                folium.TileLayer(
                    tiles=google_config["tiles"], attr=google_config["attr"],
                    name="Citra satelit (Google Maps Platform)", overlay=False, control=True, show=True,
                    max_zoom=22,
                ).add_to(fmap)
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Tiles © Esri — Sources: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
                name="Citra satelit (Esri World Imagery)", overlay=False, control=True, show=not bool(google_config),
            ).add_to(fmap)
            folium.TileLayer("OpenStreetMap", name="Peta jalan (OpenStreetMap)", overlay=False, control=True, show=False).add_to(fmap)
            if show_raster_overlay and raster_meta and raster_meta.get("bounds_wgs84"):
                rb = raster_meta["bounds_wgs84"]
                folium.raster_layers.ImageOverlay(
                    image=str(raster_preview),
                    bounds=[[float(rb["south"]), float(rb["west"])], [float(rb["north"]), float(rb["east"])]],
                    opacity=0.72,
                    name="Prediksi klasifikasi raster (AOI 40 × 40 km)",
                    interactive=False,
                    cross_origin=False,
                    zindex=2,
                ).add_to(fmap)
            boundary_options = [
                ROOT / "data" / "geojson" / "jawa_timur_AOI_ringan.geojson",
                ROOT / "data" / "geojson" / "jawa_timur_2D.geojson",
                ROOT / "data" / "geojson" / "jawa_timur.geojson",
            ]
            boundary = next((p for p in boundary_options if p.exists() and p.stat().st_size < 4_000_000), None)
            if boundary:
                boundary_fc = load_geojson(boundary)
                if boundary_fc:
                    folium.GeoJson(boundary_fc, name="Batas Jawa Timur (referensi)",
                                   style_function=lambda _: {"color": "#FFFFFF", "weight": 1.5, "fillOpacity": 0},
                                   show=True).add_to(fmap)
            for cls in map_classes:
                cfeats = [f for f in kept if str((f.get("properties") or {}).get("display_class", "")).lower() == cls]
                if not cfeats:
                    continue
                fc = {"type": "FeatureCollection", "features": cfeats}
                if map_mode == "Prediksi sampel test":
                    fields = ["sample_id", "actual_class_name", "predicted_class_name", "source_file"]
                    alias_map = {"sample_id": "ID sampel:", "actual_class_name": "Label aktual:", "predicted_class_name": "Prediksi model:", "source_file": "Sumber geometri:"}
                else:
                    fields = ["sample_id", "class_name", "source_file", "latest_clear_pixel_count"]
                    alias_map = {"sample_id": "ID sampel:", "class_name": "Kelas sampel:", "source_file": "Sumber geometri:", "latest_clear_pixel_count": "Piksel bersih:"}
                prop0 = cfeats[0].get("properties") or {}
                fields = [field for field in fields if field in prop0]
                aliases = [alias_map.get(field, field) for field in fields]
                folium.GeoJson(
                    fc, name=f"{label_for(cls)} ({len(cfeats)})",
                    style_function=lambda feat, c=COLORS[cls]: {"color": c, "weight": 1.6, "fillColor": c, "fillOpacity": 0.34},
                    highlight_function=lambda _: {"weight": 3, "fillOpacity": 0.55},
                    tooltip=folium.GeoJsonTooltip(fields=fields, aliases=aliases, sticky=False) if fields else None,
                    show=True,
                ).add_to(fmap)
            folium.LayerControl(collapsed=False).add_to(fmap)
            st_folium(fmap, height=640, use_container_width=True, returned_objects=[])
            legend_cols = st.columns(3)
            for i, cls in enumerate(map_classes):
                with legend_cols[i % 3]:
                    st.markdown(f"<span style='display:inline-block;width:12px;height:12px;background:{COLORS[cls]};border-radius:3px;margin-right:7px'></span>{label_for(cls)}", unsafe_allow_html=True)
            if google_config:
                st.caption("Basemap memakai Google Maps Platform Satellite dengan atribusi copyright dari viewport API. Untuk penggunaan publik, batasi API key ke Map Tiles API dan konfigurasi keamanan sesuai dokumentasi Google.")
            else:
                st.caption("Basemap satelit menggunakan Esri World Imagery. Jika dosen mensyaratkan Google Maps secara spesifik, set GOOGLE_MAPS_API_KEY dan aktifkan Map Tiles API pada Google Cloud. Overlay raster (jika tersedia) hanya kotak AOI pesisir 40 × 40 km termasuk perairan sekitar, bukan raster seluruh provinsi.")
            if show_raster_overlay and raster_area_summary.exists():
                st.markdown("#### Luas prediksi kelas pada AOI percontohan")
                area_df = read_csv_safe(raster_area_summary)
                if area_df is not None and not area_df.empty:
                    st.dataframe(area_df, hide_index=True, width="stretch")
                    st.caption("Luas merupakan estimasi dari jumlah piksel prediksi valid pada AOI percontohan saja, bukan statistik luas tutupan lahan seluruh Jawa Timur.")
    else:
        st.error(f"GeoJSON sampel tidak ditemukan: {SAMPLE_GEOJSON.relative_to(ROOT)}")

with tab_method:
    st.subheader("Metodologi, acuan klasifikasi, dan relevansi kebijakan")
    st.markdown("#### Alur pemrosesan")
    st.markdown("1. Mengumpulkan sampel poligon dan batas wilayah studi Jawa Timur.\n2. Mengambil reflektansi Sentinel-2A Level-2A pada 1 September–10 Oktober 2026.\n3. Menggunakan median temporal per piksel dari observasi yang memenuhi dataMask dan SCL kelas 4, 5, atau 6.\n4. Menghitung statistik poligon dan indeks spektral.\n5. Membandingkan lima algoritma dengan split acak terstratifikasi 80:20 dan CV 5-fold pada data latih.\n6. Menyajikan hasil baseline dan sebaran sampel melalui dashboard SIG.")
    st.markdown("#### Acuan klasifikasi Indonesia")
    st.markdown("**SNI 7645-1:2014 — Klasifikasi penutup lahan, Bagian 1: Skala kecil dan menengah** tercatat berstatus berlaku di katalog Badan Standardisasi Nasional. SNI menjadi acuan konseptual dan hierarki kelas penutup lahan. Enam kelas pada proyek ini adalah pengelompokan operasional untuk tujuan tugas, bukan klaim bahwa nama dan batas keenam kelas sama persis dengan kode/hierarki SNI.")
    crosswalk = pd.DataFrame([
        ("Sawah", "Pertanian/sawah", "Sawah dengan padi terus menerus dan bentuk sawah lain menurut hierarki/definisi skala SNI; validasi visual tetap diperlukan."),
        ("Bangunan", "Kawasan terbangun", "Kelompok operasional untuk permukaan terbangun; bukan setiap objek bangunan individual."),
        ("Mangrove", "Tutupan vegetasi mangrove", "Kelas vegetasi pesisir; dapat dirujuk juga pada SNI 7717:2020 untuk spesifikasi informasi geospasial mangrove."),
        ("Lahan hijau", "Kelompok vegetasi non-sawah/non-mangrove", "Kelas payung khusus proyek; dapat mencakup vegetasi berbeda dan bukan satu kelas tunggal SNI tanpa definisi rinci."),
        ("Laut", "Tubuh air laut/perairan terbuka", "Kelas operasional perairan terbuka; jangan disamakan dengan semua badan air."),
        ("Danau / ranu", "Danau/tubuh air daratan", "Kelas operasional badan air daratan; validasi konektivitas pesisir dan badan air alami diperlukan."),
    ], columns=["Kelas proyek", "Kaitan konseptual", "Catatan pemetaan ke acuan"])
    st.dataframe(crosswalk, hide_index=True, width="stretch")
    st.caption("Untuk kutipan normatif/detail hierarki, gunakan dokumen SNI resmi; katalog BSN mengonfirmasi nomor, judul, dan status standar, tetapi tidak menggantikan teks lengkap standar.")

    st.markdown("#### Rumus indeks")
    st.latex(r"NDVI = (B08 - B04)/(B08 + B04)")
    st.latex(r"NDWI = (B03 - B08)/(B03 + B08)")
    st.latex(r"MNDWI = (B03 - B11)/(B03 + B11)")
    st.latex(r"NDBI = (B11 - B08)/(B11 + B08)")

    st.markdown("#### Relevansi SIG dan kebijakan (kerangka analisis awal)")
    st.markdown("- **Sawah:** bandingkan prediksi sawah dengan peta LP2B/KP2B atau pola ruang pertanian resmi untuk menyusun pertanyaan tentang perlindungan lahan pangan. Peta ini sendiri tidak membuktikan alih fungsi lahan.\n- **Bangunan:** bandingkan dengan zona permukiman/budidaya dan jaringan infrastruktur pada RTRW/RDTR resmi; jangan menyimpulkan pelanggaran tata ruang hanya dari prediksi tutupan.\n- **Mangrove:** bandingkan dengan inventarisasi dan kawasan perlindungan mangrove resmi; pemetaan perubahan membutuhkan data multitemporal serta verifikasi.\n- **Lahan hijau:** kelas ini masih berupa kelompok umum sehingga perlu pemecahan lebih rinci sebelum dipakai untuk keputusan zonasi.\n- **Laut/perairan terbuka:** perbandingan kebijakan memerlukan dokumen rencana zonasi/penataan ruang laut yang relevan.\n- **Danau/ranu:** diperlukan inventarisasi badan air yang otoritatif dan batas air daratan agar tidak tertukar dengan tambak/laguna/perairan pesisir.")
    st.markdown("**Dokumen pembanding yang perlu disiapkan:** RTRW Provinsi Jawa Timur 2023–2043 (Perda Provinsi Jawa Timur No. 10 Tahun 2023), serta RTRW/RDTR kabupaten/kota atau dokumen zonasi laut yang benar-benar mencakup AOI. Tahap tumpang-susun belum dilakukan karena paket ini belum menyertakan layer pola ruang resmi. Periksa naskah dan lampiran peta dari portal resmi JDIH sebelum menarik kesimpulan.")
    st.markdown("[JDIH Provinsi Jawa Timur](https://jdih.jatimprov.go.id/) · [JDIH/Basis Data Peraturan BPK](https://peraturan.bpk.go.id/)")
    st.warning("Batasan: dashboard ini memvisualisasikan sampel, evaluasi model, dan raster hanya untuk AOI pesisir percontohan 40 × 40 km. Belum tersedia raster prediksi kontinu seluruh Jawa Timur atau statistik luas kelas seluruh provinsi. Evaluasi blok spasial gagal karena konsentrasi sampel bangunan, sehingga metrik random split tidak boleh dianggap sebagai bukti generalisasi spasial atau kepatuhan terhadap RTRW.")
    st.markdown("#### Referensi sumber")
    st.markdown("- [BSN — SNI 7645-1:2014, Klasifikasi penutup lahan](https://pesta.bsn.go.id/produk/detail/10269-sni7645-12014)\n- [BSN — SNI 7717:2020, Spesifikasi informasi geospasial mangrove](https://pesta.bsn.go.id/produk/detail/13110-sni77172020)\n- [ESA/Copernicus — Sentinel-2 Products Specification Document](https://sentinels.copernicus.eu/documents/d/sentinel/sentinel-2-products-specification-document-15_1)\n- [BIG — Layanan Batas Administrasi Provinsi](https://geoservices.big.go.id/gis/rest/services/STIG/Batas_Provinsi/MapServer)\n- [Google Maps Platform — Map Tiles API](https://developers.google.com/maps/documentation/tile/) dan [kebijakannya](https://developers.google.com/maps/documentation/tile/policies?hl=id).")

with st.sidebar:
    st.markdown("---")
    st.caption("Dashboard lokal • PSD UTS • Dataset/hasil tetap berada di folder proyek")

choose_path_message([COMPOSITE_CSV, DATASETS["Semua sampel (303)"], DATASETS["Tersaring kualitas (255)"],
                     SAMPLE_GEOJSON])
