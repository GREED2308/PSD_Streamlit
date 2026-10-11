# SIG Tutupan Lahan Jawa Timur — Streamlit (deployment-ready source)

Paket ini memuat aplikasi Streamlit, data tabel, laporan evaluasi, GeoJSON sampel/batas, preview raster SVM, serta metadata dan rekap luas KNN dari hasil run lengkap. GeoTIFF 10 m tidak dimasukkan karena UI menampilkan PNG preview dan rekap luas, bukan membuka raster mentah. File preview PNG KNN asli harus disalin dari output lokal sebelum push ke GitHub; file tersebut tidak tersedia pada lampiran paket sumber di sesi ini.

## Uji lokal

PowerShell, di folder hasil ekstraksi paket:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\requirements.txt
python -m streamlit run .\app.py
```

## Deploy ke Streamlit Community Cloud

1. Ekstrak paket dan buat repository GitHub baru.
2. Letakkan `app.py`, `requirements.txt`, `outputs/`, dan `data/` di root repository; jangan unggah `.env`, kredensial Copernicus, atau `secrets.toml`.
3. Push repository ke GitHub.
4. Buka https://share.streamlit.io/ dan pilih repository, branch, dan entrypoint `app.py`.
5. Setelah build selesai, buka URL `*.streamlit.app` yang disediakan dan uji seluruh tab. Hanya setelah URL diuji, tandai status deployment sebagai berhasil.

Dokumentasi resmi: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## Google Maps Satellite (opsional bila API key tersedia)

Aplikasi memakai Esri World Imagery ketika `GOOGLE_MAPS_API_KEY` tidak disediakan. Untuk Google Maps Platform Map Tiles API, siapkan project/billing dan API secara resmi. Saat deployment, tambahkan secret `GOOGLE_MAPS_API_KEY` pada Settings → Secrets, jangan menaruh key di kode atau repository, dan batasi key pada API serta referrer yang diperlukan. Pastikan branding/atribusi Google Maps dan atribusi copyright hasil viewport terlihat sesuai ketentuan API.

## Data dan batasan

- Enam kelas: sawah, bangunan, mangrove, lahan hijau, perairan terbuka/laut, danau/ranu.
- Dataset: 303 sampel (242 train/61 test) dan quality-screened 255 (204 train/51 test).
- 16 fitur: 12 band reflektansi L2A mean + NDVI, NDWI, MNDWI, NDBI.
- Raster yang dipaketkan adalah baseline SVM RBF untuk AOI pesisir 40 × 40 km, bukan seluruh Jawa Timur.
- KNN memperoleh skor testing tertinggi pada varian quality-screened. Metadata dan CSV rekap KNN sudah disertakan. Sebelum deployment, salin file `classification_KNN_quality_screened_aoi40km_offshore_preview.png` dari folder `D:\Projects\Kuliah\PSD\outputs\classification_raster_v6_aoi40km_offshore` ke `outputs/classification_raster_v6_aoi40km_offshore/` dalam paket ini. Jangan menggantinya dengan preview SVM. Raster SVM tetap ada sebagai perbandingan.
- Metrik testing adalah evaluasi satu random split dan tidak sama dengan akurasi raster per piksel.

Jangan menyertakan `.env` atau API keys pada arsip publik.


**Catatan peta:** Saat Google Maps Platform diaktifkan, periksa seluruh atribusi dan branding yang diwajibkan Google pada peta. Bila rahasia/API key tidak tersedia, dashboard menggunakan Esri World Imagery sebagai fallback.


## Menyiapkan aset KNN sebelum mengunggah

Dari PowerShell setelah ZIP diekstrak ke `D:\Projects\Kuliah\PSD_Streamlit_Deploy_UTS`, jalankan:

```powershell
$src = "D:\Projects\Kuliah\PSD\outputs\classification_raster_v6_aoi40km_offshore"
$dst = "D:\Projects\Kuliah\PSD_Streamlit_Deploy_UTS\outputs\classification_raster_v6_aoi40km_offshore"
Copy-Item "$src\classification_KNN_quality_screened_aoi40km_offshore_preview.png" $dst -Force
```

Setelah itu periksa bahwa file PNG KNN ada di folder tujuan sebelum commit ke GitHub. Metadata dan rekap KNN sudah disertakan dalam ZIP ini.


## Urutan cepat setelah ZIP diekstrak

1. Ekstrak ZIP ke folder lokal.
2. Salin preview PNG KNN asli dari proyek lokal ke `outputs/classification_raster_v6_aoi40km_offshore/` dengan nama `classification_KNN_quality_screened_aoi40km_offshore_preview.png`. Jangan rename preview SVM menjadi KNN.
3. Buat repository GitHub baru dan unggah isi folder paket (bukan file ZIP-nya), termasuk `app.py`, `requirements.txt`, `data/`, `outputs/`, dan `.streamlit/config.toml`. Jangan unggah `.env`, API key, token, folder venv, atau cache.
4. Buka `https://share.streamlit.io/`, klik **Create app**, pilih repository, branch `main`, dan entrypoint `app.py`.
5. Deploy. Uji URL `*.streamlit.app` dari browser setelah build sukses; baru setelah itu status deployment dapat dinyatakan selesai.
6. Google Maps Satellite hanya aktif jika `GOOGLE_MAPS_API_KEY` diset sebagai secret `GOOGLE_MAPS_API_KEY = \"...\"` setelah Map Tiles API disiapkan pada Google Cloud. Tanpa key, aplikasi memakai Esri World Imagery.

`PSD_PROJECT_ROOT` tidak perlu diset pada Streamlit Community Cloud karena aplikasi membaca file relatif dari folder repository.
