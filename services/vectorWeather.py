"""
vectorWeather.py — Open-Meteo → ChromaDB (collection: weather_data)

Data is filtered to only what matters for:
  - Climate risk analysis         → extreme weather, flood/drought indicators
  - ESG / sustainability teams    → climate-related physical risk metrics
  - Infrastructure & operations   → heat stress, heavy rain, wind gusts
  - Supply chain resilience       → drought, flood risk, crop/climate impact

Storage strategy:
  - ~109 daily summary docs (concise, climate-intelligence-focused)
  - Each daily doc summarises min/max/avg of all relevant variables
    + flags extreme events (heavy rain, drought, heat stress, frost risk)

Run standalone:   python vectorWeather.py
Import in main:   from vectorWeather import weather_store, weather_retriever

Perubahan dari versi sebelumnya:
  - DB_PATH + persist_directory diganti dengan ChromaDB HttpClient
  - CHROMA_HOST / CHROMA_PORT bisa diubah sesuai setup
"""

import hashlib
import os
import time
import warnings
from datetime import date

# Suppress pydantic V1 deprecation warnings on Python 3.14+
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

import openmeteo_requests
import numpy as np
import pandas as pd
import requests_cache
from retry_requests import retry

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from dotenv import load_dotenv
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / "config" / ".env")

# ─── Configuration ────────────────────────────────────────────────────────────
EMBED_MODEL    = os.getenv("EMBED_MODEL", "mxbai-embed-large")
LATITUDE       = float(os.getenv("LATITUDE", "-6.1818"))
LONGITUDE      = float(os.getenv("LONGITUDE", "106.8223"))
LOCATION_NAME  = os.getenv("LOCATION_NAME", "Jakarta")

PAST_DAYS      = int(os.getenv("PAST_DAYS", "92"))
FORECAST_DAYS  = int(os.getenv("FORECAST_DAYS", "16"))
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "300"))

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "rain",
    "evapotranspiration",
    "surface_pressure",
    "cloud_cover",
    "soil_temperature_0cm",
    "soil_temperature_6cm",
    "soil_temperature_18cm",
    "soil_moisture_0_to_1cm",
    "soil_moisture_1_to_3cm",
    "soil_moisture_3_to_9cm",
    "soil_moisture_9_to_27cm",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
]

# ─── ChromaDB server connection ───────────────────────────────────────────────
# Ganti host/port sesuai setup kamu.
# Default: server jalan di mesin yang sama (localhost:8000)
import chromadb as _chromadb
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
_chroma_client = _chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

_embeddings   = OllamaEmbeddings(model=EMBED_MODEL)
weather_store = Chroma(
    collection_name="weather_data",
    client=_chroma_client,
    embedding_function=_embeddings,
)

# No score_threshold — weather docs are structured text (not natural language)
# and would be incorrectly filtered by cosine similarity thresholds.
weather_retriever = weather_store.as_retriever(search_kwargs={"k": 8})

# ─── Open-Meteo client ────────────────────────────────────────────────────────
_cache   = requests_cache.CachedSession(".weather_cache", expire_after=3600)
_retried = retry(_cache, retries=5, backoff_factor=0.2)
_client  = openmeteo_requests.Client(session=_retried)

API_URL = "https://api.open-meteo.com/v1/forecast"


# ─── Fetch raw hourly DataFrame ───────────────────────────────────────────────

def _fetch_dataframe() -> pd.DataFrame:
    params = {
        "latitude":      LATITUDE,
        "longitude":     LONGITUDE,
        "hourly":        HOURLY_VARS,
        "past_days":     PAST_DAYS,
        "forecast_days": FORECAST_DAYS,
    }
    responses = _client.weather_api(API_URL, params=params)
    response  = responses[0]

    print(f"   Location: {LOCATION_NAME} "
          f"{response.Latitude():.4f}N {response.Longitude():.4f}E "
          f"elevation {response.Elevation():.0f} m asl")

    hourly = response.Hourly()
    data   = {"datetime": pd.date_range(
        start     = pd.to_datetime(hourly.Time(),    unit="s", utc=True),
        end       = pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq      = pd.Timedelta(seconds=hourly.Interval()),
        inclusive = "left",
    )}
    for idx, var in enumerate(HOURLY_VARS):
        data[var] = hourly.Variables(idx).ValuesAsNumpy()

    df = pd.DataFrame(data)
    df["datetime"] = df["datetime"].dt.tz_convert("Asia/Jakarta")
    df["date"]     = df["datetime"].dt.date
    return df


# ─── Aggregate hourly → daily farming summary ─────────────────────────────────

def _daily_summary_to_text(day: date, g: pd.DataFrame) -> str:
    d = str(day)

    def _s(series, func="mean", fallback=float("nan")):
        clean = series.dropna()
        if clean.empty:
            return fallback
        return getattr(clean, func)()

    t_min  = _s(g["temperature_2m"],       "min")
    t_max  = _s(g["temperature_2m"],       "max")
    t_avg  = _s(g["temperature_2m"],       "mean")
    rh_avg = _s(g["relative_humidity_2m"], "mean")
    rh_max = _s(g["relative_humidity_2m"], "max")
    dp_avg = _s(g["dew_point_2m"],         "mean")
    rain   = _s(g["rain"],                 "sum",  fallback=0.0)
    precip = _s(g["precipitation"],        "sum",  fallback=0.0)
    et     = _s(g["evapotranspiration"],   "sum",  fallback=0.0)
    cloud  = _s(g["cloud_cover"],          "mean")
    pres   = _s(g["surface_pressure"],     "mean")
    ws_max = _s(g["wind_speed_10m"],       "max",  fallback=0.0)
    ws_avg = _s(g["wind_speed_10m"],       "mean", fallback=0.0)
    gust   = _s(g["wind_gusts_10m"],       "max",  fallback=0.0)

    wind_col = g["wind_speed_10m"].dropna()
    if not wind_col.empty:
        peak_hour = g.loc[wind_col.idxmax()]
        wind_dir  = peak_hour["wind_direction_10m"]
        if pd.isna(wind_dir):
            wind_dir = float("nan")
    else:
        wind_dir = float("nan")

    sm_surf = _s(g["soil_moisture_0_to_1cm"],  "mean")
    sm_root = _s(g["soil_moisture_3_to_9cm"],  "mean")
    sm_deep = _s(g["soil_moisture_9_to_27cm"], "mean")
    st_surf = _s(g["soil_temperature_0cm"],    "mean")
    st_root = _s(g["soil_temperature_6cm"],    "mean")
    st_deep = _s(g["soil_temperature_18cm"],   "mean")

    alerts = []

    if rain < 1.0 and (sm_root == sm_root) and sm_root < 0.15:
        alerts.append("DROUGHT RISK: No rain and low root-zone soil moisture — consider irrigation.")
    elif rain < 2.0:
        alerts.append("LOW RAINFALL: Monitor soil moisture, irrigation may be needed.")

    if rain >= 50:
        alerts.append("EXTREME RAIN: >50mm — high flood risk, delay field operations.")
    elif rain >= 20:
        alerts.append("HEAVY RAIN: >20mm — monitor drainage, risk of waterlogging.")

    if t_max >= 38:
        alerts.append("EXTREME HEAT: >38C — severe heat stress for crops and livestock.")
    elif t_max >= 35:
        alerts.append("HEAT STRESS: >35C — protect sensitive crops, provide shade for livestock.")

    if t_min <= 10:
        alerts.append("COLD RISK: <10C — risk of chilling injury to tropical crops.")
    elif t_min <= 15:
        alerts.append("COOL NIGHT: <15C — monitor cold-sensitive crops and seedlings.")

    if rh_max >= 95 and t_avg >= 25:
        alerts.append("HIGH DISEASE RISK: Very high humidity + warm temp — spray fungicide preventively.")
    elif rh_avg >= 85:
        alerts.append("ELEVATED HUMIDITY: Monitor for fungal diseases (blast, blight, mildew).")

    if gust >= 60:
        alerts.append("STRONG GUSTS: >60 km/h — risk of crop lodging and structural damage.")
    elif ws_max >= 40:
        alerts.append("HIGH WIND: >40 km/h — avoid spraying pesticides/herbicides.")

    if et >= 6:
        alerts.append("HIGH WATER DEMAND: ET >6mm — increase irrigation frequency.")

    alert_text = "\n".join(alerts) if alerts else "No major agronomic alerts for this day."

    def _fmt(val, fmt, fallback="data unavailable"):
        return (fmt % val) if val == val else fallback

    lines = [
        f"=== Daily Weather Summary: {LOCATION_NAME} | {d} ===",
        "",
        "-- ATMOSPHERE --",
        f"Temperature: min {_fmt(t_min,'%.1f')}C / avg {_fmt(t_avg,'%.1f')}C / max {_fmt(t_max,'%.1f')}C",
        f"Humidity: avg {_fmt(rh_avg,'%.0f')}% / max {_fmt(rh_max,'%.0f')}%",
        f"Dew point: avg {_fmt(dp_avg,'%.1f')}C",
        f"Rainfall: {rain:.1f} mm",
        f"Total precipitation: {precip:.1f} mm",
        f"Evapotranspiration (ET0): {et:.2f} mm/day",
        f"Cloud cover: {_fmt(cloud,'%.0f')}%",
        f"Surface pressure: {_fmt(pres,'%.1f')} hPa",
        "",
        "-- WIND --",
        f"Wind speed: avg {ws_avg:.1f} km/h / max {ws_max:.1f} km/h",
        f"Wind gusts: max {gust:.1f} km/h",
        f"Wind direction at peak: {_fmt(wind_dir,'%.0f')} deg",
        "",
        "-- SOIL --",
        f"Soil temperature: surface {_fmt(st_surf,'%.1f')}C / 6cm {_fmt(st_root,'%.1f')}C / 18cm {_fmt(st_deep,'%.1f')}C",
        f"Soil moisture: surface {_fmt(sm_surf,'%.3f')} m3/m3 / root {_fmt(sm_root,'%.3f')} m3/m3 / deep {_fmt(sm_deep,'%.3f')} m3/m3",
        "",
        "-- AGRONOMIC ALERTS --",
        alert_text,
    ]
    return "\n".join(lines)


# ─── Core indexing ────────────────────────────────────────────────────────────

def index_weather(fetch_date: str):
    """Fetch Open-Meteo data, aggregate to daily summaries, store in ChromaDB."""
    print(f"\n   Fetching Open-Meteo data (batch: {fetch_date})...")
    try:
        df = _fetch_dataframe()
    except Exception as e:
        print(f"   API fetch failed: {e}")
        return

    # Remove stale docs
    try:
        old = weather_store.get(where={"fetch_date": {"$ne": fetch_date}}, limit=100_000)
        if old["ids"]:
            weather_store.delete(ids=old["ids"])
            print(f"   Removed {len(old['ids'])} outdated docs.")
    except Exception:
        pass

    documents = []
    ids       = []
    grouped   = df.groupby("date")

    for day, group in grouped:
        text   = _daily_summary_to_text(day, group)
        doc_id = _doc_id(f"{LOCATION_NAME}:{day}")
        documents.append(Document(
            page_content=text,
            metadata={
                "source":     "open_meteo",
                "location":   LOCATION_NAME,
                "date":       str(day),
                "fetch_date": fetch_date,
            },
            id=doc_id,
        ))
        ids.append(doc_id)

    batch_size = 50
    total      = len(documents)
    print(f"   Storing {total} daily weather summaries...")
    for i in range(0, total, batch_size):
        weather_store.add_documents(
            documents=documents[i:i+batch_size],
            ids=ids[i:i+batch_size],
        )
    print(f"   Weather indexed: {total} daily summaries for {fetch_date}.")


def _doc_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def is_weather_fresh(today: str) -> bool:
    try:
        result = weather_store.get(where={"fetch_date": today}, limit=1)
        return bool(result["ids"])
    except Exception:
        return False


# ─── Main server loop ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Weather Vector Server | Location: {LOCATION_NAME}")
    print(f"   ChromaDB: {CHROMA_HOST}:{CHROMA_PORT}")
    print(f"   Stores climate-intelligence daily summaries (not raw hourly data).")
    print(f"   Checks for day change every {CHECK_INTERVAL}s.\n")

    last_date = None

    while True:
        today = date.today().isoformat()

        if today != last_date:
            if is_weather_fresh(today):
                print(f"Weather data for {today} already in ChromaDB — skipping fetch.")
            else:
                index_weather(today)
            last_date = today

        time.sleep(CHECK_INTERVAL)