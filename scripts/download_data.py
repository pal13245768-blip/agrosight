"""
AgroSight – Data Download Scripts
===================================
Downloads raw data from all external APIs before ingestion.

Usage:
    python -m scripts.download_data [--source all|weather|mandi|usda|soil|kaggle]

Sources:
  weather  – OpenWeatherMap agro advisories for major Indian cities
  mandi    – data.gov.in Agmarknet mandi prices (last 30 days)
  usda     – USDA NASS crop statistics (wheat, rice, cotton, maize)
  soil     – ISRIC SoilGrids for Indian agricultural zones
  kaggle   – PlantVillage disease image dataset (requires Kaggle credentials)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.utils.config import get_settings
from app.utils.file_utils import ensure_dir
from app.utils.logger import configure_logger, logger

configure_logger()
settings = get_settings()

# ---------------------------------------------------------------------------
# Target cities for weather advisories
# ---------------------------------------------------------------------------

AGRO_CITIES = [
    {"name": "Ahmedabad", "state": "Gujarat", "lat": 23.03, "lon": 72.58},
    {"name": "Ludhiana", "state": "Punjab", "lat": 30.90, "lon": 75.85},
    {"name": "Nagpur", "state": "Maharashtra", "lat": 21.15, "lon": 79.09},
    {"name": "Patna", "state": "Bihar", "lat": 25.59, "lon": 85.14},
    {"name": "Jaipur", "state": "Rajasthan", "lat": 26.91, "lon": 75.79},
    {"name": "Bhopal", "state": "Madhya Pradesh", "lat": 23.26, "lon": 77.41},
    {"name": "Hyderabad", "state": "Telangana", "lat": 17.38, "lon": 78.49},
    {"name": "Bengaluru", "state": "Karnataka", "lat": 12.97, "lon": 77.59},
    {"name": "Kolkata", "state": "West Bengal", "lat": 22.57, "lon": 88.36},
    {"name": "Lucknow", "state": "Uttar Pradesh", "lat": 26.85, "lon": 80.95},
    {"name": "Chandigarh", "state": "Punjab/Haryana", "lat": 30.73, "lon": 76.78},
    {"name": "Surat", "state": "Gujarat", "lat": 21.19, "lon": 72.83},
    {"name": "Coimbatore", "state": "Tamil Nadu", "lat": 11.00, "lon": 76.97},
    {"name": "Indore", "state": "Madhya Pradesh", "lat": 22.72, "lon": 75.86},
    {"name": "Varanasi", "state": "Uttar Pradesh", "lat": 25.32, "lon": 82.97},
]

# Commodities for mandi price download
COMMODITIES = [
    "Wheat", "Rice", "Cotton", "Soybean", "Maize", "Sugarcane",
    "Onion", "Tomato", "Potato", "Groundnut", "Mustard", "Chilli",
    "Turmeric", "Garlic", "Ginger", "Jowar", "Bajra", "Gram",
]


# ---------------------------------------------------------------------------
# Weather download
# ---------------------------------------------------------------------------


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _fetch_city_weather(client: httpx.AsyncClient, city: dict) -> dict | None:
    if not settings.openweather_api_key:
        logger.warning("OPENWEATHER_API_KEY not set – skipping weather download")
        return None
    url = f"{settings.openweather_base_url}/weather"
    params = {
        "lat": city["lat"],
        "lon": city["lon"],
        "appid": settings.openweather_api_key,
        "units": "metric",
    }
    try:
        resp = await client.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return {
            "city": city["name"],
            "state": city["state"],
            "lat": city["lat"],
            "lon": city["lon"],
            "temperature_c": data["main"]["temp"],
            "humidity_pct": data["main"]["humidity"],
            "wind_kmh": round(data["wind"]["speed"] * 3.6, 1),
            "condition": data["weather"][0]["description"],
            "rain_1h_mm": data.get("rain", {}).get("1h", 0),
            "timestamp": data.get("dt"),
        }
    except Exception as exc:
        logger.warning(f"Weather fetch failed for {city['name']}: {exc}")
        return None


async def download_weather() -> None:
    out_dir = ensure_dir(Path(settings.data_root) / "weather" / "openweather")
    advisories: list[dict] = []
    async with httpx.AsyncClient() as client:
        tasks = [_fetch_city_weather(client, city) for city in AGRO_CITIES]
        results = await asyncio.gather(*tasks)
    advisories = [r for r in results if r is not None]
    out_path = out_dir / "agro_advisories.json"
    out_path.write_text(json.dumps(advisories, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.success(f"Weather: saved {len(advisories)} city advisories → {out_path}")


# ---------------------------------------------------------------------------
# Mandi price download
# ---------------------------------------------------------------------------


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _fetch_mandi_prices(client: httpx.AsyncClient, commodity: str) -> list[dict]:
    api_key = settings.data_gov_api_key_1
    resource_id = settings.data_gov_mandi_resource_id
    if not api_key:
        return []
    url = f"{settings.data_gov_base_url}/resource/{resource_id}"
    params = {
        "api-key": api_key,
        "format": "json",
        "filters[commodity]": commodity,
        "limit": 50,
    }
    try:
        resp = await client.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        return data.get("records", [])
    except Exception as exc:
        logger.warning(f"Mandi fetch failed for {commodity}: {exc}")
        return []


async def download_mandi_prices() -> None:
    if not settings.data_gov_api_key_1:
        logger.warning("DATA_GOV_API_KEY_1 not set – skipping mandi download")
        return
    out_dir = ensure_dir(Path(settings.data_root) / "market")
    all_records: list[dict] = []
    async with httpx.AsyncClient() as client:
        for commodity in COMMODITIES:
            records = await _fetch_mandi_prices(client, commodity)
            all_records.extend(records)
            logger.debug(f"Mandi: {len(records)} records for {commodity}")
    if all_records:
        import pandas as pd
        df = pd.DataFrame(all_records)
        out_path = out_dir / "agmarket_india.csv"
        df.to_csv(out_path, index=False)
        logger.success(f"Mandi: saved {len(all_records)} records → {out_path}")
    else:
        logger.warning("Mandi: no records fetched")


# ---------------------------------------------------------------------------
# USDA NASS download
# ---------------------------------------------------------------------------


async def download_usda_nass() -> None:
    if not settings.usda_nass_api_key:
        logger.warning("USDA_NASS_API_KEY not set – skipping USDA download")
        return
    out_dir = ensure_dir(Path(settings.data_root) / "global_stats" / "usda")
    commodities = ["WHEAT", "RICE", "COTTON", "CORN", "SOYBEANS"]
    all_records: list[dict] = []
    async with httpx.AsyncClient() as client:
        for commodity in commodities:
            url = f"{settings.usda_nass_base_url}/api_GET/"
            params = {
                "key": settings.usda_nass_api_key,
                "commodity_desc": commodity,
                "statisticcat_desc": "PRODUCTION",
                "year__GE": "2018",
                "format": "JSON",
            }
            try:
                resp = await client.get(url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                records = data.get("data", [])
                all_records.extend(records)
                logger.debug(f"USDA: {len(records)} records for {commodity}")
            except Exception as exc:
                logger.warning(f"USDA fetch failed for {commodity}: {exc}")

    if all_records:
        import pandas as pd
        df = pd.DataFrame(all_records)
        out_path = out_dir / "usda_nass_crop_production.csv"
        df.to_csv(out_path, index=False)
        logger.success(f"USDA: saved {len(all_records)} records → {out_path}")


# ---------------------------------------------------------------------------
# SoilGrids download (key Indian agricultural coordinates)
# ---------------------------------------------------------------------------

SOIL_POINTS = [
    {"name": "Punjab_Plains", "lat": 30.9, "lon": 75.8},
    {"name": "Ganga_Delta_WB", "lat": 22.5, "lon": 88.3},
    {"name": "Deccan_Plateau_MH", "lat": 19.0, "lon": 76.5},
    {"name": "Black_Soil_Vidarbha", "lat": 20.7, "lon": 78.5},
    {"name": "Red_Soil_Karnataka", "lat": 15.3, "lon": 75.7},
    {"name": "Alluvial_UP", "lat": 27.0, "lon": 80.9},
    {"name": "Sandy_Rajasthan", "lat": 27.0, "lon": 73.0},
    {"name": "Laterite_Kerala", "lat": 10.8, "lon": 76.3},
    {"name": "Coastal_Gujarat", "lat": 22.3, "lon": 72.6},
    {"name": "Hill_Soil_Himachal", "lat": 31.1, "lon": 77.2},
]


async def download_soilgrids() -> None:
    out_dir = ensure_dir(Path(settings.data_root) / "soil")
    soil_records: list[dict] = []
    properties = ["nitrogen", "phh2o", "soc", "clay", "sand", "silt", "bdod", "cec"]

    async with httpx.AsyncClient() as client:
        for point in SOIL_POINTS:
            url = f"{settings.isric_base_url}/properties/query"
            params = {
                "lon": point["lon"],
                "lat": point["lat"],
                "property": properties,
                "depth": ["0-5cm", "5-15cm", "15-30cm"],
                "value": "mean",
            }
            try:
                resp = await client.get(url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                record = {"location": point["name"], "lat": point["lat"], "lon": point["lon"]}
                layers = data.get("properties", {}).get("layers", [])
                for layer in layers:
                    prop_name = layer.get("name", "")
                    for depth_data in layer.get("depths", []):
                        depth_label = depth_data.get("label", "")
                        mean_val = depth_data.get("values", {}).get("mean")
                        record[f"{prop_name}_{depth_label}"] = mean_val
                soil_records.append(record)
                logger.debug(f"SoilGrids: fetched data for {point['name']}")
            except Exception as exc:
                logger.warning(f"SoilGrids fetch failed for {point['name']}: {exc}")

    if soil_records:
        import pandas as pd
        df = pd.DataFrame(soil_records)
        out_path = out_dir / "soilgrids_india.csv"
        df.to_csv(out_path, index=False)
        logger.success(f"SoilGrids: saved {len(soil_records)} location records → {out_path}")


# ---------------------------------------------------------------------------
# Kaggle PlantVillage (image dataset – metadata only)
# ---------------------------------------------------------------------------


async def download_kaggle_metadata() -> None:
    """
    Download PlantVillage dataset class metadata from Kaggle.
    Actual images are large (~2GB) and downloaded separately via kaggle CLI.
    """
    if not settings.kaggle_username or not settings.kaggle_key:
        logger.warning("KAGGLE credentials not set – skipping Kaggle download")
        logger.info(
            "To download PlantVillage images manually:\n"
            "  export KAGGLE_USERNAME=your_username\n"
            "  export KAGGLE_KEY=your_key\n"
            "  kaggle datasets download -d emmarex/plantdisease -p data/raw/pest_disease --unzip"
        )
        return

    # Write kaggle.json credentials
    import os
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    (kaggle_dir / "kaggle.json").write_text(
        json.dumps({"username": settings.kaggle_username, "key": settings.kaggle_key}),
        encoding="utf-8",
    )
    os.chmod(kaggle_dir / "kaggle.json", 0o600)

    logger.info("Kaggle credentials written. Run the following to download PlantVillage:")
    logger.info(
        "  kaggle datasets download -d emmarex/plantdisease "
        "-p data/raw/pest_disease --unzip"
    )


# ---------------------------------------------------------------------------
# Seed JSON knowledge files (static)
# ---------------------------------------------------------------------------


DISEASE_KNOWLEDGE = [
    {
        "disease_name": "Wheat Rust (Yellow Stripe Rust)",
        "pathogen": "Puccinia striiformis f. sp. tritici",
        "crops_affected": ["wheat", "barley"],
        "symptoms": "Yellow to orange pustules in stripes along leaf veins. Leaves turn yellow and dry.",
        "favorable_conditions": "Cool temperatures (10–15°C), high humidity, dew formation",
        "management": [
            "Spray Propiconazole 25% EC @ 0.1% or Tebuconazole 25.9% EC @ 0.1%",
            "Use resistant varieties: HD 2967, PBW 550, DBW 17",
            "Avoid late sowing (Rabi season)",
            "Remove infected plant debris after harvest",
        ],
        "resistant_varieties": ["HD 2967", "PBW 550", "DBW 17", "GW 322"],
        "kisan_advisory": "यदि पत्तियों पर पीली धारियाँ दिखें तो तुरंत फफूंदनाशक का छिड़काव करें।",
    },
    {
        "disease_name": "Rice Blast",
        "pathogen": "Magnaporthe oryzae",
        "crops_affected": ["rice"],
        "symptoms": "Diamond-shaped lesions with grey centre and brown border on leaves. Neck rot causes white heads (deadhearts).",
        "favorable_conditions": "High humidity >90%, temperature 24–28°C, excess nitrogen",
        "management": [
            "Spray Tricyclazole 75 WP @ 0.6 g/L water",
            "Isoprothiolane 40 EC @ 1.5 ml/L as preventive spray",
            "Avoid excess nitrogen fertilisation",
            "Drain standing water for 3–4 days at tillering",
        ],
        "resistant_varieties": ["Pusa Basmati 1121", "Swarna Sub1", "MTU 1010"],
        "kisan_advisory": "ब्लास्ट के लिए ट्राइसाइक्लाजोल का छिड़काव करें। अधिक नाइट्रोजन से बचें।",
    },
    {
        "disease_name": "Cotton Bollworm",
        "pathogen": "Helicoverpa armigera (insect pest)",
        "crops_affected": ["cotton", "tomato", "chickpea"],
        "symptoms": "Entry holes in bolls, green caterpillars inside bolls, shedding of young bolls",
        "favorable_conditions": "Temperature 25–30°C, dryspells after rains",
        "management": [
            "Spray Chlorantraniliprole 18.5 SC @ 0.3 ml/L",
            "Emamectin benzoate 5 SG @ 0.4 g/L",
            "Use pheromone traps (5/acre) for monitoring",
            "Spray Neem-based insecticide (NSKE 5%) as preventive measure",
        ],
        "resistant_varieties": ["Bt cotton hybrids: JKCH 1947, MRC 7017 BG II"],
        "kisan_advisory": "ਬੋਲਵਰਮ ਲਈ ਕਲੋਰਾਂਟਰਾਨਿਲੀਪ੍ਰੋਲ ਦਾ ਛਿੜਕਾਅ ਕਰੋ।",
    },
    {
        "disease_name": "Khaira Disease (Rice Zinc Deficiency)",
        "pathogen": "Zinc deficiency (abiotic)",
        "crops_affected": ["rice"],
        "symptoms": "Rusty brown spots on older leaves, stunted growth, pale yellow new leaves",
        "favorable_conditions": "Flooded soils with pH > 7.5, calcareous soils, cold weather",
        "management": [
            "Apply zinc sulphate (ZnSO4) @ 25 kg/ha as basal dose",
            "Foliar spray of 0.5% ZnSO4 + 0.25% slaked lime (2–3 sprays)",
            "Maintain proper water management – drain and re-irrigate",
        ],
        "resistant_varieties": ["Pusa Sugandh 5", "Swarna"],
        "kisan_advisory": "खैरा रोग के लिए जिंक सल्फेट 25 किग्रा/हेक्टेयर डालें।",
    },
]

SCHEME_FAQS = [
    {
        "question": "Who is eligible for PM-KISAN?",
        "answer": "All landholding farmer families with cultivable land are eligible for PM-KISAN. The scheme provides ₹6,000 per year in 3 equal instalments. Small, marginal, and large farmers are all covered, but farmers who are income tax payers, constitutional post holders, or central/state government employees (except Group D workers) are excluded. [Source: PM-KISAN official guidelines]",
    },
    {
        "question": "What subsidy is available for drip irrigation?",
        "answer": "Under PMKSY-PDMC, drip and sprinkler irrigation systems are subsidised at 55% for small/marginal farmers and 45% for other farmers. State-level additional subsidy may apply. Apply through your district Agriculture office or the PMKSY portal. [Source: PMKSY guidelines]",
    },
    {
        "question": "What is PMFBY crop insurance?",
        "answer": "Pradhan Mantri Fasal Bima Yojana (PMFBY) provides crop insurance at premium rates of 2% for Kharif crops, 1.5% for Rabi crops, and 5% for commercial/horticultural crops. The government pays the rest of the actuarial premium. Claims are settled based on yield loss assessed by the state government. [Source: PMFBY official notification]",
    },
    {
        "question": "How do I apply for Kisan Credit Card (KCC)?",
        "answer": "Visit your nearest bank (SBI, PNB, cooperative bank) with land documents, Aadhaar card, and passport-size photographs. KCC provides short-term credit up to ₹3 lakh at 4% interest rate (after government subvention) for crop production expenses. [Source: RBI KCC guidelines]",
    },
    {
        "question": "What is the MSP for wheat in 2024-25?",
        "answer": "The Minimum Support Price (MSP) for wheat for 2024-25 Rabi season is ₹2,275 per quintal, announced by the Cabinet Committee on Economic Affairs (CCEA). Procurement is done by FCI and state agencies. [Source: CACP MSP announcement 2024]",
    },
]

SOIL_CLASSIFICATION = [
    {
        "soil_type": "Black Cotton Soil (Regur)",
        "region": "Maharashtra, Madhya Pradesh, Gujarat, Andhra Pradesh",
        "characteristics": "High clay content (30–80%), high water-holding capacity, cracks when dry, self-mulching, rich in calcium and magnesium",
        "ph_range": "7.5–8.5 (alkaline)",
        "crops_best_suited": ["cotton", "soybean", "wheat", "jowar", "chickpea"],
        "management_tips": "Avoid waterlogging. Add gypsum for sodic conditions. Deep ploughing once in 3 years. Zinc application essential.",
        "fertiliser_notes": "Potassium generally not required. Phosphorus fixation high – use SSP in splits.",
    },
    {
        "soil_type": "Alluvial Soil",
        "region": "Indo-Gangetic Plains – Punjab, Haryana, UP, Bihar, West Bengal",
        "characteristics": "High fertility, sandy loam to clay loam texture, good water retention, rich in potash and lime",
        "ph_range": "6.5–8.0",
        "crops_best_suited": ["wheat", "rice", "sugarcane", "maize", "pulses", "oilseeds"],
        "management_tips": "Maintain organic matter with green manuring. Avoid burning stubble. Use balanced NPK fertilisation.",
        "fertiliser_notes": "Zinc deficiency common in rice-wheat systems. Apply ZnSO4 25 kg/ha once in 3 years.",
    },
    {
        "soil_type": "Red and Yellow Soil",
        "region": "Eastern Deccan – Odisha, Jharkhand, parts of Karnataka and Tamil Nadu",
        "characteristics": "Sandy loam texture, low water retention, low in nitrogen, phosphorus, and organic matter, good drainage",
        "ph_range": "5.5–7.5",
        "crops_best_suited": ["groundnut", "cotton", "millets", "pulses", "tobacco"],
        "management_tips": "Add organic manure (FYM 10 t/ha). Contour bunding for water conservation. Lime application where pH < 6.",
        "fertiliser_notes": "Phosphorus fertilisation critical. DAP or SSP + urea recommended.",
    },
    {
        "soil_type": "Laterite Soil",
        "region": "Kerala, Karnataka, Maharashtra (coastal), Goa, parts of Odisha",
        "characteristics": "High iron and aluminium oxides, low base saturation, hardpans on drying, very low fertility",
        "ph_range": "4.5–6.0 (acidic)",
        "crops_best_suited": ["cashew", "coconut", "rubber", "tea", "coffee", "tapioca"],
        "management_tips": "Lime application mandatory (2–3 t/ha). Heavy mulching to retain moisture. Green manuring.",
        "fertiliser_notes": "Phosphorus fixation very high. Apply rock phosphate. Boron deficiency common – apply borax.",
    },
    {
        "soil_type": "Sandy / Desert Soil",
        "region": "Rajasthan, parts of Gujarat and Haryana (Aravalli region)",
        "characteristics": "Coarse texture, very low water retention, low organic matter, high permeability, wind erosion prone",
        "ph_range": "7.0–8.5",
        "crops_best_suited": ["bajra", "moth bean", "cluster bean (guar)", "sesame", "castor"],
        "management_tips": "Windbreaks and shelter belts. Drip irrigation highly recommended. Add organic manure to improve water retention.",
        "fertiliser_notes": "Nitrogen losses high – split application mandatory. Phosphorus and micronutrients needed.",
    },
]

FERTILISER_FAQS = [
    {
        "question": "What is the price of DAP fertiliser?",
        "answer": "DAP (Di-Ammonium Phosphate) MRP is ₹1,350 per 50 kg bag as per government-fixed price 2024. Farmers can buy from authorised dealers at this price. Avoid buying from unregulated sources as adulteration is common. [Source: Department of Fertilisers, GoI]",
    },
    {
        "question": "How much urea should I apply to wheat?",
        "answer": "For wheat, recommended nitrogen dose is 120 kg N/ha. Since urea contains 46% N, apply approximately 260 kg urea/ha in splits: 1/3 as basal dose, 1/3 at first irrigation (CRI stage), 1/3 at second irrigation. Avoid single large application which increases losses. [Source: ICAR wheat production guide]",
    },
    {
        "question": "What is nano urea and how to use it?",
        "answer": "Nano Urea (liquid) by IFFCO contains 4% nitrogen in nanoscale form. Apply 2–4 ml per litre of water as foliar spray at tillering and heading stages. One 500 ml bottle can replace one bag of urea for foliar application. Do not mix with alkaline pesticides. [Source: IFFCO Nano Urea guidelines]",
    },
    {
        "question": "Can I mix DAP and MOP in the same application?",
        "answer": "Yes, DAP and MOP can be mixed as basal application and applied together before sowing. This is a common practice for providing NPK nutrients simultaneously. Ensure uniform mixing and immediate incorporation into soil. [Source: ICAR fertiliser use recommendations]",
    },
]


async def seed_knowledge_files() -> None:
    """Write static knowledge JSON files to data directories."""
    out_map = {
        Path(settings.data_root) / "pest_disease" / "disease_knowledge.json": DISEASE_KNOWLEDGE,
        Path(settings.data_root) / "government" / "scheme_faqs.json": SCHEME_FAQS,
        Path(settings.data_root) / "soil" / "indian_soil_classification.json": SOIL_CLASSIFICATION,
        Path(settings.data_root) / "fertilizer" / "fertilizer_faqs.json": FERTILISER_FAQS,
    }
    for path, data in out_map.items():
        ensure_dir(path.parent)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.success(f"Seeded: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def main(source: str) -> None:
    if source in ("all", "knowledge"):
        await seed_knowledge_files()
    if source in ("all", "weather"):
        await download_weather()
    if source in ("all", "mandi"):
        await download_mandi_prices()
    if source in ("all", "usda"):
        await download_usda_nass()
    if source in ("all", "soil"):
        await download_soilgrids()
    if source in ("all", "kaggle"):
        await download_kaggle_metadata()
    logger.success("Data download complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgroSight Data Downloader")
    parser.add_argument(
        "--source",
        default="all",
        choices=["all", "weather", "mandi", "usda", "soil", "kaggle", "knowledge"],
        help="Which data source to download",
    )
    args = parser.parse_args()
    asyncio.run(main(args.source))
