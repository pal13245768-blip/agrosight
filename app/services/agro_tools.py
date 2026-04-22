"""
AgroSight – Agent Tools
========================
Three LangGraph / LangChain tool-compatible callables:
  1. get_weather_advisory  – OpenWeatherMap current conditions + agro advisory
  2. get_mandi_price       – data.gov.in mandi price for commodity + market
  3. fertiliser_calculator – dose calculation from nutrient requirements

Each function is a plain Python async function that the LangGraph ReAct agent
calls when it decides tool use is needed.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.utils.config import get_settings
from app.utils.logger import logger

settings = get_settings()

# Static price fallback (used when API is down)
_PRICE_FALLBACK: dict[str, float] = {
    "wheat": 2275,
    "rice": 2300,
    "cotton": 7020,
    "soybean": 4600,
    "maize": 2090,
    "sugarcane": 340,
    "onion": 1200,
    "tomato": 800,
    "potato": 1200,
}


# ---------------------------------------------------------------------------
# Tool 1 – Weather advisory
# ---------------------------------------------------------------------------


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
async def get_weather_advisory(location: str) -> dict[str, Any]:
    """
    Fetch current weather for *location* (city name or lat,lon) from OpenWeatherMap
    and generate a short agronomy advisory based on conditions.

    Returns:
        dict with keys: location, temperature_c, humidity_pct, wind_kmh,
                        condition, advisory, source
    """
    if not settings.openweather_api_key:
        return {
            "error": "OpenWeatherMap API key not configured",
            "advisory": "Weather data unavailable. Please check conditions locally.",
        }

    url = f"{settings.openweather_base_url}/weather"
    params = {
        "q": location,
        "appid": settings.openweather_api_key,
        "units": "metric",
        "lang": "en",
    }

    async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

    temp_c = data["main"]["temp"]
    humidity = data["main"]["humidity"]
    wind_mps = data["wind"]["speed"]
    wind_kmh = round(wind_mps * 3.6, 1)
    condition = data["weather"][0]["description"]
    rain_1h = data.get("rain", {}).get("1h", 0)

    advisory = _generate_advisory(temp_c, humidity, wind_kmh, condition, rain_1h)

    return {
        "location": data.get("name", location),
        "temperature_c": temp_c,
        "humidity_pct": humidity,
        "wind_kmh": wind_kmh,
        "condition": condition,
        "rain_last_hour_mm": rain_1h,
        "advisory": advisory,
        "source": "OpenWeatherMap",
    }


def _generate_advisory(
    temp_c: float,
    humidity: int,
    wind_kmh: float,
    condition: str,
    rain_mm: float,
) -> str:
    """Rule-based agronomy advisory from weather parameters."""
    tips: list[str] = []

    if rain_mm > 10:
        tips.append("Heavy rainfall detected — avoid irrigation and field operations today.")
    elif rain_mm > 2:
        tips.append("Light rain recorded — irrigation likely not required for the next 24h.")
    elif humidity < 40 and temp_c > 35:
        tips.append("Hot and dry conditions — irrigate crops early morning or evening.")

    if wind_kmh > 40:
        tips.append("High winds — postpone pesticide / fertiliser spraying to avoid drift.")
    elif wind_kmh > 20:
        tips.append("Moderate winds — use directed nozzles if spraying is necessary.")

    if temp_c > 40:
        tips.append("Extreme heat — protect nurseries and seedlings with shade netting.")
    elif temp_c < 5:
        tips.append("Near-frost temperatures — protect sensitive crops and avoid planting.")

    if humidity > 85 and temp_c > 25:
        tips.append("High humidity — monitor for fungal diseases (blast, blight). Consider preventive fungicide.")

    if not tips:
        tips.append("Conditions appear normal. Continue regular field monitoring.")

    return " ".join(tips)


# ---------------------------------------------------------------------------
# Tool 2 – Mandi price lookup
# ---------------------------------------------------------------------------


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
async def get_mandi_price(commodity: str, market: str = "", state: str = "Gujarat") -> dict[str, Any]:
    """
    Fetch modal price for *commodity* at *market* from data.gov.in Agmarknet API.

    Falls back to static table if API is unreachable.
    """
    api_key = settings.data_gov_api_key_1
    resource_id = settings.data_gov_mandi_resource_id

    if not api_key:
        return _price_fallback(commodity, market)

    url = f"{settings.data_gov_base_url}/resource/{resource_id}"
    params: dict[str, Any] = {
        "api-key": api_key,
        "format": "json",
        "filters[commodity]": commodity.title(),
        "limit": 10,
    }
    if market:
        params["filters[market]"] = market.title()
    if state:
        params["filters[state]"] = state.title()

    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout, follow_redirects=True) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        records = data.get("records", [])
        if not records:
            return _price_fallback(commodity, market, note="No records found in API; using MSP fallback.")

        latest = records[0]
        return {
            "commodity": latest.get("commodity", commodity),
            "market": latest.get("market", market),
            "state": latest.get("state", state),
            "modal_price_inr": latest.get("modal_price"),
            "min_price_inr": latest.get("min_price"),
            "max_price_inr": latest.get("max_price"),
            "arrival_date": latest.get("arrival_date"),
            "source": "data.gov.in / Agmarknet",
        }

    except Exception as exc:
        logger.warning(f"Mandi API error ({exc}). Using fallback price.")
        return _price_fallback(commodity, market, note=f"API error: {exc}")


def _price_fallback(commodity: str, market: str = "", note: str = "") -> dict[str, Any]:
    key = commodity.lower().strip()
    price = _PRICE_FALLBACK.get(key)
    result: dict[str, Any] = {
        "commodity": commodity,
        "market": market or "N/A (fallback)",
        "modal_price_inr": price,
        "source": "Static MSP fallback table (data.gov.in unavailable)",
    }
    if note:
        result["note"] = note
    return result


# ---------------------------------------------------------------------------
# Tool 3 – Fertiliser dose calculator
# ---------------------------------------------------------------------------

# Nutrient content of common fertilisers (% N, P2O5, K2O)
_FERTILISER_COMPOSITION: dict[str, dict[str, float]] = {
    "urea": {"N": 46.0, "P": 0.0, "K": 0.0},
    "dap": {"N": 18.0, "P": 46.0, "K": 0.0},
    "mop": {"N": 0.0, "P": 0.0, "K": 60.0},
    "ssp": {"N": 0.0, "P": 16.0, "K": 0.0},
    "10:26:26": {"N": 10.0, "P": 26.0, "K": 26.0},
    "12:32:16": {"N": 12.0, "P": 32.0, "K": 16.0},
    "npk 20:20:0": {"N": 20.0, "P": 20.0, "K": 0.0},
}

# Per-crop nutrient requirements (kg/ha) — simplified representative values
_CROP_NUTRIENT_REQ: dict[str, dict[str, float]] = {
    "wheat": {"N": 120, "P": 60, "K": 40},
    "rice": {"N": 120, "P": 60, "K": 60},
    "cotton": {"N": 150, "P": 60, "K": 60},
    "maize": {"N": 150, "P": 75, "K": 50},
    "soybean": {"N": 20, "P": 80, "K": 40},
    "sugarcane": {"N": 250, "P": 100, "K": 120},
    "potato": {"N": 150, "P": 100, "K": 150},
    "onion": {"N": 100, "P": 60, "K": 60},
    "tomato": {"N": 120, "P": 80, "K": 60},
    "groundnut": {"N": 25, "P": 50, "K": 50},
    "mango": {"N": 100, "P": 50, "K": 100},
}


async def fertiliser_calculator(
    crop: str,
    area_acres: float,
    fertiliser: str = "urea",
    nutrient: str = "N",
) -> dict[str, Any]:
    """
    Calculate fertiliser quantity needed for *crop* on *area_acres*.

    Args:
        crop: crop name (e.g. 'wheat')
        area_acres: field area in acres
        fertiliser: fertiliser product (e.g. 'urea', 'dap')
        nutrient: which nutrient to calculate for ('N', 'P', or 'K')

    Returns:
        dict with dose_kg, bags_50kg, cost_estimate_inr, notes
    """
    crop_key = crop.lower().strip()
    fert_key = fertiliser.lower().strip()
    nut_key = nutrient.upper()

    crop_req = _CROP_NUTRIENT_REQ.get(crop_key)
    fert_comp = _FERTILISER_COMPOSITION.get(fert_key)

    if crop_req is None:
        return {
            "error": f"Crop '{crop}' not in database. Please consult local ICAR advisory.",
            "available_crops": list(_CROP_NUTRIENT_REQ.keys()),
        }

    if fert_comp is None:
        return {
            "error": f"Fertiliser '{fertiliser}' not in database.",
            "available_fertilisers": list(_FERTILISER_COMPOSITION.keys()),
        }

    if nut_key not in ("N", "P", "K"):
        return {"error": "Nutrient must be N, P, or K."}

    nutrient_content_pct = fert_comp.get(nut_key, 0)
    if nutrient_content_pct == 0:
        return {
            "error": f"{fertiliser.upper()} contains no {nut_key}. Choose appropriate fertiliser.",
        }

    area_ha = area_acres * 0.4047
    required_nutrient_kg = crop_req[nut_key] * area_ha
    fertiliser_dose_kg = required_nutrient_kg / (nutrient_content_pct / 100)
    bags_50kg = round(fertiliser_dose_kg / 50, 1)

    # Rough price estimates (INR/50kg bag, 2024 approx)
    bag_prices = {"urea": 266, "dap": 1350, "mop": 900, "ssp": 450}
    bag_price = bag_prices.get(fert_key, 1000)
    cost_inr = round(bags_50kg * bag_price)

    return {
        "crop": crop,
        "area_acres": area_acres,
        "area_ha": round(area_ha, 2),
        "nutrient": nut_key,
        "nutrient_required_kg": round(required_nutrient_kg, 1),
        "fertiliser": fertiliser.upper(),
        "fertiliser_dose_kg": round(fertiliser_dose_kg, 1),
        "bags_50kg": bags_50kg,
        "cost_estimate_inr": cost_inr,
        "notes": (
            f"Based on recommended {nut_key} dose of {crop_req[nut_key]} kg/ha for {crop}. "
            "Actual requirement may vary by soil test results. "
            "Apply in split doses as per crop growth stage."
        ),
        "source": "AgroSight nutrient requirement database",
    }
