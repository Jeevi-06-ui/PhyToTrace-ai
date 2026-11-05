# PhytoTrace AI — Weather Intelligence & Agronomic Telemetry Guide

This document explains the weather intelligence system in PhytoTrace AI, including telemetry integration, explicit user consent mechanisms, tomato-specific microclimate risk calculations, and failover options.

---

## 1. System Overview

Tomato leaf-miner (*Liriomyza spp.* and *Tuta absoluta*) population dynamics and larval feeding intensity correlate strongly with ambient canopy temperature, relative humidity, and rainfall patterns.

PhytoTrace incorporates real-time hyper-local weather telemetry to contextualize computer vision inspections with actionable agronomic advisories:

```
    ┌───────────────────────────┐
    │  Browser Geolocation API  │  (Explicit user consent required)
    └─────────────┬─────────────┘
                  │ lat, lon (e.g. 12.9716, 77.5946)
                  ▼
    ┌───────────────────────────┐
    │ WeatherIntelligenceService │  (phytotrace.services.weather)
    └─────────────┬─────────────┘
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
  [ Open-Meteo API ]   [ Google Maps Weather (Optional) ]
  (High-accuracy,      (Enterprise SLA fallback)
   free, zero-key)
        │
        ▼
  [ Tomato Agronomic Rules Engine ]
  - Humidity Thresholds (RH >= 75%)
  - Thermal Stress (T > 32°C)
  - Precipitation Drainage Warnings
```

---

## 2. Privacy & Explicit Geolocation Consent

In accordance with strict privacy principles:
- **No Automatic Geotracking:** The application NEVER queries the browser `navigator.geolocation` API upon page load.
- **Explicit User Action:** On the `/weather` screen, a clean privacy banner informs the user why location is requested. Only when the user clicks **"Allow Location"** is the browser prompt displayed.
- **Default Baseline:** If location permission is declined or not granted, PhytoTrace defaults to **Bengaluru, Karnataka (Lat: 12.9716, Lon: 77.5946)**, the headquarters of ICAR-IIHR and Presidency University.

---

## 3. Configuration & Fallback Options

### Option A: Open-Meteo (Active by Default)
- **Status:** Pre-configured and operational with zero setup.
- **Endpoint:** `https://api.open-meteo.com/v1/forecast`
- **Telemetry Gathered:**
  - `temperature_2m`
  - `relative_humidity_2m`
  - `precipitation`
  - `wind_speed_10m`
  - 5-day daily forecast (max temp, min temp, precipitation sum)

### Option B: Google Maps Platform Weather API (Optional)
If required for enterprise deployments:
1. Enable the Weather API in the [Google Cloud Console](https://console.cloud.google.com/).
2. Obtain an API key.
3. Add to `.env`:
   ```bash
   GOOGLE_MAPS_WEATHER_API_KEY="AIzaSyYourMapsWeatherKey"
   ```

---

## 4. Tomato Microclimate Risk Model

The `WeatherIntelligenceService` computes three domain-specific risk alerts:

| Condition | Threshold | Agronomic Implication & Action |
| :--- | :--- | :--- |
| **High Canopy Humidity** | $\text{RH} \ge 75\%$ | Accelerates leaf-miner egg hatching and induces secondary fungal sporulation (*Alternaria solani*) within serpentine mines. **Action:** Increase ventilation and cease overhead sprinkler wetting. |
| **Optimal Humidity** | $\text{RH} < 75\%$ | Normal vegetative conditions; monitor according to weekly schedule. |
| **Precipitation / Runoff** | $\text{Rain} > 2.0\text{ mm}$ | Saturated root zones impede nutrient uptake. **Action:** Check bed drainage ditches and pause scheduled surface fertigation. |
| **Heat Stress** | $\text{Temp} > 32^\circ\text{C}$ | Surpasses optimum vegetative growth range (21–29°C). Increases flower drop. **Action:** Increase morning irrigation duration. |

---

## 5. Verification Checklist

Run the weather integration tests:

```bash
.venv/bin/pytest tests/test_api_v1_and_services.py -k "test_v1_weather"
```

Expected output:
```
tests/test_api_v1_and_services.py::test_v1_weather_current PASSED
tests/test_api_v1_and_services.py::test_v1_weather_crop_advice PASSED
```

