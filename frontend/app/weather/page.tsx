"use client";

import { useEffect, useState, useRef } from "react";
import { 
  CloudSun, 
  Droplets, 
  Wind, 
  CloudRain, 
  Thermometer, 
  MapPin, 
  AlertTriangle, 
  CheckCircle2, 
  Info, 
  RefreshCw, 
  Compass,
  Sprout,
  Sun,
  Cloud,
  CloudLightning,
  ShieldCheck,
  Bookmark,
  Sparkles,
  Navigation,
  Calendar,
  Layers,
  Activity,
  Search
} from "lucide-react";
import { useLanguage } from "@/context/LanguageContext";
import { useAuth } from "@/context/AuthContext";
import StatCard from "@/components/StatCard";

interface AgronomicCalculation {
  reference_evapotranspiration_mm_day: number;
  crop_coefficient_kc: number;
  crop_water_demand_etc_mm_day: number;
  net_irrigation_depth_mm: number;
  estimated_daily_liters: number;
  irrigation_system_efficiency_pct: number;
  farm_area_acres: number;
  growth_stage: string;
  methodology: string;
}

interface AgronomicGuidance {
  irrigation_advice: string;
  pest_scouting_recommendation: string;
  fertilizer_guidance: string;
  agronomic_calculation: AgronomicCalculation | null;
  alerts: Array<{
    type: "warning" | "advisory" | "info" | string;
    title: string;
    message: string;
  }>;
  ai_explanation: string | null;
  citations: string[];
  language?: string;
}

interface WeatherData {
  location_name: string;
  latitude: number;
  longitude: number;
  timestamp: string;
  source: string;
  provider_status?: string;
  current: {
    temperature_c: number;
    feels_like_c?: number;
    humidity_pct: number;
    rainfall_mm: number | null;
    precipitation_probability?: number;
    wind_kmh: number;
    condition: string | null;
    condition_icon?: string;
    observation_time?: string;
  };
  hourly: Array<{
    time: string;
    temperature: number;
    humidity?: number;
    precipitation_prob: number | null;
    wind_kmh?: number;
    condition?: string;
  }>;
  daily: Array<{
    date: string;
    max_temp: number;
    min_temp: number;
    rainfall_mm: number | null;
    precipitation_prob?: number;
    condition?: string;
    condition_icon?: string;
  }>;
  history_24h?: Array<{
    time: string;
    temperature: number;
    rainfall_mm: number | null;
  }>;
  public_alerts?: Array<{
    title: string;
    description: string;
    severity?: string;
  }>;
  agronomic_guidance: AgronomicGuidance;
}

export default function WeatherPage() {
  const { t, locale } = useLanguage();
  const { idToken, user } = useAuth();

  const [weather, setWeather] = useState<WeatherData | null>(null);
  const [loading, setLoading] = useState(false);
  const [calculatingAdvice, setCalculatingAdvice] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saveStatus, setSaveStatus] = useState<string | null>(null);

  // Location and permission state
  const [permissionState, setPermissionState] = useState<"prompt" | "granted" | "manual" | "dismissed">("prompt");
  const [latitude, setLatitude] = useState("12.9716");
  const [longitude, setLongitude] = useState("77.5946");
  const [locationName, setLocationName] = useState("Bengaluru, Karnataka");

  // Farmer Profile Inputs
  const [plantingDate, setPlantingDate] = useState("");
  const [growthStage, setGrowthStage] = useState("flowering");
  const [soilType, setSoilType] = useState("red_loam");
  const [irrigationMethod, setIrrigationMethod] = useState("drip");
  const [farmArea, setFarmArea] = useState("1.5");
  const [farmAreaUnit, setFarmAreaUnit] = useState("acres");
  const [recentIrrigation, setRecentIrrigation] = useState("Yesterday morning (2 hours drip)");
  const [fertilizerHistory, setFertilizerHistory] = useState("NPK 10:26:26 applied 6 days ago");
  const [soilMoisture, setSoilMoisture] = useState("65");
  const [soilPh, setSoilPh] = useState("6.5");

  const [isResolvingLocation, setIsResolvingLocation] = useState(false);
  const skipNextGeocodeRef = useRef(false);

  // Initial load
  useEffect(() => {
    void fetchWeather(Number(latitude), Number(longitude), locationName);
  }, [locale]);

  // Debounced reverse geocoding when latitude / longitude change manually
  useEffect(() => {
    if (skipNextGeocodeRef.current) {
      skipNextGeocodeRef.current = false;
      return;
    }
    const latNum = parseFloat(latitude);
    const lonNum = parseFloat(longitude);
    if (isNaN(latNum) || isNaN(lonNum) || latNum < -90 || latNum > 90 || lonNum < -180 || lonNum > 180) {
      return;
    }

    const timer = setTimeout(async () => {
      try {
        setIsResolvingLocation(true);
        const res = await fetch(`/api/v1/weather/reverse-geocode?lat=${latNum}&lon=${lonNum}&lang=${locale}`);
        if (res.ok) {
          const data = await res.json();
          if (data && data.location_name) {
            setLocationName(data.location_name);
          }
        }
      } catch {
        // silent network fallback
      } finally {
        setIsResolvingLocation(false);
      }
    }, 600);

    return () => clearTimeout(timer);
  }, [latitude, longitude, locale]);

  async function fetchWeather(lat: number, lon: number, location: string) {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(
        `/api/v1/weather/current?lat=${lat}&lon=${lon}&location=${encodeURIComponent(location)}&lang=${locale}`
      );
      if (!response.ok) throw new Error("Unable to retrieve live weather.");
      const data: WeatherData = await response.json();
      setWeather(data);
      if (data.location_name) {
        setLocationName(data.location_name);
      }
    } catch {
      setError("Unable to retrieve live weather. Please check network connection.");
    } finally {
      setLoading(false);
    }
  }

  function handleUseMyLocation() {
    if (!navigator.geolocation) {
      setError("Device geolocation is not supported by your browser. Please enter coordinates manually.");
      setPermissionState("manual");
      return;
    }
    setLoading(true);
    navigator.geolocation.getCurrentPosition(
      async (pos) => {
        const lat = pos.coords.latitude;
        const lon = pos.coords.longitude;
        skipNextGeocodeRef.current = true;
        setLatitude(lat.toFixed(4));
        setLongitude(lon.toFixed(4));
        setPermissionState("granted");

        let detectedName = "Field Location (GPS)";
        try {
          const res = await fetch(`/api/v1/weather/reverse-geocode?lat=${lat}&lon=${lon}&lang=${locale}`);
          if (res.ok) {
            const data = await res.json();
            if (data?.location_name) {
              detectedName = data.location_name;
            }
          }
        } catch {}

        setLocationName(detectedName);
        void fetchWeather(lat, lon, detectedName);
      },
      (err) => {
        setError(`Location access declined (${err.message}). You can enter your farm location manually.`);
        setPermissionState("manual");
        setLoading(false);
      },
      { timeout: 10000, enableHighAccuracy: false }
    );
  }

  async function handleSearchPlaceName(placeQuery: string) {
    if (!placeQuery || placeQuery.trim().length < 2) return;
    setIsResolvingLocation(true);
    try {
      const res = await fetch(`/api/v1/weather/geocode?q=${encodeURIComponent(placeQuery.trim())}&lang=${locale}`);
      if (res.ok) {
        const data = await res.json();
        if (data?.results && data.results.length > 0) {
          const match = data.results[0];
          skipNextGeocodeRef.current = true;
          setLatitude(match.latitude.toFixed(4));
          setLongitude(match.longitude.toFixed(4));
          setLocationName(match.location_name);
          void fetchWeather(match.latitude, match.longitude, match.location_name);
          return;
        } else {
          setError(`No locations found for "${placeQuery}". Try entering taluk/district.`);
        }
      }
    } catch {
      setError("Location lookup failed. Check network connection.");
    } finally {
      setIsResolvingLocation(false);
    }
  }

  async function handleUpdateCropAdvice(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!weather) return;
    setCalculatingAdvice(true);
    setSaveStatus(null);
    try {
      const payload = {
        latitude: weather.latitude,
        longitude: weather.longitude,
        location_name: weather.location_name,
        crop: "Tomato",
        planting_date: plantingDate || null,
        growth_stage: growthStage,
        soil_type: soilType,
        irrigation_method: irrigationMethod,
        farm_area: farmArea ? Number(farmArea) : null,
        farm_area_unit: farmAreaUnit,
        recent_irrigation: recentIrrigation || null,
        fertilizer_history: fertilizerHistory || null,
        soil_moisture_pct: soilMoisture ? Number(soilMoisture) : null,
        soil_ph: soilPh ? Number(soilPh) : null,
        language: locale,
        save_to_profile: Boolean(idToken),
      };

      const response = await fetch("/api/v1/weather/crop-advice", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(idToken ? { Authorization: `Bearer ${idToken}` } : {}),
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) throw new Error("Advice calculation failed");
      const result = await response.json();

      setWeather((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          agronomic_guidance: result.agronomic_guidance,
        };
      });

      if (idToken) {
        setSaveStatus(t("weather.profile_saved", "Farm weather profile saved to account."));
      }
    } catch {
      setError("Could not recalculate crop guidance. Please try again.");
    } finally {
      setCalculatingAdvice(false);
    }
  }

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-12">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-4 border-b border-[#294B3C]">
        <div>
          <span className="text-xs font-bold uppercase tracking-wider text-[#6AE6A0] flex items-center gap-1.5">
            <Sprout className="w-3.5 h-3.5 text-[#6AE6A0]" />
            Agrometeorological Intelligence & Farmer Care
          </span>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-[#F2FAF5] tracking-tight mt-1 flex items-center gap-2.5">
            <CloudSun className="w-7 h-7 text-[#6AE6A0]" />
            <span>{t("weather.title", "Weather & Crop Intelligence")}</span>
          </h1>
          <p className="text-xs sm:text-sm text-[#A6B9AE] mt-1 max-w-2xl">
            {t("weather.subtitle", "Localized agro-meteorological telemetry and tomato crop protection guidance.")}
          </p>
        </div>

        {weather && (
          <div className="flex flex-col sm:items-end gap-1">
            <span className="text-[11px] font-mono text-[#6AE6A0] bg-[#152C22] px-3 py-1 rounded-full border border-[#294B3C] flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-[#6AE6A0] animate-pulse" />
              {weather.source}
            </span>
            <span className="text-[10px] text-[#A6B9AE]">
              Updated: {new Date(weather.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </span>
          </div>
        )}
      </div>

      {/* Task 2: Explicit Location Permission Card */}
      {permissionState === "prompt" && (
        <div className="p-5 rounded-2xl bg-[#152C22] border-2 border-[#6AE6A0]/40 shadow-xl flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div className="flex items-start gap-3.5">
            <div className="w-10 h-10 rounded-xl bg-[#10231B] border border-[#294B3C] flex items-center justify-center text-[#6AE6A0] shrink-0">
              <Navigation className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-[#F2FAF5]">
                {t("weather.permission_title", "Allow location access to receive weather-based tomato crop guidance.")}
              </h3>
              <p className="text-xs text-[#A6B9AE] mt-0.5 leading-relaxed">
                {t("weather.permission_desc", "PhytoTrace calculates pest hatching risks, canopy humidity alerts, and irrigation schedules based on your local microclimate.")}
              </p>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2.5 shrink-0 w-full md:w-auto">
            <button
              onClick={handleUseMyLocation}
              className="px-4 py-2 rounded-xl bg-[#6AE6A0] hover:bg-[#5BD690] text-[#0A1814] text-xs font-bold transition flex items-center gap-1.5 shadow-md"
            >
              <Navigation className="w-3.5 h-3.5" />
              {t("weather.use_loc_btn", "Use My Location")}
            </button>
            <button
              onClick={() => setPermissionState("manual")}
              className="px-4 py-2 rounded-xl border border-[#294B3C] hover:bg-[#1A382B] text-[#F2FAF5] text-xs font-semibold transition flex items-center gap-1.5"
            >
              <MapPin className="w-3.5 h-3.5 text-[#6AE6A0]" />
              {t("weather.manual_btn", "Enter Farm Location")}
            </button>
            <button
              onClick={() => setPermissionState("dismissed")}
              className="px-3 py-2 text-xs text-[#A6B9AE] hover:text-[#F2FAF5] transition"
            >
              {t("weather.not_now_btn", "Skip")}
            </button>
          </div>
        </div>
      )}

      {/* Manual Location Form */}
      {(permissionState === "manual" || permissionState === "granted" || permissionState === "dismissed") && (
        <form
          className="flex flex-wrap items-end gap-3 rounded-2xl border border-[#294B3C] bg-[#10231B] p-4 shadow-sm"
          onSubmit={(e) => {
            e.preventDefault();
            void fetchWeather(Number(latitude), Number(longitude), locationName);
          }}
        >
          <div className="flex-1 min-w-[200px]">
            <label className="text-[11px] font-bold text-[#A6B9AE] uppercase tracking-wider block mb-1">
              {t("weather.location_name", "Farm / Location Name")}
            </label>
            <div className="relative flex items-center">
              <MapPin className="w-4 h-4 text-[#6AE6A0] absolute left-3 top-2.5" />
              <input
                className="w-full rounded-xl border border-[#294B3C] bg-[#152C22] pl-9 pr-20 py-2 text-xs text-[#F2FAF5] focus:outline-none focus:border-[#6AE6A0]"
                required
                maxLength={96}
                placeholder={t("weather.search_placeholder", "Search city, taluk (e.g. Kolar, Hubli)...")}
                value={locationName}
                onChange={(e) => setLocationName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    void handleSearchPlaceName(locationName);
                  }
                }}
              />
              <button
                type="button"
                onClick={() => void handleSearchPlaceName(locationName)}
                disabled={isResolvingLocation || loading}
                className="absolute right-1 px-2.5 py-1 rounded-lg bg-[#294B3C] hover:bg-[#3B6653] text-[#F2FAF5] text-[10px] font-bold transition flex items-center gap-1 shadow-sm"
                title="Search location coordinates"
              >
                {isResolvingLocation ? (
                  <RefreshCw className="w-3 h-3 animate-spin text-[#6AE6A0]" />
                ) : (
                  <Search className="w-3 h-3 text-[#6AE6A0]" />
                )}
                <span>{t("weather.search_btn", "Search")}</span>
              </button>
            </div>
          </div>

          <div className="w-[120px]">
            <label className="text-[11px] font-bold text-[#A6B9AE] uppercase tracking-wider block mb-1">
              {t("weather.latitude", "Latitude")}
            </label>
            <input
              className="w-full rounded-xl border border-[#294B3C] bg-[#152C22] p-2 text-xs text-[#F2FAF5] focus:outline-none focus:border-[#6AE6A0]"
              required
              type="number"
              step="any"
              min="-90"
              max="90"
              value={latitude}
              onChange={(e) => setLatitude(e.target.value)}
            />
          </div>

          <div className="w-[120px]">
            <label className="text-[11px] font-bold text-[#A6B9AE] uppercase tracking-wider block mb-1">
              {t("weather.longitude", "Longitude")}
            </label>
            <input
              className="w-full rounded-xl border border-[#294B3C] bg-[#152C22] p-2 text-xs text-[#F2FAF5] focus:outline-none focus:border-[#6AE6A0]"
              required
              type="number"
              step="any"
              min="-180"
              max="180"
              value={longitude}
              onChange={(e) => setLongitude(e.target.value)}
            />
          </div>

          <button
            disabled={loading}
            type="submit"
            className="px-4 py-2 rounded-xl bg-[#6AE6A0] hover:bg-[#5BD690] text-[#0A1814] text-xs font-bold transition flex items-center gap-1.5 h-[34px]"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
            {t("weather.get_forecast", "Get Forecast")}
          </button>

          <button
            disabled={loading}
            type="button"
            onClick={handleUseMyLocation}
            className="px-3.5 py-2 rounded-xl border border-[#294B3C] hover:bg-[#1A382B] text-xs text-[#F2FAF5] transition flex items-center gap-1.5 h-[34px]"
          >
            <Navigation className="w-3.5 h-3.5 text-[#6AE6A0]" />
            {t("weather.use_loc_btn", "Use GPS")}
          </button>
        </form>
      )}

      {error && (
        <div className="p-4 rounded-xl bg-[#271515] border border-[#F17878]/40 text-[#F17878] text-xs flex items-center gap-2.5">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {loading ? (
        <div className="p-16 text-center text-xs text-[#A6B9AE] animate-pulse flex flex-col items-center gap-3">
          <RefreshCw className="w-8 h-8 text-[#6AE6A0] animate-spin" />
          <span>Synchronizing agrometeorological telemetry for {locationName}...</span>
        </div>
      ) : !weather ? (
        <div className="p-16 text-center text-xs text-[#A6B9AE] bg-[#152C22] rounded-2xl border border-[#294B3C]">
          {t("weather.choose_location", "Choose a location to retrieve a forecast.")}
        </div>
      ) : (
        <div className="space-y-8">
          {/* Task 3: Telemetry Stat Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
            <StatCard
              label={t("weather.temp", "Temperature")}
              value={`${weather.current.temperature_c.toFixed(1)}°C`}
              subtext={`${t("weather.feels_like", "Feels like")} ${weather.current.feels_like_c != null ? weather.current.feels_like_c.toFixed(1) : weather.current.temperature_c.toFixed(1)}°C`}
              icon={Thermometer}
              trend={{
                value: weather.current.temperature_c > 32 ? t("weather.heat_stress", "Heat Stress") : t("weather.moderate_temp", "Optimal"),
                isPositive: weather.current.temperature_c <= 30,
              }}
            />

            <StatCard
              label={t("weather.humidity", "Relative Humidity")}
              value={`${weather.current.humidity_pct}%`}
              subtext={weather.current.humidity_pct > 75 ? t("weather.blight_risk", "Blight Spore Risk") : "Normal Canopy"}
              icon={Droplets}
              trend={{
                value: weather.current.humidity_pct > 75 ? t("weather.blight_risk", "Elevated Risk") : "Normal",
                isPositive: weather.current.humidity_pct <= 75,
              }}
              highlight={weather.current.humidity_pct > 75}
            />

            <StatCard
              label={t("weather.rain", "Rainfall & POP")}
              value={weather.current.rainfall_mm == null ? "0.0 mm" : `${weather.current.rainfall_mm.toFixed(1)} mm`}
              subtext={`${weather.current.precipitation_probability ?? 0}% ${t("weather.precip_prob", "rain probability")}`}
              icon={CloudRain}
              trend={{
                value: (weather.current.rainfall_mm ?? 0) > 2 ? "Precipitation" : "Dry Canopy",
                isNeutral: true,
              }}
            />

            <StatCard
              label={t("weather.wind", "Wind Speed")}
              value={`${weather.current.wind_kmh.toFixed(1)} km/h`}
              subtext={t("weather.air_circulation", "Air circulation")}
              icon={Wind}
              trend={{
                value: weather.current.wind_kmh > 20 ? "High Wind" : "Calm",
                isPositive: weather.current.wind_kmh <= 20,
              }}
            />

            <div className="rounded-2xl border border-[#294B3C] bg-[#152C22] p-4 flex flex-col justify-between shadow-sm">
              <span className="text-[11px] font-bold text-[#A6B9AE] uppercase tracking-wider">Current Condition</span>
              <div className="my-1">
                <span className="text-lg font-extrabold text-[#F2FAF5] block leading-tight">
                  {weather.current.condition || "Clear sky"}
                </span>
                <span className="text-[11px] text-[#6AE6A0] font-medium flex items-center gap-1 mt-1">
                  <Sun className="w-3.5 h-3.5" />
                  Tomato Growing Window
                </span>
              </div>
              <span className="text-[10px] text-[#A6B9AE] block truncate">
                Observed at {weather.location_name}
              </span>
            </div>
          </div>

          {/* Task 5 & 6: Agronomic Crop-Care Alerts & Gemini Guidance */}
          <div className="bg-[#152C22] rounded-2xl border border-[#294B3C] p-6 shadow-lg space-y-5">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-[#294B3C] gap-2">
              <div>
                <span className="text-[11px] font-bold uppercase tracking-wider text-[#6AE6A0] flex items-center gap-1.5">
                  <ShieldCheck className="w-4 h-4 text-[#6AE6A0]" />
                  {t("weather.crop_advisory_title", "Evidence-Based Agronomic Crop Advisory")}
                </span>
                <h2 className="text-lg font-bold text-[#F2FAF5] mt-0.5">
                  Targeted Tomato Management · {weather.location_name}
                </h2>
              </div>
              <span className="text-xs text-[#A6B9AE] font-mono">
                ICAR-IIHR Best Practice Standard
              </span>
            </div>

            {/* Alert Banners */}
            <div className="space-y-3">
              {weather.agronomic_guidance.alerts.map((alert, idx) => (
                <div
                  key={idx}
                  className={`p-4 rounded-xl border flex items-start gap-3.5 ${
                    alert.type === "warning"
                      ? "bg-[#271E14] border-[#F1BA65]/50 text-[#F1BA65]"
                      : "bg-[#10231B] border-[#6AE6A0]/40 text-[#F2FAF5]"
                  }`}
                >
                  <AlertTriangle className={`w-5 h-5 shrink-0 mt-0.5 ${alert.type === "warning" ? "text-[#F1BA65]" : "text-[#6AE6A0]"}`} />
                  <div>
                    <span className="font-bold text-xs block mb-1">
                      {alert.title}
                    </span>
                    <p className="text-xs text-[#E0ECE5] leading-relaxed">
                      {alert.message}
                    </p>
                  </div>
                </div>
              ))}
            </div>

            {/* Irrigation & Pest Scouting 2-Column Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
              <div className="p-4 rounded-xl bg-[#10231B] border border-[#294B3C] flex flex-col justify-between">
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <Droplets className="w-4 h-4 text-[#6AE6A0]" />
                    <span className="text-xs font-bold text-[#6AE6A0]">
                      {t("weather.irrigation_advice_title", "Irrigation Scheduling & Water Budget")}
                    </span>
                  </div>
                  <p className="text-xs text-[#A6B9AE] leading-relaxed">
                    {weather.agronomic_guidance.irrigation_advice}
                  </p>
                </div>

                {/* Quantitative Agronomic Calculation Card if Area is Provided */}
                {weather.agronomic_guidance.agronomic_calculation && (
                  <div className="mt-3 pt-3 border-t border-[#294B3C] text-[11px] space-y-1">
                    <div className="flex justify-between text-[#F2FAF5]">
                      <span>Daily Crop Demand (ETc):</span>
                      <span className="font-mono font-bold text-[#6AE6A0]">
                        {weather.agronomic_guidance.agronomic_calculation.crop_water_demand_etc_mm_day} mm/day
                      </span>
                    </div>
                    <div className="flex justify-between text-[#F2FAF5]">
                      <span>Net Irrigation Needed:</span>
                      <span className="font-mono font-bold text-[#6AE6A0]">
                        {weather.agronomic_guidance.agronomic_calculation.net_irrigation_depth_mm} mm
                      </span>
                    </div>
                    <div className="flex justify-between text-[#F2FAF5]">
                      <span>Estimated Water Volume:</span>
                      <span className="font-mono font-bold text-[#6AE6A0]">
                        {weather.agronomic_guidance.agronomic_calculation.estimated_daily_liters.toLocaleString()} Liters / day
                      </span>
                    </div>
                    <span className="text-[10px] text-[#A6B9AE] block mt-1">
                      Basis: {weather.agronomic_guidance.agronomic_calculation.methodology}
                    </span>
                  </div>
                )}
              </div>

              <div className="p-4 rounded-xl bg-[#10231B] border border-[#294B3C]">
                <div className="flex items-center gap-2 mb-2">
                  <Activity className="w-4 h-4 text-[#8EF0AF]" />
                  <span className="text-xs font-bold text-[#8EF0AF]">
                    {t("weather.pest_scouting_title", "Pest & Disease Scouting Protocol")}
                  </span>
                </div>
                <p className="text-xs text-[#A6B9AE] leading-relaxed">
                  {weather.agronomic_guidance.pest_scouting_recommendation}
                </p>
                <div className="mt-3 pt-3 border-t border-[#294B3C]">
                  <span className="text-[11px] font-bold text-[#F2FAF5] block mb-1">
                    {t("weather.fertilizer_guidance_title", "Nutrient & Fertilizer Guidance")}
                  </span>
                  <p className="text-[11px] text-[#A6B9AE] leading-relaxed">
                    {weather.agronomic_guidance.fertilizer_guidance}
                  </p>
                </div>
              </div>
            </div>

            {/* Gemini Horticultural Explanation */}
            {weather.agronomic_guidance.ai_explanation && (
              <div className="p-4 rounded-xl bg-gradient-to-br from-[#10231B] to-[#152C22] border border-[#6AE6A0]/40">
                <div className="flex items-center gap-2 mb-2">
                  <Sparkles className="w-4 h-4 text-[#6AE6A0]" />
                  <span className="text-xs font-bold text-[#6AE6A0]">
                    {t("weather.ai_explanation_title", "Horticultural AI Explanation")}
                  </span>
                </div>
                <p className="text-xs text-[#E0ECE5] leading-relaxed whitespace-pre-line">
                  {weather.agronomic_guidance.ai_explanation}
                </p>
              </div>
            )}

            {/* Citations and Attribution */}
            <div className="pt-3 border-t border-[#294B3C] flex flex-wrap items-center justify-between text-[11px] text-[#A6B9AE] gap-2">
              <span className="flex items-center gap-1.5">
                <Info className="w-3.5 h-3.5 text-[#6AE6A0]" />
                Sources: {weather.agronomic_guidance.citations.join(" · ")}
              </span>
              <span>{t("weather.provider_attribution", "Weather data by Google & Agrometeorological Services")}</span>
            </div>
          </div>

          {/* Task 4: Farmer Profile Inputs Form */}
          <div className="bg-[#152C22] rounded-2xl border border-[#294B3C] p-6 shadow-lg space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-[#294B3C]">
              <div>
                <h3 className="text-base font-bold text-[#F2FAF5]">
                  Farm Profile & Agronomic Parameters
                </h3>
                <p className="text-xs text-[#A6B9AE]">
                  Enter your plot specifications to tailor water volume and disease models.
                </p>
              </div>
              <span className="text-xs font-mono text-[#6AE6A0] bg-[#10231B] px-3 py-1 rounded-full border border-[#294B3C]">
                Crop: Tomato
              </span>
            </div>

            <form onSubmit={handleUpdateCropAdvice} className="space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
                <div>
                  <label className="font-bold text-[#A6B9AE] block mb-1">
                    {t("weather.crop_stage", "Plant Growth Stage")}
                  </label>
                  <select
                    value={growthStage}
                    onChange={(e) => setGrowthStage(e.target.value)}
                    className="w-full rounded-xl border border-[#294B3C] bg-[#10231B] p-2 text-xs text-[#F2FAF5] focus:outline-none focus:border-[#6AE6A0]"
                  >
                    <option value="nursery">{t("weather.stage_seedling", "Nursery / Seedling (0-20 days)")}</option>
                    <option value="vegetative">{t("weather.stage_vegetative", "Vegetative Canopy (21-45 days)")}</option>
                    <option value="flowering">{t("weather.stage_flowering", "Flowering Phase (46-65 days)")}</option>
                    <option value="fruit_set">{t("weather.stage_fruiting", "Fruit Set & Sizing (66-85 days)")}</option>
                    <option value="harvesting">{t("weather.stage_harvesting", "Harvesting Phase (86+ days)")}</option>
                  </select>
                </div>

                <div>
                  <label className="font-bold text-[#A6B9AE] block mb-1">
                    {t("weather.soil_type", "Soil Type")}
                  </label>
                  <select
                    value={soilType}
                    onChange={(e) => setSoilType(e.target.value)}
                    className="w-full rounded-xl border border-[#294B3C] bg-[#10231B] p-2 text-xs text-[#F2FAF5] focus:outline-none focus:border-[#6AE6A0]"
                  >
                    <option value="red_loam">{t("weather.soil_red_loam", "Red Sandy Loam")}</option>
                    <option value="clay_loam">{t("weather.soil_clay_loam", "Clay Loam")}</option>
                    <option value="black_soil">{t("weather.soil_black", "Black Soil")}</option>
                    <option value="sandy_loam">{t("weather.soil_sandy_loam", "Sandy Loam")}</option>
                  </select>
                </div>

                <div>
                  <label className="font-bold text-[#A6B9AE] block mb-1">
                    {t("weather.irrigation_type", "Irrigation Method")}
                  </label>
                  <select
                    value={irrigationMethod}
                    onChange={(e) => setIrrigationMethod(e.target.value)}
                    className="w-full rounded-xl border border-[#294B3C] bg-[#10231B] p-2 text-xs text-[#F2FAF5] focus:outline-none focus:border-[#6AE6A0]"
                  >
                    <option value="drip">{t("weather.irrig_drip", "Drip Irrigation")}</option>
                    <option value="furrow">{t("weather.irrig_furrow", "Furrow Irrigation")}</option>
                    <option value="sprinkler">{t("weather.irrig_sprinkler", "Sprinkler Irrigation")}</option>
                    <option value="manual">{t("weather.irrig_manual", "Manual Watering")}</option>
                  </select>
                </div>

                <div>
                  <label className="font-bold text-[#A6B9AE] block mb-1">
                    {t("weather.farm_area", "Farm / Plot Area")}
                  </label>
                  <div className="flex gap-2">
                    <input
                      type="number"
                      step="any"
                      min="0.1"
                      value={farmArea}
                      onChange={(e) => setFarmArea(e.target.value)}
                      className="w-full rounded-xl border border-[#294B3C] bg-[#10231B] p-2 text-xs text-[#F2FAF5] focus:outline-none focus:border-[#6AE6A0]"
                    />
                    <select
                      value={farmAreaUnit}
                      onChange={(e) => setFarmAreaUnit(e.target.value)}
                      className="rounded-xl border border-[#294B3C] bg-[#10231B] px-2 text-xs text-[#F2FAF5]"
                    >
                      <option value="acres">{t("weather.unit_acres", "Acres")}</option>
                      <option value="guntas">{t("weather.unit_guntas", "Guntas")}</option>
                      <option value="hectares">{t("weather.unit_hectares", "Hectares")}</option>
                      <option value="m2">{t("weather.unit_sqm", "m²")}</option>
                    </select>
                  </div>
                </div>

                <div>
                  <label className="font-bold text-[#A6B9AE] block mb-1">
                    Planting Date <span className="text-[10px] text-[#6AE6A0]">{t("weather.optional", "(Optional)")}</span>
                  </label>
                  <input
                    type="date"
                    value={plantingDate}
                    onChange={(e) => setPlantingDate(e.target.value)}
                    className="w-full rounded-xl border border-[#294B3C] bg-[#10231B] p-2 text-xs text-[#F2FAF5] focus:outline-none focus:border-[#6AE6A0]"
                  />
                </div>

                <div>
                  <label className="font-bold text-[#A6B9AE] block mb-1">
                    {t("weather.watering_history", "Recent Watering History")} <span className="text-[10px] text-[#6AE6A0]">{t("weather.optional", "(Optional)")}</span>
                  </label>
                  <input
                    value={recentIrrigation}
                    onChange={(e) => setRecentIrrigation(e.target.value)}
                    placeholder="e.g. Yesterday morning (2 hrs)"
                    className="w-full rounded-xl border border-[#294B3C] bg-[#10231B] p-2 text-xs text-[#F2FAF5] focus:outline-none focus:border-[#6AE6A0]"
                  />
                </div>

                <div>
                  <label className="font-bold text-[#A6B9AE] block mb-1">
                    {t("weather.soil_moisture", "Measured Soil Moisture (%)")} <span className="text-[10px] text-[#6AE6A0]">{t("weather.optional", "(Optional)")}</span>
                  </label>
                  <input
                    type="number"
                    min="0"
                    max="100"
                    value={soilMoisture}
                    onChange={(e) => setSoilMoisture(e.target.value)}
                    placeholder="e.g. 65"
                    className="w-full rounded-xl border border-[#294B3C] bg-[#10231B] p-2 text-xs text-[#F2FAF5] focus:outline-none focus:border-[#6AE6A0]"
                  />
                </div>

                <div>
                  <label className="font-bold text-[#A6B9AE] block mb-1">
                    {t("weather.fertilizer_history", "Fertilizer History")} <span className="text-[10px] text-[#6AE6A0]">{t("weather.optional", "(Optional)")}</span>
                  </label>
                  <input
                    value={fertilizerHistory}
                    onChange={(e) => setFertilizerHistory(e.target.value)}
                    placeholder="e.g. NPK 10:26:26 applied 6 days ago"
                    className="w-full rounded-xl border border-[#294B3C] bg-[#10231B] p-2 text-xs text-[#F2FAF5] focus:outline-none focus:border-[#6AE6A0]"
                  />
                </div>
              </div>

              <div className="flex flex-wrap items-center justify-between pt-2 gap-3">
                <button
                  disabled={calculatingAdvice}
                  type="submit"
                  className="px-5 py-2.5 rounded-xl bg-[#6AE6A0] hover:bg-[#5BD690] text-[#0A1814] text-xs font-bold transition flex items-center gap-2 shadow-md"
                >
                  <Sparkles className={`w-4 h-4 ${calculatingAdvice ? "animate-spin" : ""}`} />
                  {calculatingAdvice ? "Recalculating Guidance..." : t("weather.recalculate", "Update Crop Recommendations")}
                </button>

                {saveStatus && (
                  <span className="text-xs text-[#6AE6A0] flex items-center gap-1.5 font-medium">
                    <CheckCircle2 className="w-4 h-4" />
                    {saveStatus}
                  </span>
                )}
                {!idToken && (
                  <span className="text-[11px] text-[#A6B9AE]">
                    Guest mode active. Sign in to automatically persist preferences to MongoDB Atlas.
                  </span>
                )}
              </div>
            </form>
          </div>

          {/* Task 3: 24-Hour Hourly Forecast Progression */}
          <div className="bg-[#152C22] rounded-2xl border border-[#294B3C] p-6 shadow-lg space-y-4">
            <h3 className="text-base font-bold text-[#F2FAF5] flex items-center gap-2">
              <Calendar className="w-4 h-4 text-[#6AE6A0]" />
              {t("weather.hourly", "24-Hour Hourly Telemetry")}
            </h3>
            <div className="flex gap-3 overflow-x-auto pb-2 scrollbar-thin scrollbar-thumb-[#294B3C]">
              {weather.hourly.map((h, i) => (
                <div
                  key={i}
                  className="min-w-[85px] bg-[#10231B] border border-[#294B3C] rounded-xl p-3 text-center flex flex-col justify-between shrink-0"
                >
                  <span className="text-xs text-[#A6B9AE] block font-mono font-medium">
                    {h.time}
                  </span>
                  <div className="my-2">
                    <span className="text-sm font-extrabold text-[#F2FAF5] block">
                      {h.temperature.toFixed(0)}°C
                    </span>
                    <span className="text-[10px] text-[#6AE6A0] block">
                      {h.condition || "Clear"}
                    </span>
                  </div>
                  <div className="pt-2 border-t border-[#294B3C] text-[10px] text-[#A6B9AE] flex items-center justify-center gap-1">
                    <CloudRain className="w-3 h-3 text-[#6AE6A0]" />
                    <span>{h.precipitation_prob ?? 0}%</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Task 3: 10-Day Agricultural Weather Outlook */}
          <div className="bg-[#152C22] rounded-2xl border border-[#294B3C] p-6 shadow-lg space-y-4">
            <h3 className="text-base font-bold text-[#F2FAF5] flex items-center gap-2">
              <CloudSun className="w-4 h-4 text-[#6AE6A0]" />
              {t("weather.forecast", "10-Day Agricultural Weather Outlook")}
            </h3>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3">
              {weather.daily.map((d, idx) => (
                <div
                  key={idx}
                  className="bg-[#10231B] rounded-xl border border-[#294B3C] p-3 text-center flex flex-col justify-between hover:border-[#6AE6A0]/40 transition"
                >
                  <div>
                    <span className="text-xs font-bold text-[#F2FAF5] block mb-1">
                      {d.date}
                    </span>
                    <span className="text-xs font-mono font-bold text-[#6AE6A0]">
                      {d.max_temp.toFixed(0)}° / {d.min_temp.toFixed(0)}°C
                    </span>
                    <span className="text-[11px] text-[#A6B9AE] block mt-1 truncate">
                      {d.condition || "Mainly clear"}
                    </span>
                  </div>

                  <div className="mt-3 pt-2 border-t border-[#294B3C] flex items-center justify-between text-[11px] text-[#A6B9AE] px-1">
                    <div className="flex items-center gap-1">
                      <CloudRain className="w-3.5 h-3.5 text-[#6AE6A0]" />
                      <span>{d.rainfall_mm == null ? "0.0 mm" : `${d.rainfall_mm.toFixed(1)} mm`}</span>
                    </div>
                    {d.precipitation_prob != null && (
                      <span className="font-mono text-[10px] text-[#6AE6A0]">{d.precipitation_prob}%</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Task 3: Available Recent Weather History (Past 24 Hours) */}
          {weather.history_24h && weather.history_24h.length > 0 && (
            <div className="bg-[#152C22] rounded-2xl border border-[#294B3C] p-6 shadow-lg space-y-4">
              <h3 className="text-base font-bold text-[#F2FAF5] flex items-center gap-2">
                <Compass className="w-4 h-4 text-[#6AE6A0]" />
                {t("weather.past_24h", "Past 24-Hour Telemetry History")}
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 gap-3">
                {weather.history_24h.map((hist, i) => (
                  <div key={i} className="bg-[#10231B] border border-[#294B3C] rounded-xl p-3 text-center">
                    <span className="text-xs font-mono text-[#A6B9AE] block mb-1">{hist.time}</span>
                    <span className="text-xs font-bold text-[#F2FAF5] block">{hist.temperature.toFixed(1)}°C</span>
                    <span className="text-[11px] text-[#6AE6A0] block mt-1">
                      {hist.rainfall_mm ? `${hist.rainfall_mm} mm` : "0.0 mm"}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
