# PhytoTrace AI — Gemini 2.5 Flash Setup & Agricultural Grounding Guide

This document describes how to configure the Google Gemini API for the **PhytoTrace Agri Assistant**, including prompt grounding in ICAR-IIHR tomato IPM protocols, cost controls, safety rules, and deterministic local fallbacks.

---

## 1. Role in PhytoTrace Architecture

> [!IMPORTANT]
> **Gemini Vision is NEVER used as a substitute for real segmentation models.**
> Leaf boundary detection and mine lesion localization are performed exclusively by our trained **Lightweight U-Net** and **YOLOv8n-Seg** models.
> 
> Gemini 2.5 Flash is employed strictly as an **evidence-based conversational agronomist**, translating real geometric measurements and microclimate telemetry into actionable, grounded IPM advice for farmers.

```
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│     Trained U-Net & YOLOv8      │       │     Local Weather Telemetry     │
│   (Real leaf & mine geometry)   │       │   (Open-Meteo microclimate)     │
└────────────────┬────────────────┘       └────────────────┬────────────────┘
                 │                                         │
                 └───────────────────┬─────────────────────┘
                                     │
                                     ▼
                ┌─────────────────────────────────────────┐
                │          AgriAssistantService           │
                │     (phytotrace.services.assistant)     │
                │                                         │
                │  ICAR-IIHR Grounded Knowledge Base:     │
                │  - Yellow sticky trap thresholds        │
                │  - Neem formulation (Azadirachtin)      │
                │  - Parasitoid Nesidiocoris tenuis       │
                │  - Biological safety warnings           │
                └────────────────────┬────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
         [ Gemini 2.5 Flash API ]         [ Deterministic Fallback ]
        (Personalized & multilingual)     (Rule-based expert system)
```

---

## 2. Google AI Studio API Key Provisioning

### Step 2.1: Generate an API Key
1. Visit [Google AI Studio](https://aistudio.google.com/).
2. Sign in with your Google account.
3. Click **Get API key** in the top left navigation.
4. Click **Create API key** -> Select or create a Google Cloud Project.
5. Copy the generated key (e.g., `AIzaSy...`).

### Step 2.2: Configure Environment
Add your key to the root `.env`:

```bash
# Google Gemini API
GEMINI_API_KEY="AIzaSyYourActualGoogleAIStudioKeyHere"
GEMINI_MODEL="gemini-2.5-flash"
```

---

## 3. Agronomic Grounding & Safety Constraints

The assistant prompt is strictly constrained by ICAR-IIHR (Hessaraghatta, Bengaluru) protocols:

1. **Species Uncertainty Rule:** The assistant is explicitly prohibited from asserting *Tuta absoluta* or *Liriomyza* with 100% certainty solely from leaf mine geometry, unless pupal or adult specimen traits are verified.
2. **Conservative IPM Escalation:**
   - **Low damage (< 5%):** Cultural control, manual leaflet removal, installation of yellow/pheromone sticky traps (15–20 traps/ha).
   - **Moderate damage (5–15%):** Botanical spray (Azadirachtin 10,000 ppm @ 1–2 ml/L) or bio-agents (*Bacillus thuringiensis* / *Trichogramma pretiosum*).
   - **Severe damage (> 15%):** Consult local KVK / ICAR-IIHR plant pathologist; rotate chemical MoA (Spinosad or Chlorantraniliprole) strictly adhering to pre-harvest intervals (PHI).
3. **Multilingual Fluency:**
   - Supports **English**, **Kannada (ಕನ್ನಡ)**, and **Hindi (हिंदी)** with accurate native agricultural terminology (e.g., "ಎಲೆ ಸುರಂಗ ಹುಳು", "पत्ती सुरंगक कीट").

---

## 4. Deterministic Local Fallback (Zero-Failure Guarantee)

If `GEMINI_API_KEY` is missing, invalid, or reaches a rate limit:
- The system automatically engages the built-in ICAR-IIHR expert rule engine.
- Generates structured, cited diagnostic responses without delay.
- Guaranteed 100% uptime during hackathon jury evaluation and offline field demos.

---

## 5. Verification

Execute the assistant endpoint test:

```bash
.venv/bin/pytest tests/test_api_v1_and_services.py -k "test_assistant_chat"
```

Expected output:
```
tests/test_api_v1_and_services.py::test_assistant_chat_endpoint PASSED
```

