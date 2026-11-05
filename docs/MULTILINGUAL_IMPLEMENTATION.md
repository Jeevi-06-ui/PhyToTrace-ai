# PhytoTrace AI — Multilingual Architecture & Internationalization (i18n)

This document describes the internationalization (i18n) framework in PhytoTrace AI, supporting **English**, **Kannada (ಕನ್ನಡ)**, and **Hindi (हिंदी)** for tomato growers, agricultural extension officers, and academic evaluators.

---

## 1. Agricultural Necessity & Design Philosophy

India is the world's second-largest tomato producer, with major cultivation hubs in Karnataka (Kolar, Chikkaballapura, Belagavi) and across central/northern India.

Most smallholder farmers are fluent in their native language (Kannada in Karnataka, Hindi across central/northern states). PhytoTrace eliminates the linguistic barrier by ensuring:
1. **Accurate Agricultural Terminology:** Terminology is translated with botanical and entomological precision (e.g., *Tomato Leafminer* = "ಟೊಮೆಟೊ ಎಲೆ ಸುರಂಗ ಹುಳು" in Kannada, "टमाटर पत्ती सुरंगक कीट" in Hindi).
2. **Instant In-Browser Switching:** Language toggling does not reload the page or disrupt an active image scan.
3. **Multilingual AI Agronomist:** The PhytoTrace Agri Assistant converses naturally in the farmer's chosen language while maintaining scientific adherence to ICAR-IIHR guidelines.

---

## 2. Frontend Architecture

The localization system is built natively using lightweight React Context and typed JSON message catalogs:

```
frontend/
├── messages/
│   ├── en.json      # English (Source of Truth)
│   ├── kn.json      # Kannada (ಕನ್ನಡ)
│   └── hi.json      # Hindi (हिंदी)
├── context/
│   └── LanguageContext.tsx  # React Context Provider & useLanguage hook
└── components/
    └── AppHeader.tsx        # Pill Toggle Switch (EN | ಕನ್ನಡ | हिंदी)
```

### The `useLanguage` Hook
Components import `useLanguage()` to access translations:

```tsx
import { useLanguage } from "@/context/LanguageContext";

export function SeverityBadge({ tier }: { tier: string }) {
  const { t } = useLanguage();
  return (
    <span className="badge">
      {t(`scan.severity_${tier}`)}
    </span>
  );
}
```

### State Persistence
The selected language is stored in browser `localStorage` under the key `phytotrace_locale`. Upon revisit, the user's preferred language is instantly restored without layout shift.

---

## 3. Translation Key Schema

The dictionary structure is organized into logical namespaces:

| Namespace | Purpose | Example Keys |
| :--- | :--- | :--- |
| `nav` | Sidebar & navigation links | `dashboard`, `scanner`, `weather`, `assistant`, `research` |
| `header` | Header controls & authentication | `guest_mode`, `sign_in`, `sign_out`, `status_healthy` |
| `scanner` | Upload & evidence analysis UI | `dropzone_title`, `leaf_boundary`, `mine_damage`, `severity_tier` |
| `weather` | Weather intelligence telemetry | `title`, `humidity`, `temperature`, `rainfall`, `grant_permission` |
| `assistant` | Conversational agronomist | `title`, `placeholder`, `prompt_chip_1`, `clear_chat` |

---

## 4. How to Add a New Language (e.g., Telugu / தமிழ்)

Adding support for another regional Indian language takes under 15 minutes:

### Step 4.1: Create Translation JSON
Create `frontend/messages/te.json` (for Telugu):
```json
{
  "nav": {
    "dashboard": "డ్యాష్‌బోర్డ్",
    "scanner": "AI స్కానర్",
    "weather": "వాతావరణం",
    "assistant": "అసిస్టెంట్",
    "inspections": "తనిఖీలు",
    "research": "పరిశోధన"
  }
}
```

### Step 4.2: Register in `LanguageContext.tsx`
Import the new dictionary and update the `Language` type:

```tsx
import te from "../messages/te.json";

export type Language = "en" | "kn" | "hi" | "te";

const translations: Record<Language, any> = { en, kn, hi, te };
```

### Step 4.3: Add Selector to `AppHeader.tsx`
Add the language pill to the selector list:
```tsx
const languages: { code: Language; label: string }[] = [
  { code: "en", label: "EN" },
  { code: "kn", label: "ಕನ್ನಡ" },
  { code: "hi", label: "हिंदी" },
  { code: "te", label: "తెలుగు" },
];
```

---

## 5. Backend Multilingual Support

When the user asks questions to the **PhytoTrace Agri Assistant** (`/api/v1/assistant/chat`), the request carries the active `language` code (`en`, `kn`, or `hi`).
The backend injects explicit linguistic instructions into the model system prompt:
- Kannada: *"Respond in fluent, clear Kannada (ಕನ್ನಡ) script using standard regional agricultural terms."*
- Hindi: *"Respond in clear Hindi (हिंदी) in Devanagari script suitable for Indian tomato farmers."*

