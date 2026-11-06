"use client";

import React, { useEffect, useState, useRef } from "react";
import { Volume2, VolumeX, Mic, Radio, Loader2 } from "lucide-react";

interface VoiceNarrationProps {
  diagnosisName?: string | null;
  accuracy?: number | null;
  damagePercentage?: number | null;
  severityTier?: string | null;
  explanation?: string | null;
  activeLocale: "en" | "kn" | "hi";
  className?: string;
}

export default function VoiceNarrationButton({
  diagnosisName,
  accuracy,
  damagePercentage,
  severityTier,
  explanation,
  activeLocale,
  className = "",
}: VoiceNarrationProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [selectedLang, setSelectedLang] = useState<"en" | "kn" | "hi">(activeLocale || "en");
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [isSupported, setIsSupported] = useState(true);
  const synthRef = useRef<SpeechSynthesis | null>(null);

  useEffect(() => {
    setSelectedLang(activeLocale || "en");
  }, [activeLocale]);

  useEffect(() => {
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      synthRef.current = window.speechSynthesis;
      const updateVoices = () => {
        if (synthRef.current) {
          setVoices(synthRef.current.getVoices());
        }
      };
      updateVoices();
      window.speechSynthesis.onvoiceschanged = updateVoices;
      return () => {
        if (synthRef.current) {
          synthRef.current.cancel();
        }
      };
    } else {
      setIsSupported(false);
    }
  }, []);

  const buildNarrationText = (lang: "en" | "kn" | "hi"): string => {
    const accStr = accuracy != null ? Math.round(accuracy) : null;
    const dmgStr = damagePercentage != null ? damagePercentage.toFixed(1) : null;

    if (lang === "kn") {
      let text = "ಟೊಮೆಟೊ ಸಸ್ಯ ಎಲೆಯ ರೋಗನಿರ್ಣಯ ವರದಿ. ";
      if (diagnosisName) {
        text += `ಪತ್ತೆಯಾದ ರೋಗಲಕ್ಷಣ: ${diagnosisName}. `;
      }
      if (accStr != null) {
        text += `ಕೃತಕ ಬುದ್ಧಿಮತ್ತೆ ಮಾದರಿಯ ನಿಖರತೆ ಶೇಕಡಾ ${accStr}. `;
      }
      if (dmgStr != null) {
        text += `ಎಲೆಯ ಮೇಲ್ಮೈ ಹಾನಿಯ ಪ್ರಮಾಣ ಶೇಕಡಾ ${dmgStr}. `;
      }
      text += "ಭಾರತೀಯ ತೋಟಗಾರಿಕಾ ಸಂಶೋಧನಾ ಸಂಸ್ಥೆಯ ಮಾರ್ಗಸೂಚಿಯ ಪ್ರಕಾರ: ಹಳದಿ ಅಂಟು ಬಲೆಗಳನ್ನು ಅಳವಡಿಸಿ, ರೋಗಪೀಡಿತ ಕೆಳಭಾಗದ ಎಲೆಗಳನ್ನು ತೆಗೆಯಿರಿ ಮತ್ತು ನಿರಂತರ ಕಣ್ಗಾವಲು ವಹಿಸಿ. ಅನಗತ್ಯ ಕೀಟನಾಶಕ ಸಿಂಪಡಿಸಬೇಡಿ.";
      return text;
    }

    if (lang === "hi") {
      let text = "टमाटर के पौधे का स्वास्थ्य निदान। ";
      if (diagnosisName) {
        text += `पहचाना गया लक्षण: ${diagnosisName}। `;
      }
      if (accStr != null) {
        text += `AI मॉडल की सटीकता ${accStr} प्रतिशत है। `;
      }
      if (dmgStr != null) {
        text += `पत्ती पर मापा गया नुकसान ${dmgStr} प्रतिशत है। `;
      }
      text += "भारतीय बागवानी अनुसंधान संस्थान के दिशानिर्देश: पीले चिपचिपे ट्रैप लगाएं, संक्रमित निचली पत्तियों को हटाएं और नियमित निगरानी करें। अनावश्यक कीटनाशक न छिड़कें।";
      return text;
    }

    // Default English
    let text = "Tomato leaf health diagnosis. ";
    if (diagnosisName) {
      text += `Identified condition: ${diagnosisName}. `;
    }
    if (accStr != null) {
      text += `Model detection confidence is ${accStr} percent. `;
    }
    if (dmgStr != null) {
      text += `Measured visible leaf damage is ${dmgStr} percent. `;
    }
    text += "According to ICAR-IIHR integrated pest management protocols: Deploy yellow sticky traps, scout lower canopy leaflets regularly, and avoid prophylactic broad-spectrum chemicals.";
    return text;
  };

  const handleSpeak = (langToSpeak = selectedLang) => {
    if (!synthRef.current) return;

    if (isPlaying) {
      synthRef.current.cancel();
      setIsPlaying(false);
      return;
    }

    synthRef.current.cancel(); // Stop any pending speech

    const text = buildNarrationText(langToSpeak);
    const utterance = new SpeechSynthesisUtterance(text);

    // Map appropriate locale tags
    const langCode = langToSpeak === "kn" ? "kn-IN" : langToSpeak === "hi" ? "hi-IN" : "en-IN";
    utterance.lang = langCode;

    // Pick matching system voice if available
    if (voices.length > 0) {
      const match = voices.find(
        (v) => v.lang.toLowerCase() === langCode.toLowerCase() || v.lang.toLowerCase().startsWith(langToSpeak)
      );
      if (match) {
        utterance.voice = match;
      }
    }

    utterance.rate = 0.95; // Slightly slower for clear rural comprehension
    utterance.pitch = 1.0;

    utterance.onstart = () => setIsPlaying(true);
    utterance.onend = () => setIsPlaying(false);
    utterance.onerror = () => setIsPlaying(false);

    synthRef.current.speak(utterance);
  };

  const handleStop = () => {
    if (synthRef.current) {
      synthRef.current.cancel();
      setIsPlaying(false);
    }
  };

  if (!isSupported) {
    return null;
  }

  return (
    <div className={`flex flex-wrap items-center gap-2 p-2 rounded-xl bg-[#0F241C] border border-[#294B3C]/80 ${className}`}>
      {/* Primary voice toggle button */}
      <button
        type="button"
        onClick={() => handleSpeak()}
        className={`px-3 py-1.5 rounded-lg text-xs font-bold transition flex items-center gap-1.5 shadow-sm ${
          isPlaying
            ? "bg-[#F17878] text-[#0A1814] hover:bg-[#ff9292]"
            : "bg-[#6AE6A0] text-[#0A1814] hover:bg-[#8EF0AF]"
        }`}
        title={isPlaying ? "Stop Audio" : "Listen to diagnosis explanation"}
      >
        {isPlaying ? (
          <>
            <VolumeX className="w-4 h-4 animate-pulse" />
            <span>
              {selectedLang === "kn" ? "ಧ್ವನಿ ನಿಲ್ಲಿಸಿ" : selectedLang === "hi" ? "आवाज़ बंद करें" : "Stop Voice"}
            </span>
          </>
        ) : (
          <>
            <Volume2 className="w-4 h-4" />
            <span>
              {selectedLang === "kn"
                ? "ಧ್ವನಿ ವಿವರಣೆ ಕೇಳಿ"
                : selectedLang === "hi"
                ? "निदान सुनें"
                : "Listen to Diagnosis"}
            </span>
          </>
        )}
      </button>

      {/* Language Pills for Voice Selection */}
      <div className="flex items-center gap-1 bg-[#152C22] p-0.5 rounded-lg border border-[#294B3C]">
        <button
          type="button"
          onClick={() => {
            setSelectedLang("en");
            if (isPlaying) {
              handleStop();
              setTimeout(() => handleSpeak("en"), 150);
            }
          }}
          className={`px-2 py-1 rounded text-[11px] font-bold transition ${
            selectedLang === "en"
              ? "bg-[#6AE6A0] text-[#0A1814]"
              : "text-[#A6B9AE] hover:text-[#F2FAF5]"
          }`}
        >
          EN
        </button>
        <button
          type="button"
          onClick={() => {
            setSelectedLang("kn");
            if (isPlaying) {
              handleStop();
              setTimeout(() => handleSpeak("kn"), 150);
            }
          }}
          className={`px-2 py-1 rounded text-[11px] font-bold transition ${
            selectedLang === "kn"
              ? "bg-[#6AE6A0] text-[#0A1814]"
              : "text-[#A6B9AE] hover:text-[#F2FAF5]"
          }`}
        >
          ಕನ್ನಡ
        </button>
        <button
          type="button"
          onClick={() => {
            setSelectedLang("hi");
            if (isPlaying) {
              handleStop();
              setTimeout(() => handleSpeak("hi"), 150);
            }
          }}
          className={`px-2 py-1 rounded text-[11px] font-bold transition ${
            selectedLang === "hi"
              ? "bg-[#6AE6A0] text-[#0A1814]"
              : "text-[#A6B9AE] hover:text-[#F2FAF5]"
          }`}
        >
          हिंदी
        </button>
      </div>

      {isPlaying && (
        <span className="flex items-center gap-1 text-[11px] font-mono text-[#6AE6A0] animate-pulse ml-1">
          <span className="w-2 h-2 rounded-full bg-[#6AE6A0] animate-ping" />
          <span>
            {selectedLang === "kn"
              ? "ಕನ್ನಡ ಧ್ವನಿ ಸಕ್ರಿಯವಾಗಿದೆ..."
              : selectedLang === "hi"
              ? "हिंदी ऑडियो चल रहा है..."
              : "Audio playing..."}
          </span>
        </span>
      )}
    </div>
  );
}

