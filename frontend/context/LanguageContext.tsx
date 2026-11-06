"use client";

import React, { createContext, useContext, useEffect, useState } from "react";
import enMessages from "../messages/en.json";
import knMessages from "../messages/kn.json";
import hiMessages from "../messages/hi.json";

export type Locale = "en" | "kn" | "hi";

interface LanguageContextType {
  locale: Locale;
  setLocale: (l: Locale) => void;
  t: (keyPath: string, fallback?: string) => string;
}

const dictionaries: Record<Locale, any> = {
  en: enMessages,
  kn: knMessages,
  hi: hiMessages,
};

const LanguageContext = createContext<LanguageContextType>({
  locale: "en",
  setLocale: () => {},
  t: (keyPath: string, fallback?: string) => fallback || keyPath,
});

export function LanguageProvider({ children }: { children: React.ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>("en");

  useEffect(() => {
    const saved = localStorage.getItem("phytotrace_locale") as Locale;
    if (saved && (saved === "en" || saved === "kn" || saved === "hi")) {
      setLocaleState(saved);
    }
  }, []);

  useEffect(() => { document.documentElement.lang = locale; }, [locale]);

  const setLocale = (l: Locale) => {
    setLocaleState(l);
    localStorage.setItem("phytotrace_locale", l);
  };

  const t = (keyPath: string, fallback?: string): string => {
    const dict = dictionaries[locale] || dictionaries["en"];
    const parts = keyPath.split(".");
    let curr = dict;
    for (const part of parts) {
      if (curr && typeof curr === "object" && part in curr) {
        curr = curr[part];
      } else {
        // Fallback to English
        let fallbackCurr = dictionaries["en"];
        for (const p of parts) {
          if (fallbackCurr && typeof fallbackCurr === "object" && p in fallbackCurr) {
            fallbackCurr = fallbackCurr[p];
          } else {
            return fallback || keyPath;
          }
        }
        return typeof fallbackCurr === "string" ? fallbackCurr : fallback || keyPath;
      }
    }
    return typeof curr === "string" ? curr : fallback || keyPath;
  };

  return (
    <LanguageContext.Provider value={{ locale, setLocale, t }}>
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguage() {
  return useContext(LanguageContext);
}

