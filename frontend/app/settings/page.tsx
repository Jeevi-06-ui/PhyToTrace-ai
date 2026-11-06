"use client";

import Link from "next/link";
import { useState } from "react";
import { useAuth } from "@/context/AuthContext";
import { useLanguage } from "@/context/LanguageContext";

export default function SettingsPage() {
  const { user, idToken, signInWithGoogle, signOut, authError } = useAuth();
  const { locale, setLocale, t } = useLanguage();
  const [status, setStatus] = useState("");

  async function savePreferences() {
    if (!idToken) return;
    try {
      const response = await fetch("/api/v1/users/preferences", {
        method: "PUT",
        headers: { Authorization: `Bearer ${idToken}`, "Content-Type": "application/json" },
        body: JSON.stringify({ language: locale }),
      });
      setStatus(response.ok ? t("settings.saved", "Language saved to your account.") : t("settings.error", "Account preferences are unavailable."));
    } catch {
      setStatus(t("settings.error", "Account preferences are unavailable."));
    }
  }

  return (
    <section className="max-w-2xl space-y-6">
      <h1 className="text-3xl font-bold">{t("settings.title", "Profile and settings")}</h1>
      <div className="space-y-4 rounded-2xl border border-[#294B3C] bg-[#152C22] p-6">
        {user ? (
          <>
            <p className="font-bold">{user.displayName}</p>
            <p className="text-sm text-[#A6B9AE]">{user.email}</p>
            <button className="rounded-lg border border-[#294B3C] p-2" onClick={signOut}>{t("settings.sign_out", "Sign out")}</button>
          </>
        ) : (
          <div className="flex flex-wrap gap-3">
            <button className="rounded-lg bg-[#6AE6A0] p-2 text-[#0A1814]" onClick={signInWithGoogle}>{t("settings.google", "Sign in with Google")}</button>
            <Link className="rounded-lg border border-[#294B3C] p-2" href="/analyze">{t("settings.guest", "Continue as guest")}</Link>
          </div>
        )}
        {authError && <p role="alert">{authError}</p>}
      </div>
      <div className="space-y-4 rounded-2xl border border-[#294B3C] bg-[#152C22] p-6">
        <label className="block">{t("settings.language", "Display language")}
          <select value={locale} onChange={(event) => setLocale(event.target.value as typeof locale)} className="ml-3 rounded bg-[#10231B] p-2">
            <option value="en">English</option><option value="kn">ಕನ್ನಡ</option><option value="hi">हिंदी</option>
          </select>
        </label>
        {idToken && <button className="rounded-lg border border-[#294B3C] p-2" onClick={savePreferences}>{t("settings.save", "Save to account")}</button>}
        {status && <p role="status">{status}</p>}
      </div>
    </section>
  );
}
