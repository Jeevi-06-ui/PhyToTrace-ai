"use client";

import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { getApps, initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider, onIdTokenChanged, signInWithPopup, signOut as firebaseSignOut } from "firebase/auth";

const firebaseConfig = {
  apiKey: process.env.NEXT_PUBLIC_FIREBASE_API_KEY,
  authDomain: process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN,
  projectId: process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID,
  storageBucket: process.env.NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: process.env.NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID,
};

function firebaseAuth() {
  if (!firebaseConfig.apiKey || !firebaseConfig.authDomain || !firebaseConfig.projectId) return null;
  return getAuth(getApps()[0] ?? initializeApp(firebaseConfig));
}

interface UserProfile {
  uid: string;
  email: string | null;
  displayName: string;
  isGuest: boolean;
  photoURL?: string | null;
}

interface AuthContextType {
  user: UserProfile | null;
  isGuest: boolean;
  idToken: string | null;
  authError: string | null;
  signInWithGoogle: () => Promise<void>;
  signOut: () => Promise<void>;
  continueAsGuest: () => void;
}

const AuthContext = createContext<AuthContextType>({
  user: null, isGuest: true, idToken: null, authError: null,
  signInWithGoogle: async () => {}, signOut: async () => {}, continueAsGuest: () => {},
});

export function AuthProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<{ user: UserProfile; token: string } | null>(null);
  const [authError, setAuthError] = useState<string | null>(null);

  useEffect(() => {
    const auth = firebaseAuth();
    if (!auth) return;
    let active = true;
    const unsubscribe = onIdTokenChanged(auth, async (account) => {
      if (!account) {
        setSession(null);
        return;
      }
      try {
        const token = await account.getIdToken();
        if (active && auth.currentUser === account) {
          setSession({
            token, user: { uid: account.uid, email: account.email, photoURL: account.photoURL,
              displayName: account.displayName ?? account.email ?? "Farmer", isGuest: false },
          });
        }
      } catch {
        if (active) setSession(null);
      }
    });
    return () => {
      active = false;
      unsubscribe();
    };
  }, []);

  const signInWithGoogle = async () => {
    setAuthError(null);
    try {
      const auth = firebaseAuth();
      if (!auth) {
        setAuthError("Google sign-in is unavailable until Firebase is configured.");
        return;
      }
      const provider = new GoogleAuthProvider();
      provider.setCustomParameters({ prompt: "select_account" });
      await signInWithPopup(auth, provider);
    } catch (err: unknown) {
      const fbErr = err as { code?: string; message?: string };
      console.error("Firebase Google Sign-In error:", fbErr?.code, fbErr?.message, err);

      if (fbErr?.code === "auth/popup-blocked") {
        setAuthError("Sign-in popup was blocked by your browser. Please allow popups for this site and try again.");
      } else if (fbErr?.code === "auth/popup-closed-by-user") {
        setAuthError("Sign-in window was closed before completing.");
      } else if (fbErr?.code === "auth/cancelled-popup-request") {
        setAuthError("A sign-in request is already in progress.");
      } else if (fbErr?.code === "auth/operation-not-allowed") {
        setAuthError("Google Sign-In is not enabled in Firebase Console (Authentication → Sign-in method → Google).");
      } else if (fbErr?.code === "auth/unauthorized-domain") {
        const domain = typeof window !== "undefined" ? window.location.hostname : "this domain";
        setAuthError(`Domain '${domain}' is not authorized in Firebase Console (Authentication → Settings → Authorized domains).`);
      } else if (fbErr?.code === "auth/network-request-failed") {
        setAuthError("Network error contacting Firebase Authentication. Please check your internet connection.");
      } else {
        setAuthError(fbErr?.message || "Google sign-in did not finish. Try again or continue as a guest.");
      }
    }
  };

  const signOut = async () => {
    setAuthError(null);
    try {
      const auth = firebaseAuth();
      if (auth) await firebaseSignOut(auth);
      setSession(null);
    } catch {
      setAuthError("Sign-out did not finish. Please try again.");
    }
  };

  return (
    <AuthContext.Provider value={{
      user: session?.user ?? null, isGuest: !session, idToken: session?.token ?? null,
      authError, signInWithGoogle, signOut, continueAsGuest: () => { void signOut(); },
    }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
