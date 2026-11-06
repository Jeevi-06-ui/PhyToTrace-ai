"use client";

import { useState, useRef, useEffect } from "react";
import { 
  Bot, 
  Send, 
  Sparkles, 
  User, 
  ShieldCheck, 
  Info, 
  CheckCircle2, 
  RefreshCw,
  BookOpen,
  ArrowRight
} from "lucide-react";
import { useLanguage } from "@/context/LanguageContext";
import { useAuth } from "@/context/AuthContext";

interface ChatMessage {
  role: "user" | "assistant";
  text: string;
  source?: string;
  citations?: string[];
  timestamp?: string;
}

export default function AssistantPage() {
  const { t, locale } = useLanguage();
  const { idToken } = useAuth();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputMessage, setInputMessage] = useState("");
  const [isSending, setIsSending] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Initial welcome message
    setMessages([
      {
        role: "assistant",
        text:
          locale === "kn"
            ? "ನಮಸ್ಕಾರ! ನಾನು ನಿಮ್ಮ ಫೈಟೋಟ್ರೇಸ್ ಕೃಷಿ ಸಹಾಯಕ (PhytoTrace Agri Assistant). ಐಸಿಎಆರ್-ಐಐಎಚ್ಆರ್ (ICAR-IIHR) ಮಾರ್ಗಸೂಚಿಗಳ ಆಧಾರದ ಮೇಲೆ ಟೊಮೆಟೊ ಎಲೆ ಕೊರೆಯುವ ಕೀಟ ನಿಯಂತ್ರಣ, ಕೃಷಿ ಪದ್ಧತಿ ಮತ್ತು ಹವಾಮಾನ ಸಲಹೆಗಳನ್ನು ನೀಡಬಲ್ಲೆ. ನಿಮ್ಮ ಪ್ರಶ್ನೆಯನ್ನು ಕೇಳಿ!"
            : locale === "hi"
            ? "नमस्ते! मैं आपका फाइटोट्रेस कृषि सहायक (PhytoTrace Agri Assistant) हूँ। ICAR-IIHR दिशानिर्देशों के आधार पर टमाटर फसल में लीफ माइनर नियंत्रण, सिंचाई और सुरक्षा पर सलाह देने के लिए उपलब्ध हूँ। अपना प्रश्न पूछें!"
            : "Welcome to PhytoTrace Agri Assistant. I am an evidence-based agricultural specialist grounded in ICAR-IIHR tomato pest management guidelines. How can I assist your crop monitoring or scan analysis today?",
      },
    ]);
  }, [locale]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSendMessage = async (textToSend?: string) => {
    const msg = textToSend || inputMessage;
    if (!msg.trim() || isSending) return;

    const userEntry: ChatMessage = {
      role: "user",
      text: msg,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };

    setMessages((prev) => [...prev, userEntry]);
    if (!textToSend) setInputMessage("");
    setIsSending(true);

    try {
      const resp = await fetch("/api/v1/assistant/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json", ...(idToken ? { Authorization: `Bearer ${idToken}` } : {}) },
        body: JSON.stringify({
          message: msg,
          session_id: "tomato-scouting",
          inspection_id: idToken ? new URLSearchParams(window.location.search).get("inspection") : undefined,
          language: locale,
        }),
      });

      if (!resp.ok) throw new Error("Assistant response failed");
      const data = await resp.json();

      const assistantEntry: ChatMessage = {
        role: "assistant",
        text: data.message,
        source: data.source,
        citations: data.citations || [],
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      };

      setMessages((prev) => [...prev, assistantEntry]);
    } catch (e) {
      console.error("Chat error:", e);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text:
            locale === "kn"
              ? "ಕ್ಷಮಿಸಿ, ಪ್ರತಿಕ್ರಿಯೆ ಪಡೆಯಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಮತ್ತೊಮ್ಮೆ ಪ್ರಯತ್ನಿಸಿ."
              : "Unable to retrieve assistance at this moment. Please verify your connection or try again shortly.",
        },
      ]);
    } finally {
      setIsSending(false);
    }
  };

  const suggestedQuestions = [
    t("assistant.q1", "How do I control leaf miner without heavy chemicals?"),
    t("assistant.q2", "What is the difference between Tuta absoluta and Liriomyza mines?"),
    t("assistant.q3", "How should I manage irrigation in 75% relative humidity?"),
    t("assistant.q4", "My scan shows 98% damage, what are the immediate steps?"),
  ];

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div className="pb-4 border-b border-[#294B3C] flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold uppercase tracking-wider text-[#6AE6A0]">
              Evidence-Based Decision Support
            </span>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-[#10231B] border border-[#294B3C] text-[#A6B9AE]">
              ICAR-IIHR Grounded
            </span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-[#F2FAF5] tracking-tight mt-1 flex items-center gap-2.5">
            <Bot className="w-7 h-7 text-[#6AE6A0]" />
            <span>{t("assistant.title", "PhytoTrace Agri Assistant")}</span>
          </h1>
          <p className="text-xs sm:text-sm text-[#A6B9AE] mt-1">
            {t("assistant.subtitle", "Grounded decision support powered by ICAR-IIHR tomato pest management guidelines.")}
          </p>
        </div>
      </div>

      {/* Suggested Prompt Chips */}
      <div className="bg-[#152C22] p-4 rounded-2xl border border-[#294B3C]">
        <div className="flex items-center gap-1.5 text-xs font-bold text-[#F2FAF5] uppercase tracking-wider mb-2.5">
          <Sparkles className="w-3.5 h-3.5 text-[#6AE6A0]" />
          <span>{t("assistant.suggested_title", "Suggested Questions")}</span>
        </div>
        <div className="flex flex-wrap gap-2">
          {suggestedQuestions.map((q, idx) => (
            <button
              key={idx}
              onClick={() => handleSendMessage(q)}
              disabled={isSending}
              className="text-xs text-left px-3 py-1.5 rounded-xl bg-[#10231B] border border-[#294B3C] hover:border-[#6AE6A0]/50 text-[#A6B9AE] hover:text-[#F2FAF5] transition-all"
            >
              {q}
            </button>
          ))}
        </div>
      </div>

      {/* Chat Messages Box */}
      <div className="bg-[#152C22] rounded-2xl border border-[#294B3C] overflow-hidden flex flex-col min-h-[440px] max-h-[580px] shadow-xl">
        <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-4">
          {messages.map((m, idx) => {
            const isUser = m.role === "user";
            return (
              <div
                key={idx}
                className={`flex gap-3 ${isUser ? "justify-end" : "justify-start"}`}
              >
                {!isUser && (
                  <div className="w-8 h-8 rounded-xl bg-[#10231B] border border-[#294B3C] flex items-center justify-center text-[#6AE6A0] shrink-0 mt-0.5">
                    <Bot className="w-4 h-4" />
                  </div>
                )}

                <div
                  className={`max-w-[85%] rounded-2xl p-4 text-xs leading-relaxed space-y-2 ${
                    isUser
                      ? "bg-[#1D4032] text-[#F2FAF5] border border-[#6AE6A0]/30 rounded-tr-xs"
                      : "bg-[#10231B] text-[#F2FAF5] border border-[#294B3C] rounded-tl-xs"
                  }`}
                >
                  <p className="whitespace-pre-wrap">{m.text}</p>

                  {/* Grounded Citation & Source */}
                  {m.source && (
                    <div className="mt-2 pt-2 border-t border-[#294B3C] text-[10px] text-[#A6B9AE] flex flex-col gap-1">
                      <div className="flex items-center gap-1.5 font-semibold text-[#6AE6A0]">
                        <BookOpen className="w-3 h-3" />
                        <span>Source: {m.source}</span>
                      </div>
                      {m.citations && m.citations.length > 0 && (
                        <div className="text-[#768D80] list-disc list-inside">
                          {m.citations.map((c, i) => (
                            <div key={i}><a href={c} target="_blank" rel="noreferrer">ICAR-IIHR crop weather advisory</a></div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}

                  {m.timestamp && (
                    <div className="text-[10px] text-[#768D80] text-right">
                      {m.timestamp}
                    </div>
                  )}
                </div>

                {isUser && (
                  <div className="w-8 h-8 rounded-xl bg-[#1D4032] border border-[#6AE6A0]/40 flex items-center justify-center text-[#6AE6A0] shrink-0 mt-0.5">
                    <User className="w-4 h-4" />
                  </div>
                )}
              </div>
            );
          })}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Bar */}
        <div className="p-3 bg-[#10231B] border-t border-[#294B3C]">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleSendMessage();
            }}
            className="flex items-center gap-2"
          >
            <input
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              placeholder={t("assistant.input_placeholder", "Ask about leaf-miners, pest management, sticky traps...")}
              className="flex-1 px-4 py-2.5 rounded-xl bg-[#152C22] border border-[#294B3C] text-xs text-[#F2FAF5] placeholder-[#768D80] focus:outline-hidden focus:border-[#6AE6A0]"
            />

            <button
              type="submit"
              disabled={!inputMessage.trim() || isSending}
              className="px-4 py-2.5 rounded-xl bg-[#6AE6A0] hover:bg-[#8EF0AF] text-[#0A1814] font-bold text-xs flex items-center gap-1.5 transition-all disabled:opacity-50 disabled:cursor-not-allowed shrink-0"
            >
              {isSending ? (
                <RefreshCw className="w-4 h-4 animate-spin" />
              ) : (
                <Send className="w-4 h-4" />
              )}
              <span className="hidden sm:inline">{t("assistant.send_btn", "Send")}</span>
            </button>
          </form>
        </div>
      </div>

      {/* Disclaimers & Regulatory Warning */}
      <div className="p-4 rounded-2xl bg-[#10231B] border border-[#294B3C] text-xs text-[#A6B9AE] flex items-start gap-3">
        <Info className="w-4 h-4 text-[#6AE6A0] shrink-0 mt-0.5" />
        <span className="leading-relaxed">
          <strong>Agricultural Safety & Regulatory Notice:</strong> Recommendations are synthesized from published ICAR-IIHR guidelines and Central Insecticides Board (CIBRC) schedules. Always verify pest identification with local agricultural university extension specialists before applying synthetic chemical sprays.
        </span>
      </div>
    </div>
  );
}

