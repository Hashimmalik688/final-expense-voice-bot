"use client";

import { useCallback, useEffect, useState } from "react";

// ---------------------------------------------------------------------------
// Types (mirrors the FastAPI /health response shape)
// ---------------------------------------------------------------------------
type ComponentStatus = "ok" | "degraded" | "registered" | "disconnected" | string;

interface HealthData {
  status: string;
  components: {
    llm: ComponentStatus;
    tts: ComponentStatus;
    sip: ComponentStatus;
  };
  active_calls: number;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
const BADGE: Record<string, string> = {
  ok: "bg-emerald-900 text-emerald-300 border border-emerald-700",
  registered: "bg-emerald-900 text-emerald-300 border border-emerald-700",
  degraded: "bg-red-900 text-red-300 border border-red-700",
  disconnected: "bg-red-900 text-red-300 border border-red-700",
  healthy: "bg-emerald-900 text-emerald-300 border border-emerald-700",
};

function statusBadge(s: string) {
  return BADGE[s] ?? "bg-gray-800 text-gray-300 border border-gray-600";
}

function dot(s: string) {
  if (["ok", "registered", "healthy"].includes(s))
    return "w-2 h-2 rounded-full bg-emerald-400 inline-block mr-2";
  if (["degraded", "disconnected"].includes(s))
    return "w-2 h-2 rounded-full bg-red-400 inline-block mr-2";
  return "w-2 h-2 rounded-full bg-yellow-400 animate-pulse inline-block mr-2";
}

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------
function StatusCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl bg-gray-900 border border-gray-800 p-5">
      <p className="text-xs font-semibold uppercase tracking-widest text-gray-500 mb-2">
        {label}
      </p>
      <span className={`text-xs font-bold px-2 py-1 rounded-full ${statusBadge(value)}`}>
        <span className={dot(value)} />
        {value}
      </span>
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-xl bg-gray-900 border border-gray-800 p-5">
      <p className="text-xs font-semibold uppercase tracking-widest text-gray-500 mb-2">
        {label}
      </p>
      <p className="text-3xl font-bold">{value}</p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main dashboard
// ---------------------------------------------------------------------------
export default function DashboardPage() {
  const [health, setHealth] = useState<HealthData | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [toast, setToast] = useState("");

  const fetchHealth = useCallback(async () => {
    try {
      const res = await fetch("/api/bot/health");
      if (res.ok) {
        setHealth(await res.json());
        setLastUpdated(new Date().toLocaleTimeString());
      }
    } catch {
      // server unreachable — keep stale data
    }
  }, []);

  useEffect(() => {
    fetchHealth();
    const id = setInterval(fetchHealth, 5_000);
    return () => clearInterval(id);
  }, [fetchHealth]);

  async function action(endpoint: string, label: string) {
    setLoading(true);
    setToast("");
    try {
      const res = await fetch(`/api/bot/${endpoint}`, { method: "POST" });
      const data = await res.json();
      setToast(`${label}: ${data.status ?? "done"}`);
    } catch {
      setToast(`${label}: failed — bot unreachable`);
    }
    setLoading(false);
    setTimeout(() => setToast(""), 4_000);
  }

  return (
    <main className="min-h-screen p-8 max-w-4xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight">
          Final Expense Voice Bot
        </h1>
        <p className="text-gray-400 mt-1 text-sm">
          Real-time monitoring dashboard
          {lastUpdated && (
            <span className="ml-3 text-gray-600">
              · updated {lastUpdated}
            </span>
          )}
        </p>
      </div>

      {/* Status cards */}
      {health ? (
        <>
          <h2 className="text-xs font-semibold uppercase tracking-widest text-gray-500 mb-3">
            Service Health
          </h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 mb-6">
            <StatusCard label="LLM (vLLM)" value={health.components.llm} />
            <StatusCard label="TTS (CosyVoice)" value={health.components.tts} />
            <StatusCard label="SIP" value={health.components.sip} />
          </div>

          <div className="grid grid-cols-2 gap-4 mb-10">
            <MetricCard label="Active Calls" value={health.active_calls} />
            <MetricCard label="Bot Status" value={health.status} />
          </div>
        </>
      ) : (
        <div className="rounded-xl bg-gray-900 border border-gray-800 p-6 mb-10 text-gray-400">
          Connecting to bot at{" "}
          <code className="text-gray-300">:9000</code>…
        </div>
      )}

      {/* Controls */}
      <h2 className="text-xs font-semibold uppercase tracking-widest text-gray-500 mb-3">
        Controls
      </h2>
      <div className="flex flex-wrap gap-3">
        <button
          disabled={loading}
          onClick={() => action("reload/script", "Script")}
          className="px-4 py-2 rounded-lg bg-blue-700 hover:bg-blue-600 disabled:opacity-40 text-sm font-semibold transition-colors"
        >
          ↺ Reload Sales Script
        </button>
        <button
          disabled={loading}
          onClick={() => action("reload/knowledge", "Knowledge")}
          className="px-4 py-2 rounded-lg bg-violet-700 hover:bg-violet-600 disabled:opacity-40 text-sm font-semibold transition-colors"
        >
          ↺ Reload Knowledge Base
        </button>
        <button
          disabled={loading}
          onClick={fetchHealth}
          className="px-4 py-2 rounded-lg bg-gray-800 hover:bg-gray-700 disabled:opacity-40 text-sm font-semibold transition-colors"
        >
          ⟳ Refresh Now
        </button>
      </div>

      {/* Toast */}
      {toast && (
        <p className="mt-4 text-sm text-emerald-400">{toast}</p>
      )}

      {/* Footer */}
      <p className="mt-16 text-xs text-gray-700">
        Stack: Llama-3.1-8B-Instruct · Parakeet TDT 0.6B · CosyVoice2 0.5B ·
        Silero VAD · Twilio SIP
      </p>
    </main>
  );
}
