/**
 * Proxy route — forwards /api/bot/<path> to the FastAPI voicebot server.
 *
 * During Docker: API_URL=http://voicebot:9000
 * During local dev: API_URL=http://localhost:9000 (set in frontend/.env.local)
 */
import { NextRequest, NextResponse } from "next/server";

const API_URL = process.env.API_URL ?? "http://localhost:9000";

async function proxy(req: NextRequest, path: string[], method: string) {
  const upstream = `${API_URL}/${path.join("/")}`;
  try {
    const res = await fetch(upstream, {
      method,
      headers: { "Content-Type": "application/json" },
      body: method !== "GET" ? (req.body ? await req.text() : undefined) : undefined,
      // short timeout so the dashboard doesn't hang if bot is down
      signal: AbortSignal.timeout(5_000),
    });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (err) {
    return NextResponse.json(
      { error: "Bot unreachable", detail: String(err) },
      { status: 503 },
    );
  }
}

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ path: string[] }> },
) {
  const { path } = await params;
  return proxy(req, path, "GET");
}

export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ path: string[] }> },
) {
  const { path } = await params;
  return proxy(req, path, "POST");
}
