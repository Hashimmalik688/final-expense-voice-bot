# syntax=docker/dockerfile:1
# =============================================================================
# Voice Bot monitoring frontend — Next.js 15 / Tailwind 4
#
# Based on the Dockerfile pattern from ShayneP/local-voice-ai
# =============================================================================

FROM node:22-slim AS base

# Enable pnpm via corepack
RUN corepack enable && corepack prepare pnpm@9.15.9 --activate

ENV PNPM_HOME=/pnpm
ENV PATH="$PNPM_HOME:$PATH"

# ---------------------------------------------------------------------------
# deps — install production + dev dependencies with cache mount
# ---------------------------------------------------------------------------
FROM base AS deps
WORKDIR /app
COPY package.json pnpm-lock.yaml* ./
RUN --mount=type=cache,id=pnpm,target=/pnpm/store \
    pnpm install --frozen-lockfile

# ---------------------------------------------------------------------------
# builder — compile Next.js standalone output
# ---------------------------------------------------------------------------
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN pnpm build

# ---------------------------------------------------------------------------
# runner — minimal production image
# ---------------------------------------------------------------------------
FROM base AS runner
WORKDIR /app

ENV NODE_ENV=production

# Non-privileged user (adopted from ShayneP/local-voice-ai)
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/app" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    nextuser

COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextuser:nextuser /app/.next/standalone ./
COPY --from=builder --chown=nextuser:nextuser /app/.next/static ./.next/static

USER nextuser

EXPOSE 3000
ENV PORT=3000 HOSTNAME="0.0.0.0"

HEALTHCHECK --interval=10s --timeout=3s --retries=5 \
  CMD curl -fsS http://localhost:3000/ || exit 1

CMD ["node", "server.js"]
