# ── Stage 1: build the React UI ───────────────────────────────────
FROM node:20-slim AS ui-builder
ARG CODEX_VERSION=0.145.0
WORKDIR /ui
COPY ui/mira/package.json ui/mira/package-lock.json ./
RUN npm ci --no-audit --no-fund
RUN npm install --global --no-audit --no-fund "@openai/codex@${CODEX_VERSION}"
COPY ui/mira/ ./
RUN npm run build

# ── Stage 2: backend + bundled UI ─────────────────────────────────
FROM python:3.12-slim
LABEL org.opencontainers.image.source="https://github.com/miracodeai/mira"
LABEL org.opencontainers.image.description="Self-hostable AI code reviewer"
LABEL org.opencontainers.image.licenses="Apache-2.0"

ARG TARGETARCH
# Antigravity CLI is bundled for the optional antigravity backend. Pinned to a
# release-manifest version/build (with SHA-512 verification, mirroring the
# upstream install script) so image rebuilds do not silently change the review
# runtime — same policy as CODEX_VERSION above. To bump: take version, build
# id, and per-arch sha512 from
# https://antigravity-cli-auto-updater-974169037036.us-central1.run.app/manifests/linux_arm64.json
ARG AGY_VERSION=1.1.28
ARG AGY_BUILD=5576113066475520

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir "/app[serve,bedrock]"

# Codex CLI is bundled for the optional codex-cli backend. Keep the version
# pinned above so image rebuilds do not silently change the review runtime.
COPY --from=ui-builder /usr/local/bin/node /usr/local/bin/node
COPY --from=ui-builder /usr/local/lib/node_modules/@openai/codex /usr/local/lib/node_modules/@openai/codex
RUN ln -s /usr/local/lib/node_modules/@openai/codex/bin/codex.js /usr/local/bin/codex

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends curl ca-certificates; \
    if [ -n "${TARGETARCH:-}" ]; then agy_target="${TARGETARCH}"; \
    else agy_target="$(uname -m | sed 's/x86_64/amd64/; s/aarch64/arm64/')"; fi; \
    case "${agy_target}" in \
        arm64) agy_dir=arm; agy_arch=arm64; agy_sha512=2cc17c2a2dfe5c7d2f8088ac16ccfef270933c7844c3c717b3e4f4f94873f865ebddc5943508cbe54b4976c0f9dc478ca71b8c26b8c08d6758d9508788235530 ;; \
        amd64) agy_dir=x64; agy_arch=x64;  agy_sha512=a855c623426fe901088bfe2b4d559b01f5b479a7c5ec0e41acfe129d4fe7deee9fbd9057072cc8e8c19ed65474cafa4f1d10fcdc14fbb85f82f20c3ec94295d4 ;; \
        *) echo "Unsupported architecture for Antigravity CLI: ${agy_target}" >&2; exit 1 ;; \
    esac; \
    curl -fsSL -o /tmp/agy.tar.gz \
        "https://storage.googleapis.com/antigravity-public/antigravity-cli/${AGY_VERSION}-${AGY_BUILD}/linux-${agy_dir}/cli_linux_${agy_arch}.tar.gz"; \
    echo "${agy_sha512}  /tmp/agy.tar.gz" | sha512sum -c -; \
    tar -xzf /tmp/agy.tar.gz -C /tmp antigravity; \
    install -m 0755 /tmp/antigravity /usr/local/bin/agy; \
    rm -rf /tmp/agy.tar.gz /tmp/antigravity; \
    apt-get purge -y curl; \
    apt-get autoremove -y; \
    rm -rf /var/lib/apt/lists/*; \
    agy --help >/dev/null

# Pull the built UI in from stage 1. webhooks.create_app() picks this up
# automatically and serves it at / with SPA fallback.
COPY --from=ui-builder /ui/dist /app/ui_dist

EXPOSE 8000
# ENTRYPOINT (not CMD) so `docker run … image --config /app/mira.yaml`
# appends the args to `mira serve` instead of replacing the command.
ENTRYPOINT ["mira", "serve"]
