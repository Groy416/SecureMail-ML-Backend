#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

: "${IMAGE_TAG:=latest}"
export IMAGE_TAG

if [[ -f .ghcr.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .ghcr.env
  set +a
fi

if [[ ! -f .env ]]; then
  echo "Missing $PWD/.env; run 'cp env.example .env' and set SECUREMAIL_API_KEY." >&2
  exit 1
fi

if [[ -n "${GHCR_TOKEN:-}" ]]; then
  : "${GHCR_USERNAME:?GHCR_USERNAME is required when GHCR_TOKEN is set}"
  printf '%s' "$GHCR_TOKEN" | docker login ghcr.io \
    --username "$GHCR_USERNAME" \
    --password-stdin >/dev/null
fi

docker compose config >/dev/null
docker compose pull api
docker compose up -d --no-build --remove-orphans

container_id="$(docker compose ps -q api)"
if [[ -z "$container_id" ]]; then
  echo "API container was not created." >&2
  exit 1
fi

for _ in {1..30}; do
  status="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$container_id" 2>/dev/null || true)"
  case "$status" in
    healthy)
      echo "SecureMail API is healthy (${IMAGE_TAG})."
      exit 0
      ;;
    unhealthy)
      docker compose logs --tail=100 api >&2
      exit 1
      ;;
  esac
  sleep 2
done

docker compose logs --tail=100 api >&2
exit 1
