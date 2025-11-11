#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$ROOT_DIR/infra/docker/compose.yaml"

export DB_URL="${DB_URL:-postgresql://postgres:postgres@localhost:5432/screenalytics}"
export REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"
export S3_ENDPOINT="${S3_ENDPOINT:-http://localhost:9000}"
export S3_ACCESS_KEY="${S3_ACCESS_KEY:-minio}"
export S3_SECRET_KEY="${S3_SECRET_KEY:-miniosecret}"
export S3_BUCKET="${S3_BUCKET:-screenalytics}"

docker compose -f "$COMPOSE_FILE" up -d

echo "Ensuring pgvector extension exists..."
psql "$DB_URL" -c "CREATE EXTENSION IF NOT EXISTS vector;" >/dev/null

echo "Applying migrations..."
for f in "$ROOT_DIR"/db/migrations/*.sql; do
  echo "  -> $f"
  psql "$DB_URL" -f "$f" >/dev/null
done

cat <<MSG
Environment variables for this shell:
  export DB_URL=$DB_URL
  export REDIS_URL=$REDIS_URL
  export S3_ENDPOINT=$S3_ENDPOINT
  export S3_ACCESS_KEY=$S3_ACCESS_KEY
  export S3_SECRET_KEY=$S3_SECRET_KEY
  export S3_BUCKET=$S3_BUCKET

Reminder: run "source tools/dev-up.sh" so the exports persist in your current shell.
MSG
