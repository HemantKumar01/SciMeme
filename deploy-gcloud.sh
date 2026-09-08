#!/usr/bin/env bash

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Build and deploy the SciMemeX web application to Google Cloud Run.

Usage:
  ./deploy-gcloud.sh [PROJECT_ID] [REGION] [SERVICE_NAME]

Defaults:
  PROJECT_ID    $GOOGLE_CLOUD_PROJECT, then the active gcloud project
  REGION        $GCLOUD_REGION, then us-central1
  SERVICE_NAME  $CLOUD_RUN_SERVICE, then scimeme

Example:
  ./deploy-gcloud.sh my-project us-central1 scimeme

The service is public, uses request-based CPU, scales to zero, and is capped at
one small instance. These settings are intended to keep light use within Cloud
Run's monthly free tier, but Google Cloud does not provide a hard zero-cost cap.
The script reads OPEN_SOURCE_API_KEY from .env, stores it in Secret Manager,
and exposes it to the Cloud Run service. Users can alternatively provide their
own OpenAI API key.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v gcloud >/dev/null 2>&1; then
  echo "Error: gcloud is not installed. Install the Google Cloud CLI first." >&2
  exit 1
fi

readonly SCIMEMEX_ENV_FILE="$SCRIPT_DIR/.env"
if [[ ! -f "$SCIMEMEX_ENV_FILE" ]]; then
  echo "Error: $SCIMEMEX_ENV_FILE does not exist. Copy .env.example to .env and add OPEN_SOURCE_API_KEY." >&2
  exit 1
fi

scimemex_api_key=""
while IFS= read -r scimemex_env_line || [[ -n "$scimemex_env_line" ]]; do
  scimemex_env_line="${scimemex_env_line%$'\r'}"
  if [[ "$scimemex_env_line" =~ ^[[:space:]]*(export[[:space:]]+)?OPEN_SOURCE_API_KEY[[:space:]]*=(.*)$ ]]; then
    scimemex_api_key="${BASH_REMATCH[2]}"
    scimemex_api_key="${scimemex_api_key#"${scimemex_api_key%%[![:space:]]*}"}"
    scimemex_api_key="${scimemex_api_key%"${scimemex_api_key##*[![:space:]]}"}"
    if [[ "$scimemex_api_key" == \"*\" && "$scimemex_api_key" == *\" ]]; then
      scimemex_api_key="${scimemex_api_key:1:-1}"
    elif [[ "$scimemex_api_key" == \'*\' && "$scimemex_api_key" == *\' ]]; then
      scimemex_api_key="${scimemex_api_key:1:-1}"
    fi
    break
  fi
done < "$SCIMEMEX_ENV_FILE"

if [[ -z "$scimemex_api_key" || "$scimemex_api_key" == "your-api-key" ]]; then
  echo "Error: set OPEN_SOURCE_API_KEY to a real value in $SCIMEMEX_ENV_FILE." >&2
  exit 1
fi
readonly SCIMEMEX_API_KEY="$scimemex_api_key"
unset scimemex_api_key scimemex_env_line

project_id="${1:-${GOOGLE_CLOUD_PROJECT:-}}"
if [[ -z "$project_id" ]]; then
  project_id="$(gcloud config get-value project 2>/dev/null || true)"
fi
if [[ -z "$project_id" || "$project_id" == "(unset)" ]]; then
  echo "Error: pass PROJECT_ID or configure one with: gcloud config set project PROJECT_ID" >&2
  exit 1
fi

readonly PROJECT_ID="$project_id"
readonly REGION="${2:-${GCLOUD_REGION:-us-central1}}"
readonly SERVICE_NAME="${3:-${CLOUD_RUN_SERVICE:-scimeme}}"
readonly MODEL_API_SECRET_NAME="${SCIMEMEX_SECRET_NAME:-scimemex-open-source-api-key}"

if [[ ! "$SERVICE_NAME" =~ ^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$ ]]; then
  echo "Error: SERVICE_NAME must start with a letter and contain only lowercase letters, digits, or hyphens." >&2
  exit 1
fi
if [[ ! "$MODEL_API_SECRET_NAME" =~ ^[A-Za-z]([A-Za-z0-9_-]{0,253}[A-Za-z0-9])?$ ]]; then
  echo "Error: SCIMEMEX_SECRET_NAME is not a valid Secret Manager secret name." >&2
  exit 1
fi
if [[ "$PROJECT_ID" =~ [[:space:]] || "$REGION" =~ [[:space:]] ]]; then
  echo "Error: PROJECT_ID and REGION cannot contain whitespace." >&2
  exit 1
fi

active_account="$(gcloud auth list --filter=status:ACTIVE --limit=1 --format='value(account)')"
if [[ -z "$active_account" ]]; then
  echo "Error: gcloud is not authenticated. Run: gcloud auth login" >&2
  exit 1
fi

echo "Deploying SciMemeX"
echo "  account: $active_account"
echo "  project: $PROJECT_ID"
echo "  region:  $REGION"
echo "  service: $SERVICE_NAME"
echo "  limits:  scale 0–1, 1 vCPU, 512 MiB, concurrency 2, request-based CPU"
echo "  models:  shared credential loaded securely from .env"

gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  --project "$PROJECT_ID" \
  --quiet

if ! gcloud secrets describe "$MODEL_API_SECRET_NAME" --project "$PROJECT_ID" >/dev/null 2>&1; then
  gcloud secrets create "$MODEL_API_SECRET_NAME" \
    --project "$PROJECT_ID" \
    --replication-policy automatic \
    --quiet
fi

printf '%s' "$SCIMEMEX_API_KEY" | gcloud secrets versions add "$MODEL_API_SECRET_NAME" \
  --project "$PROJECT_ID" \
  --data-file=- \
  --quiet

runtime_service_account="$(
  gcloud run services describe "$SERVICE_NAME" \
    --project "$PROJECT_ID" \
    --region "$REGION" \
    --format='value(spec.template.spec.serviceAccountName)' 2>/dev/null || true
)"
if [[ -z "$runtime_service_account" ]]; then
  project_number="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
  runtime_service_account="${project_number}-compute@developer.gserviceaccount.com"
fi
readonly RUNTIME_SERVICE_ACCOUNT="$runtime_service_account"
unset runtime_service_account project_number

gcloud secrets add-iam-policy-binding "$MODEL_API_SECRET_NAME" \
  --project "$PROJECT_ID" \
  --member="serviceAccount:$RUNTIME_SERVICE_ACCOUNT" \
  --role=roles/secretmanager.secretAccessor \
  --quiet >/dev/null

# Support both current service-level scaling flags and older gcloud releases.
scaling_flags=(--min-instances=0 --max-instances=1)
deploy_help="$(CLOUDSDK_PAGER="" gcloud run deploy --help 2>/dev/null || true)"
if grep -q -- '--max=MAX' <<<"$deploy_help"; then
  scaling_flags=(--min=0 --max=1)
fi

gcloud run deploy "$SERVICE_NAME" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --source "$SCRIPT_DIR" \
  --allow-unauthenticated \
  --ingress all \
  --port 8080 \
  --cpu 1 \
  --memory 512Mi \
  --concurrency 2 \
  --timeout 3600 \
  --cpu-throttling \
  --no-cpu-boost \
  --update-secrets="OPEN_SOURCE_API_KEY=$MODEL_API_SECRET_NAME:latest" \
  "${scaling_flags[@]}" \
  --labels app=scimemex,managed-by=deploy-gcloud \
  --quiet

service_url="$(
  gcloud run services describe "$SERVICE_NAME" \
    --project "$PROJECT_ID" \
    --region "$REGION" \
    --format='value(status.url)'
)"

echo "Deployment complete: $service_url"
echo "Health endpoint:     $service_url/api/health"
