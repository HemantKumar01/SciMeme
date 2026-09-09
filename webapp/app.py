from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from .pipeline import MAX_PDF_BYTES, MAX_SEARCH_ITERATIONS, run_pipeline_events

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
load_dotenv(APP_DIR.parent / ".env")

BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1").strip()
if not re.fullmatch(r"[a-z]{2}(?:-gov)?-[a-z]+-\d", BEDROCK_REGION):
    raise RuntimeError("BEDROCK_REGION is invalid.")
BUILT_IN_BASE_URL = f"https://bedrock-mantle.{BEDROCK_REGION}.api.aws/v1"
BUILT_IN_MODEL_OPTIONS = [
    {"id": "zai.glm-5", "label": "GLM 5"},
    {
        "id": "qwen.qwen3-vl-235b-a22b-instruct",
        "label": "Qwen3 VL 235B A22B",
    },
    {"id": "meta.llama3-70b-instruct-v1:0", "label": "Llama 3 70B Instruct"},
]
BUILT_IN_MODELS = [option["id"] for option in BUILT_IN_MODEL_OPTIONS]
MODEL_EXCLUSIONS = (
    "audio",
    "realtime",
    "transcribe",
    "tts",
    "image",
    "embedding",
    "moderation",
    "search",
    "computer",
    "codex",
    "deep-research",
)
PREFERRED_MODELS = [
    "gpt-5.6-luna",
    "gpt-5.6-terra",
    "gpt-5.6-sol",
    "gpt-5.4-mini",
    "gpt-5-mini",
    "gpt-4.1-mini",
    "gpt-4o-mini",
]

app = FastAPI(
    title="SciMemeX Studio",
    description="Generate a scientific meme from a research-paper PDF.",
    version="3.1.0",
)


class ModelRequest(BaseModel):
    api_key: str = Field(min_length=12, max_length=500)


def _is_text_model(model_id: str) -> bool:
    lowered = model_id.casefold()
    if any(excluded in lowered for excluded in MODEL_EXCLUSIONS):
        return False
    return bool(
        re.match(r"^(?:gpt-(?:4o|4\.1|5|6)(?:[-.]|$)|o[34](?:[-.]|$))", lowered)
    )


def _sort_models(model_ids: list[str]) -> list[str]:
    order = {model: index for index, model in enumerate(PREFERRED_MODELS)}
    return sorted(
        set(model_ids), key=lambda model: (order.get(model, len(order)), model)
    )


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "scope": "exploration_exploitation_through_final_meme"}


def _bedrock_api_key() -> str:
    return os.getenv("OPEN_SOURCE_API_KEY", "").strip()


@app.get("/api/providers")
async def providers() -> dict[str, object]:
    return {
        "default_provider": "open_source",
        "open_source": {
            "available": bool(_bedrock_api_key()),
            "provider": "Amazon Bedrock",
            "region": BEDROCK_REGION,
            "models": BUILT_IN_MODEL_OPTIONS,
        },
    }


@app.post("/api/models")
async def list_models(request: ModelRequest) -> dict[str, list[str]]:
    client = AsyncOpenAI(api_key=request.api_key, timeout=30.0, max_retries=0)
    try:
        page = await client.models.list()
        models = _sort_models(
            [item.id for item in page.data if _is_text_model(item.id)]
        )
        if not models:
            raise HTTPException(
                status_code=502,
                detail="No compatible text models were found for this key.",
            )
        return {"models": models}
    except HTTPException:
        raise
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        if status == 401:
            detail = "OpenAI rejected the API key. Verify the key and project access."
        elif status == 429:
            detail = "OpenAI rate or quota limits were reached. Check project billing and retry."
        else:
            detail = "The OpenAI model list could not be loaded. Check the key and network, then retry."
        raise HTTPException(
            status_code=status if status in {401, 429} else 502, detail=detail
        ) from exc
    finally:
        await client.close()


@app.post("/api/run")
async def run_pipeline(
    paper: UploadFile = File(...),
    provider: Literal["open_source", "openai"] = Form("open_source"),
    api_key: str = Form("", max_length=500),
    innovation_model: str = Form(..., min_length=2, max_length=100),
    concisio_model: str = Form(..., min_length=2, max_length=100),
    tsa_model: str = Form(..., min_length=2, max_length=100),
    generation_model: str = Form(..., min_length=2, max_length=100),
    critic_model: str = Form(..., min_length=2, max_length=100),
    top_k: int = Form(5, ge=1, le=5),
    max_iterations: int = Form(MAX_SEARCH_ITERATIONS, ge=1, le=MAX_SEARCH_ITERATIONS),
) -> StreamingResponse:
    if paper.content_type not in {"application/pdf", "application/x-pdf"} and not (
        paper.filename or ""
    ).lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Upload a PDF file.")
    contents = await paper.read(MAX_PDF_BYTES + 1)
    await paper.close()
    if len(contents) > MAX_PDF_BYTES:
        raise HTTPException(
            status_code=413, detail="The PDF exceeds the 25 MB upload limit."
        )

    base_url: str | None = None
    if provider == "open_source":
        selected_models = {
            innovation_model,
            concisio_model,
            tsa_model,
            generation_model,
            critic_model,
        }
        unsupported_models = selected_models.difference(BUILT_IN_MODELS)
        if unsupported_models:
            raise HTTPException(
                status_code=422,
                detail="Choose a model available in the open source models tab.",
            )
        api_key = _bedrock_api_key()
        if not api_key:
            raise HTTPException(
                status_code=503,
                detail="Amazon Bedrock is not configured on this server. Set OPEN_SOURCE_API_KEY in .env.",
            )
        base_url = BUILT_IN_BASE_URL
    elif len(api_key.strip()) < 12:
        raise HTTPException(status_code=422, detail="Enter your OpenAI API key.")
    else:
        api_key = api_key.strip()

    events = run_pipeline_events(
        pdf_bytes=contents,
        filename=Path(paper.filename or "paper.pdf").name,
        api_key=api_key,
        base_url=base_url,
        innovation_model=innovation_model,
        concisio_model=concisio_model,
        tsa_model=tsa_model,
        generation_model=generation_model,
        critic_model=critic_model,
        top_k=top_k,
        max_iterations=max_iterations,
    )
    return StreamingResponse(
        events,
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
