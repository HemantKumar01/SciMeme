from __future__ import annotations

import asyncio
import base64
import io
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator
from urllib.parse import quote, urlparse

import httpx
from openai import AsyncOpenAI
from PIL import Image, ImageDraw, ImageFont

try:
    import pymupdf
except ImportError:  # PyMuPDF < 1.24 compatibility
    import fitz as pymupdf  # type: ignore[no-redef]


MAX_PDF_BYTES = 25 * 1024 * 1024
MAX_PAGES = 100
MAX_AGENT_INPUT_CHARS = 72_000
TEMPLATE_LIBRARY_PATH = Path(__file__).resolve().parents[1] / "data" / "meme_template_dataset.json"
IMGFLIP_CATALOG_URL = "https://api.imgflip.com/get_memes"
MAX_TEMPLATE_IMAGE_BYTES = 8 * 1024 * 1024
MAX_SEARCH_ITERATIONS = 6
MAX_AGENT_ATTEMPTS = 4
AGENT_RETRY_BASE_SECONDS = 1.0
BEDROCK_MANTLE_HOST_PREFIX = "bedrock-mantle."
BEDROCK_GLM_MODEL = "zai.glm-5"
BEDROCK_QWEN_VL_MODEL = "qwen.qwen3-vl-235b-a22b-instruct"
BEDROCK_LLAMA_MODEL = "meta.llama3-70b-instruct-v1:0"
BEDROCK_CONVERSE_MODELS = {BEDROCK_LLAMA_MODEL}
TEXT_ONLY_BEDROCK_MODELS = {BEDROCK_GLM_MODEL, BEDROCK_LLAMA_MODEL}
REASONING_BEDROCK_MODELS = {BEDROCK_GLM_MODEL, BEDROCK_QWEN_VL_MODEL}

INNOVATION_SYSTEM_PROMPT = """You are a comparative research analyst. Your task is to deeply analyze a research paper to extract the main conceptual and technical differences between past work and this paper's contributions.

Structure your response as:
Prior Work (Old Ideas): Summarize how this problem was tackled in previous literature. Include model names, strategies, or limitations if possible.
Proposed Approach (New Idea): What method, model, or strategy is introduced by this paper?
Core Differences: A comparative analysis answering:
- What exactly changes in methodology, architecture, or objective?
- Why is this a meaningful improvement?
- Are there tradeoffs or assumptions introduced?

Be precise, technical, and avoid generic summaries. Rely only on information in the supplied paper excerpts. Do not introduce external facts. Return only the requested structured analysis."""

CONCISIO_SYSTEM_PROMPT = """Your task is to read a structured comparison between prior work and a new method, and produce a single paragraph that captures the essential difference in approach or idea.

Your output must:
- Clearly state what was done before.
- Explain what this paper does differently.
- Highlight why this difference is important.
- Be technical but readable and under 100 words.

Avoid vagueness. Do not invent anything not present in the input. Output only the paragraph, without a heading or filler text."""

TSA_SYSTEM_PROMPT = """You are the Template Selector Agent in SciMemeX. Given a paper's key ideas, contrastive feedback when available, and a fixed list of meme templates, choose the top-k templates that best highlight the contrast between prior work and the paper's contributions.

Prefer templates with at least two conceptual roles or panels, a clear visual structure, and a natural fit for a prior-work versus proposed-work contrast. Judge semantic and rhetorical fit, not template popularity. Select only exact template names from the provided library.

Return strict JSON with this shape and no markdown fences:
{"selections":[{"template_name":"exact library name","reasoning":"brief reason","contrast_mapping":"how old and new ideas map onto the template"}]}"""

CAPTION_SYSTEM_PROMPT = """You are the SciMemeX Generator Agent. Generate and place concise scientific meme captions for a graduate-level research audience.

Inspect every supplied template image. For each template, write exactly the requested number of text boxes and place each one using normalized image coordinates from 0 to 1000: x and y are the top-left corner; width and height define its bounding box. Prefer existing blank or dedicated caption regions. Keep every box inside the image, avoid faces and important visual elements, and associate text with the correct panel or character. Use font_size 20-40, expressed as thousandths of the image's shorter side; prefer 24-32 and use smaller sizes for longer text. Captions must contrast prior work with the paper's new contribution, preserve technical accuracy, fit the template's visual roles, and remain readable without the paper. Prefer a sharp observation over forced humor. Do not use slurs, disparagement, emojis, headings, or unsupported claims. Return only the requested structured output."""

FIDELITY_SYSTEM_PROMPT = """You are the Scientific Fidelity Critic in a scientific-meme pipeline. Compare each candidate only with the supplied research summary. Score 1-5: 5 preserves the precise core contribution; 4 is correct but omits nuance; 3 conveys only the general topic; 2 is misleading; 1 is inaccurate or unrelated. Do not judge humor. Give brief actionable feedback and return only the requested structured output."""

CLARITY_SYSTEM_PROMPT = """You are the Clarity Critic in a scientific-meme pipeline. Judge whether graduate students or early-career researchers can recover the intended contrast from each template-caption pair. Score 1-3: 3 clear, 2 partly clear or missing important nuance, 1 unclear. Give brief actionable feedback and return only the requested structured output."""

ENGAGEMENT_SYSTEM_PROMPT = """You are the Engagement Critic in a scientific-meme pipeline. Judge template-caption synergy, wit, memorability, and likely resonance with a technical audience. Score 1-5. Assign 1 to anything offensive or disparaging. Do not reward scientific distortion. Give brief actionable feedback and return only the requested structured output."""

CONTRASTIVE_SYSTEM_PROMPT = """You are the Contrastive Feedback-Guided Meme Generator in SciMemeX. Generate one candidate for every supplied template. Use the best candidate and its evaluation as positive evidence; use the worst candidate and its evaluation as negative evidence. Keep the winner's strengths while avoiding the loser's weaknesses. Preserve scientific fidelity, clarity, template fit, respectful humor, and normalized coordinates. Inspect each supplied image and return only the requested structured output."""


@dataclass(slots=True)
class ExtractedPaper:
    filename: str
    page_count: int
    character_count: int
    sections: dict[str, str]
    agent_input: str
    agent_input_truncated: bool


class PipelineError(RuntimeError):
    """A user-presentable pipeline failure."""


def _clean_text(value: str) -> str:
    value = value.replace("\x00", " ").replace("\u00ad", "")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n[ \t]+", "\n", value)
    return re.sub(r"\n{3,}", "\n\n", value).strip()


SECTION_NAMES: dict[str, tuple[str, ...]] = {
    "abstract": ("abstract",),
    "introduction": ("introduction",),
    "related_work": (
        "related work",
        "related works",
        "background",
        "literature review",
        "previous work",
    ),
    "methodology": (
        "methodology",
        "methods",
        "proposed approach",
    ),
    "conclusion": (
        "conclusion",
        "conclusions",
        "conclusion and future work",
        "discussion and conclusion",
    ),
    "evaluation": (
        "evaluation",
        "experiments",
        "experimental setup",
        "results",
        "results and discussion",
    ),
    "limitations": ("limitations", "limitation", "future work"),
    "acknowledgments": ("acknowledgments", "acknowledgements"),
    "appendix": ("appendix", "appendices"),
    "references": ("references", "bibliography"),
}


def _heading_markers(text: str) -> list[tuple[int, int, str]]:
    aliases = sorted(
        ((alias, canonical) for canonical, names in SECTION_NAMES.items() for alias in names),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    alias_pattern = "|".join(re.escape(alias) for alias, _ in aliases)
    pattern = re.compile(
        rf"(?im)^\s*(?:[0-9]+(?:\.[0-9]+)*[.)]?\s+)?(?P<name>{alias_pattern})\s*[:.]?\s*$"
    )
    alias_map = {alias.casefold(): canonical for alias, canonical in aliases}
    return [
        (match.start(), match.end(), alias_map[match.group("name").casefold()])
        for match in pattern.finditer(text)
    ]


def _extract_sections(text: str) -> dict[str, str]:
    markers = _heading_markers(text)
    boundaries = {
        "abstract": {"introduction", "related_work", "methodology", "evaluation", "conclusion", "references"},
        "related_work": {"methodology", "evaluation", "conclusion", "limitations", "references"},
        "methodology": {"evaluation", "conclusion", "limitations", "references"},
        "conclusion": {"limitations", "acknowledgments", "appendix", "references"},
    }
    sections: dict[str, str] = {}
    for target in ("abstract", "related_work", "methodology", "conclusion"):
        candidates = [item for item in markers if item[2] == target]
        if not candidates:
            continue
        marker = candidates[-1] if target == "conclusion" else candidates[0]
        marker_index = markers.index(marker)
        next_boundary = next(
            (item for item in markers[marker_index + 1 :] if item[2] in boundaries[target]),
            None,
        )
        end = next_boundary[0] if next_boundary else len(text)
        section = _clean_text(text[marker[1] : end])
        if section:
            sections[target] = section

    if "abstract" not in sections:
        inline = re.search(r"(?is)\babstract\b\s*[:.—-]?\s*(.+?)(?=\b(?:1\s+)?introduction\b)", text)
        if inline:
            sections["abstract"] = _clean_text(inline.group(1))
    return sections


def extract_pdf(
    pdf_bytes: bytes,
    filename: str,
    *,
    max_pdf_bytes: int = MAX_PDF_BYTES,
) -> ExtractedPaper:
    if not pdf_bytes.startswith(b"%PDF"):
        raise PipelineError("The uploaded file is not a valid PDF.")
    if len(pdf_bytes) > max_pdf_bytes:
        size_limit_mb = max_pdf_bytes // (1024 * 1024)
        raise PipelineError(f"The PDF exceeds the {size_limit_mb} MB size limit.")

    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise PipelineError("The PDF could not be opened. It may be damaged or encrypted.") from exc

    try:
        if document.needs_pass:
            raise PipelineError("Password-protected PDFs are not supported.")
        if document.page_count > MAX_PAGES:
            raise PipelineError(f"The PDF has more than the {MAX_PAGES}-page limit.")
        # Native PDF block order is usually more reliable for two-column research papers.
        # Sorting by coordinates can interleave columns and attach headings to unrelated text.
        pages = [page.get_text("text", sort=False) for page in document]
    finally:
        document.close()

    full_text = _clean_text("\n\n".join(pages))
    if len(full_text) < 100:
        raise PipelineError("No usable text was found. Scanned PDFs need OCR before upload.")

    sections = _extract_sections(full_text)
    preferred = [
        ("Abstract", sections.get("abstract", "")),
        ("Related Work", sections.get("related_work", "")),
        ("Methodology", sections.get("methodology", "")),
        ("Conclusion", sections.get("conclusion", "")),
    ]
    selected = [f"## {heading}\n{content}" for heading, content in preferred if content]
    agent_input = "\n\n".join(selected)
    if len(agent_input) < 1_200:
        agent_input = full_text
    truncated = len(agent_input) > MAX_AGENT_INPUT_CHARS
    agent_input = agent_input[:MAX_AGENT_INPUT_CHARS]

    return ExtractedPaper(
        filename=filename,
        page_count=len(pages),
        character_count=len(full_text),
        sections=sections,
        agent_input=agent_input,
        agent_input_truncated=truncated,
    )


def load_template_library() -> dict[str, dict[str, str]]:
    try:
        library = json.loads(TEMPLATE_LIBRARY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError("The bundled template dataset could not be loaded.") from exc
    if not isinstance(library, dict) or not all(
        isinstance(key, str)
        and isinstance(value, dict)
        and isinstance(value.get("description"), str)
        for key, value in library.items()
    ):
        raise PipelineError("The bundled template library has an unexpected format.")
    return library


def display_template_name(filename: str) -> str:
    name = re.sub(r"\.(?:jpe?g|png|webp)$", "", filename.strip(), flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", name.replace("_", " ")).strip()


def _canonical_template_name(value: str) -> str:
    value = re.sub(r"\.(?:jpe?g|png|webp)$", "", value.strip(), flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _library_for_prompt(library: dict[str, dict[str, str]]) -> str:
    def compact_description(value: str) -> str:
        cleaned = _clean_text(value).replace("\n", " ")
        if len(cleaned) <= 220:
            return cleaned
        shortened = cleaned[:220].rsplit(" ", 1)[0]
        return shortened.rstrip(" ,;:-") + "…"

    return "\n\n".join(
        f"TEMPLATE: {name}\nDESCRIPTION: {compact_description(details['description'])}"
        for name, details in library.items()
    )


async def fetch_imgflip_catalog() -> dict[str, dict[str, Any]]:
    """Return public Imgflip template metadata keyed by normalized name."""
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=False) as client:
            response = await client.get(IMGFLIP_CATALOG_URL)
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise PipelineError("The public meme-template catalog could not be loaded. Try again shortly.") from exc

    memes = payload.get("data", {}).get("memes", []) if payload.get("success") else []
    catalog: dict[str, dict[str, Any]] = {}
    for meme in memes:
        if not isinstance(meme, dict):
            continue
        name = str(meme.get("name", "")).strip()
        image_url = str(meme.get("url", "")).strip()
        parsed = urlparse(image_url)
        if not name or parsed.scheme != "https" or parsed.hostname != "i.imgflip.com":
            continue
        try:
            entry = {
                "template_id": str(meme["id"]),
                "template_name": name,
                "image_url": image_url,
                "width": int(meme.get("width", 0)),
                "height": int(meme.get("height", 0)),
                "box_count": max(1, min(int(meme.get("box_count", 2)), 6)),
            }
        except (KeyError, TypeError, ValueError):
            continue
        catalog[_canonical_template_name(name)] = entry
    if not catalog:
        raise PipelineError("The public meme-template catalog returned no usable templates.")
    return catalog


def resolve_renderable_templates(
    selections: list[dict[str, str]], catalog: dict[str, dict[str, Any]], *, require_all: bool = True
) -> tuple[list[dict[str, Any]], list[str]]:
    renderable: list[dict[str, Any]] = []
    unavailable: list[str] = []
    for selection in selections:
        name = selection["template_name"]
        canonical_name = _canonical_template_name(name)
        catalog_entry = catalog.get(canonical_name)
        if catalog_entry is None and len(canonical_name) >= 8:
            prefix_matches = [
                entry
                for catalog_name, entry in catalog.items()
                if catalog_name.startswith(canonical_name) or canonical_name.startswith(catalog_name)
            ]
            if len(prefix_matches) == 1:
                catalog_entry = prefix_matches[0]
        if catalog_entry is None:
            unavailable.append(name)
            continue
        renderable.append(
            {
                **catalog_entry,
                "description": selection["description"],
                "tsa_reasoning": selection["reasoning"],
                "contrast_mapping": selection["contrast_mapping"],
            }
        )
    if require_all and unavailable:
        raise PipelineError(
            "Selected templates are unavailable in Imgflip's public catalog: " + ", ".join(unavailable)
        )
    return renderable, unavailable


def _clean_caption(value: Any) -> str:
    text = re.sub(r"^\s*(?:line|box|caption)?\s*\d+\s*[:.)-]\s*", "", str(value), flags=re.I)
    return re.sub(r"\s+", " ", text).strip().strip('"')


def _text_box_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "x": {"type": "integer", "minimum": 0, "maximum": 950},
            "y": {"type": "integer", "minimum": 0, "maximum": 950},
            "width": {"type": "integer", "minimum": 50, "maximum": 1000},
            "height": {"type": "integer", "minimum": 40, "maximum": 1000},
            "font_size": {"type": "integer", "minimum": 20, "maximum": 40},
            "align": {"type": "string", "enum": ["left", "center", "right"]},
        },
        "required": ["text", "x", "y", "width", "height", "font_size", "align"],
        "additionalProperties": False,
    }


def _normalize_text_boxes(value: Any, *, expected: int) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != expected:
        return []
    boxes: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            return []
        text = _clean_caption(item.get("text", ""))
        if not text:
            return []
        try:
            x = max(0, min(950, int(item.get("x"))))
            y = max(0, min(950, int(item.get("y"))))
            width = max(50, min(1000 - x, int(item.get("width"))))
            height = max(40, min(1000 - y, int(item.get("height"))))
            font_size = max(20, min(40, int(item.get("font_size"))))
        except (TypeError, ValueError):
            return []
        align = str(item.get("align", "center"))
        if align not in {"left", "center", "right"}:
            align = "center"
        boxes.append(
            {
                "text": text,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "font_size": font_size,
                "align": align,
            }
        )
    return boxes


def normalize_caption_candidates(
    raw: str, templates: list[dict[str, Any]], *, prefix: str = "initial"
) -> list[dict[str, Any]]:
    parsed = _parse_json_object(raw)
    supplied = parsed.get("candidates")
    if not isinstance(supplied, list):
        raise PipelineError("The caption generator returned an unexpected response.")

    by_name = {_canonical_template_name(item["template_name"]): item for item in templates}
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in supplied:
        if not isinstance(item, dict):
            continue
        template = by_name.get(_canonical_template_name(str(item.get("template_name", ""))))
        if template is None or template["template_id"] in seen:
            continue
        text_boxes = _normalize_text_boxes(item.get("text_boxes"), expected=template["box_count"])
        if not text_boxes:
            continue
        captions = [box["text"] for box in text_boxes]
        seen.add(template["template_id"])
        candidates.append(
            {
                "candidate_id": f"{prefix}_{len(candidates) + 1}",
                "template": template,
                "captions": captions,
                "text_boxes": text_boxes,
                "coordinate_system": "normalized_0_1000_top_left",
                "generation_note": _clean_caption(item.get("generation_note", "")),
            }
        )
    if not candidates:
        raise PipelineError("No caption candidate matched the selected templates. Try another model.")
    return candidates


def _review_schema(candidate_ids: list[str], *, max_score: int = 5) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "reviews": {
                "type": "array",
                "minItems": len(candidate_ids),
                "maxItems": len(candidate_ids),
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {"type": "string", "enum": candidate_ids},
                        "score": {"type": "integer", "minimum": 1, "maximum": max_score},
                        "reasoning": {"type": "string"},
                        "feedback": {"type": "string"},
                    },
                    "required": ["candidate_id", "score", "reasoning", "feedback"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["reviews"],
        "additionalProperties": False,
    }


def _reviews_by_id(raw: str, candidate_ids: list[str], *, max_score: int) -> dict[str, dict[str, Any]]:
    parsed = _parse_json_object(raw)
    reviews = parsed.get("reviews")
    if not isinstance(reviews, list):
        raise PipelineError("A critique agent returned an unexpected response.")
    expected = set(candidate_ids)
    normalized: dict[str, dict[str, Any]] = {}
    for review in reviews:
        if not isinstance(review, dict):
            continue
        candidate_id = str(review.get("candidate_id", ""))
        if candidate_id not in expected or candidate_id in normalized:
            continue
        try:
            score = int(review.get("score"))
        except (TypeError, ValueError):
            continue
        if not 1 <= score <= max_score:
            continue
        normalized[candidate_id] = {
            "score": score,
            "reasoning": _clean_caption(review.get("reasoning", "")),
            "feedback": _clean_caption(review.get("feedback", "")),
        }
    if set(normalized) != expected:
        raise PipelineError("A critique agent did not evaluate every candidate. Try another model.")
    return normalized


def _candidate_payload(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": item["candidate_id"],
            "template_name": item["template"]["template_name"],
            "template_description": item["template"]["description"],
            "captions": item["captions"],
            "text_boxes": item["text_boxes"],
            "coordinate_system": item["coordinate_system"],
        }
        for item in candidates
    ]


def _uses_bedrock_endpoint(client: AsyncOpenAI) -> bool:
    parsed = urlparse(str(client.base_url))
    return bool(parsed.hostname and parsed.hostname.startswith(BEDROCK_MANTLE_HOST_PREFIX))


def _bedrock_region(client: AsyncOpenAI) -> str:
    hostname = urlparse(str(client.base_url)).hostname or ""
    if hostname.startswith(BEDROCK_MANTLE_HOST_PREFIX):
        return hostname.removeprefix(BEDROCK_MANTLE_HOST_PREFIX).removesuffix(".api.aws")
    raise PipelineError("The Amazon Bedrock endpoint is invalid.")


def _bedrock_user_text(user_input: Any) -> str:
    if isinstance(user_input, str):
        return user_input
    parts: list[str] = []
    for message in user_input:
        for item in message.get("content", []):
            if item.get("type") == "input_text":
                parts.append(str(item.get("text", "")))
    return "\n\n".join(part for part in parts if part)


async def _bedrock_converse_completion(
    client: AsyncOpenAI,
    *,
    model: str,
    instructions: str,
    user_input: Any,
    max_output_tokens: int,
) -> str:
    api_key = getattr(client, "api_key", "")
    if not isinstance(api_key, str) or not api_key:
        raise PipelineError("The Amazon Bedrock API key is unavailable.")
    region = _bedrock_region(client)
    endpoint = (
        f"https://bedrock-runtime.{region}.amazonaws.com/model/"
        f"{quote(model, safe='')}/converse"
    )
    payload = {
        "system": [{"text": instructions}],
        "messages": [
            {
                "role": "user",
                "content": [{"text": _bedrock_user_text(user_input)}],
            }
        ],
        # Llama 3 70B supports at most 2,048 generated tokens.
        "inferenceConfig": {"maxTokens": min(max_output_tokens, 2_048)},
    }
    async with httpx.AsyncClient(timeout=180.0) as http_client:
        response = await http_client.post(
            endpoint,
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        )
        response.raise_for_status()
    try:
        blocks = response.json()["output"]["message"]["content"]
        return "\n".join(str(block["text"]) for block in blocks if "text" in block)
    except (KeyError, TypeError, ValueError) as exc:
        raise _UnreadableAgentResponse(
            f"Model {model} returned an unexpected Amazon Bedrock response."
        ) from exc


def _image_data_url(image_bytes: bytes) -> str:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            mime_type = Image.MIME.get(image.format or "", "image/jpeg")
    except (OSError, ValueError) as exc:
        raise PipelineError("A meme-template image could not be prepared for the selected model.") from exc
    return f"data:{mime_type};base64," + base64.b64encode(image_bytes).decode("ascii")


async def critique_candidates(
    client: AsyncOpenAI,
    *,
    model: str,
    research_idea: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidate_ids = [item["candidate_id"] for item in candidates]
    candidate_json = json.dumps(_candidate_payload(candidates), ensure_ascii=False, indent=2)
    base_input = f"CORE RESEARCH CONTRAST:\n{research_idea}\n\nCANDIDATES:\n{candidate_json}"
    five_point_schema = _review_schema(candidate_ids, max_score=5)
    clarity_schema = _review_schema(candidate_ids, max_score=3)

    fidelity_raw, clarity_raw, engagement_raw = await asyncio.gather(
        call_agent(
            client,
            model=model,
            instructions=FIDELITY_SYSTEM_PROMPT,
            user_input=base_input,
            max_output_tokens=2_000,
            response_schema=five_point_schema,
            schema_name="fidelity_reviews",
        ),
        call_agent(
            client,
            model=model,
            instructions=CLARITY_SYSTEM_PROMPT,
            user_input=base_input + "\n\nFor this critic, use scores 1-3 only.",
            max_output_tokens=2_000,
            response_schema=clarity_schema,
            schema_name="clarity_reviews",
        ),
        call_agent(
            client,
            model=model,
            instructions=ENGAGEMENT_SYSTEM_PROMPT,
            user_input=base_input,
            max_output_tokens=2_000,
            response_schema=five_point_schema,
            schema_name="engagement_reviews",
        ),
    )
    fidelity = _reviews_by_id(fidelity_raw, candidate_ids, max_score=5)
    clarity = _reviews_by_id(clarity_raw, candidate_ids, max_score=3)
    engagement = _reviews_by_id(engagement_raw, candidate_ids, max_score=5)

    evaluated: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        scores = {
            "fidelity": fidelity[candidate_id],
            "clarity": clarity[candidate_id],
            "engagement": engagement[candidate_id],
        }
        total = sum(review["score"] for review in scores.values())
        evaluated.append(
            {
                **candidate,
                "evaluation": {
                    **scores,
                    "total_score": total,
                    "normalized_score": round(total / 13 * 100, 1),
                },
            }
        )
    return evaluated


def _font_path() -> str | None:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    )
    return next((path for path in candidates if Path(path).exists()), None)


def _load_font(path: str | None, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.truetype(path, size) if path else ImageFont.load_default(size=size)


def _caption_regions(width: int, height: int, count: int) -> list[tuple[int, int, int, int]]:
    margin = max(8, int(min(width, height) * 0.025))
    if count == 2:
        region_height = max(int(height * 0.26), 60)
        return [
            (margin, margin, width - margin, min(height, margin + region_height)),
            (margin, max(margin, height - margin - region_height), width - margin, height - margin),
        ]
    if count == 4 and width > height * 1.25:
        half_width, half_height = width // 2, height // 2
        return [
            (margin, margin, half_width - margin, half_height - margin),
            (half_width + margin, margin, width - margin, half_height - margin),
            (margin, half_height + margin, half_width - margin, height - margin),
            (half_width + margin, half_height + margin, width - margin, height - margin),
        ]
    cell_height = height / count
    return [
        (margin, int(index * cell_height) + margin, width - margin, int((index + 1) * cell_height) - margin)
        for index in range(count)
    ]


def _wrapped_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    words = text.upper().split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        proposed = f"{current} {word}"
        if draw.textbbox((0, 0), proposed, font=font, stroke_width=2)[2] <= max_width:
            current = proposed
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def render_meme_image(template_bytes: bytes, text_boxes: list[dict[str, Any]] | list[str]) -> bytes:
    try:
        image = Image.open(io.BytesIO(template_bytes)).convert("RGB")
    except Exception as exc:
        raise PipelineError("The selected meme-template image could not be decoded.") from exc
    if image.width < 200 or image.height < 150 or image.width * image.height > 20_000_000:
        raise PipelineError("The selected meme-template image has unsupported dimensions.")

    draw = ImageDraw.Draw(image)
    if text_boxes and all(isinstance(item, str) for item in text_boxes):
        fallback_regions = _caption_regions(image.width, image.height, len(text_boxes))
        boxes: list[dict[str, Any]] = []
        for caption, (left, top, right, bottom) in zip(text_boxes, fallback_regions):
            boxes.append(
                {
                    "text": caption,
                    "x": round(left / image.width * 1000),
                    "y": round(top / image.height * 1000),
                    "width": round((right - left) / image.width * 1000),
                    "height": round((bottom - top) / image.height * 1000),
                    "font_size": 34,
                    "align": "center",
                }
            )
    else:
        boxes = _normalize_text_boxes(text_boxes, expected=len(text_boxes))
    if not boxes:
        raise PipelineError("The final caption has no valid text coordinates.")

    font_path = _font_path()
    shorter_side = min(image.width, image.height)
    for box in boxes:
        left = round(box["x"] / 1000 * image.width)
        top = round(box["y"] / 1000 * image.height)
        right = round((box["x"] + box["width"]) / 1000 * image.width)
        bottom = round((box["y"] + box["height"]) / 1000 * image.height)
        inset = max(3, round(shorter_side * 0.006))
        left, top = left + inset, top + inset
        right, bottom = right - inset, bottom - inset
        max_width, max_height = max(right - left, 40), max(bottom - top, 30)
        start_size = max(14, min(64, round(shorter_side * box["font_size"] / 1000)))
        lines: list[str] = []
        font = _load_font(font_path, start_size)
        for size in range(start_size, 11, -1):
            font = _load_font(font_path, size)
            lines = _wrapped_lines(draw, box["text"], font, max_width)
            line_height = draw.textbbox((0, 0), "Ag", font=font, stroke_width=2)[3] + max(2, size // 8)
            widest = max(
                (draw.textbbox((0, 0), line, font=font, stroke_width=2)[2] for line in lines),
                default=0,
            )
            if len(lines) * line_height <= max_height and widest <= max_width:
                break
        line_height = draw.textbbox((0, 0), "Ag", font=font, stroke_width=2)[3] + max(2, font.size // 8)
        y = top + max(0, (max_height - len(lines) * line_height) // 2)
        stroke = max(2, font.size // 12)
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font, stroke_width=stroke)
            line_width = bbox[2] - bbox[0]
            if box["align"] == "left":
                x = left
            elif box["align"] == "right":
                x = right - line_width
            else:
                x = left + max(0, (max_width - line_width) // 2)
            draw.text((x, y), line, font=font, fill="white", stroke_width=stroke, stroke_fill="black")
            y += line_height

    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


async def download_template_image(image_url: str) -> bytes:
    parsed = urlparse(image_url)
    if parsed.scheme != "https" or parsed.hostname != "i.imgflip.com":
        raise PipelineError("The selected template has an invalid image source.")
    try:
        async with httpx.AsyncClient(timeout=25.0, follow_redirects=False) as client:
            response = await client.get(image_url)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise PipelineError("The selected meme-template image could not be downloaded.") from exc
    content_type = response.headers.get("content-type", "")
    if not content_type.startswith("image/") or len(response.content) > MAX_TEMPLATE_IMAGE_BYTES:
        raise PipelineError("The selected meme-template image response was invalid.")
    return response.content


def _parse_json_object(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        # Some reasoning models can place analysis before the final answer. That
        # analysis may itself contain braces, so slicing from the first opening
        # brace to the last closing brace can turn a valid final object into
        # invalid JSON. Decode at every object boundary and use the first complete
        # JSON object instead.
        decoder = json.JSONDecoder()
        value = None
        for match in re.finditer(r"\{", cleaned):
            try:
                candidate, _ = decoder.raw_decode(cleaned, match.start())
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                value = candidate
                break
        if value is None:
            if "{" not in cleaned:
                raise _UnreadableAgentResponse(
                    "An agent returned an unreadable response. Try another model."
                )
            raise _UnreadableAgentResponse("An agent returned invalid JSON. Try another model.")
    if not isinstance(value, dict):
        raise _UnreadableAgentResponse("An agent returned an unexpected response shape.")
    return value


def normalize_tsa_output(
    raw: str, library: dict[str, dict[str, str]], top_k: int
) -> list[dict[str, str]]:
    parsed = _parse_json_object(raw)
    selections = parsed.get("selections")
    if not isinstance(selections, list):
        raise PipelineError("TSA did not return a selections list.")

    by_display_name: dict[str, tuple[str, str, str]] = {}
    for name, details in library.items():
        entry = (name, details["description"], details.get("source_filename", ""))
        by_display_name[_canonical_template_name(name)] = entry
        by_display_name[_canonical_template_name(display_template_name(name))] = entry
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in selections:
        if not isinstance(item, dict):
            continue
        requested = str(item.get("template_name", "")).strip()
        match = by_display_name.get(_canonical_template_name(requested))
        if not match or match[0] in seen:
            continue
        source_name, description, source_filename = match
        seen.add(source_name)
        normalized.append(
            {
                "template_name": display_template_name(source_name),
                "source_filename": source_filename,
                "description": description,
                "reasoning": str(item.get("reasoning", "")).strip(),
                "contrast_mapping": str(item.get("contrast_mapping", "")).strip(),
            }
        )
        if len(normalized) == top_k:
            break
    if len(normalized) != top_k:
        raise PipelineError(
            f"TSA returned {len(normalized)} valid library selections; {top_k} were requested. "
            "Try another model or run again."
        )
    return normalized


class _EmptyAgentResponse(PipelineError):
    pass


class _UnreadableAgentResponse(PipelineError):
    pass


def _matches_json_schema(value: Any, schema: dict[str, Any]) -> bool:
    """Validate the JSON Schema subset used by this pipeline."""
    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(value, dict):
            return False
        properties = schema.get("properties", {})
        if any(key not in value for key in schema.get("required", [])):
            return False
        if schema.get("additionalProperties") is False and any(
            key not in properties for key in value
        ):
            return False
        return all(
            key not in value or _matches_json_schema(value[key], child_schema)
            for key, child_schema in properties.items()
        )
    if expected_type == "array":
        if not isinstance(value, list):
            return False
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get(
            "maxItems", float("inf")
        ):
            return False
        item_schema = schema.get("items")
        return not item_schema or all(_matches_json_schema(item, item_schema) for item in value)
    if expected_type == "string" and not isinstance(value, str):
        return False
    if expected_type == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
        return False
    if expected_type == "number" and (
        not isinstance(value, (int, float)) or isinstance(value, bool)
    ):
        return False
    if expected_type == "boolean" and not isinstance(value, bool):
        return False
    return "enum" not in schema or value in schema["enum"]


def _is_retryable_agent_error(exc: Exception) -> bool:
    if isinstance(exc, (_EmptyAgentResponse, _UnreadableAgentResponse)):
        return True
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in {400, 401, 403, 404, 422}:
        return False
    if status in {408, 409, 425, 429} or (isinstance(status, int) and status >= 500):
        return True
    if exc.__class__.__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
    }:
        return True
    return status is None


async def _request_agent_with_backoff(
    operation: Any,
    model: str,
    *,
    response_schema: dict[str, Any] | None = None,
) -> str:
    for attempt in range(MAX_AGENT_ATTEMPTS):
        try:
            output = await operation()
            if not isinstance(output, str) or not output.strip():
                raise _EmptyAgentResponse(f"Model {model} returned no text.")
            if response_schema is not None:
                parsed = _parse_json_object(output)
                if not _matches_json_schema(parsed, response_schema):
                    raise _UnreadableAgentResponse(
                        f"Model {model} returned JSON that did not match the requested structure."
                    )
            return output.strip()
        except Exception as exc:
            is_last_attempt = attempt == MAX_AGENT_ATTEMPTS - 1
            if is_last_attempt or not _is_retryable_agent_error(exc):
                raise
            await asyncio.sleep(AGENT_RETRY_BASE_SECONDS * (2**attempt))
    raise AssertionError("Agent retry loop exited unexpectedly.")


async def call_agent(
    client: AsyncOpenAI,
    *,
    model: str,
    instructions: str,
    user_input: Any,
    max_output_tokens: int,
    response_schema: dict[str, Any] | None = None,
    schema_name: str = "agent_output",
) -> str:
    if _uses_bedrock_endpoint(client):
        chat_instructions = instructions
        if model in BEDROCK_CONVERSE_MODELS:
            if response_schema is not None:
                chat_instructions += (
                    "\n\nReturn only valid JSON matching this JSON Schema; do not use markdown fences:\n"
                    + json.dumps(response_schema, ensure_ascii=False)
                )

            async def create_converse_completion() -> str:
                return await _bedrock_converse_completion(
                    client,
                    model=model,
                    instructions=chat_instructions,
                    user_input=user_input,
                    max_output_tokens=max_output_tokens,
                )

            return await _request_agent_with_backoff(
                create_converse_completion,
                model,
                response_schema=response_schema,
            )

        messages: list[dict[str, Any]] = [{"role": "system", "content": chat_instructions}]
        if isinstance(user_input, str):
            messages.append({"role": "user", "content": user_input})
        else:
            for message in user_input:
                converted_content: list[dict[str, Any]] = []
                for item in message.get("content", []):
                    if item.get("type") == "input_text":
                        converted_content.append({"type": "text", "text": item["text"]})
                    elif item.get("type") == "input_image":
                        converted_content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": item["image_url"],
                                    "detail": item.get("detail", "auto"),
                                },
                            }
                        )
                messages.append({"role": message.get("role", "user"), "content": converted_content})
        async def create_chat_completion() -> str:
            request: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": max_output_tokens,
            }
            if model in REASONING_BEDROCK_MODELS:
                request["reasoning_effort"] = "low"
            if response_schema is not None:
                request["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "description": "Structured output for the current SciMemeX pipeline stage.",
                        "schema": response_schema,
                        "strict": True,
                    },
                }
            completion = await client.chat.completions.create(
                **request,
            )
            if not completion.choices:
                return ""
            return completion.choices[0].message.content or ""

        return await _request_agent_with_backoff(
            create_chat_completion,
            model,
            response_schema=response_schema,
        )

    request: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": user_input,
        "max_output_tokens": max_output_tokens,
        "store": False,
    }
    if model.casefold().startswith("gpt-5"):
        request["reasoning"] = {"effort": "low"}
    if response_schema is not None:
        request["text"] = {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "description": "Structured output for the current SciMemeX pipeline stage.",
                "schema": response_schema,
                "strict": True,
            }
        }
    async def create_response() -> str:
        response = await client.responses.create(**request)
        return response.output_text or ""

    return await _request_agent_with_backoff(
        create_response,
        model,
        response_schema=response_schema,
    )


def _event(event: str, **payload: Any) -> str:
    return json.dumps({"event": event, **payload}, ensure_ascii=False) + "\n"


def _friendly_api_error(exc: Exception, *, open_source: bool = False) -> str:
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    service_name = "Amazon Bedrock" if open_source else "OpenAI"
    if status == 400:
        return f"{service_name} rejected the request. The selected model may not support the requested input."
    if status == 401:
        return f"{service_name} rejected the API credentials. Verify the server configuration, then try again."
    if status == 403:
        return f"{service_name} does not allow access to the selected model. Choose another model."
    if status == 404:
        return "The selected model was not found. Reload the model list and try again."
    if status == 429:
        return f"{service_name} rate or quota limits were reached. Wait briefly, then retry."
    if status and status >= 500:
        return f"{service_name} is temporarily unavailable. Try the run again in a moment."
    if exc.__class__.__name__ in {"APIConnectionError", "APITimeoutError"}:
        return f"{service_name} could not be reached. Check the network connection and try again."
    return "The model request failed. Verify the selected models and try again."


def _selection_schema(template_names: list[str], top_k: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "selections": {
                "type": "array",
                "minItems": top_k,
                "maxItems": top_k,
                "items": {
                    "type": "object",
                    "properties": {
                        "template_name": {"type": "string", "enum": template_names},
                        "reasoning": {"type": "string"},
                        "contrast_mapping": {"type": "string"},
                    },
                    "required": ["template_name", "reasoning", "contrast_mapping"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["selections"],
        "additionalProperties": False,
    }


def _candidate_schema(templates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "minItems": len(templates),
                "maxItems": len(templates),
                "items": {
                    "type": "object",
                    "properties": {
                        "template_name": {
                            "type": "string",
                            "enum": [item["template_name"] for item in templates],
                        },
                        "text_boxes": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 6,
                            "items": _text_box_schema(),
                        },
                        "generation_note": {"type": "string"},
                    },
                    "required": ["template_name", "text_boxes", "generation_note"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["candidates"],
        "additionalProperties": False,
    }


def _feedback_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate["candidate_id"],
        "template_name": candidate["template"]["template_name"],
        "captions": candidate["captions"],
        "text_boxes": candidate["text_boxes"],
        "evaluation": candidate["evaluation"],
    }


async def select_templates(
    client: AsyncOpenAI,
    *,
    model: str,
    research_idea: str,
    library: dict[str, dict[str, str]],
    top_k: int,
    iteration: int,
    best: dict[str, Any] | None = None,
    worst: dict[str, Any] | None = None,
) -> dict[str, Any]:
    feedback = ""
    if best is not None and worst is not None:
        feedback = (
            "\n\nCONTRASTIVE FEEDBACK FROM THE PREVIOUS ROUND:\n"
            + json.dumps(
                {"best": _feedback_candidate(best), "worst": _feedback_candidate(worst)},
                ensure_ascii=False,
                indent=2,
            )
        )
    raw = await call_agent(
        client,
        model=model,
        instructions=TSA_SYSTEM_PROMPT,
        user_input=(
            f"TEMPLATE DATASET ({len(library)} templates):\n{_library_for_prompt(library)}\n\n"
            f"SEARCH ROUND: {iteration}\n"
            f"Select exactly {top_k} templates.\n\n"
            f"CORE RESEARCH CONTRAST:\n{research_idea}{feedback}"
        ),
        max_output_tokens=4_000,
        response_schema=_selection_schema(list(library), top_k),
        schema_name=f"tsa_round_{iteration}",
    )
    selections = normalize_tsa_output(raw, library, top_k)
    return {
        "iteration": iteration,
        "model": model,
        "requested_top_k": top_k,
        "library_size": len(library),
        "selections": selections,
    }


async def generate_candidate_batch(
    client: AsyncOpenAI,
    *,
    model: str,
    research_idea: str,
    templates: list[dict[str, Any]],
    iteration: int,
    best: dict[str, Any] | None = None,
    worst: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    template_data = [
        {
            "template_name": item["template_name"],
            "box_count": item["box_count"],
            "image_width": item["width"],
            "image_height": item["height"],
            "description": item["description"],
            "contrast_mapping": item["contrast_mapping"],
        }
        for item in templates
    ]
    feedback = ""
    instructions = CAPTION_SYSTEM_PROMPT
    if best is not None and worst is not None:
        instructions = CONTRASTIVE_SYSTEM_PROMPT
        feedback = (
            "\n\nBEST CANDIDATE AND POSITIVE FEEDBACK:\n"
            + json.dumps(_feedback_candidate(best), ensure_ascii=False, indent=2)
            + "\n\nWORST CANDIDATE AND NEGATIVE FEEDBACK:\n"
            + json.dumps(_feedback_candidate(worst), ensure_ascii=False, indent=2)
        )
    bedrock_endpoint = _uses_bedrock_endpoint(client)
    text_only_model = bedrock_endpoint and model in TEXT_ONLY_BEDROCK_MODELS
    image_urls: list[str] = []
    if bedrock_endpoint and not text_only_model:
        image_bytes = await asyncio.gather(
            *(download_template_image(template["image_url"]) for template in templates)
        )
        image_urls = [_image_data_url(contents) for contents in image_bytes]
    elif not bedrock_endpoint:
        image_urls = [template["image_url"] for template in templates]

    image_guidance = (
        "This model cannot inspect images. Infer practical caption placement from each template's "
        "name, description, dimensions, and box count; keep every normalized box inside the image."
        if text_only_model
        else "Generate captions and normalized coordinates together by inspecting the corresponding images."
    )
    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                f"CORE RESEARCH CONTRAST:\n{research_idea}"
                f"{feedback}\n\n"
                "Create one candidate for every listed template. The text_boxes array must contain "
                f"exactly that template's box_count objects. {image_guidance}\n\n"
                "TEMPLATES:\n"
                + json.dumps(template_data, ensure_ascii=False, indent=2)
            ),
        }
    ]
    for index, template in enumerate(templates, start=1):
        content.append(
            {
                "type": "input_text",
                "text": f"{'TEMPLATE' if text_only_model else 'IMAGE'} {index}: {template['template_name']}",
            }
        )
        if image_urls:
            content.append({"type": "input_image", "image_url": image_urls[index - 1], "detail": "high"})
    raw = await call_agent(
        client,
        model=model,
        instructions=instructions,
        user_input=[{"role": "user", "content": content}],
        max_output_tokens=4_000,
        response_schema=_candidate_schema(templates),
        schema_name=f"meme_candidates_round_{iteration}",
    )
    return normalize_caption_candidates(raw, templates, prefix=f"round_{iteration}")


def _ranked_round(
    iteration: int,
    phase: str,
    selection: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    ranked = sorted(candidates, key=lambda item: item["evaluation"]["total_score"], reverse=True)
    return {
        "iteration": iteration,
        "phase": phase,
        "selection": selection,
        "candidates": ranked,
        "best_candidate_id": ranked[0]["candidate_id"],
        "best_score": ranked[0]["evaluation"]["total_score"],
        "worst_candidate_id": ranked[-1]["candidate_id"],
        "worst_score": ranked[-1]["evaluation"]["total_score"],
    }


async def run_pipeline_events(
    *,
    pdf_bytes: bytes,
    filename: str,
    api_key: str,
    base_url: str | None = None,
    innovation_model: str,
    concisio_model: str,
    tsa_model: str,
    generation_model: str,
    critic_model: str,
    top_k: int,
    max_iterations: int = MAX_SEARCH_ITERATIONS,
    max_pdf_bytes: int = MAX_PDF_BYTES,
) -> AsyncIterator[str]:
    client: AsyncOpenAI | None = None
    try:
        if not 1 <= max_iterations <= MAX_SEARCH_ITERATIONS:
            raise PipelineError(f"Search iterations must be between 1 and {MAX_SEARCH_ITERATIONS}.")

        yield _event("stage_started", stage="extraction", progress=1, message="Reading the paper")
        paper = extract_pdf(pdf_bytes, filename, max_pdf_bytes=max_pdf_bytes)
        extraction_output = asdict(paper)
        yield _event(
            "stage_completed",
            stage="extraction",
            progress=8,
            message=f"Extracted {paper.page_count} pages",
            data=extraction_output,
        )

        client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=180.0, max_retries=0)

        yield _event(
            "stage_started",
            stage="innovation",
            progress=9,
            message=f"Innovative Reflections · {innovation_model}",
        )
        innovation = await call_agent(
            client,
            model=innovation_model,
            instructions=INNOVATION_SYSTEM_PROMPT,
            user_input=(
                "Analyze these research-paper excerpts. Preserve specific method names, objectives, "
                "results, tradeoffs, and limitations when supplied.\n\n" + paper.agent_input
            ),
            max_output_tokens=3_000,
        )
        innovation_output = {"model": innovation_model, "content": innovation}
        yield _event(
            "stage_completed",
            stage="innovation",
            progress=16,
            message="Contrastive analysis complete",
            data=innovation_output,
        )

        yield _event(
            "stage_started",
            stage="concisio",
            progress=17,
            message=f"Concisio · {concisio_model}",
        )
        concisio = await call_agent(
            client,
            model=concisio_model,
            instructions=CONCISIO_SYSTEM_PROMPT,
            user_input="Create the compact contrastive synthesis from this analysis:\n\n" + innovation,
            max_output_tokens=900,
        )
        concisio_output = {"model": concisio_model, "content": concisio}
        yield _event(
            "stage_completed",
            stage="concisio",
            progress=24,
            message="Concise synthesis complete",
            data=concisio_output,
        )

        yield _event(
            "stage_started",
            stage="template_catalog",
            progress=25,
            message="Loading the meme template dataset",
        )
        complete_library = load_template_library()
        catalog = await fetch_imgflip_catalog()
        library = {
            name: details
            for name, details in complete_library.items()
            if _canonical_template_name(name) in catalog
        }
        if len(library) < top_k:
            raise PipelineError("The template dataset and public Imgflip catalog have too few matches.")
        catalog_output = {
            "dataset_size": len(complete_library),
            "renderable_templates": len(library),
            "source": "Bundled SciMemeX dataset matched to Imgflip public catalog",
        }
        yield _event(
            "stage_completed",
            stage="template_catalog",
            progress=27,
            message=f"Loaded {len(library)} renderable templates",
            data=catalog_output,
        )

        history: list[dict[str, Any]] = []
        all_time_best: dict[str, Any] | None = None
        all_time_best_score = -1
        previous_best: dict[str, Any] | None = None
        previous_worst: dict[str, Any] | None = None
        total_search_units = (max_iterations + 1) * 3
        completed_units = 0

        for iteration in range(max_iterations + 1):
            phase = "initial_exploration" if iteration == 0 else "contrastive_exploration"
            round_label = "Initial exploration" if iteration == 0 else f"Contrastive iteration {iteration}"

            start_progress = 27 + round(completed_units / total_search_units * 60)
            yield _event(
                "stage_started",
                stage=f"selection_{iteration}",
                progress=start_progress,
                message=f"{round_label}: selecting templates",
            )
            selection = await select_templates(
                client,
                model=tsa_model,
                research_idea=concisio,
                library=library,
                top_k=top_k,
                iteration=iteration,
                best=previous_best,
                worst=previous_worst,
            )
            renderable, _ = resolve_renderable_templates(selection["selections"], catalog)
            completed_units += 1
            progress_value = 27 + round(completed_units / total_search_units * 60)
            yield _event(
                "stage_completed",
                stage=f"selection_{iteration}",
                progress=progress_value,
                message=f"{round_label}: selected {len(renderable)} templates",
                data=selection,
            )

            yield _event(
                "stage_started",
                stage=f"generation_{iteration}",
                progress=progress_value,
                message=f"{round_label}: generating candidates",
            )
            candidates = await generate_candidate_batch(
                client,
                model=generation_model,
                research_idea=concisio,
                templates=renderable,
                iteration=iteration,
                best=previous_best,
                worst=previous_worst,
            )
            completed_units += 1
            progress_value = 27 + round(completed_units / total_search_units * 60)
            yield _event(
                "stage_completed",
                stage=f"generation_{iteration}",
                progress=progress_value,
                message=f"{round_label}: generated {len(candidates)} candidates",
                data={"iteration": iteration, "model": generation_model, "candidates": candidates},
            )

            yield _event(
                "stage_started",
                stage=f"evaluation_{iteration}",
                progress=progress_value,
                message=f"{round_label}: scoring fidelity, clarity, and engagement",
            )
            evaluated = await critique_candidates(
                client,
                model=critic_model,
                research_idea=concisio,
                candidates=candidates,
            )
            round_output = _ranked_round(iteration, phase, selection, evaluated)
            ranked = round_output["candidates"]
            round_best, previous_worst = ranked[0], ranked[-1]
            if round_best["evaluation"]["total_score"] > all_time_best_score:
                all_time_best = round_best
                all_time_best_score = round_best["evaluation"]["total_score"]
            # Exploit the strongest candidate found across every completed round,
            # while using the current round's weakest candidate as negative evidence.
            previous_best = all_time_best
            round_output["all_time_best_candidate_id"] = all_time_best["candidate_id"]
            round_output["all_time_best_score"] = all_time_best_score
            history.append(round_output)

            completed_units += 1
            progress_value = 27 + round(completed_units / total_search_units * 60)
            yield _event(
                "stage_completed",
                stage=f"evaluation_{iteration}",
                progress=progress_value,
                message=f"{round_label}: candidates ranked",
                data=round_output,
            )

        if all_time_best is None:
            raise PipelineError("The search produced no valid meme candidates.")

        yield _event(
            "stage_started",
            stage="final_evaluation",
            progress=89,
            message="Running the final detailed evaluation",
        )
        final_evaluated = await critique_candidates(
            client,
            model=critic_model,
            research_idea=concisio,
            candidates=[{key: value for key, value in all_time_best.items() if key != "evaluation"}],
        )
        final_candidate = final_evaluated[0]
        final_evaluation_output = {
            "model": critic_model,
            "candidate_id": final_candidate["candidate_id"],
            "evaluation": final_candidate["evaluation"],
        }
        yield _event(
            "stage_completed",
            stage="final_evaluation",
            progress=95,
            message="Final evaluation complete",
            data=final_evaluation_output,
        )

        yield _event(
            "stage_started",
            stage="rendering",
            progress=96,
            message="Rendering the best meme",
        )
        template_bytes = await download_template_image(final_candidate["template"]["image_url"])
        meme_bytes = render_meme_image(template_bytes, final_candidate["text_boxes"])
        image_data_url = "data:image/png;base64," + base64.b64encode(meme_bytes).decode("ascii")
        final_meme_output = {
            "candidate_id": final_candidate["candidate_id"],
            "template_id": final_candidate["template"]["template_id"],
            "template_name": final_candidate["template"]["template_name"],
            "captions": final_candidate["captions"],
            "text_boxes": final_candidate["text_boxes"],
            "coordinate_system": final_candidate["coordinate_system"],
            "evaluation": final_candidate["evaluation"],
            "renderer": "local_pillow",
            "template_source": "Imgflip public catalog",
            "mime_type": "image/png",
            "image_data_url": image_data_url,
        }
        yield _event(
            "stage_completed",
            stage="rendering",
            progress=99,
            message="Final meme rendered",
            data={key: value for key, value in final_meme_output.items() if key != "image_data_url"},
        )

        export = {
            "schema_version": "3.0",
            "pipeline_scope": "scimemex_exploration_exploitation_through_final_meme",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "paper": {
                "filename": paper.filename,
                "page_count": paper.page_count,
                "character_count": paper.character_count,
            },
            "configuration": {
                "models": {
                    "innovative_reflections": innovation_model,
                    "concisio": concisio_model,
                    "template_selector": tsa_model,
                    "meme_generator": generation_model,
                    "evaluation_agents": critic_model,
                },
                "top_k": top_k,
                "contrastive_iterations": max_iterations,
                "api_key_included": False,
            },
            "steps": {
                "extraction": extraction_output,
                "innovative_reflections": innovation_output,
                "concisio": concisio_output,
                "template_catalog": catalog_output,
                "initial_exploration": history[0],
                "contrastive_iterations": history[1:],
                "final_evaluation": final_evaluation_output,
                "final_meme": final_meme_output,
            },
        }
        yield _event(
            "pipeline_completed",
            progress=100,
            message="Exploration–exploitation search complete",
            data=export,
        )
    except PipelineError as exc:
        yield _event("pipeline_error", message=str(exc))
    except Exception as exc:
        yield _event("pipeline_error", message=_friendly_api_error(exc, open_source=base_url is not None))
    finally:
        if client is not None:
            await client.close()
