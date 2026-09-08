from __future__ import annotations

import json
import os
import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pymupdf
from PIL import Image

from webapp.app import BUILT_IN_MODELS, _bedrock_api_key, _is_text_model, _sort_models
from webapp.pipeline import (
    PipelineError,
    _extract_sections,
    _library_for_prompt,
    _parse_json_object,
    call_agent,
    display_template_name,
    extract_pdf,
    generate_candidate_batch,
    load_template_library,
    normalize_tsa_output,
    render_meme_image,
    resolve_renderable_templates,
    run_pipeline_events,
)


def make_pdf() -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Abstract\nOld systems used recurrence. This paper introduces direct attention.\n"
        "1 Introduction\nSequence models have traditionally processed tokens step by step.\n"
        "2 Related Work\nPrior approaches relied on RNNs and convolutions.\n"
        "3 Methodology\nThe proposed model uses self-attention and parallel computation.\n"
        "5 Conclusion\nAttention removes recurrence and improves parallel training.\n"
        "References\nExample reference.",
        fontsize=10,
    )
    contents = document.tobytes()
    document.close()
    return contents


class ExtractionTests(unittest.TestCase):
    def test_extracts_relevant_sections(self) -> None:
        paper = extract_pdf(make_pdf(), "example.pdf")
        self.assertEqual(paper.page_count, 1)
        self.assertIn("abstract", paper.sections)
        self.assertIn("related_work", paper.sections)
        self.assertIn("methodology", paper.sections)
        self.assertIn("conclusion", paper.sections)
        self.assertIn("self-attention", paper.agent_input)

    def test_rejects_non_pdf(self) -> None:
        with self.assertRaises(PipelineError):
            extract_pdf(b"plain text", "bad.pdf")

    def test_ignores_figure_labels_when_finding_section_boundaries(self) -> None:
        sections = _extract_sections(
            "Abstract\nSummary.\n1 Introduction\nIntro.\n2 Related Work\nPrior.\n"
            "3 Methodology\nMethod begins.\nAbstract\nScores\nand feedback\nMethod continues.\n"
            "4 Evaluation\nEvaluation text.\n6 Conclusion\nDone.\nReferences\nRef."
        )
        self.assertIn("Method continues", sections["methodology"])
        self.assertNotIn("Evaluation text", sections["methodology"])


class TsaTests(unittest.TestCase):
    def test_template_prompt_fits_llama_3_context(self) -> None:
        self.assertLess(len(_library_for_prompt(load_template_library())), 30_000)

    def test_extracts_json_after_reasoning_that_contains_braces(self) -> None:
        result = _parse_json_object(
            '<think>Compare {old} with {new}; then answer.</think>\n'
            '{"selections":[{"template_name":"Drake Hotline Bling"}]}'
        )
        self.assertEqual(result["selections"][0]["template_name"], "Drake Hotline Bling")

    def test_preserves_periods_that_are_part_of_template_names(self) -> None:
        self.assertEqual(display_template_name("Buff Doge vs. Cheems"), "Buff Doge vs. Cheems")
        self.assertEqual(display_template_name("Buff_Doge_vs._Cheems.jpg"), "Buff Doge vs. Cheems")

    def test_resolves_legacy_truncated_template_name(self) -> None:
        catalog = {
            "buffdogevscheems": {
                "template_id": "247375501",
                "template_name": "Buff Doge vs. Cheems",
                "image_url": "https://i.imgflip.com/43a45p.png",
                "width": 937,
                "height": 720,
                "box_count": 2,
            }
        }
        selections = [
            {
                "template_name": "Buff Doge vs",
                "description": "Old versus new.",
                "reasoning": "Direct comparison.",
                "contrast_mapping": "Old and new map to the two dogs.",
            }
        ]
        renderable, unavailable = resolve_renderable_templates(selections, catalog)
        self.assertFalse(unavailable)
        self.assertEqual(renderable[0]["template_name"], "Buff Doge vs. Cheems")

    def test_normalizes_exact_library_names(self) -> None:
        library = load_template_library()
        names = list(library)[:3]
        selections = [
            {
                "template_name": name.rsplit(".", 1)[0].replace("_", " "),
                "reasoning": "Fits the comparison.",
                "contrast_mapping": "Old versus new.",
            }
            for name in names
        ]
        result = normalize_tsa_output(json.dumps({"selections": selections}), library, 3)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["source_filename"], library[names[0]]["source_filename"])
        self.assertIn("description", result[0])

    def test_rejects_unknown_template_names(self) -> None:
        with self.assertRaises(PipelineError):
            normalize_tsa_output(
                '{"selections":[{"template_name":"Not in library"}]}',
                load_template_library(),
                1,
            )


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_export_contains_all_steps_but_not_key(self) -> None:
        template_name = "Gru's Plan"
        tsa_json = json.dumps(
            {
                "selections": [
                    {
                        "template_name": template_name,
                        "reasoning": "Good contrast.",
                        "contrast_mapping": "Prior versus proposed.",
                    }
                ]
            }
        )
        unstructured_outputs = iter(
            [
                "Prior Work: recurrence\nProposed Approach: attention\nCore Differences: parallel processing",
                "Prior work processed tokens recurrently; this paper uses direct attention for parallel sequence modeling.",
            ]
        )

        async def fake_agent(*args, **kwargs):
            schema_name = kwargs.get("schema_name", "agent_output")
            if kwargs.get("response_schema") is None:
                return next(unstructured_outputs)
            if schema_name.startswith("tsa_round_"):
                return tsa_json
            if schema_name.startswith("meme_candidates_round_"):
                content = kwargs["user_input"][0]["content"]
                self.assertEqual(sum(item["type"] == "input_image" for item in content), 1)
                round_number = schema_name.rsplit("_", 1)[-1]
                return json.dumps(
                    {
                        "candidates": [
                            {
                                "template_name": template_name,
                                "text_boxes": [
                                    {"text": f"Old: recurrence {round_number}", "x": 40, "y": 40, "width": 920, "height": 180, "font_size": 32, "align": "center"},
                                    {"text": f"New: direct attention {round_number}", "x": 40, "y": 780, "width": 920, "height": 180, "font_size": 32, "align": "center"},
                                ],
                                "generation_note": "Direct contrast.",
                            }
                        ]
                    }
                )
            if schema_name.endswith("_reviews"):
                ids = kwargs["response_schema"]["properties"]["reviews"]["items"]["properties"]["candidate_id"]["enum"]
                score = 3 if schema_name == "clarity_reviews" else 5
                return json.dumps(
                    {
                        "reviews": [
                            {"candidate_id": candidate_id, "score": score, "reasoning": "Works.", "feedback": "Keep it concise."}
                            for candidate_id in ids
                        ]
                    }
                )
            raise AssertionError(f"Unexpected schema: {schema_name}")

        source = BytesIO()
        Image.new("RGB", (640, 480), "#777777").save(source, "PNG")
        catalog = {
            "grusplan": {
                "template_id": "131087935",
                "template_name": template_name,
                "image_url": "https://i.imgflip.com/26jxvz.jpg",
                "width": 640,
                "height": 480,
                "box_count": 2,
            }
        }
        with (
            patch("webapp.pipeline.call_agent", new=fake_agent),
            patch("webapp.pipeline.fetch_imgflip_catalog", new=AsyncMock(return_value=catalog)),
            patch("webapp.pipeline.download_template_image", new=AsyncMock(return_value=source.getvalue())),
        ):
            lines = [
                line
                async for line in run_pipeline_events(
                    pdf_bytes=make_pdf(),
                    filename="example.pdf",
                    api_key="sk-test-secret-value",
                    innovation_model="gpt-test-a",
                    concisio_model="gpt-test-b",
                    tsa_model="gpt-test-c",
                    generation_model="gpt-test-d",
                    critic_model="gpt-test-e",
                    top_k=1,
                    max_iterations=1,
                )
            ]
        events = [json.loads(line) for line in lines]
        progress_values = [event["progress"] for event in events if "progress" in event]
        self.assertEqual(progress_values, sorted(progress_values))
        completed = next(event for event in events if event["event"] == "pipeline_completed")
        export = completed["data"]
        self.assertEqual(export["pipeline_scope"], "scimemex_exploration_exploitation_through_final_meme")
        self.assertEqual(
            set(export["steps"]),
            {
                "extraction",
                "innovative_reflections",
                "concisio",
                "template_catalog",
                "initial_exploration",
                "contrastive_iterations",
                "final_evaluation",
                "final_meme",
            },
        )
        self.assertEqual(len(export["steps"]["contrastive_iterations"]), 1)
        self.assertEqual(export["configuration"]["contrastive_iterations"], 1)
        self.assertEqual(export["steps"]["final_evaluation"]["evaluation"]["total_score"], 13)
        self.assertTrue(export["steps"]["final_meme"]["image_data_url"].startswith("data:image/png;base64,"))
        self.assertEqual(export["steps"]["final_meme"]["coordinate_system"], "normalized_0_1000_top_left")
        self.assertEqual(len(export["steps"]["final_meme"]["text_boxes"]), 2)
        self.assertNotIn("sk-test-secret-value", json.dumps(export))
        self.assertFalse(export["configuration"]["api_key_included"])


class RenderingTests(unittest.TestCase):
    def test_renders_captioned_png(self) -> None:
        source = BytesIO()
        Image.new("RGB", (640, 480), "#555555").save(source, "JPEG")
        result = render_meme_image(
            source.getvalue(),
            [
                {"text": "Prior work", "x": 50, "y": 50, "width": 400, "height": 180, "font_size": 28, "align": "left"},
                {"text": "New method", "x": 550, "y": 750, "width": 400, "height": 180, "font_size": 28, "align": "right"},
            ],
        )
        rendered = Image.open(BytesIO(result))
        self.assertEqual(rendered.format, "PNG")
        self.assertEqual(rendered.size, (640, 480))


class ModelFilterTests(unittest.TestCase):
    def test_keeps_text_models_and_filters_specialized_models(self) -> None:
        self.assertTrue(_is_text_model("gpt-5-mini"))
        self.assertTrue(_is_text_model("o3-mini"))
        self.assertFalse(_is_text_model("gpt-3.5-turbo"))
        self.assertFalse(_is_text_model("gpt-image-1"))
        self.assertFalse(_is_text_model("gpt-4o-realtime-preview"))

    def test_preferred_models_sort_first(self) -> None:
        self.assertEqual(_sort_models(["gpt-z", "gpt-5.6-luna"])[0], "gpt-5.6-luna")

    def test_built_in_models_and_environment_key(self) -> None:
        self.assertEqual(
            BUILT_IN_MODELS,
            [
                "zai.glm-5",
                "qwen.qwen3-vl-235b-a22b-instruct",
                "meta.llama3-70b-instruct-v1:0",
            ],
        )
        with patch.dict(os.environ, {"OPEN_SOURCE_API_KEY": "server-secret"}, clear=False):
            self.assertEqual(_bedrock_api_key(), "server-secret")


class ChatAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_uses_converse_for_llama_3(self) -> None:
        client = SimpleNamespace(
            api_key="bedrock-secret",
            base_url="https://bedrock-mantle.us-east-1.api.aws/v1/",
        )
        schema = {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        }
        with patch(
            "webapp.pipeline._bedrock_converse_completion",
            new=AsyncMock(return_value='{"ok":true}'),
        ) as converse:
            output = await call_agent(
                client,
                model="meta.llama3-70b-instruct-v1:0",
                instructions="Return JSON.",
                user_input="Hello",
                max_output_tokens=4_000,
                response_schema=schema,
            )
        self.assertEqual(output, '{"ok":true}')
        request = converse.await_args.kwargs
        self.assertEqual(request["model"], "meta.llama3-70b-instruct-v1:0")
        self.assertIn("JSON Schema", request["instructions"])

    async def test_uses_chat_completions_for_built_in_models(self) -> None:
        create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))]
            )
        )
        client = SimpleNamespace(
            base_url="https://bedrock-mantle.us-east-1.api.aws/v1/",
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        )
        output = await call_agent(
            client,
            model="zai.glm-5",
            instructions="Answer briefly.",
            user_input="Hello",
            max_output_tokens=100,
            response_schema={"type": "object"},
        )
        self.assertEqual(output, '{"ok":true}')
        request = create.await_args.kwargs
        self.assertEqual(request["model"], "zai.glm-5")
        self.assertEqual(request["messages"][0]["role"], "system")
        self.assertEqual(request["reasoning_effort"], "low")
        self.assertEqual(request["response_format"]["type"], "json_schema")
        self.assertEqual(request["response_format"]["json_schema"]["schema"], {"type": "object"})
        self.assertTrue(request["response_format"]["json_schema"]["strict"])
        self.assertNotIn("JSON Schema", request["messages"][0]["content"])

    async def test_retries_empty_responses_with_exponential_backoff(self) -> None:
        empty = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=""))])
        success = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Recovered"))])
        create = AsyncMock(side_effect=[empty, empty, success])
        client = SimpleNamespace(
            base_url="https://bedrock-mantle.us-east-1.api.aws/v1/",
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        )
        with patch("webapp.pipeline.asyncio.sleep", new=AsyncMock()) as sleep:
            output = await call_agent(
                client,
                model="zai.glm-5",
                instructions="Answer briefly.",
                user_input="Hello",
                max_output_tokens=100,
            )
        self.assertEqual(output, "Recovered")
        self.assertEqual(create.await_count, 3)
        self.assertEqual([call.args[0] for call in sleep.await_args_list], [1.0, 2.0])

    async def test_retries_unreadable_structured_responses(self) -> None:
        unreadable = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="not json"))]
        )
        success = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))]
        )
        create = AsyncMock(side_effect=[unreadable, success])
        client = SimpleNamespace(
            base_url="https://bedrock-mantle.us-east-1.api.aws/v1/",
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        )
        with patch("webapp.pipeline.asyncio.sleep", new=AsyncMock()) as sleep:
            output = await call_agent(
                client,
                model="zai.glm-5",
                instructions="Return JSON.",
                user_input="Hello",
                max_output_tokens=100,
                response_schema={"type": "object"},
            )
        self.assertEqual(output, '{"ok":true}')
        sleep.assert_awaited_once_with(1.0)

    async def test_retries_json_that_does_not_match_the_schema(self) -> None:
        wrong_shape = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"type":"object"}'))]
        )
        success = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"selections":["A"]}'))]
        )
        create = AsyncMock(side_effect=[wrong_shape, success])
        client = SimpleNamespace(
            base_url="https://bedrock-mantle.us-east-1.api.aws/v1/",
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        )
        schema = {
            "type": "object",
            "properties": {
                "selections": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 1,
                    "items": {"type": "string", "enum": ["A"]},
                }
            },
            "required": ["selections"],
            "additionalProperties": False,
        }
        with patch("webapp.pipeline.asyncio.sleep", new=AsyncMock()) as sleep:
            output = await call_agent(
                client,
                model="zai.glm-5",
                instructions="Return JSON.",
                user_input="Hello",
                max_output_tokens=100,
                response_schema=schema,
            )
        self.assertEqual(output, '{"selections":["A"]}')
        sleep.assert_awaited_once_with(1.0)

    async def test_retries_unclassified_provider_errors(self) -> None:
        success = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Recovered"))]
        )
        create = AsyncMock(side_effect=[RuntimeError("temporary provider failure"), success])
        client = SimpleNamespace(
            base_url="https://bedrock-mantle.us-east-1.api.aws/v1/",
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        )
        with patch("webapp.pipeline.asyncio.sleep", new=AsyncMock()) as sleep:
            output = await call_agent(
                client,
                model="zai.glm-5",
                instructions="Answer briefly.",
                user_input="Hello",
                max_output_tokens=100,
            )
        self.assertEqual(output, "Recovered")
        sleep.assert_awaited_once_with(1.0)


class GenerationInputTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.template = {
            "template_id": "247375501",
            "template_name": "Buff Doge vs. Cheems",
            "image_url": "https://i.imgflip.com/43a45p.png",
            "width": 937,
            "height": 720,
            "box_count": 1,
            "description": "Two dogs used for comparison.",
            "contrast_mapping": "Old versus new.",
            "tsa_reasoning": "Direct contrast.",
        }
        self.candidate_json = json.dumps(
            {
                "candidates": [
                    {
                        "template_name": "Buff Doge vs. Cheems",
                        "text_boxes": [
                            {
                                "text": "Old versus new",
                                "x": 40,
                                "y": 40,
                                "width": 920,
                                "height": 180,
                                "font_size": 28,
                                "align": "center",
                            }
                        ],
                        "generation_note": "Direct contrast.",
                    }
                ]
            }
        )

    async def test_glm_uses_text_only_template_metadata(self) -> None:
        client = SimpleNamespace(base_url="https://bedrock-mantle.us-east-1.api.aws/v1/")
        with patch("webapp.pipeline.call_agent", new=AsyncMock(return_value=self.candidate_json)) as agent:
            await generate_candidate_batch(
                client,
                model="zai.glm-5",
                research_idea="Old versus new.",
                templates=[self.template],
                iteration=0,
            )
        content = agent.await_args.kwargs["user_input"][0]["content"]
        self.assertFalse(any(item["type"] == "input_image" for item in content))

    async def test_other_built_in_models_receive_inline_images(self) -> None:
        source = BytesIO()
        Image.new("RGB", (32, 32), "#777777").save(source, "PNG")
        client = SimpleNamespace(base_url="https://bedrock-mantle.us-east-1.api.aws/v1/")
        with (
            patch("webapp.pipeline.download_template_image", new=AsyncMock(return_value=source.getvalue())),
            patch("webapp.pipeline.call_agent", new=AsyncMock(return_value=self.candidate_json)) as agent,
        ):
            await generate_candidate_batch(
                client,
                model="qwen.qwen3-vl-235b-a22b-instruct",
                research_idea="Old versus new.",
                templates=[self.template],
                iteration=0,
            )
        content = agent.await_args.kwargs["user_input"][0]["content"]
        image_item = next(item for item in content if item["type"] == "input_image")
        self.assertTrue(image_item["image_url"].startswith("data:image/png;base64,"))


if __name__ == "__main__":
    unittest.main()
