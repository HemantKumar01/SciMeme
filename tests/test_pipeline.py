from __future__ import annotations

import json
import unittest
from io import BytesIO
from unittest.mock import AsyncMock, patch

import pymupdf
from PIL import Image

from webapp.app import _is_text_model, _sort_models
from webapp.pipeline import (
    PipelineError,
    _extract_sections,
    extract_pdf,
    load_template_library,
    normalize_tsa_output,
    render_meme_image,
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


if __name__ == "__main__":
    unittest.main()
