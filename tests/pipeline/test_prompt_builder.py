# tests/pipeline/test_prompt_builder.py

import pytest
from pathlib import Path

from twoprompt.pipeline.prompt_builder import (
    build_direct_mcq_prompt,
    build_free_text_prompt,
    build_option_matching_prompt,
    load_prompt_templates,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = _REPO_ROOT / "prompts"
_TEMPLATES = load_prompt_templates("v1", _PROMPTS_DIR)


class TestBuildDirectMcqPrompt:
    """Tests for build_direct_mcq_prompt."""

    def test_includes_question_options_and_letter_instruction(self):
        question = "Which number has one factor?"
        option_a = "one"
        option_b = "two"
        option_c = "three"
        option_d = "four"

        prompt = build_direct_mcq_prompt(
            _TEMPLATES["direct_mcq"], question, option_a, option_b, option_c, option_d
        )

        assert question in prompt
        assert "Respond with only the letter." in prompt
        assert prompt.index("A. one") < prompt.index("B. two")
        assert prompt.index("B. two") < prompt.index("C. three")
        assert prompt.index("C. three") < prompt.index("D. four")


class TestBuildFreeTextPrompt:
    """Tests for build_free_text_prompt."""

    def test_includes_question_and_excludes_options(self):
        question = "Which number has one factor?"

        actual = build_free_text_prompt(_TEMPLATES["free_text"], question)

        assert question in actual
        assert "Options:" not in actual
        assert "A." not in actual
        assert "B." not in actual
        assert "C." not in actual
        assert "D." not in actual


class TestBuildOptionMatchingPrompt:
    """Tests for build_option_matching_prompt."""

    def test_includes_question_free_text_options_and_letter_instruction(self):
        question = "Which number has one factor?"
        option_a = "one"
        option_b = "two"
        option_c = "three"
        option_d = "four"
        free_response = "one"

        prompt = build_option_matching_prompt(
            _TEMPLATES["option_matching"],
            question,
            free_response,
            option_a,
            option_b,
            option_c,
            option_d,
        )

        assert "Select the option that best matches the reference answer in the context of the question.".lower() in prompt.lower()
        assert question in prompt
        assert "Respond with only the letter." in prompt
        assert prompt.index("A. one") < prompt.index("B. two")
        assert prompt.index("B. two") < prompt.index("C. three")
        assert prompt.index("C. three") < prompt.index("D. four")
        assert free_response in prompt


class TestLoadPromptTemplates:
    """Tests for load_prompt_templates version isolation."""

    def test_v1_loads_original_stage1(self):
        t = load_prompt_templates("v1", _PROMPTS_DIR)
        assert "Respond with a short direct answer only." in t["free_text"]

    def test_v1_loads_original_stage2(self):
        t = load_prompt_templates("v1", _PROMPTS_DIR)
        assert "Select the option that best matches the reference answer" in t["option_matching"]
        assert "Respond with only the letter." in t["option_matching"]

    def test_v2_changes_only_stage1(self):
        v1 = load_prompt_templates("v1", _PROMPTS_DIR)
        v2 = load_prompt_templates("v2", _PROMPTS_DIR)
        # Stage 1 must differ
        assert v2["free_text"] != v1["free_text"]
        assert "concise answer phrase" in v2["free_text"]
        # Stage 2 must be identical to v1
        assert v2["option_matching"] == v1["option_matching"]
        # direct_mcq must be identical to v1
        assert v2["direct_mcq"] == v1["direct_mcq"]

    def test_v3_changes_only_stage2(self):
        v1 = load_prompt_templates("v1", _PROMPTS_DIR)
        v3 = load_prompt_templates("v3", _PROMPTS_DIR)
        # Stage 2 must differ
        assert v3["option_matching"] != v1["option_matching"]
        assert "Prioritize meaning over exact wording." in v3["option_matching"]
        assert "Respond with only A, B, C, or D." in v3["option_matching"]
        # Stage 1 must be identical to v1
        assert v3["free_text"] == v1["free_text"]
        # direct_mcq must be identical to v1
        assert v3["direct_mcq"] == v1["direct_mcq"]

    def test_unknown_version_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="v99"):
            load_prompt_templates("v99", _PROMPTS_DIR)
