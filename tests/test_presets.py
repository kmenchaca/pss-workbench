"""Tests for the presets module."""

from pss.config import PSSConfig
from pss.presets import (
    PRESETS,
    Preset,
    apply_preset_to_config,
    get_preset,
    list_presets,
)


class TestPresetDefinitions:
    """Tests for preset definitions."""

    def test_all_presets_exist(self):
        """Test that all expected presets are defined."""
        expected = {"explore", "write", "research", "draft", "investigate", "debug", "reasoning"}
        assert set(PRESETS.keys()) == expected

    def test_presets_have_required_fields(self):
        """Test that all presets have the required fields."""
        for name, preset in PRESETS.items():
            assert preset.name == name
            assert preset.description
            assert preset.provider in ["anthropic", "openrouter"]
            assert preset.model
            assert preset.soft_gate_tokens > 0
            assert preset.hard_gate_tokens > preset.soft_gate_tokens
            assert preset.total_max > 0
            assert preset.max_contexts > 0
            assert preset.system_prompt

    def test_explore_preset_is_cheap_and_wide(self):
        """Test explore preset has appropriate settings."""
        preset = PRESETS["explore"]
        # Should use cheap model
        assert "llama" in preset.model.lower() or "8b" in preset.model.lower()
        # Should have quick gates for wide exploration
        assert preset.soft_gate_tokens <= 5000
        # Should allow many contexts
        assert preset.max_contexts >= 10

    def test_research_preset_is_deep(self):
        """Test research preset has appropriate settings."""
        preset = PRESETS["research"]
        # Should have longer gates for depth
        assert preset.soft_gate_tokens >= 10000
        # Should have higher budget
        assert preset.total_max >= 100000
        # Fewer contexts (depth over breadth)
        assert preset.max_contexts <= 10

    def test_write_preset_is_moderate(self):
        """Test write preset has moderate settings."""
        preset = PRESETS["write"]
        # Moderate gates
        assert 5000 <= preset.soft_gate_tokens <= 15000
        # Moderate contexts
        assert 5 <= preset.max_contexts <= 15

    def test_draft_preset_is_focused(self):
        """Test draft preset has focused settings."""
        preset = PRESETS["draft"]
        # Should have few contexts (focused output)
        assert preset.max_contexts <= 5

    def test_reasoning_preset_has_early_gates(self):
        """Test reasoning preset has earlier gates than exploration presets."""
        preset = PRESETS["reasoning"]
        explore_preset = PRESETS["explore"]
        # Should have earlier gates than explore (but allow enough thinking time)
        assert preset.soft_gate_tokens <= explore_preset.soft_gate_tokens
        assert preset.hard_gate_tokens <= explore_preset.hard_gate_tokens
        # Should allow multiple parallel approaches
        assert preset.max_contexts >= 5


class TestGetPreset:
    """Tests for get_preset function."""

    def test_get_existing_preset(self):
        """Test getting an existing preset."""
        preset = get_preset("explore")
        assert preset is not None
        assert preset.name == "explore"

    def test_get_nonexistent_preset(self):
        """Test getting a nonexistent preset returns None."""
        preset = get_preset("nonexistent")
        assert preset is None

    def test_get_all_presets(self):
        """Test that all presets can be retrieved."""
        for name in PRESETS.keys():
            preset = get_preset(name)
            assert preset is not None
            assert preset.name == name


class TestListPresets:
    """Tests for list_presets function."""

    def test_list_returns_all_presets(self):
        """Test that list_presets returns all presets."""
        presets = list_presets()
        assert len(presets) == len(PRESETS)

    def test_list_returns_preset_objects(self):
        """Test that list_presets returns Preset objects."""
        presets = list_presets()
        for preset in presets:
            assert isinstance(preset, Preset)


class TestApplyPresetToConfig:
    """Tests for apply_preset_to_config function."""

    def test_apply_preset_sets_all_fields(self):
        """Test that applying a preset sets all expected fields."""
        config = PSSConfig()
        preset = get_preset("explore")

        apply_preset_to_config(preset, config)

        assert config.provider == preset.provider
        assert config.model == preset.model
        assert config.soft_gate_tokens == preset.soft_gate_tokens
        assert config.hard_gate_tokens == preset.hard_gate_tokens
        assert config.total_max == preset.total_max
        assert config.max_contexts == preset.max_contexts
        assert config.system_prompt == preset.system_prompt

    def test_apply_preset_overwrites_existing(self):
        """Test that preset overwrites existing config values."""
        config = PSSConfig(
            provider="anthropic",
            model="claude-3-opus",
            soft_gate_tokens=1000,
        )
        preset = get_preset("explore")

        apply_preset_to_config(preset, config)

        # Should be overwritten
        assert config.provider == preset.provider
        assert config.model == preset.model
        assert config.soft_gate_tokens == preset.soft_gate_tokens

    def test_apply_preset_preserves_other_fields(self):
        """Test that preset doesn't affect unrelated config fields."""
        config = PSSConfig(
            parallel=True,
            diversity_enabled=False,
            verification_strategy="programmatic",
        )
        preset = get_preset("explore")

        apply_preset_to_config(preset, config)

        # Should be preserved
        assert config.parallel is True
        assert config.diversity_enabled is False
        assert config.verification_strategy == "programmatic"


class TestSystemPrompts:
    """Tests for system prompts in presets."""

    def test_explore_prompt_encourages_branching(self):
        """Test explore prompt encourages branching."""
        preset = PRESETS["explore"]
        prompt_lower = preset.system_prompt.lower()
        assert "branch" in prompt_lower
        assert "breadth" in prompt_lower or "wide" in prompt_lower

    def test_research_prompt_encourages_depth(self):
        """Test research prompt encourages depth."""
        preset = PRESETS["research"]
        prompt_lower = preset.system_prompt.lower()
        assert "deep" in prompt_lower or "thorough" in prompt_lower
        assert "rigorous" in prompt_lower or "systematic" in prompt_lower

    def test_write_prompt_mentions_writing(self):
        """Test write prompt is writing-focused."""
        preset = PRESETS["write"]
        prompt_lower = preset.system_prompt.lower()
        assert "writ" in prompt_lower  # write, writer, writing

    def test_draft_prompt_mentions_quality(self):
        """Test draft prompt focuses on quality."""
        preset = PRESETS["draft"]
        prompt_lower = preset.system_prompt.lower()
        assert "quality" in prompt_lower or "polished" in prompt_lower
