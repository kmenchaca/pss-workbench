"""Tests for config loading."""

import tempfile

from pss.config import PSSConfig, load_config


def test_default_config():
    """Test that default config has sensible values."""
    config = PSSConfig()
    assert config.provider == "openrouter"
    assert config.soft_gate_tokens == 15000
    assert config.hard_gate_tokens == 50000
    assert config.max_contexts == 20


def test_load_missing_file():
    """Test loading a nonexistent config file returns defaults."""
    config = load_config("/nonexistent/path/config.yaml")
    assert config.provider == "openrouter"


def test_load_yaml_config():
    """Test loading config from YAML."""
    yaml_content = """
provider:
  name: anthropic
  model: claude-haiku-4-5-20251001

gates:
  soft_gate_tokens: 10000
  hard_gate_tokens: 40000

budget:
  max_contexts: 10
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        config = load_config(f.name)

    assert config.provider == "anthropic"
    assert config.model == "claude-haiku-4-5-20251001"
    assert config.soft_gate_tokens == 10000
    assert config.hard_gate_tokens == 40000
    assert config.max_contexts == 10


def test_load_flat_config():
    """Test loading flat-style config."""
    yaml_content = """
provider: openrouter
model: meta-llama/llama-3.1-8b-instruct
soft_gate_tokens: 5000
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        config = load_config(f.name)

    assert config.provider == "openrouter"
    assert config.model == "meta-llama/llama-3.1-8b-instruct"
    assert config.soft_gate_tokens == 5000


def test_load_verification_config():
    """Test loading verification settings from YAML."""
    yaml_content = """
verification:
  strategy: programmatic
  test_command: npm test
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        config = load_config(f.name)

    assert config.verification_strategy == "programmatic"
    assert config.test_command == "npm test"


def test_default_verification_is_none():
    """Test that verification defaults to 'none'."""
    config = PSSConfig()
    assert config.verification_strategy == "none"
    assert config.test_command is None
