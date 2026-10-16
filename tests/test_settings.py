"""Settings persistence: the Whisper prompt default and its migration."""

import json
import re

from vachan.core.settings import DEFAULT_PROMPT, OLD_DEFAULT_PROMPTS, Settings


def test_prompt_is_vocabulary_without_numbers():
    # Whisper echoes the prompt; numbers in it once produced references nobody said
    assert not re.search(r"\d", DEFAULT_PROMPT)


def test_old_default_prompt_is_replaced(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"initial_prompt": next(iter(OLD_DEFAULT_PROMPTS))}), encoding="utf-8")
    assert Settings.load(path).initial_prompt == DEFAULT_PROMPT


def test_custom_prompt_is_kept(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"initial_prompt": "मेरी शब्दावली"}), encoding="utf-8")
    assert Settings.load(path).initial_prompt == "मेरी शब्दावली"
