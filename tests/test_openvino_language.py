"""OpenVINO transcripts are tagged with the language Whisper actually wrote.

Measured 2026-10-16: with an initial_prompt OpenVINO GenAI ignores the language token, and
on 1 s partials it reports "ur"/"ko"; the old code then fell back to "en", so the operator
window said "English" above Devanagari text.
"""

import numpy as np
import pytest

from vachan.core.settings import Settings
from vachan.stt.openvino_transcriber import OpenVinoTranscriber


class FakeResult:
    def __init__(self, text, language):
        self.text, self.language = text, language

    def __str__(self):
        return self.text


class FakePipe:
    def __init__(self, text, language):
        self.result = FakeResult(text, language)

    def generate(self, *args, **kwargs):
        return self.result


def transcriber(text, reported):
    t = OpenVinoTranscriber(Settings(), "OpenVINO/whisper-small-int8-ov", "GPU")
    t._pipe, t.device = FakePipe(text, reported), "openvino-gpu"
    return t


@pytest.mark.parametrize("reported", ["<|ko|>", "<|en|>", "", "<|ur|>", "<|hi|>"])
def test_devanagari_text_is_hindi(reported):
    t = transcriber("यूहन्ना, रचित सुसमाचार, पहला, ध्याय, दूसरा, वचन.", reported)
    assert t.transcribe(np.zeros(16000, np.float32)).language == "hi"


@pytest.mark.parametrize("reported", ["<|ko|>", "<|hi|>", ""])
def test_latin_text_is_english(reported):
    t = transcriber("Turn to John chapter 3 verse 16", reported)
    assert t.transcribe(np.zeros(16000, np.float32)).language == "en"


def test_mixed_line_follows_the_majority():
    t = transcriber("आज हम Psalms 91 verse 2 पढ़ेंगे और देखेंगे", "<|en|>")
    assert t.transcribe(np.zeros(16000, np.float32)).language == "hi"
