"""Card -> Show -> output window, driven by the real pipeline (offscreen Qt, fake STT)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from tests.test_pipeline import FakeSTT, bible  # noqa: E402, F401  (fixture)
from vachan.core.settings import Settings  # noqa: E402
from vachan.output.verse_window import VerseDisplayWindow  # noqa: E402
from vachan.parser import ReferenceParser  # noqa: E402
from vachan.ui.main_window import ControlWindow  # noqa: E402


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def win(app, bible):  # noqa: F811
    stt = FakeSTT([])
    stt.fallback_reason = ""
    w = ControlWindow(Settings(), stt, ReferenceParser(), bible, VerseDisplayWindow("Midnight", "both"))
    w.settings.save = lambda *a: None  # never touch the user's settings.json
    w.pipeline.events.suggestion = w._on_suggestion  # direct call instead of a queued signal
    w.pipeline._clock = w.clock = Clock()
    yield w
    w.output.close()
    w.close()


def tick(w, seconds):
    """Silence: time passes, the pipeline may emit a guess."""
    w.clock.t += seconds
    s = w.pipeline._check_pending()
    if s:
        w._on_suggestion(s)


def handle(w, text):
    w.clock.t += 1.0
    _, s = w.pipeline._handle_text(text)
    if s:
        w._on_suggestion(s)


def card(w):
    return w.sug_ref.text().split("  ·  ")[0]


def on_air(w):
    return w.output.current.reference.format_reference()


def test_psalms_91_verse_2_card_and_output_agree(win):
    handle(win, "Psalms 91")
    handle(win, "verse 2.")
    assert card(win) == "Psalms 91:2"
    pending = win.pending
    win.approve()
    assert win.output.current is pending  # Show uses the card's own object, no re-parse
    assert on_air(win) == "Psalms 91:2"


def test_chapter_only_card_is_marked_as_guess_never_silent(win):
    handle(win, "Turn to Psalms 91")
    assert win.pending is None  # still waiting for the verse
    tick(win, 3.5)
    assert card(win) == "Psalms 91:1  (verse? guessed)"
    assert any("guess" in w for w in win.sug_warn.text().splitlines())
    assert win.conf_badge.text().startswith("Review")


def test_upgrade_replaces_pending_guess_card(win):
    handle(win, "Turn to Psalms 91")
    tick(win, 3.5)
    handle(win, "verse 2")
    assert card(win) == "Psalms 91:2"
    assert win.pending.action == "upgrade"


def test_upgrade_replaces_guess_already_on_air(win):
    handle(win, "Turn to Psalms 91")
    tick(win, 3.5)
    win.approve()
    assert on_air(win) == "Psalms 91:1"
    handle(win, "and verse 2 says")
    assert on_air(win) == "Psalms 91:2"
    assert "Upgraded on air" in win.status.text()


def test_upgrade_does_not_replace_a_spoken_verse_on_air(win):
    handle(win, "Turn to John 3:16")
    win.approve()
    handle(win, "Turn to Psalms 91")
    tick(win, 3.5)
    handle(win, "verse 2")
    assert on_air(win) == "John 3:16"
    assert card(win) == "Psalms 91:2"


def test_next_verse_is_relative_to_verse_on_screen(win):
    handle(win, "Turn to Psalms 91 verse 2")
    win.approve()
    handle(win, "Turn to Romans 8")  # chapter-only guess pending; must not hijack 'next verse'
    handle(win, "now the next verse")
    assert card(win) == "Psalms 91:3"


def test_next_verse_after_upgrade_on_air(win):
    handle(win, "Turn to Psalms 91")
    tick(win, 3.5)
    win.approve()
    handle(win, "verse 2")
    handle(win, "the next verse")
    assert card(win) == "Psalms 91:3"


def test_show_ignores_later_state_changes(win):
    handle(win, "Turn to Psalms 91 verse 2")
    shown_on_card = card(win)
    win.pipeline.set_context(None)
    win.pipeline._state = type(win.pipeline._state)()
    win.approve()
    assert on_air(win) == shown_on_card


@pytest.mark.parametrize("run", range(10))
def test_ten_repeated_runs(win, run):
    texts = [["Psalms 91 verse 2"], ["भजन संहिता 91 पद 2"], ["Bhajan Sanhita 91 pad do"],
             ["Psalms 91", "verse 2"], ["भजन संहिता 91", "पद दो"]][run % 5]  # fmt: skip
    for text in texts:
        handle(win, text)
    assert card(win) == "Psalms 91:2"
    win.approve()
    assert on_air(win) == "Psalms 91:2"
