"""Initial test to verify test harness works."""

from vachan import __version__


def test_version() -> None:
    """Verify package version is set."""
    assert __version__ == "0.1.0"
