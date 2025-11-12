"""Pytest configuration for client tests.

Handles napari/Qt event loop cleanup quirks.

Known Issues:
- Napari tests may have Qt event loop cleanup errors during teardown. These are
  artifacts of napari's canvas deletion and are not actual test failures. The test
  assertions pass, but pytest-qt catches exceptions in the event loop. This is a
  known napari issue (https://github.com/napari/napari/issues/).
"""

import pytest


def pytest_configure(config):
    """Configure pytest markers for napari tests."""
    config.addinivalue_line(
        "markers", "napari: mark test as using napari viewer (may have Qt cleanup warnings)"
    )
