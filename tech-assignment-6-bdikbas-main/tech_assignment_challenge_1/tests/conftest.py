"""Pytest configuration for Challenge 1 tests.

Extra credit tests are skipped by default.
To run them:
  uv run -m pytest tests/test_features.py -v --run-ec
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-ec",
        action="store_true",
        default=False,
        help="Run extra credit tests (TODOs C, D, E, F)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "extra_credit: mark test as extra credit (skipped unless --run-ec is passed)",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-ec"):
        skip_ec = pytest.mark.skip(reason="Extra credit — run with --run-ec to enable")
        for item in items:
            if "extra_credit" in item.keywords:
                item.add_marker(skip_ec)
