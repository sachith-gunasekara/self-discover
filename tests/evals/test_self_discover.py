import pytest

from evals.helpers.config import config

def test_config_py():
    assert "MODEL" in config
    assert "MODE" in config
    assert "PATHS" in config
    assert "EVAL" in config
    assert "CURRENTS" in config