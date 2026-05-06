import pytest

from claude_code_toolbox.safety.permissions import (
    PathDeniedError,
    default_policy,
)


def test_denylist_blocks_secrets():
    p = default_policy()
    assert p.is_denied(".env")
    assert p.is_denied("secrets/api_key.txt")
    assert p.is_denied("data/raw/lightcurve.fits")
    assert p.is_denied("configs/production.yaml")


def test_write_allowlist():
    p = default_policy()
    assert p.can_write("astro_toolbox/new_tool.py")
    assert p.can_write("tests/test_x.py")
    assert p.can_write("papers/drafts/v1.tex")
    assert p.can_write("runs/abc/claude_code/r.json")
    assert not p.can_write("configs/production.yaml")
    assert not p.can_write("data/raw/foo.fits")
    assert not p.can_write("scripts/random.py")  # outside allowlist


def test_can_read_anywhere_except_deny():
    p = default_policy()
    assert p.can_read("README.md")
    assert p.can_read("astro_toolbox/desi.py")
    assert not p.can_read(".env")
    assert not p.can_read("secrets/x")


def test_check_write_raises():
    p = default_policy()
    with pytest.raises(PathDeniedError):
        p.check_write("configs/production.yaml")
    p.check_write("astro_toolbox/x.py")  # no raise
