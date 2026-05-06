import pytest

from claude_code_toolbox.safety.output_guard import (
    PaperRefinementGuardError,
    find_new_claims,
    guard_paper_refinement,
)


ORIGINAL = (
    "We measured the orbital period of 0.123 days. "
    "The companion mass is 0.45 Msun. "
    "No significant rv excess was found."
)


def test_pure_rewording_passes():
    refined = (
        "The orbital period was determined to be 0.123 days. "
        "We find a companion mass of 0.45 Msun. "
        "No significant rv excess was detected."
    )
    guard_paper_refinement(ORIGINAL, refined)  # must not raise


def test_new_sigma_claim_blocked():
    refined = ORIGINAL + " We detect a 5 sigma flare."
    with pytest.raises(PaperRefinementGuardError):
        guard_paper_refinement(ORIGINAL, refined)


def test_new_redshift_claim_blocked():
    refined = ORIGINAL + " The system is at redshift z=0.42."
    with pytest.raises(PaperRefinementGuardError):
        guard_paper_refinement(ORIGINAL, refined)


def test_new_detection_claim_blocked():
    refined = ORIGINAL + " We report the discovery of a planetary companion."
    with pytest.raises(PaperRefinementGuardError):
        guard_paper_refinement(ORIGINAL, refined)


def test_find_new_claims_returns_offending_sentences():
    refined = ORIGINAL + " A 7 sigma signal was found."
    flagged = find_new_claims(ORIGINAL, refined)
    assert any("7 sigma" in s for s in flagged)
