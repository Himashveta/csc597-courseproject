"""Tests for the bug-classification oracle."""

from polyfuzz.oracle import classify_outcome
from polyfuzz.oracle.oracle import BugClass


def test_clean_exit_is_not_a_bug():
    v = classify_outcome(0, "", False)
    assert v.bug_class == BugClass.NONE
    assert not v.is_bug()


def test_timeout_is_hang_not_bug():
    v = classify_outcome(-1, "", True)
    assert v.bug_class == BugClass.HANG
    assert not v.is_bug()


def test_sigsegv_is_crash():
    v = classify_outcome(-11, "", False)
    assert v.bug_class == BugClass.CRASH
    assert v.signal_name == "SIGSEGV"
    assert v.is_bug()


def test_ubsan_recover_mode_still_classified_as_bug_when_rc_zero():
    """UBSan with -fno-sanitize-recover prints the banner and exits 0.
    The oracle must inspect stderr regardless of return code."""
    stderr = "mock_compiler.c:180:30: runtime error: division by zero\n"
    v = classify_outcome(0, stderr, False)
    assert v.bug_class == BugClass.SANITIZER
    assert v.is_bug()


def test_asan_banner_is_sanitizer_bug():
    stderr = "==12345==ERROR: AddressSanitizer: heap-buffer-overflow"
    v = classify_outcome(-6, stderr, False)
    assert v.bug_class == BugClass.SANITIZER
    assert v.signal_name == "SIGABRT"


def test_torch_check_is_assertion():
    stderr = ('Traceback (most recent call last):\n'
              'RuntimeError: TORCH_CHECK failed: dimension mismatch')
    v = classify_outcome(1, stderr, False)
    assert v.bug_class == BugClass.ASSERTION


def test_import_error_is_mutator_artefact_not_bug():
    """Mutator-induced ImportError is noise from the fuzzer, not a
    target bug. We must not log it as a bug or it floods the report."""
    stderr = ("Traceback (most recent call last):\n"
              "ImportError: cannot import name 'OP_REDUCE' from 'mock_compiler'")
    v = classify_outcome(1, stderr, False)
    assert v.bug_class == BugClass.NONE
    assert not v.is_bug()


def test_syntax_error_is_mutator_artefact():
    stderr = "SyntaxError: invalid syntax"
    v = classify_outcome(1, stderr, False)
    assert v.bug_class == BugClass.NONE


def test_sanitizer_takes_priority_over_mutator_artefact():
    """If a seed both has a syntax error AND triggers UBSan (somehow),
    UBSan wins — sanitizer banners are always interesting."""
    stderr = ("ImportError: nope\n"
              "runtime error: division by zero")
    v = classify_outcome(1, stderr, False)
    # Sanitizer check happens first.
    assert v.bug_class == BugClass.SANITIZER
