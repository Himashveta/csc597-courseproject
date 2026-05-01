"""Oracle: turn a subprocess outcome into a bug verdict.

We do NOT run differential testing here — that's a separate harness
concern and is left to the user's seed scripts (a seed can compare
torch.compile against eager and exit non-zero on mismatch). What this
oracle does is classify the *exit signal*:

  - exit 0                                : pass
  - exit nonzero, stderr matches assertion: assertion / internal error
  - signal SIGSEGV / SIGABRT              : crash
  - signal SIGSEGV with ASAN/UBSan banner : sanitizer crash (high-confidence)
  - timeout                               : potential hang (recorded but not bug)
  - any other nonzero exit                : runtime error

The oracle returns an OracleVerdict that the fuzzer logs and uses to
boost a seed's fitness when a bug is found.
"""

from __future__ import annotations

import dataclasses
import enum
import re
import signal
from typing import Optional


class BugClass(str, enum.Enum):
    NONE = "none"
    CRASH = "crash"
    SANITIZER = "sanitizer"
    ASSERTION = "assertion"
    RUNTIME_ERROR = "runtime_error"
    HANG = "hang"


@dataclasses.dataclass
class OracleVerdict:
    bug_class: BugClass
    return_code: int
    signal_name: Optional[str]
    summary: str

    def is_bug(self) -> bool:
        return self.bug_class not in (BugClass.NONE, BugClass.HANG)


# Patterns that indicate something interesting in stderr.
_SANITIZER_RE = re.compile(
    r"(AddressSanitizer|LeakSanitizer|UndefinedBehaviorSanitizer|"
    r"==\d+==ERROR|runtime error:)",
    re.IGNORECASE,
)
_ASSERTION_RE = re.compile(
    r"(TORCH_CHECK|TORCH_INTERNAL_ASSERT|CHECK failed|"
    r"Assertion .* failed|assert\s+\w+|XlaRuntimeError|OrtFail)",
)
# Mutator-artefact patterns: things we explicitly want to NOT count
# as bugs because they're caused by the fuzzer rewriting the seed
# rather than by the system under test misbehaving.
_MUTATOR_ARTEFACT_RE = re.compile(
    r"^(ImportError|ModuleNotFoundError|SyntaxError|IndentationError|"
    r"NameError|TabError):",
    re.MULTILINE,
)


def classify_outcome(
    return_code: int,
    stderr: str,
    timed_out: bool,
) -> OracleVerdict:
    if timed_out:
        return OracleVerdict(
            bug_class=BugClass.HANG,
            return_code=return_code,
            signal_name=None,
            summary="timeout",
        )

    # Sanitizer reports get checked first, regardless of return code:
    # UBSan in recover-on mode prints "runtime error: ..." and exits 0,
    # but it's still a high-confidence bug we want to capture.
    if _SANITIZER_RE.search(stderr):
        sig_name = _signal_name(return_code) if return_code < 0 else None
        return OracleVerdict(
            bug_class=BugClass.SANITIZER,
            return_code=return_code,
            signal_name=sig_name,
            summary="sanitizer report",
        )

    if return_code == 0:
        return OracleVerdict(
            bug_class=BugClass.NONE,
            return_code=0,
            signal_name=None,
            summary="ok",
        )

    # Mutator artefacts (ImportError from a token swap that didn't
    # land in an import block, SyntaxError from an unparseable rewrite)
    # are not bugs in the SUT — they're noise from the fuzzer.
    if _MUTATOR_ARTEFACT_RE.search(stderr):
        return OracleVerdict(
            bug_class=BugClass.NONE,
            return_code=return_code,
            signal_name=None,
            summary="mutator artefact",
        )

    # Negative return codes from subprocess.run mean killed by signal.
    if return_code < 0:
        sig_name = _signal_name(return_code)
        if sig_name in {"SIGSEGV", "SIGBUS", "SIGFPE", "SIGILL", "SIGABRT"}:
            return OracleVerdict(
                bug_class=BugClass.CRASH,
                return_code=return_code,
                signal_name=sig_name,
                summary=f"crashed via {sig_name}",
            )
        return OracleVerdict(
            bug_class=BugClass.RUNTIME_ERROR,
            return_code=return_code,
            signal_name=sig_name,
            summary=f"signal {sig_name}",
        )

    if _ASSERTION_RE.search(stderr):
        return OracleVerdict(
            bug_class=BugClass.ASSERTION,
            return_code=return_code,
            signal_name=None,
            summary="assertion / internal check",
        )

    return OracleVerdict(
        bug_class=BugClass.RUNTIME_ERROR,
        return_code=return_code,
        signal_name=None,
        summary=f"exit {return_code}",
    )


def _signal_name(rc: int) -> Optional[str]:
    if rc >= 0:
        return None
    try:
        return signal.Signals(-rc).name
    except (ValueError, AttributeError):
        return None
