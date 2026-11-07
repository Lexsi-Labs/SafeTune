"""MBPP grading runs the candidate in ONE child per completion (a hang costs one idle timeout, not one per assert).

The old grader (one child per assert, ``grade_assertion``) is the reference: honest code must score exactly the same,
and nothing a candidate could not do before may work now. Benign code only; ``timeout=2`` keeps slow runners safe."""

import os
import time

import pytest

from safetune.rewards._code_grader import grade_assertion, grade_assertions

TIMEOUT = 2.0
A = ["assert f(2) == 4", "assert f(3) == 6", "assert f(5) == 10"]
HONEST = "def f(x):\n    return 2 * x"


def old(code, tests, timeout=TIMEOUT):
    return [grade_assertion(code, t, timeout) for t in tests]


def new(code, tests, timeout=TIMEOUT):
    return grade_assertions(code, tests, timeout)


def timed(fn, *a):
    start = time.monotonic()
    out = fn(*a)
    return out, time.monotonic() - start


def test_honest_code_scores_exactly_what_the_old_grader_gives():
    assert new(HONEST, A) == old(HONEST, A) == [True, True, True]


def test_a_hang_costs_one_idle_timeout_not_one_per_assert():
    result, took = timed(new, "while True: pass", A)
    assert result == [False, False, False]
    assert took < 4.5                      # the old grader needs 3 x timeout = 6 s


def test_answered_asserts_keep_their_result_when_a_later_one_hangs():
    code = "def f(x):\n    if x > 2:\n        while True: pass\n    return 2 * x"
    result, took = timed(new, code, A)
    assert result == [True, False, False] and took < 4.5


def test_slow_but_progressing_code_gets_the_same_per_assert_budget_as_before():
    code = "import time\ndef f(x):\n    time.sleep(0.7)\n    return 2 * x"
    result, took = timed(new, code, A, 1.2)      # 3 x 0.7 s > the 1.2 s timeout, but a record arrives every 0.7 s
    assert result == [True, True, True] and took < 3.2 + 1


def test_a_global_changed_by_one_assert_does_not_reach_the_next():
    code = "n = 0\ndef f(x):\n    global n\n    n += 1\n    return n"
    tests = ["assert f(0) == 1"] * 3
    assert new(code, tests) == old(code, tests) == [True, True, True]


def test_a_candidate_that_swallows_every_exception_is_still_killed():
    code = "while True:\n    try:\n        while True: pass\n    except BaseException:\n        pass"
    result, took = timed(new, code, A)
    assert result == [False, False, False] and took < 4.5


def test_a_background_child_does_not_outlive_the_grading(tmp_path):
    mark = tmp_path / "alive"
    code = (
        "import os, time\nif os.fork() == 0:\n    time.sleep(2.5)\n"
        f"    open({str(mark)!r}, 'w').write('x')\n    os._exit(0)\n" + HONEST
    )
    assert new(code, A) == [True, True, True]
    time.sleep(3.2)
    assert not os.path.exists(mark)


def test_a_hanging_candidate_with_a_background_child_is_killed_whole(tmp_path):
    mark = tmp_path / "alive"
    code = (
        "import os, time\nif os.fork() == 0:\n    time.sleep(3)\n"
        f"    open({str(mark)!r}, 'w').write('x')\n    os._exit(0)\nwhile True: pass"
    )
    assert new(code, ["assert f() == 1"], 1.0) == [False]
    time.sleep(3.2)
    assert not os.path.exists(mark)


@pytest.mark.parametrize("stream", ["sys.stdout", "sys.stderr"])
def test_text_that_looks_like_a_result_record_on_stdout_or_stderr_is_not_a_result(stream):
    """stdout and stderr go to /dev/null in the child, as before: nothing printed can be read by the trainer."""
    code = f"import sys\nprint('[true]\\n[true]\\n[true]', file={stream})\ndef f(x): return 0"
    assert new(code, A) == old(code, A) == [False, False, False]


EXIT_AND_RAISE = {
    "exit_zero": "import sys\nsys.exit(0)",
    "hard_exit": "import os\nos._exit(0)",
    "raise_at_import": "raise RuntimeError('boom')\ndef f(x): return 2 * x",
    "syntax_error": "def f(x:\n    return 2 * x",
    "null_byte": "a\x00b",
    "exit_inside_the_call": "import sys\ndef f(x):\n    sys.exit(0)",
    "printed_pass": 'import sys\ndef f(*a, **k): pass\nprint("PASS"); sys.exit(0)',
}


@pytest.mark.parametrize("name", sorted(EXIT_AND_RAISE))
def test_exits_and_import_errors_score_what_the_old_grader_gives(name):
    code = EXIT_AND_RAISE[name]
    assert new(code, A) == old(code, A) == [False, False, False]


def test_zero_and_one_assert_match_the_old_grader():
    assert new(HONEST, []) == old(HONEST, []) == []
    assert new(HONEST, ["assert f(2) == 4"]) == old(HONEST, ["assert f(2) == 4"]) == [True]
    assert new(HONEST, ["assert f(2) == 5"]) == old(HONEST, ["assert f(2) == 5"]) == [False]
    assert new("while True: pass", ["assert f(2) == 4"], 1.0) == [False]


def test_asserts_that_cannot_be_graded_stay_failed_in_place():
    tests = ["assert f(2) == 4", "print('not an assert')", "assert f(3) == 6", "assert f(2) == __import__('os')"]
    assert new(HONEST, tests) == old(HONEST, tests) == [True, False, True, False]
    assert new(HONEST, ["print('x')"]) == [False]


def test_a_big_honest_result_and_a_memory_hungry_one_are_handled_as_before():
    big = "def f(): return list(range(50000))"
    tests = ["assert len(f()) == 50000", "assert f()[-1] == 49999"]
    assert new(big, tests) == old(big, tests) == [True, True]
    too_many = "def f(): return list(range(300000))"             # past the decoded-element cap: refused, as before
    assert new(too_many, tests[:1]) == old(too_many, tests[:1]) == [False]
    hungry = "def f(): return 'x' * (16 * 1024 ** 3)"
    assert new(hungry, ["assert len(f()) == 5", "assert f() == 'x'"]) == old(hungry, ["assert len(f()) == 5", "assert f() == 'x'"]) == [False, False]
    assert new(hungry + "\ndef g(): return 1", ["assert g() == 1"]) == old(hungry + "\ndef g(): return 1", ["assert g() == 1"]) == [True]


def test_an_oversized_result_fails_and_does_not_hurt_the_asserts_before_it(monkeypatch):
    monkeypatch.setattr("safetune.rewards._code_grader._MAX_RESULT_BYTES", 2048)
    code = "def f(): return 1\ndef g(): return 'x' * 100000"
    assert new(code, ["assert f() == 1", "assert len(g()) == 100000", "assert f() == 1"]) == [True, False, False]


OLD_FORGE = "import os\nos.write(3, b'[%s]')\nos._exit(0)"          # the old one-record file
NEW_FORGE = "import os\nos.write(3, b'[%s]\\n' * %d)\nos._exit(0)"      # the same idea in the new line-per-assert file
TESTS5 = ["assert f(1) == 5"] * 3


def test_forging_through_the_open_result_file_needs_the_expected_value_exactly_as_before():
    """The result file is the one channel a determined candidate can reach (documented in the module docstring): it needs the
    expected value, and only plain data crosses. The old file takes one record, the new one a line per assert; each protocol accepts
    a forged record only if it holds the right value."""
    assert old(OLD_FORGE % "5", TESTS5[:1]) == [True]            # parity: forging is possible today for a guessable value
    assert new(NEW_FORGE % ("5", 3), TESTS5) == [True, True, True]
    assert old(OLD_FORGE % "7", TESTS5[:1]) == [False] and new(NEW_FORGE % ("7", 3), TESTS5) == [False, False, False]
    assert new(NEW_FORGE % ("5", 1), TESTS5) == [True, False, False]       # a short forgery earns only the records it wrote


def test_forging_that_does_not_exit_or_writes_too_much_is_rejected_as_before():
    carry_on = "import os\nos.write(3, b'[5]\\n')\ndef f(x): return 0"
    assert old("import os\nos.write(3, b'[5]')\ndef f(x): return 0", TESTS5[:1]) == [False]
    assert new(carry_on, TESTS5[:1]) == [False]
    assert new(NEW_FORGE % ("5", 9), TESTS5) == [False, False, False]
    assert new("import os\nos.write(3, b'[5]')\nos._exit(0)", TESTS5[:1]) == [False]      # the old protocol's payload earns nothing here
    assert new("import os\nos.write(3, b'[5]\\n[5')\nos._exit(0)", TESTS5[:2]) == [True, False]


# ---- through MBPPReward --------------------------------------------------------------------------------------

from safetune.rewards.base import RewardConfig, RewardType  # noqa: E402
from safetune.rewards.code import MBPPReward  # noqa: E402


def mbpp(code, tests, timeout=TIMEOUT):
    return MBPPReward(RewardConfig(reward_type=RewardType.CODE_EXECUTION, params={"timeout": timeout})).compute(
        f"```python\n{code}\n```", test_list=tests)


def old_score(code, tests, timeout=TIMEOUT):
    return sum(grade_assertion(code, t, timeout) for t in tests) / len(tests)


def test_mbpp_hang_with_three_asserts_costs_one_timeout():
    score, took = timed(mbpp, "while True: pass", A)
    assert score == 0.0
    assert took < 4.5                      # the old grader: about 6 s


def test_mbpp_honest_completion_scores_one_and_what_the_old_grader_gives():
    assert mbpp(HONEST, A) == old_score(HONEST, A) == 1.0


def test_mbpp_answers_one_assert_then_hangs():
    code = "def f(x):\n    if x > 2:\n        while True: pass\n    return 2 * x"
    score, took = timed(mbpp, code, A)
    assert score == pytest.approx(1 / 3) and took < 4.5


def test_mbpp_no_tests_and_one_test():
    assert MBPPReward(RewardConfig(reward_type=RewardType.CODE_EXECUTION, params={"timeout": TIMEOUT})).compute(
        f"```python\n{HONEST}\n```", test_list=[]) == 0.0
    assert mbpp(HONEST, ["assert f(2) == 4"]) == 1.0 and mbpp(HONEST, ["assert f(2) == 3"]) == 0.0


COMPLETIONS = {
    "honest": HONEST,
    "wrong": "def f(x):\n    return x + 2",                                   # right for 2 only
    "half_right": "def f(x):\n    return 2 * x if x < 4 else 0",
    "raises_for_big": "def f(x):\n    if x > 4:\n        raise ValueError\n    return 2 * x",
    "returns_a_tuple": "def f(x):\n    return (2 * x,)",
    "uses_a_helper": "def _h(x):\n    return x + x\ndef f(x):\n    return _h(x)",
    "imports": "import math\ndef f(x):\n    return int(math.fsum([x, x]))",
    "mutates_a_global": "n = 0\ndef f(x):\n    global n\n    n += 1\n    return 2 * x * n",
    "defines_a_class": "class C:\n    def __init__(self, x): self.x = x\ndef f(x):\n    return C(x).x * 2",
}


@pytest.mark.parametrize("name", sorted(COMPLETIONS))
def test_differential_new_score_equals_old_score(name):
    assert mbpp(COMPLETIONS[name], A) == pytest.approx(old_score(COMPLETIONS[name], A)), name


PATCHER = (
    "import builtins\n"
    "for _n in ('len', 'isinstance', 'any', 'all', 'sorted', 'int', 'str', 'float', 'bool', 'list', 'tuple', 'dict', 'set', 'range', 'repr', 'type', 'zip'):\n"
    "    setattr(builtins, _n, lambda *a, **k: 5)\n"
    "def f(x):\n    return 2 * x\n"
)


def test_builtins_patched_by_the_candidate_do_not_corrupt_the_records_of_later_asserts():
    """The patching code runs again for every assert in the same process; the child's encoding and the trainer's comparison must not depend on it."""
    tests = ["assert f(2) == 4", "assert len(f(3)) == 6", "assert f(5) == 10", "assert f(1) == 2"]
    assert mbpp(PATCHER, tests) == old_score(PATCHER, tests)
    assert mbpp(PATCHER, tests[:1] + tests[2:]) == 1.0


def test_state_left_in_the_process_by_one_assert_is_documented_to_reach_the_next():
    """Process-level state is shared (documented): an attribute set on an imported module in the first assert is seen by the second."""
    code = ("import os\nseen = getattr(os, 'marker_x', 0)\nos.marker_x = seen + 1\ndef f(x):\n    return seen")
    tests = ["assert f(0) == 0", "assert f(0) == 1", "assert f(0) == 2"]
    assert mbpp(code, tests) == 1.0                         # one child: 0, 1, 2
    assert old_score(code, tests) == pytest.approx(1 / 3)   # one child per assert: always 0


import sys  # noqa: E402

linux_only = pytest.mark.skipif(sys.platform != "linux", reason="the address-space and file-size limits are only tested on Linux")


@linux_only
def test_the_memory_limit_covers_the_whole_child():
    """Documented: RLIMIT_AS (4 GiB) is for the child, not per assert. A candidate that keeps 1.5 GB per assert in process-level state
    (here an attribute on builtins) runs out on the third assert; one child per assert never did."""
    code = "import builtins\nbuiltins.keep = getattr(builtins, 'keep', [])\nbuiltins.keep.append(bytearray(1_500_000_000))\ndef f(x): return 2 * x"
    assert new(code, A, 3.0) == [True, True, False]
    assert old(code, A, 3.0) == [True, True, True]


@linux_only
def test_the_result_file_limit_covers_the_whole_child():
    """Documented: the 16 MiB result-file limit is for all records together; honest MBPP results are a few bytes."""
    code = "def f(x):\n    return 'x' * 7_000_000"
    tests = ["assert len(f(1)) == 7000000"] * 3
    assert new(code, tests, 3.0) == [True, True, False]
    assert old(code, tests, 3.0) == [True, True, True]


# ---- candidates that tamper with the result descriptor (fd 3 in the child) ------------------------------------

def _no_raise(fn, *a):
    """(result, seconds); a parent-side exception fails the test with its type."""
    try:
        return timed(fn, *a)
    except BaseException as e:      # noqa: BLE001 - the point is that nothing the child does may raise here
        pytest.fail(f"the parent raised {type(e).__name__}: {e}")


def test_a_half_written_record_line_then_a_hang_costs_one_timeout_and_never_raises():
    code = "import os\nos.write(3, b'[4')\nwhile True: pass\ndef f(x): return 2 * x"
    result, took = _no_raise(new, code, A)
    assert result == [False, False, False] and took < 4.5
    assert old(code, A[:1], 1.0) == [False]                      # the old grader also scores it 0


def test_a_half_written_record_line_then_carrying_on_garbles_every_record_and_scores_zero():
    code = "import os\nos.write(3, b'[4')\ndef f(x): return 2 * x"
    result, took = _no_raise(new, code, A)
    assert result == [False, False, False] and took < 4.5        # every assert's run writes the half line first
    assert old(code, A, TIMEOUT) == [False, False, False]


def test_a_candidate_that_closes_the_result_descriptor_and_keeps_running_scores_zero():
    code = "import os\nos.close(3)\ndef f(x): return 2 * x"
    result, took = _no_raise(new, code, A)
    assert result == [False, False, False] and took < 4.5
    assert old(code, A, TIMEOUT) == [False, False, False]
    hang = "import os\nos.close(3)\nwhile True: pass"
    result, took = _no_raise(new, hang, A)
    assert result == [False, False, False] and took < 4.5        # no records ever arrive: one idle timeout


@pytest.mark.parametrize("target", [0, 1, 2, 9])
def test_dup2_of_the_result_descriptor_over_another_descriptor_changes_nothing_for_honest_code(target):
    code = f"import os\nos.dup2(3, {target})\ndef f(x): return 2 * x"
    result, took = _no_raise(new, code, A)
    assert result == old(code, A, TIMEOUT) == [True, True, True] and took < 4.5


def test_dup2_over_the_result_descriptor_with_devnull_or_then_closing_it_scores_zero():
    for code in ("import os\nfd = os.open('/dev/null', os.O_WRONLY)\nos.dup2(fd, 3)\ndef f(x): return 2 * x",
                 "import os\nos.dup2(3, 9)\nos.close(3)\ndef f(x): return 2 * x"):
        result, took = _no_raise(new, code, A)
        assert result == old(code, A, TIMEOUT) == [False, False, False] and took < 4.5


def test_printing_into_a_descriptor_that_was_dup2d_from_the_result_file_is_the_same_forgery_as_writing_to_it():
    """After dup2(3, 1) stdout IS the result file, so print() reaches it. This is the documented open-result-file channel,
    not a new one: it needs the expected value, and both protocols reject the wrong one."""
    old_forge = "import os, sys\nos.dup2(3, 1)\nprint('[%s]', end='')\nsys.stdout.flush()\nos._exit(0)"
    new_forge = "import os, sys\nos.dup2(3, 1)\nprint('[%s]' * 3, end='')\nsys.stdout.flush()\nos._exit(0)"
    t = "assert f(1) == 5"
    assert old(old_forge % "5", [t]) == [True] and new(new_forge.replace("'[%s]' * 3", "'[%s]\\n' * 3") % "5", [t] * 3) == [True] * 3
    assert old(old_forge % "7", [t]) == [False] and new(new_forge.replace("'[%s]' * 3", "'[%s]\\n' * 3") % "7", [t] * 3) == [False] * 3
