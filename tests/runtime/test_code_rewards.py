"""Code rewards pay for correct answers, not for faking a pass."""

import os
import time

import pytest

from safetune.rewards.base import RewardConfig, RewardType
from safetune.rewards.code import CodeCorrectnessReward, CodeExecutionReward, MBPPReward

ASSERTS = [
    "assert is_not_prime(2) == False",
    "assert is_not_prime(10) == True",
    "assert is_not_prime(35) == True",
]
DICTS = [{"input": 2, "expected_output": False}, {"input": 10, "expected_output": True}]

GOOD = """```python
def is_not_prime(n):
    return any(n % i == 0 for i in range(2, int(n**0.5) + 1))
```"""
WRONG = "```python\ndef is_not_prime(n):\n    return False\n```"

CHEATS = {
    "printed_pass": '```python\nimport sys\ndef f(*a, **k): pass\nprint("PASS"); sys.exit(0)\n```',
    "exit_zero": "```python\nimport sys\nsys.exit(0)\n```",
    "hard_exit": "```python\nimport os\nos._exit(0)\n```",
    "always_equal": (
        "class _Always:\n    def __eq__(self, other): return True\n"
        "def is_not_prime(*a, **k): return _Always()"
    ),
    "bool_subclass": (
        "class B(int):\n    def __eq__(self, other): return True\n"
        "def is_not_prime(*a, **k): return B()"
    ),
    "hang": "```python\nwhile True: pass\n```",
}


def _config(**params):
    return RewardConfig(reward_type=RewardType.CODE_EXECUTION, params={"timeout": 2.0, **params})


def _scores(text):
    return (
        CodeExecutionReward(_config()).compute(text, test_cases=ASSERTS),
        CodeExecutionReward(_config()).compute(text, test_cases=DICTS),
        MBPPReward(_config()).compute(text, test_list=ASSERTS),
    )


@pytest.mark.parametrize("name", sorted(CHEATS))
def test_cheats_score_zero(name):
    assert _scores(CHEATS[name]) == (0.0, 0.0, 0.0)


def test_correct_answer_scores_one():
    assert _scores(GOOD) == (1.0, 1.0, 1.0)


def test_wrong_answer_scores_what_it_earns():
    exec_score, dict_score, mbpp_score = _scores(WRONG)
    assert exec_score == mbpp_score == pytest.approx(1 / 3)
    assert dict_score == 0.5


def test_no_python_on_path_still_grades(monkeypatch):
    monkeypatch.setenv("PATH", "")
    assert _scores(GOOD) == (1.0, 1.0, 1.0)


def test_correctness_with_no_tests_scores_zero():
    assert CodeCorrectnessReward(_config()).compute(GOOD) == 0.0
    assert CodeCorrectnessReward(_config()).compute("") == 0.0


def test_execution_with_no_tests_needs_a_full_run():
    reward = CodeExecutionReward(_config())
    assert reward.compute("```python\nx = 1\n```") == 1.0
    assert reward.compute(CHEATS["exit_zero"]) == 0.0
    assert reward.compute(CHEATS["hard_exit"]) == 0.0


def test_expected_value_never_reaches_the_child():
    # Reads the harness's job, and would pass if the expected value were there. The
    # number is split so the candidate's own source doesn't contain it.
    code = (
        "import __main__\njob = str(__main__.job)\ntarget = '123' + '45'\n"
        "def f(): return int(target) if target in job else 0"
    )
    assert MBPPReward(_config()).compute(code, test_list=["assert f() == 12345"]) == 0.0


@pytest.mark.parametrize(
    "test",
    [
        "assert math.isclose(area(2), 4.0, rel_tol=0.001)",
        "assert set(pair(1)) == {1, 2}",
        "assert pair(1) == (1, 2)",
        "assert data() == {'a': [1, 2.5]}",
        "assert 1 in pair(1)",
        "assert not empty()",
        "assert area(2)",
        "assert empty() is None",
        "assert area(1) < area(2)",
    ],
)
def test_assert_shapes_grade_correct_answers(test):
    code = (
        "import math\n"
        "def area(r): return float(r * r)\n"
        "def pair(n): return (n, n + 1)\n"
        "def data(): return {'a': [1, 2.5]}\n"
        "def empty(): return None"
    )
    assert MBPPReward(_config()).compute(code, test_list=[test]) == 1.0


def test_assert_shape_it_cant_grade_fails():
    assert MBPPReward(_config()).compute(GOOD, test_list=["print('not an assert')"]) == 0.0


def test_imports_above_the_first_def_are_kept():
    code = "import math\ndef f(x): return math.floor(x)"
    assert CodeExecutionReward(_config()).compute(code, test_cases=["assert f(1.5) == 1"]) == 1.0


def _writes_result(payload):
    return (
        "import __main__, os\n"
        f"__main__.result.write({payload!r})\n__main__.result.close()\nos._exit(0)"
    )


@pytest.mark.parametrize(
    "payload",
    [
        "[" * 200000 + "]" * 200000,
        "[" + '{"list":[' * 900 + "]}" * 900 + "]",
        "[]",
        "[1, 2]",
        '{"f": "1.0"}',
        '[{"nope": 1}]',
        '[{"dict": [[[], 1]]}]',
        "not json",
    ],
)
def test_a_tampered_result_never_raises_into_the_trainer(payload):
    code = _writes_result(payload)
    assert MBPPReward(_config()).compute(code, test_list=["assert f(1) == 1"]) == 0.0
    assert CodeExecutionReward(_config()).compute(code, test_cases=DICTS) == 0.0


def test_result_path_is_not_in_argv_or_the_working_directory():
    code = (
        "import os, sys\n"
        "assert sys.argv == ['candidate']\n"
        "assert os.listdir('.') == []\n"
        "def f(): return 1"
    )
    assert MBPPReward(_config()).compute(code, test_list=["assert f() == 1"]) == 1.0


def test_oversized_result_fails(monkeypatch):
    monkeypatch.setattr("safetune.rewards._code_grader._MAX_RESULT_BYTES", 1024)
    code = "def f(): return 'x' * 100000"
    assert MBPPReward(_config()).compute(code, test_list=["assert f() == 'x'"]) == 0.0


def test_trainer_environment_does_not_reach_the_child(monkeypatch):
    monkeypatch.setenv("SAFETUNE_TEST_SECRET", "marker")
    code = "import os\ndef f(): return os.environ.get('SAFETUNE_TEST_SECRET')"
    assert MBPPReward(_config()).compute(code, test_list=["assert f() is None"]) == 1.0


def test_io_test_with_none_input():
    code = "def f(x): return x is None"
    case = [{"input": None, "expected_output": True}]
    assert CodeExecutionReward(_config()).compute(code, test_cases=case) == 1.0


def test_grandchild_dies_with_the_timeout(tmp_path):
    mark = tmp_path / "alive"
    code = (
        "import os, time\nif os.fork() == 0:\n    time.sleep(3)\n"
        f"    open({str(mark)!r}, 'w').write('x')\n    os._exit(0)\nwhile True: pass"
    )
    reward = MBPPReward(RewardConfig(reward_type=RewardType.CODE_EXECUTION, params={"timeout": 1.0}))
    assert reward.compute(code, test_list=["assert f() == 1"]) == 0.0
    time.sleep(3)
    assert not os.path.exists(mark)


def test_source_that_breaks_the_compiler_scores_zero():
    assert CodeExecutionReward(_config()).compute("(" * 200000, test_cases=ASSERTS) == 0.0
    assert CodeExecutionReward(_config()).compute("a\x00b", test_cases=ASSERTS) == 0.0


def _writes_payload(payload_expr):
    # Writes the encoded result directly, so the child never builds the container itself.
    return (
        "import __main__, json, os\n"
        f"__main__.result.write(json.dumps({payload_expr}))\n"
        "__main__.result.close()\nos._exit(0)"
    )


def test_colliding_hash_set_cannot_hang_the_trainer():
    # Multiples of 2**61 - 1 all hash to 0: building a set of them in the trainer is quadratic.
    code = _writes_payload('[{"set": [k * (2**61 - 1) for k in range(1, 200001)]}]')
    start = time.time()
    assert MBPPReward(_config()).compute(code, test_list=["assert f() != 0"]) == 0.0
    assert time.time() - start < 20


def test_many_nested_elements_score_zero():
    code = _writes_payload('[{"list": [{"list": [1] * 10} for _ in range(20000)]}]')
    assert MBPPReward(_config()).compute(code, test_list=["assert f() != 0"]) == 0.0


def test_reasonable_containers_still_decode():
    code = "def f():\n    return {i: [i, i + 0.5] for i in range(1000)}"
    assert MBPPReward(_config()).compute(code, test_list=["assert f() != 0"]) == 1.0
