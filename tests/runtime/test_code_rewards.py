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


# --- the trainer decides the boolean structure; the child can't change the builtins ---

def _mbpp(code, *tests):
    return MBPPReward(_config()).compute(code, test_list=list(tests))


FIVE = "def f(): return [1, 2, 3]\n"  # three items, so `len(f()) == 5` is false

REDEFINED = {
    "len": "def len(x): return 5\n",
    "abs": "def abs(x): return 5\n",
    "isinstance": "def isinstance(a, b): return True\n",
    "sorted": "def sorted(x, **k): return [1, 2, 3, 4, 5]\n",
}


@pytest.mark.parametrize("name", sorted(REDEFINED))
def test_redefined_builtin_does_not_change_the_test(name):
    test = {
        "len": "assert len(f()) == 5",
        "abs": "assert abs(f()[0]) == 5",
        "isinstance": "assert isinstance(f(), tuple)",
        "sorted": "assert sorted(f()) == [1, 2, 3, 4, 5]",
    }[name]
    assert _mbpp(FIVE + REDEFINED[name], test) == 0.0


def test_patched_builtins_do_not_change_the_test():
    code = FIVE + "import builtins\nbuiltins.len = lambda x: 5\nbuiltins.isinstance = lambda a, b: True\n"
    assert _mbpp(code, "assert len(f()) == 5") == 0.0
    assert _mbpp(code, "assert isinstance(f(), tuple)") == 0.0
    assert _mbpp(code, "assert len(f()) == 3") == 1.0


def test_name_clash_with_a_builtin_fails_with_a_warning(caplog):
    with caplog.at_level("WARNING"):
        assert _mbpp("def sorted(x): return [1]\n", "assert sorted([2, 1]) == [1, 2]") == 0.0
    assert "builtin's name" in caplog.text


COMPOUND_CODE = (
    "def once(n): return 1 if n == 1 else 0\n"
    "def f(n): return n if n == 1 else 99 if n == 3 else 0\n"
    "def two(): return 2\n"
    "class Always:\n    def __eq__(self, o): return True\n"
    "def obj(): return Always()\n"
)


@pytest.mark.parametrize(
    "test, score",
    [
        ("assert f(1) == 1 and f(2) == 99", 0.0),
        ("assert f(1) == 1 and f(3) == 99", 1.0),
        ("assert f(1) == 1 or garbage()", 1.0),
        ("assert garbage() or f(1) == 1", 1.0),
        ("assert f(2) == 5 or garbage()", 0.0),
        ("assert not (f(1) == 2)", 1.0),
        ("assert not (f(1) == 1)", 0.0),
        ("assert not garbage()", 0.0),
        ("assert not (garbage() or f(2) == 5)", 0.0),
        ("assert 1 < two() < 3", 1.0),
        ("assert 1 < two() < 2", 0.0),
        ("assert 3 < two() < 5", 0.0),
        ("assert two() == 2 and (f(1) == 1 or f(2) == 1)", 1.0),
        ("assert (obj() == 1) is True", 0.0),
        ("assert all(obj() == i for i in range(3))", 0.0),
        ("assert obj() == 1 and two() == 2", 0.0),
        ("assert f(3) + 1 == 100", 1.0),
        ("assert f(1) + 1 == 2", 1.0),
        ("assert f(1) + 1 == 3", 0.0),
        ("assert isinstance(two(), int) and any([two()]) and all([two()])", 1.0),
        ("assert sorted([two(), 1]) == [1, 2] and len(f(1) * [1]) == 1", 1.0),
        ("assert two().bit_length() == 2", 0.0),
        ("assert two()._x == 1", 0.0),
        ("assert open('x')", 0.0),
        ("assert (lambda: True)()", 0.0),
    ],
)
def test_compound_asserts_are_decided_in_the_trainer(test, score):
    assert _mbpp(COMPOUND_CODE, test) == score


def test_compound_assert_that_cant_be_decomposed_fails_with_a_warning(caplog):
    with caplog.at_level("WARNING"):
        assert _mbpp(COMPOUND_CODE, "assert all(obj() == i for i in range(3))") == 0.0
    assert "Can't grade" in caplog.text


TAMPER = {
    "frame": (
        "import sys\ndef f():\n    return [1]\n"
        "fr = sys._getframe(0)\nwhile fr.f_back is not None:\n    fr = fr.f_back\n"
        "for k, v in list(fr.f_locals.items()):\n    if isinstance(v, dict) and 'len' in v: v['len'] = lambda x: 5\n"
    ),
    "gc": (
        "import gc\ndef f():\n    return [1]\n"
        "for o in gc.get_objects():\n    if isinstance(o, dict) and o.get('len') is len: o['len'] = lambda x: 5\n"
    ),
}


@pytest.mark.parametrize("name", sorted(TAMPER))
def test_candidate_walking_frames_or_gc_cannot_change_a_test(name):
    assert _mbpp(TAMPER[name], "assert len(f()) == 5") == 0.0
    assert _mbpp(TAMPER[name], "assert len(f()) == 1") == 1.0


def test_object_with_a_lying_class_or_eq_is_rejected():
    code = (
        "class L(list):\n    __class__ = int\n    def __eq__(self, o): return True\n"
        "def f(): return L([1])\n"
    )
    assert _mbpp(code, "assert f() == [2]", "assert isinstance(f(), list)", "assert f() == f()") == 0.0


def test_patched_predicates_in_the_candidate_namespace_do_not_apply():
    code = "def isinstance(a, b): return True\ndef any(x): return True\ndef f(): return 5\n"
    assert _mbpp(code, "assert isinstance(f(), list)") == 0.0
    assert _mbpp(code, "assert any(f())") == 0.0
    code = "import builtins\nbuiltins.isinstance = lambda a, b: True\nbuiltins.any = lambda x: True\ndef f(): return []\n"
    assert _mbpp(code, "assert isinstance(f(), list)", "assert any(f())") == 0.5


BIG = "def f(): return 10**4000\ndef n(): return 5\ndef s(): return 'x' * 1000000\ndef sl(): return ['a'] * 9000\n"


@pytest.mark.parametrize(
    "test",
    [
        "assert s() == 'x' * 10**9",
        "assert len(s()) * 10**9 == 0",
        "assert '%99999999d' % n() == ''",
        "assert '%s' % n() == '5'",
        "assert n() ** 10**6 == 0",
        "assert 2 ** f() == 0",
        "assert sum(range(10**10)) == f()",
        "assert [0] * f() == []",
        "assert f() << 10**7 == 0",
        "assert ('b' * 20).join(sl()) == ''",
        "assert 'a'.ljust(10**9) == ''",
        "assert 'a'.replace('a', 'b' * 10**6) == ''",
        "assert '{0.__class__}'.format(1) == ''",
        "assert f'{n():99999999}' == ''",
        "assert sorted(s()) == []",
        "assert list(s()) == []",
    ],
)
def test_resource_hungry_asserts_fail_fast(test):
    import tracemalloc

    tracemalloc.start()
    start = time.time()
    try:
        assert _mbpp(BIG, test) == 0.0
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert time.time() - start < 5
    assert peak < 60 * 1024 * 1024  # the trainer never builds the huge value


def test_normal_arithmetic_and_sizes_still_pass():
    assert _mbpp(BIG, "assert len(s()) == 1000000", "assert n() * 2 == 10", "assert n() % 3 == 2",
                 "assert n() ** 2 == 25", "assert [0] * n() == [0] * 5", "assert ','.join(['a', 'b']) == 'a,b'") == 1.0


HASHES = (
    "M = 2**61 - 1\n"
    "def f(): return [M * i for i in range(1, 9000)]\n"
    "def g(): return [M * i for i in range(1, 9000)]\n"
    "def d(): return list(range(9000))\n"
)


@pytest.mark.parametrize(
    "test",
    [
        "assert set(f()) == {1}",
        "assert frozenset(f()) == frozenset()",
        "assert dict.fromkeys(f()) == {}",
        "assert set(f()) | set(g()) == set()",
        "assert set(f()).union(g()) == set()",
        "assert set([1]).issubset(f())",
        "assert len(set(f())) == 9000",
    ],
)
def test_colliding_hashes_cannot_make_the_trainer_build_a_slow_set(test):
    start = time.time()
    assert _mbpp(HASHES, test) == 0.0
    assert time.time() - start < 5


def test_distinct_values_still_go_through_sets():
    assert _mbpp(HASHES, "assert len(set(d())) == 9000", "assert set(d()) | {-1} != set()",
                 "assert len(frozenset(d())) == 9000",
                 "assert d().count(5) == 1", "assert len(dict(enumerate(d()))) == 9000") == 1.0


def test_colliding_hashes_are_refused_before_the_set_is_built():
    from safetune.rewards import _code_grader

    bad = [(2**61 - 1) * i for i in range(1, 9000)]
    for test in ("assert set(f()) == {1}", "assert frozenset(f()) == frozenset()", "assert len(set(f())) == 1"):
        calls, check = _code_grader._plan_assert(test)
        start = time.time()
        with pytest.raises(_code_grader._Fail):
            check([bad])
        assert time.time() - start < 0.2


COLLIDE = [(2**61 - 1) * i for i in range(1, 9000)]
DISTINCT = list(range(9000))
DVALUES = {i: (2**61 - 1) * i for i in range(1, 9000)}


def _evaluate(test, **named):
    """Run the trainer-side check on values given by candidate-call name."""
    from safetune.rewards import _code_grader

    calls, check = _code_grader._plan_assert(test)
    values = [named[c.split("(")[0]] for c in calls]
    start = time.time()
    try:
        return check(values)
    finally:
        assert time.time() - start < 1


@pytest.mark.parametrize(
    "test, named",
    [
        ("assert len(set(f().values())) == 1", dict(f=DVALUES)),
        ("assert len(frozenset(f().values())) == 1", dict(f=DVALUES)),
        ("assert len(set(zip(f(), g()))) == 1", dict(f=COLLIDE, g=COLLIDE)),
        ("assert len(set(zip(g(), f()))) == 1", dict(f=COLLIDE, g=COLLIDE)),
        ("assert len(dict(zip(f(), g()))) == 1", dict(f=COLLIDE, g=DISTINCT)),
        ("assert len(set(reversed(f()))) == 1", dict(f=COLLIDE)),
        ("assert len(set(sorted(f().values()))) == 1", dict(f=DVALUES)),
        ("assert len(set([1]).union(zip(g(), g()))) == 1", dict(g=COLLIDE)),
        ("assert len(set([1]).union(g().values())) == 1", dict(g=DVALUES)),
        ("assert set([1]).issubset(g().values())", dict(g=DVALUES)),
        ("assert len(f().keys() | g().keys()) == 1", dict(f=DVALUES, g=DVALUES)),
    ],
)
def test_colliding_hashes_through_lazy_iterables_are_refused(test, named):
    from safetune.rewards import _code_grader

    with pytest.raises((_code_grader._Fail, TypeError)):
        _evaluate(test, **named)


@pytest.mark.parametrize(
    "test, named, want",
    [
        ("assert len(set(f().values())) == 9000", dict(f={i: i for i in range(9000)}), True),
        ("assert len(frozenset(f().items())) == 8999", dict(f=DVALUES), True),
        ("assert len(set(zip(f(), g()))) == 9000", dict(f=DISTINCT, g=DISTINCT), True),
        ("assert len(dict(zip(f(), g()))) == 9000", dict(f=DISTINCT, g=COLLIDE + [0]), True),
        ("assert len(dict(enumerate(f()))) == 8999", dict(f=COLLIDE), True),
        ("assert len(set(enumerate(f()))) == 8999", dict(f=COLLIDE), True),
        ("assert len(set(reversed(f()))) == 9000", dict(f=DISTINCT), True),
        ("assert sorted(f()) == sorted(f())", dict(f=COLLIDE), True),
        ("assert len(set(f().keys()) | set(g().keys())) == 8999", dict(f=DVALUES, g=DVALUES), True),
    ],
)
def test_distinct_values_through_lazy_iterables_still_work(test, named, want):
    assert _evaluate(test, **named) is want


SET_A, SET_B, SET_C = (set(COLLIDE[i * 3000:(i + 1) * 3000]) for i in range(3))
HALF = (set(COLLIDE[:4000]), set(COLLIDE[4000:8000]))
PAIRS = [(k, 1) for k in COLLIDE]


@pytest.mark.parametrize(
    "test, named",
    [
        ("assert len(f() | g()) == 1", dict(f=HALF[0], g=HALF[1])),
        ("assert len(f() & g()) == 1", dict(f=HALF[0], g=HALF[1])),
        ("assert len(f() - g()) == 1", dict(f=HALF[0], g=HALF[1])),
        ("assert len(f() ^ g()) == 1", dict(f=HALF[0], g=HALF[1])),
        ("assert len(f().union(g())) == 1", dict(f=HALF[0], g=HALF[1])),
        ("assert len(f().intersection(g())) == 1", dict(f=HALF[0], g=HALF[1])),
        ("assert len(f().difference(g())) == 1", dict(f=HALF[0], g=HALF[1])),
        ("assert len(f().symmetric_difference(g())) == 1", dict(f=HALF[0], g=HALF[1])),
        ("assert len(f().union(g(), h())) == 1", dict(f=SET_A, g=SET_B, h=SET_C)),
        ("assert len(set(f()) | set(g()) | set(h())) == 1", dict(f=list(SET_A), g=list(SET_B), h=list(SET_C))),
        ("assert len(dict(f())) == 1", dict(f=PAIRS + [()])),
        ("assert len(dict(f())) == 1", dict(f=PAIRS + [(1, 2, 3)])),
        ("assert len(dict(f())) == 1", dict(f=PAIRS + [5])),
        ("assert len(dict(f())) == 1", dict(f=PAIRS)),
        ("assert len(dict(f())) == 1", dict(f=frozenset(PAIRS))),
        ("assert len(dict(f())) == 1", dict(f=set(PAIRS))),
    ],
)
def test_set_operators_methods_and_dict_pairs_refuse_colliding_hashes(test, named):
    from safetune.rewards import _code_grader

    with pytest.raises((_code_grader._Fail, TypeError)):
        _evaluate(test, **named)


def test_distinct_sets_and_pairs_still_work():
    a, b = set(range(4000)), set(range(2000, 6000))
    assert _evaluate("assert len(f() | g()) == 6000", f=a, g=b) is True
    assert _evaluate("assert len(f() & g()) == 2000", f=a, g=b) is True
    assert _evaluate("assert len(f().union(g(), g())) == 6000", f=a, g=b) is True
    assert _evaluate("assert len(dict(f())) == 9000", f=[(i, 1) for i in range(9000)]) is True
    assert _evaluate("assert len(dict(f())) == 9000", f=frozenset((i, 1) for i in range(9000))) is True


@pytest.mark.parametrize(
    "pair",
    [frozenset({1, 2}), {1, 2}, "ab", {1: 2, 3: 4}, b"ab", [1]],
)
def test_dict_needs_every_element_to_be_a_pair_of_a_list_or_tuple(pair):
    from safetune.rewards import _code_grader

    with pytest.raises(_code_grader._Fail):
        _evaluate("assert len(dict(f())) == 1", f=[(1, 2), pair])
    with pytest.raises(_code_grader._Fail):
        _evaluate("assert len(dict(f())) == 1", f=PAIRS[:5000] + [pair])


def test_pairs_whose_keys_collide_in_a_frozenset_pair_are_refused():
    from safetune.rewards import _code_grader

    with pytest.raises(_code_grader._Fail):
        _evaluate("assert len(dict(f())) == 1", f=[frozenset({k, 1}) for k in COLLIDE])
    assert _evaluate("assert dict(f()) == {1: 2}", f=[(1, 2)]) is True


@pytest.mark.parametrize(
    "test",
    [
        "assert len(set(f()).union(g())) == 18000",
        "assert len(dict(zip(f(), g()))) == 9000",
    ],
)
def test_two_large_honest_results_in_one_assert_still_pass(test):
    code = "def f(): return list(range(9000))\ndef g(): return list(range(9000, 18000))"
    assert MBPPReward(_config()).compute(code, test_list=[test]) == 1.0


REPEATS = (
    "def f(): return [0] * 9000\n"
    "def g(): return [2] * 9000\n"
    "def h(): return [1] * 100 + [1.0] * 100 + [True] * 100\n"
    "def p(): return [(1, 2)] * 9000\n"
    "def s(): return ['a'] * 9000\n"
)


@pytest.mark.parametrize(
    "test",
    [
        "assert set(f()) == {0}",
        "assert frozenset(f()) == frozenset({0})",
        "assert set(f()).union(g()) == {0, 2}",
        "assert set(f()) | set(g()) == {0, 2}",
        "assert dict(p()) == {1: 2}",
        "assert set(h()) == {1}",
        "assert set(s()) == {'a'}",
        "assert len(set(f())) == 1",
    ],
)
def test_many_equal_values_are_not_a_collision(test):
    start = time.time()
    assert _mbpp(REPEATS, test) == 1.0
    assert time.time() - start < 5


def test_distinct_colliding_values_are_still_refused_and_equal_ones_are_not():
    from safetune.rewards import _code_grader

    assert not _code_grader._too_many_alike([0] * 9000)
    assert not _code_grader._too_many_alike([1] * 100 + [1.0] * 100 + [True] * 100)
    assert _code_grader._too_many_alike(COLLIDE)
    assert _code_grader._too_many_alike(COLLIDE + COLLIDE)
    for test, named in (
        ("assert len(set(f())) == 1", dict(f=COLLIDE)),
        ("assert len(frozenset(f())) == 1", dict(f=COLLIDE)),
        ("assert len(dict(f())) == 1", dict(f=PAIRS)),
        ("assert len(set(f()).union(g())) == 1", dict(f=[1], g=COLLIDE)),
        ("assert len(set(f()) | set(g())) == 1", dict(f=[1], g=COLLIDE)),
    ):
        with pytest.raises(_code_grader._Fail):
            _evaluate(test, **named)


def test_imported_helper_before_the_function_does_not_hijack_the_call():
    from safetune.rewards._code_grader import grade_io

    code = "from functools import lru_cache\ndef add(a, b):\n    return a + b"
    assert grade_io(code, {"input": [2, 3], "expected_output": 5}, 5.0)
