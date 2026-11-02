"""Grade model-written code by the values it returns.

The candidate runs in a child process. The child sends back plain data, and the
trainer does the comparison, so printed text, the exit code and a custom
``__eq__`` can't earn a pass. A literal expected value stays in the trainer and
never reaches the child.

This limits what a completion can score. It isn't a sandbox: the child can still
read the user's files, use the network and signal other processes, the trainer
included. A determined candidate can also find the open result file and write
values into it: that needs the expected value to score, and only plain data
crosses, but it can pass a test that wants a constant such as ``True``.
"""

import ast
import json
import logging
import math
import operator
import os
import signal
import subprocess
import sys
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_MAX_RESULT_BYTES = 16 * 1024 * 1024
# Ints that hash alike make building a set or dict quadratic: 10,000 colliding ints cost about
# a second, 100,000 about half a minute. So the trainer-side decode is bounded by element
# count too, not only by file size.
_MAX_DECODED_ELEMENTS = 10_000
_MAX_MEMORY_BYTES = 4 * 1024 * 1024 * 1024

# Runs in the child. It encodes exact builtin types only, so a subclass or an
# object with its own __eq__ can't cross. The candidate can replace anything
# here, but all that gets it is a value of its own choosing.
_HARNESS = r"""
import inspect, json, os, sys
try:
    import resource
    for limit, value in ((resource.RLIMIT_AS, int(sys.argv[3])), (resource.RLIMIT_FSIZE, int(sys.argv[4]))):
        resource.setrlimit(limit, (value, value))
except (ImportError, ValueError, OSError):
    pass  # not every platform enforces these
with open(sys.argv[1]) as f:
    job = json.load(f)
os.unlink(sys.argv[1])
result = open(sys.argv[2], "w")
sys.argv = ["candidate"]
def enc(x):
    t = type(x)
    if x is None or t in (bool, int, str): return x
    if t is float: return {"f": repr(x)}
    if t in (list, tuple, set, frozenset): return {t.__name__: [enc(i) for i in x]}
    if t is dict: return {"dict": [[enc(k), enc(v)] for k, v in x.items()]}
    raise TypeError(t)
ns = {"__name__": "__main__"}
exec(compile(job["code"], "<candidate>", "exec"), ns)
values = [enc(eval(e, ns)) for e in job["exprs"]]
if job["has_call"]:
    fs = [v for k, v in ns.items() if not k.startswith("_") and inspect.isfunction(v)]
    fs = fs or [v for k, v in ns.items() if not k.startswith("_") and callable(v)]
    inp = job["call"]
    values.append(enc(fs[0](*inp) if type(inp) in (list, tuple) else fs[0](inp)))
json.dump(values, result)
result.close()
"""

_CONTAINERS = {"list": list, "tuple": tuple, "set": set, "frozenset": frozenset}

_COMPARE = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}


def _decode(value: Any, budget: List[int]) -> Any:
    budget[0] -= 1
    if budget[0] < 0:
        raise ValueError("result has too many elements")
    if isinstance(value, dict):
        (kind, payload), = value.items()
        if kind == "f":
            return float(payload)
        if kind == "dict":
            return {_decode(k, budget): _decode(v, budget) for k, v in payload}
        return _CONTAINERS[kind](_decode(i, budget) for i in payload)
    return value


_NO_CALL = object()


def run_candidate(
    code: str,
    timeout: float,
    exprs: Tuple[str, ...] = (),
    call: Any = _NO_CALL,
) -> Optional[List[Any]]:
    """Run ``code`` in a child and return the values it produced.

    Returns ``None`` unless the child got through the candidate and every
    expression, which covers a crash, a timeout, ``sys.exit``, ``os._exit`` and a
    value that isn't plain data. ``call`` holds the arguments for the first public
    function the candidate defines. Nothing the child does should raise here.
    """
    has_call = call is not _NO_CALL
    expected = len(exprs) + has_call
    # Two directories, so the working directory doesn't list the result file.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp, \
            tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as work:
        harness, job, result = (os.path.join(tmp, n) for n in ("h.py", "job.json", "out.json"))
        proc = None
        try:
            with open(harness, "w") as f:
                f.write(_HARNESS)
            with open(job, "w") as f:
                json.dump(
                    {"code": code, "exprs": list(exprs), "has_call": has_call, "call": None if not has_call else call}, f
                )
            proc = subprocess.Popen(
                [sys.executable, "-I", harness, job, result, str(_MAX_MEMORY_BYTES), str(_MAX_RESULT_BYTES)],
                cwd=work,
                env={"PATH": os.environ.get("PATH", ""), "HOME": work},
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            proc.wait(timeout=timeout)
            if os.path.getsize(result) > _MAX_RESULT_BYTES:
                return None
            with open(result) as f:
                raw = json.load(f)
            if not isinstance(raw, list) or len(raw) != expected:
                return None
            budget = [_MAX_DECODED_ELEMENTS]
            return [_decode(v, budget) for v in raw]
        # The result file is the child's to write, so anything can be in it.
        except (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError, MemoryError,
                subprocess.SubprocessError):
            return None
        finally:
            if proc is not None:
                # ponytail: the process group misses a child that calls setsid.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (OSError, AttributeError):
                    proc.kill()
                proc.wait()


def _expression(node: ast.expr) -> Tuple[Optional[str], Any]:
    """Return (source to run in the child, None), or (None, literal kept here)."""
    try:
        return None, ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return ast.unparse(node), None


def _plan_assert(assertion: str) -> Optional[Tuple[List[Tuple[Optional[str], Any]], Callable[..., bool]]]:
    """Split an assert into its operands and the check the trainer applies."""
    try:
        body = ast.parse(assertion.strip()).body
    except (SyntaxError, ValueError):
        return None
    if len(body) != 1 or not isinstance(body[0], ast.Assert):
        return None
    test = body[0].test
    if isinstance(test, ast.Compare) and len(test.ops) == 1 and type(test.ops[0]) in _COMPARE:
        return [_expression(test.left), _expression(test.comparators[0])], _COMPARE[type(test.ops[0])]
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return [_expression(test.operand)], operator.not_
    if isinstance(test, ast.Call) and ast.unparse(test.func) == "math.isclose" and len(test.args) == 2:
        try:
            kwargs = {k.arg: ast.literal_eval(k.value) for k in test.keywords}
        except (ValueError, TypeError, SyntaxError):
            return None
        return [_expression(a) for a in test.args], lambda a, b: math.isclose(a, b, **kwargs)
    return [_expression(test)], bool


def grade_assertion(code: str, assertion: str, timeout: float) -> bool:
    """Whether ``assertion`` holds for values produced by ``code``."""
    plan = _plan_assert(assertion)
    if plan is None:
        logger.warning("Can't grade test, counting it as failed: %.80s", assertion)
        return False
    operands, check = plan
    values = run_candidate(code, timeout, exprs=tuple(src for src, _ in operands if src is not None))
    if values is None:
        return False
    produced = iter(values)
    args = [next(produced) if src is not None else literal for src, literal in operands]
    try:
        return bool(check(*args))
    except (TypeError, ValueError, RecursionError, OverflowError):
        return False


def grade_io(code: str, test_case: Dict[str, Any], timeout: float) -> bool:
    """Whether the candidate's function maps the test's input to its output."""
    values = run_candidate(code, timeout, call=test_case.get("input"))
    if values is None:
        return False
    try:
        return bool(values[-1] == test_case.get("expected_output"))
    except RecursionError:
        return False
