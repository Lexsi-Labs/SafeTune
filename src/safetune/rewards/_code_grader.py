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

Only the candidate's own calls (``f(3)``, ``mod.f(3)``, a constant it defines) run in
the child, as separate expressions whose results come back as plain data. The
trainer evaluates the rest of the assert, on plain data, with a whitelist of
builtins, ``math`` functions and methods: ``and``, ``or``, ``not``, chained
comparisons, arithmetic, ``len``, ``sorted``, ``isinstance`` and the like all run
here, so a candidate can't change them by redefining names or patching
``builtins``. An assert that uses anything else fails with a warning. A task
whose required function shares a builtin's name can't be graded either: its
tests fail, with a warning.

What remains: the candidate can still return a forged plain value, so a test
whose expected value it can guess is not protected.
"""

import ast
import builtins
import json
import logging
import math
import operator
import os
import signal
import subprocess
import sys
import tempfile
from collections import Counter
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
def _main():
    import inspect, json, os, sys
    # Only the candidate's own calls are evaluated here; the trainer evaluates the rest of a test.
    # Names used after the candidate runs are bound first, so patching builtins doesn't break the encoding.
    _eval, _exec, _compile, _open, _dump = eval, exec, compile, open, json.dump
    _type, _bool, _int, _str, _float, _repr = type, bool, int, str, float, repr
    _list, _tuple, _set, _frozenset, _dict = list, tuple, set, frozenset, dict
    _Exception, _TypeError, _isfunction, _callable = Exception, TypeError, inspect.isfunction, callable
    try:
        import resource
        for limit, value in ((resource.RLIMIT_AS, int(sys.argv[3])), (resource.RLIMIT_FSIZE, int(sys.argv[4]))):
            resource.setrlimit(limit, (value, value))
    except (ImportError, ValueError, OSError):
        pass  # not every platform enforces these
    with _open(sys.argv[1]) as f:
        job = json.load(f)
    os.unlink(sys.argv[1])
    result = _open(sys.argv[2], "w")
    sys.argv = ["candidate"]

    def enc(x):
        t = _type(x)
        if x is None or t is _bool or t is _int or t is _str: return x
        if t is _float: return {"f": _repr(x)}
        if t is _list or t is _tuple or t is _set or t is _frozenset: return {t.__name__: [enc(i) for i in x]}
        if t is _dict: return {"dict": [[enc(k), enc(v)] for k, v in x.items()]}
        raise _TypeError(t)

    ns = {"__name__": "__main__"}
    _exec(_compile(job["code"], "<candidate>", "exec"), ns)
    values = []
    for e in job["exprs"]:
        try:
            values.append(enc(_eval(e, ns)))
        except _Exception:
            values.append({"e": 0})
    if job["has_call"]:
        fs = [v for k, v in ns.items() if k[:1] != "_" and _isfunction(v)]
        fs = fs or [v for k, v in ns.items() if k[:1] != "_" and _callable(v)]
        inp = job["call"]
        values.append(enc(fs[0](*inp) if _type(inp) in (_list, _tuple) else fs[0](inp)))
    _dump(values, result)
    result.close()

_main()
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


class _Failed:
    """A candidate call that raised or returned something that isn't plain data."""


_FAILED = _Failed()


def _hashed(items: List[Any]) -> List[Any]:
    if items and max(Counter(map(hash, items)).values()) > _MAX_SAME_HASH:
        raise ValueError("too many colliding hashes")
    return items


def _decode(value: Any, budget: List[int]) -> Any:
    budget[0] -= 1
    if budget[0] < 0:
        raise ValueError("result has too many elements")
    if isinstance(value, dict):
        (kind, payload), = value.items()
        if kind == "e":
            return _FAILED
        if kind == "f":
            return float(payload)
        if kind == "dict":
            pairs = [(_decode(k, budget), _decode(v, budget)) for k, v in payload]
            _hashed([k for k, _ in pairs])
            return dict(pairs)
        items = [_decode(i, budget) for i in payload]
        if kind in ("set", "frozenset"):
            _hashed(items)
        return _CONTAINERS[kind](items)
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


class _Fail(Exception):
    """A part of the test could not be evaluated, which fails it."""


class _Undecomposable(Exception):
    pass


_FUNCS = {
    n: getattr(builtins, n)
    for n in ("len sum min max abs sorted set frozenset list tuple dict int float str bool isinstance "
              "any all round reversed zip enumerate divmod").split()
}
_MATH = {n: getattr(math, n) for n in "isclose sqrt floor ceil fabs isnan isinf isfinite pi e inf nan".split()}
_METHODS = frozenset(
    "count index keys values items get startswith endswith lower upper strip lstrip rstrip split join "
    "isdigit isalpha isalnum isupper islower find issubset issuperset union intersection "
    "difference symmetric_difference copy".split()
)
_PLAIN = (bool, int, float, str, list, tuple, set, frozenset, dict, type(None))
_BUILTIN_NAMES = frozenset(dir(builtins))
# Every container, string and repeat the trainer builds or reads while evaluating a test is held to
# this many elements, and an assert may visit this many nodes, so a forged value or a big literal
# can't make the trainer allocate or loop for long.
_MAX_REPEAT = 10**5
_MAX_NODES = 10_000
_MAX_SAME_HASH = 64


class _Vals(list):
    """The child's values for one evaluation, plus the count of nodes visited."""

    visited = 0
    touched = 0


def _small(v: "_Vals", *things: Any) -> None:
    """Refuse a big container or string, a total past the cap for this assert, and hash collisions."""
    for t in things:
        if type(t) in (str, list, tuple, set, frozenset, dict):
            if len(t) > _MAX_REPEAT:
                raise _Fail
            v.touched += len(t)
            # Values that hash alike make building a set or dict quadratic.
            if type(t) is not str and t and max(Counter(map(hash, t)).values()) > _MAX_SAME_HASH:
                raise _Fail
    if v.touched > 10 * _MAX_REPEAT:
        raise _Fail

Ev = Callable[[List[Any]], Any]


def _is_chain(node: ast.expr) -> bool:
    while isinstance(node, ast.Attribute):
        node = node.value
    return isinstance(node, ast.Name)


def _candidate(node: ast.expr) -> bool:
    """Whether ``node`` is the candidate's: a name it defines, or a call to one (``f(1)``, ``mod.f(1)``)."""
    if isinstance(node, ast.Call):
        return _is_chain(node.func) and _candidate(node.func)
    if isinstance(node, ast.Attribute):
        return _is_chain(node) and _candidate(node.value)
    if isinstance(node, ast.Name):
        if node.id in _FUNCS or node.id == "math":
            return False
        if node.id in _BUILTIN_NAMES:
            raise _Undecomposable(node.id)
        return True
    return False


def _binop(op: ast.operator, a: Any, b: Any, v: "_Vals") -> Any:
    _small(v, a, b)
    if isinstance(op, ast.Mult):
        for seq, n in ((a, b), (b, a)):
            if isinstance(seq, (str, list, tuple)) and type(n) in (int, bool):
                if len(seq) * abs(n) > _MAX_REPEAT:
                    raise _Fail
    elif isinstance(op, ast.Add):
        if isinstance(a, (str, list, tuple)) and isinstance(b, (str, list, tuple)) and len(a) + len(b) > _MAX_REPEAT:
            raise _Fail
    elif isinstance(op, ast.Pow):
        if type(a) is int and type(b) is int and (abs(b) > 1000 or a.bit_length() * abs(b) > 10_000):
            raise _Fail
    elif isinstance(op, ast.Mod):
        if type(a) not in (int, float) or type(b) not in (int, float):  # no printf-style formatting
            raise _Fail
    return _BINOPS[type(op)](a, b)


_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.BitAnd: operator.and_, ast.BitOr: operator.or_, ast.BitXor: operator.xor,
}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Invert: operator.invert, ast.Not: operator.not_}


def _compile(node: ast.expr, calls: List[str]) -> Ev:
    run = _compile_node(node, calls)

    def counted(v: Any) -> Any:
        v.visited += 1
        if v.visited > _MAX_NODES:
            raise _Fail
        return run(v)

    return counted


def _compile_node(node: ast.expr, calls: List[str]) -> Ev:
    """Turn a test expression into a function of the child's values.

    Each maximal candidate call becomes one expression for the child; everything
    else is evaluated here, on plain data, by this whitelist.
    """
    try:
        literal = ast.literal_eval(node)
        return lambda v: literal
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        pass
    if _candidate(node):
        calls.append(ast.unparse(node))
        i = len(calls) - 1

        def get(v: List[Any]) -> Any:
            if v[i] is _FAILED:
                raise _Fail
            return v[i]

        return get
    if isinstance(node, ast.BoolOp):
        subs = [_compile(x, calls) for x in node.values]
        if isinstance(node.op, ast.And):
            def conj(v: List[Any]) -> Any:
                r: Any = True
                for s in subs:
                    r = s(v)
                    if not r:
                        break
                return r

            return conj

        def disj(v: List[Any]) -> Any:  # an `or` holds if any side holds, even if another raised
            failed, r = False, False
            for s in subs:
                try:
                    r = s(v)
                except (_Fail, TypeError, ValueError, RecursionError, OverflowError, ZeroDivisionError):
                    failed = True
                    continue
                if r:
                    return r
            if failed:
                raise _Fail
            return r

        return disj
    if isinstance(node, ast.UnaryOp):
        x, op = _compile(node.operand, calls), _UNARY[type(node.op)]
        return lambda v: op(x(v))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        a, b = _compile(node.left, calls), _compile(node.right, calls)
        return lambda v: _binop(node.op, a(v), b(v), v)
    if isinstance(node, ast.Compare):
        ops = [_COMPARE[type(op)] for op in node.ops]
        parts = [_compile(x, calls) for x in [node.left, *node.comparators]]

        def compare(v: List[Any]) -> bool:
            left = parts[0](v)
            for op, part in zip(ops, parts[1:]):
                right = part(v)
                if not op(left, right):
                    return False
                left = right
            return True

        return compare
    if isinstance(node, ast.IfExp):
        t, a, b = (_compile(x, calls) for x in (node.test, node.body, node.orelse))
        return lambda v: a(v) if t(v) else b(v)
    if isinstance(node, ast.Subscript):
        value, index = _compile(node.value, calls), _compile_slice(node.slice, calls)
        return lambda v: value(v)[index(v)]
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)) and not any(isinstance(x, ast.Starred) for x in node.elts):
        items, make = [_compile(x, calls) for x in node.elts], {ast.Tuple: tuple, ast.List: list, ast.Set: set}[type(node)]
        return lambda v: make(i(v) for i in items)
    if isinstance(node, ast.Dict) and all(k is not None for k in node.keys):
        pairs = [(_compile(k, calls), _compile(x, calls)) for k, x in zip(node.keys, node.values)]
        return lambda v: {k(v): x(v) for k, x in pairs}
    if isinstance(node, ast.Name) and node.id in _FUNCS:
        thing = _FUNCS[node.id]
        return lambda v: thing
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "math" \
            and node.attr in _MATH:
        thing = _MATH[node.attr]
        return lambda v: thing
    if isinstance(node, ast.Call) and not any(isinstance(a, ast.Starred) for a in node.args) \
            and all(k.arg for k in node.keywords):
        args = [_compile(a, calls) for a in node.args]
        kwargs = [(k.arg, _compile(k.value, calls)) for k in node.keywords]
        func = node.func
        if isinstance(func, ast.Name) and func.id in _FUNCS:
            def call(v: List[Any]) -> Any:
                values = [a(v) for a in args]
                if func.id != "len":
                    _small(v, *values)
                return _FUNCS[func.id](*values, **{k: a(v) for k, a in kwargs})

            return call
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "math" \
                and func.attr in _MATH and callable(_MATH[func.attr]):
            return lambda v: _MATH[func.attr](*(a(v) for a in args), **{k: a(v) for k, a in kwargs})
        if isinstance(func, ast.Attribute) and func.attr in _METHODS:
            receiver = _compile(func.value, calls)

            def method(v: List[Any]) -> Any:
                r = receiver(v)
                if type(r) not in _PLAIN:
                    raise _Fail
                values = [a(v) for a in args]
                _small(v, r, *values)
                if func.attr == "join" and (len(values) != 1 or not hasattr(values[0], "__len__")
                                            or len(r) * len(values[0]) > _MAX_REPEAT):
                    raise _Fail
                return getattr(r, func.attr)(*values, **{k: a(v) for k, a in kwargs})

            return method
    raise _Undecomposable(ast.unparse(node))


def _compile_slice(node: ast.expr, calls: List[str]) -> Ev:
    if isinstance(node, ast.Slice):
        parts = [_compile(x, calls) if x is not None else (lambda v: None) for x in (node.lower, node.upper, node.step)]
        return lambda v: slice(*(p(v) for p in parts))
    return _compile(node, calls)


def _plan_assert(assertion: str) -> Optional[Tuple[List[str], Callable[[List[Any]], bool]]]:
    """Split an assert into the candidate calls the child runs and the check the trainer applies."""
    try:
        body = ast.parse(assertion.strip()).body
        if len(body) != 1 or not isinstance(body[0], ast.Assert):
            return None
        calls: List[str] = []
        ev = _compile(body[0].test, calls)
    except (SyntaxError, ValueError, MemoryError, RecursionError, _Undecomposable):
        return None
    return calls, lambda values: bool(ev(_Vals(values)))


def _bound_names(code: str) -> set:
    """Names the candidate binds at module level by ``def``, ``class`` or assignment."""
    try:
        body = ast.parse(code).body
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return set()
    names = set()
    for stmt in body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(stmt.name)
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            names.update(n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name))
    return names


def grade_assertion(code: str, assertion: str, timeout: float) -> bool:
    """Whether ``assertion`` holds for values produced by ``code``."""
    plan = _plan_assert(assertion)
    if plan is None:
        logger.warning("Can't grade test, counting it as failed: %.80s", assertion)
        return False
    calls, check = plan
    used = {n.id for n in ast.walk(ast.parse(assertion.strip())) if isinstance(n, ast.Name)}
    if used & (set(_FUNCS) | {"math"}) & _bound_names(code):
        logger.warning("Test uses a builtin's name that the candidate defines, counting it as failed: %.80s", assertion)
        return False
    values = run_candidate(code, timeout, exprs=tuple(calls))
    if values is None:
        return False
    try:
        return check(values)
    except (_Fail, TypeError, ValueError, RecursionError, OverflowError, ZeroDivisionError, MemoryError, KeyError, IndexError, AttributeError):
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
