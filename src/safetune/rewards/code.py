"""
Code-related reward functions for SafeTune.
"""

from typing import List, Optional, Dict, Any
import logging
import re
import subprocess
import tempfile
import os
from .base import RewardFunction, RewardConfig

logger = logging.getLogger(__name__)

class CodeSyntaxReward(RewardFunction):
    """Reward for code syntax correctness."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        code_blocks = re.findall(r'```[\s\S]*?```', text)
        if not code_blocks: return 0.0
        
        correct_count = 0
        for block in code_blocks:
            code = block.strip('```').strip()
            if self._check_syntax(code):
                correct_count += 1
        return correct_count / len(code_blocks)
    
    def _check_syntax(self, code: str) -> bool:
        try:
            compile(code, '<string>', 'exec')
            return True
        except (SyntaxError, IndentationError):
            return False


class CodeExecutionReward(RewardFunction):
    """Reward for code execution success."""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self.timeout = config.params.get('timeout', 5.0)
        self.test_cases = config.params.get('test_cases', [])

    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        test_cases = kwargs.get('test_cases', self.test_cases)
        code = self._extract_code(text)
        if not code: return 0.0
        
        if test_cases:
            return self._validate_with_test_cases(code, test_cases)
        return 1.0 if self._execute_code_safely(code) else 0.0

    def _extract_code(self, text: str) -> str:
        blocks = re.findall(r'```python\s*(.*?)```', text, re.DOTALL) or re.findall(r'```\s*(.*?)```', text, re.DOTALL)
        if blocks:
            for b in blocks:
                if 'def ' in b: return b.strip()
            return blocks[0].strip()
        lines = text.split('\n')
        code_lines = []
        in_fn = False
        for l in lines:
            if l.strip().startswith('def '): in_fn = True
            if in_fn: code_lines.append(l)
        return '\n'.join(code_lines) if code_lines else text.strip()

    def _execute_code_safely(self, code: str) -> bool:
        # SECURITY: run model-generated code OUT-OF-PROCESS with a timeout, NOT
        # via an in-process exec() with full builtins (which is an
        # arbitrary-code-execution vector in the reward process — it could read
        # files, open sockets, etc.). This matches the subprocess isolation the
        # test-runner methods below already use.
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            result = subprocess.run(
                ['python', temp_file], capture_output=True, text=True,
                timeout=self.timeout,
            )
            return result.returncode == 0
        except Exception:
            return False
        finally:
            if temp_file:
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass

    def _validate_with_test_cases(self, code: str, test_cases: List[Any]) -> float:
        if not test_cases: return 0.0
        passed = 0
        for tc in test_cases:
            try:
                if isinstance(tc, str) and tc.strip().startswith('assert'):
                    if self._run_assertion_test(code, tc): passed += 1
                elif isinstance(tc, dict):
                    if self._run_dict_test(code, tc): passed += 1
            except Exception:
                continue
        return passed / len(test_cases)

    def _run_assertion_test(self, code: str, assertion: str) -> bool:
        test_code = f'{code}\n\n{assertion}\nprint("PASS")'
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_code)
                temp_file = f.name
            result = subprocess.run(['python', temp_file], capture_output=True, text=True, timeout=self.timeout)
            os.unlink(temp_file)
            return result.returncode == 0 and "PASS" in result.stdout
        except Exception:
            try: os.unlink(temp_file)
            except Exception: pass
            return False

    def _run_dict_test(self, code: str, test_case: Dict[str, Any]) -> bool:
        test_input = test_case.get('input')
        expected_output = test_case.get('expected_output')
        test_code = f'''{code}
functions = {{k: v for k, v in globals().items() if callable(v) and not k.startswith('_')}}
main_func = list(functions.values())[0] if functions else None
if not main_func: exit(1)
inp = {repr(test_input)}
exp = {repr(expected_output)}
act = main_func(*inp) if isinstance(inp, (list, tuple)) else main_func(inp)
if act == exp: print("PASS")
'''
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_code)
                temp_file = f.name
            result = subprocess.run(['python', temp_file], capture_output=True, text=True, timeout=self.timeout)
            os.unlink(temp_file)
            return result.returncode == 0 and "PASS" in result.stdout
        except Exception:
            try: os.unlink(temp_file)
            except Exception: pass
            return False


class CodeCompletenessReward(RewardFunction):
    """Reward for code completeness."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        code_blocks = re.findall(r'```[\s\S]*?```', text)
        if not code_blocks: return 0.0
        scores = []
        for block in code_blocks:
            code = block.strip('```').strip()
            score = 0.0
            if re.search(r'def\s+\w+\s*\(', code): score += 0.3
            if re.search(r'import\s+\w+', code): score += 0.2
            if re.search(r'return\s+', code): score += 0.3
            if len(code.split('\n')) > 3: score += 0.2
            scores.append(min(1.0, score))
        return sum(scores) / len(scores)


class CodeQualityReward(RewardFunction):
    """Reward function for code quality assessment."""
    
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        code = self._extract_code(text)
        if not code: return 0.0
        score = 0.0
        if '"""' in code or "'''" in code: score += 0.3
        if re.search(r'#.*', code): score += 0.2
        lines = code.split('\n')
        indents = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
        if len(set(indents)) > 1: score += 0.2
        if len(lines) > 5: score += 0.3
        return min(1.0, score)

    def _extract_code(self, text: str) -> str:
        blocks = re.findall(r'```[\s\S]*?```', text)
        return blocks[0].strip('```').strip() if blocks else text.strip()


class CodeCorrectnessReward(RewardFunction):
    """Alias for CodeExecutionReward for backward compatibility."""
    def __init__(self, config: RewardConfig):
        self.exec_reward = CodeExecutionReward(config)
        super().__init__(config)
    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        return self.exec_reward.compute(text, reference, **kwargs)


class MBPPReward(RewardFunction):
    """Reward for MBPP-style code generation.

    MBPP problems ship with three Python ``assert`` statements as the test
    suite. The reward extracts a code block from ``text``, executes it together
    with the asserts in a sandboxed subprocess, and returns the fraction of
    asserts that pass. Tests are taken from ``kwargs['test_list']`` (a list of
    assert strings) or from ``config.params['test_list']``.
    """

    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self.timeout = float(config.params.get("timeout", 5.0))
        self.test_list: List[str] = list(config.params.get("test_list", []))

    def compute(self, text: str, reference: Optional[str] = None, **kwargs) -> float:
        tests: List[str] = list(kwargs.get("test_list", self.test_list))
        code = self._extract_code(text)
        if not code or not tests:
            return 0.0
        passed = 0
        for test in tests:
            program = f"{code}\n{test}\n"
            try:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as fh:
                    fh.write(program)
                    fh.flush()
                    path = fh.name
                result = subprocess.run(
                    ["python", path],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                if result.returncode == 0:
                    passed += 1
            except (subprocess.TimeoutExpired, OSError):
                pass
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass
        return passed / len(tests)

    def _extract_code(self, text: str) -> str:
        blocks = re.findall(r"```(?:python)?\n?([\s\S]*?)```", text)
        return blocks[0].strip() if blocks else text.strip()
