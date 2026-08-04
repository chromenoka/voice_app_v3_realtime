"""Run token-budget tests without third-party test dependencies."""

from __future__ import annotations

import importlib.util
from pathlib import Path


test_file = Path(__file__).parent / "tests" / "test_token_budget.py"
spec = importlib.util.spec_from_file_location("token_budget_tests", test_file)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

tests = [
    getattr(module, name)
    for name in dir(module)
    if name.startswith("test_")
]
for test in tests:
    test()

print(f"{len(tests)} token tests: PASS")
