"""Adapter that runs the pytest-style test modules under unittest too.

Owner: Ali. Mahya wrote tests/test_comments_cleaning.py and
tests/test_build_recommendation_splits.py as plain `def test_*()` functions
with bare asserts, which is valid pytest but invisible to
`python -m unittest discover -s tests` -- the command the README documents.
Before this adapter existed both files silently contributed zero tests.

Rather than rewriting her files (they are hers, and they run correctly under
pytest), this module imports them and wraps each `test_*` function in a
generated TestCase. Both runners now execute exactly the same assertions:

    python -m unittest discover -s tests     # via this adapter
    pytest                                   # directly, unchanged

Neither module uses fixtures, parametrize, or pytest.raises, so a plain call
is a faithful execution of each test.
"""

from __future__ import annotations

import importlib
import unittest
from typing import Any, Callable

WRAPPED_MODULES = (
    "tests.test_comments_cleaning",
    "tests.test_build_recommendation_splits",
)


def _test_functions(module: Any) -> list[tuple[str, Callable[[], None]]]:
    """Every zero-argument `test_*` callable defined in that module.

    Functions imported from elsewhere are skipped via __module__, so a helper
    pulled in by name cannot be mistaken for one of the module's own tests.
    """
    found = []
    for name in sorted(dir(module)):
        if not name.startswith("test_"):
            continue
        candidate = getattr(module, name)
        if not callable(candidate):
            continue
        if getattr(candidate, "__module__", None) != module.__name__:
            continue
        found.append((name, candidate))
    return found


def _build_case(module_name: str) -> type[unittest.TestCase]:
    module = importlib.import_module(module_name)
    functions = _test_functions(module)
    if not functions:
        raise AssertionError(
            f"{module_name} exposes no test_* functions; the adapter would "
            "silently pass and hide the very problem it exists to fix"
        )

    namespace: dict[str, Any] = {
        "__doc__": f"pytest-style tests from {module_name}, run under unittest."
    }
    for name, function in functions:
        # default arg binds the current function rather than the loop variable
        def method(self: unittest.TestCase, _fn: Callable[[], None] = function) -> None:
            _fn()

        method.__name__ = name
        method.__doc__ = function.__doc__
        namespace[name] = method

    short = module_name.rsplit(".", 1)[-1]
    class_name = "".join(part.title() for part in short.split("_")) + "Adapted"
    return type(class_name, (unittest.TestCase,), namespace)


CommentsCleaningAdapted = _build_case(WRAPPED_MODULES[0])
RecommendationSplitsAdapted = _build_case(WRAPPED_MODULES[1])


if __name__ == "__main__":
    unittest.main()
