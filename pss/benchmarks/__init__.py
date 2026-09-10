"""Benchmark tasks for PSS evaluation."""

from pss.benchmarks.game_of_24 import GAME_OF_24_PROBLEMS, verify_24_solution
from pss.benchmarks.brainstorm import BRAINSTORM_PROBLEMS, count_unique_ideas
from pss.benchmarks.bugfind import BUGFIND_PROBLEMS, check_bug_found

__all__ = [
    "GAME_OF_24_PROBLEMS",
    "verify_24_solution",
    "BRAINSTORM_PROBLEMS",
    "count_unique_ideas",
    "BUGFIND_PROBLEMS",
    "check_bug_found",
]
