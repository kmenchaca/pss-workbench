"""HumanEval benchmark for PSS evaluation.

HumanEval is a benchmark of 164 hand-written Python programming problems.
Each problem includes a function signature, docstring, and test cases.

Reference: Chen et al., "Evaluating Large Language Models Trained on Code" (2021)

Cost estimate: ~$0.05 for 10 problems with PSS (Llama 8B via OpenRouter)

Note: This is a simplified version with a subset of problems. For the full
benchmark, use the official HumanEval dataset.
"""

from __future__ import annotations

import re
import sys
import traceback
from dataclasses import dataclass, field
from io import StringIO
from typing import Any


@dataclass
class HumanEvalProblem:
    """A HumanEval programming problem."""

    task_id: str
    prompt: str  # Function signature + docstring
    canonical_solution: str  # Reference solution
    test_cases: str  # Test code to run
    entry_point: str  # Function name to test
    difficulty: str  # "easy", "medium", "hard"


# Sample problems from HumanEval
HUMANEVAL_PROBLEMS: list[HumanEvalProblem] = [
    # Easy problems
    HumanEvalProblem(
        task_id="HumanEval/0",
        prompt='''def has_close_elements(numbers: list[float], threshold: float) -> bool:
    """Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
''',
        canonical_solution='''    for i, n1 in enumerate(numbers):
        for n2 in numbers[i+1:]:
            if abs(n1 - n2) < threshold:
                return True
    return False''',
        test_cases='''
assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False
assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True
assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
''',
        entry_point="has_close_elements",
        difficulty="easy",
    ),
    HumanEvalProblem(
        task_id="HumanEval/1",
        prompt='''def separate_paren_groups(paren_string: str) -> list[str]:
    """Input to this function is a string containing multiple groups of nested parentheses.
    Your goal is to separate those group into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other.
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """
''',
        canonical_solution='''    result = []
    current_string = []
    current_depth = 0
    for c in paren_string:
        if c == '(':
            current_depth += 1
            current_string.append(c)
        elif c == ')':
            current_depth -= 1
            current_string.append(c)
            if current_depth == 0:
                result.append(''.join(current_string))
                current_string = []
    return result''',
        test_cases='''
assert separate_paren_groups('(()()) ((())) () ((())()())') == ['(()())', '((()))', '()', '((())()())']
assert separate_paren_groups('() (()) ((())) (((())))') == ['()', '(())', '((()))', '(((())))']
assert separate_paren_groups('(()(()))') == ['(()(()))']
''',
        entry_point="separate_paren_groups",
        difficulty="easy",
    ),
    HumanEvalProblem(
        task_id="HumanEval/2",
        prompt='''def truncate_number(number: float) -> float:
    """Given a positive floating point number, it can be decomposed into
    an integer part (largest integer smaller than given number) and decimals
    (leftover part always smaller than 1).
    Return the decimal part of the number.
    >>> truncate_number(3.5)
    0.5
    """
''',
        canonical_solution='''    return number % 1.0''',
        test_cases='''
assert truncate_number(3.5) == 0.5
assert abs(truncate_number(1.25) - 0.25) < 1e-6
assert abs(truncate_number(123.0) - 0.0) < 1e-6
''',
        entry_point="truncate_number",
        difficulty="easy",
    ),
    HumanEvalProblem(
        task_id="HumanEval/4",
        prompt='''from typing import List

def mean_absolute_deviation(numbers: List[float]) -> float:
    """For a given list of input numbers, calculate Mean Absolute Deviation
    around the mean of this dataset.
    Mean Absolute Deviation is the average absolute difference between each
    element and a centerpoint (mean in this case):
    MAD = average | x - x_mean |
    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])
    1.0
    """
''',
        canonical_solution='''    mean = sum(numbers) / len(numbers)
    return sum(abs(x - mean) for x in numbers) / len(numbers)''',
        test_cases='''
assert abs(mean_absolute_deviation([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-6
assert abs(mean_absolute_deviation([1.0, 2.0, 3.0, 4.0, 5.0]) - 1.2) < 1e-6
''',
        entry_point="mean_absolute_deviation",
        difficulty="easy",
    ),
    # Medium problems
    HumanEvalProblem(
        task_id="HumanEval/6",
        prompt='''from typing import List

def parse_nested_parens(paren_string: str) -> List[int]:
    """Input to this function is a string represented multiple groups for nested parentheses separated by spaces.
    For each of the group, output the deepest level of nesting of parentheses.
    E.g. (()()) has maximum two levels of nesting while ((())) has three.
    >>> parse_nested_parens('(()()) ((())) () ((())()())')
    [2, 3, 1, 3]
    """
''',
        canonical_solution='''    def max_depth(s):
        depth = max_depth = 0
        for c in s:
            if c == '(':
                depth += 1
                max_depth = max(max_depth, depth)
            elif c == ')':
                depth -= 1
        return max_depth
    return [max_depth(g) for g in paren_string.split() if g]''',
        test_cases='''
assert parse_nested_parens('(()()) ((())) () ((())()())') == [2, 3, 1, 3]
assert parse_nested_parens('() (()) ((())) (((())))') == [1, 2, 3, 4]
assert parse_nested_parens('(()(()))') == [3]
''',
        entry_point="parse_nested_parens",
        difficulty="medium",
    ),
    HumanEvalProblem(
        task_id="HumanEval/10",
        prompt='''def is_palindrome(string: str) -> bool:
    """Test if given string is a palindrome"""
    return string == string[::-1]

def make_palindrome(string: str) -> str:
    """Find the shortest palindrome that begins with a supplied string.
    Algorithm idea is simple:
    - Find the longest postfix of supplied string that is a palindrome.
    - Append to the end of the string reverse of a string prefix that comes before the palindromic suffix.
    >>> make_palindrome('')
    ''
    >>> make_palindrome('cat')
    'catac'
    >>> make_palindrome('cata')
    'catac'
    """
''',
        canonical_solution='''    if not string:
        return ''
    for i in range(len(string)):
        if is_palindrome(string[i:]):
            return string + string[:i][::-1]
    return string + string[:-1][::-1]''',
        test_cases='''
assert make_palindrome('') == ''
assert make_palindrome('cat') == 'catac'
assert make_palindrome('cata') == 'catac'
assert make_palindrome('a') == 'a'
assert make_palindrome('xyx') == 'xyx'
assert make_palindrome('abcd') == 'abcdcba'
''',
        entry_point="make_palindrome",
        difficulty="medium",
    ),
    HumanEvalProblem(
        task_id="HumanEval/12",
        prompt='''from typing import List, Optional

def longest(strings: List[str]) -> Optional[str]:
    """Out of list of strings, return the longest one. Return the first one in case of multiple
    strings of the same length. Return None in case the input list is empty.
    >>> longest([])

    >>> longest(['a', 'b', 'c'])
    'a'
    >>> longest(['a', 'bb', 'ccc'])
    'ccc'
    """
''',
        canonical_solution='''    if not strings:
        return None
    return max(strings, key=len)''',
        test_cases='''
assert longest([]) == None
assert longest(['a', 'b', 'c']) == 'a'
assert longest(['a', 'bb', 'ccc']) == 'ccc'
assert longest(['aaa', 'bb', 'c']) == 'aaa'
''',
        entry_point="longest",
        difficulty="medium",
    ),
    HumanEvalProblem(
        task_id="HumanEval/14",
        prompt='''from typing import List

def all_prefixes(string: str) -> List[str]:
    """Return list of all prefixes from shortest to longest of the input string
    >>> all_prefixes('abc')
    ['a', 'ab', 'abc']
    """
''',
        canonical_solution='''    return [string[:i+1] for i in range(len(string))]''',
        test_cases='''
assert all_prefixes('abc') == ['a', 'ab', 'abc']
assert all_prefixes('') == []
assert all_prefixes('x') == ['x']
assert all_prefixes('asdf') == ['a', 'as', 'asd', 'asdf']
''',
        entry_point="all_prefixes",
        difficulty="medium",
    ),
    # Hard problems
    HumanEvalProblem(
        task_id="HumanEval/32",
        prompt='''import math

def poly(xs: list, x: float):
    """Evaluates polynomial with coefficients xs at point x.
    return xs[0] + xs[1] * x + xs[2] * x^2 + .... xs[n] * x^n
    """
    return sum([coeff * math.pow(x, i) for i, coeff in enumerate(xs)])

def find_zero(xs: list):
    """xs are coefficients of a polynomial.
    find_zero find x such that poly(xs, x) = 0.
    find_zero returns only one zero point, even if there are many.
    Moreover, find_zero only takes list xs having even number of coefficients
    and largest non zero coefficient as it guarantees a solution.
    >>> round(find_zero([1, 2]), 2)  # f(x) = 1 + 2x
    -0.5
    >>> round(find_zero([-6, 11, -6, 1]), 2)  # (x - 1) * (x - 2) * (x - 3) = -6 + 11x - 6x^2 + x^3
    1.0
    """
''',
        canonical_solution='''    begin, end = -1., 1.
    while poly(xs, begin) * poly(xs, end) > 0:
        begin *= 2.
        end *= 2.
    while end - begin > 1e-10:
        center = (begin + end) / 2.
        if poly(xs, begin) * poly(xs, center) > 0:
            begin = center
        else:
            end = center
    return begin''',
        test_cases='''
assert abs(find_zero([1, 2]) - (-0.5)) < 1e-4
assert abs(find_zero([-6, 11, -6, 1]) - 1.0) < 1e-4 or abs(find_zero([-6, 11, -6, 1]) - 2.0) < 1e-4 or abs(find_zero([-6, 11, -6, 1]) - 3.0) < 1e-4
''',
        entry_point="find_zero",
        difficulty="hard",
    ),
    HumanEvalProblem(
        task_id="HumanEval/37",
        prompt='''def sort_even(l: list):
    """This function takes a list l and returns a list l' such that
    l' is identical to l in the odd indicies, while its values at the even indicies are equal
    to the values of the even indicies of l, but sorted.
    >>> sort_even([1, 2, 3])
    [1, 2, 3]
    >>> sort_even([5, 6, 3, 4])
    [3, 6, 5, 4]
    """
''',
        canonical_solution='''    evens = sorted(l[::2])
    result = []
    for i, x in enumerate(l):
        if i % 2 == 0:
            result.append(evens[i // 2])
        else:
            result.append(x)
    return result''',
        test_cases='''
assert sort_even([1, 2, 3]) == [1, 2, 3]
assert sort_even([5, 6, 3, 4]) == [3, 6, 5, 4]
assert sort_even([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10]) == [-10, 3, -5, 2, -3, 3, 5, 0, 9, 1, 123]
''',
        entry_point="sort_even",
        difficulty="hard",
    ),
    HumanEvalProblem(
        task_id="HumanEval/38",
        prompt='''def encode_cyclic(s: str):
    """returns encoded string by cycling groups of three characters."""
    groups = [s[(3 * i):min((3 * i + 3), len(s))] for i in range((len(s) + 2) // 3)]
    groups = [(group[1:] + group[0]) if len(group) == 3 else group for group in groups]
    return "".join(groups)

def decode_cyclic(s: str):
    """takes as input string encoded with encode_cyclic function. Returns decoded string.
    """
''',
        canonical_solution='''    groups = [s[(3 * i):min((3 * i + 3), len(s))] for i in range((len(s) + 2) // 3)]
    groups = [(group[-1] + group[:-1]) if len(group) == 3 else group for group in groups]
    return "".join(groups)''',
        test_cases='''
for s in ["", "a", "ab", "abc", "abcd", "abcde", "abcdef", "abcdefghijklmnop"]:
    assert decode_cyclic(encode_cyclic(s)) == s
''',
        entry_point="decode_cyclic",
        difficulty="hard",
    ),
    HumanEvalProblem(
        task_id="HumanEval/45",
        prompt='''def triangle_area(a, h):
    """Given length of a side and height, return area of a triangle.
    >>> triangle_area(5, 3)
    7.5
    """
''',
        canonical_solution='''    return a * h / 2.0''',
        test_cases='''
assert triangle_area(5, 3) == 7.5
assert triangle_area(2, 2) == 2.0
assert triangle_area(10, 8) == 40.0
''',
        entry_point="triangle_area",
        difficulty="easy",
    ),
]


def format_humaneval_prompt(problem: HumanEvalProblem) -> str:
    """Format a HumanEval problem as a prompt."""
    return f"""Complete the following Python function. Only write the function body (the implementation).
Do not include the function signature or docstring - those are already provided.

```python
{problem.prompt}```

Write ONLY the implementation code that goes inside the function.
Start your code with proper indentation (4 spaces).

Your response should look like:
```python
    # implementation here
    return result
```"""


def extract_code(text: str, problem: HumanEvalProblem) -> str | None:
    """Extract the function implementation from model output."""
    # Try to find code in markdown code blocks
    code_pattern = r"```(?:python)?\s*([\s\S]*?)```"
    matches = re.findall(code_pattern, text)

    for match in matches:
        code = match.strip()
        if code:
            # Check if it looks like function body (indented code)
            if code.startswith("    ") or code.startswith("\t"):
                return code
            # Check if it's the full function
            if f"def {problem.entry_point}" in code:
                # Extract just the body
                lines = code.split("\n")
                body_lines = []
                in_body = False
                for line in lines:
                    if in_body:
                        body_lines.append(line)
                    elif line.strip().startswith("def "):
                        in_body = True
                    elif line.strip().startswith('"""') or line.strip().startswith("'''"):
                        # Skip docstring
                        pass
                if body_lines:
                    return "\n".join(body_lines)

    # Fallback: look for indented code anywhere
    lines = text.split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        if line.startswith("    ") or line.startswith("\t"):
            code_lines.append(line)
            in_code = True
        elif in_code and line.strip() == "":
            code_lines.append(line)
        elif in_code:
            break

    if code_lines:
        return "\n".join(code_lines)

    return None


def verify_humaneval_solution(output: str, problem: HumanEvalProblem) -> bool:
    """
    Verify that an output contains a correct solution to a HumanEval problem.

    Args:
        output: The model's output text
        problem: The HumanEval problem

    Returns:
        True if the solution passes all test cases
    """
    code_body = extract_code(output, problem)
    if code_body is None:
        return False

    # Construct the full code with the function definition
    full_code = problem.prompt + code_body

    # Add the test cases
    test_code = full_code + "\n\n" + problem.test_cases

    # Try to execute
    try:
        # Create a restricted namespace
        namespace: dict[str, Any] = {"__builtins__": __builtins__}

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            exec(test_code, namespace)
            return True
        except AssertionError:
            return False
        except Exception as e:
            # Other errors (syntax, runtime) mean the code is wrong
            return False
        finally:
            sys.stdout = old_stdout

    except Exception:
        return False


def get_problems_by_difficulty(difficulty: str) -> list[HumanEvalProblem]:
    """Get problems filtered by difficulty."""
    return [p for p in HUMANEVAL_PROBLEMS if p.difficulty == difficulty]


def create_verifier(problem: HumanEvalProblem):
    """Create a verifier function for a specific problem."""
    return lambda output: verify_humaneval_solution(output, problem)


# Cost estimation
def estimate_cost(num_problems: int, branches_per_problem: int = 5) -> dict:
    """
    Estimate the cost of running the HumanEval benchmark.

    Args:
        num_problems: Number of problems to run
        branches_per_problem: Average branches per PSS run

    Returns:
        Dict with cost breakdown
    """
    # Llama 8B via OpenRouter pricing
    input_cost_per_1m = 0.055
    output_cost_per_1m = 0.055

    # Code problems tend to have longer outputs
    tokens_per_branch = 3000  # prompt + response (more than math)
    total_tokens = num_problems * branches_per_problem * tokens_per_branch

    exploration_cost = (total_tokens / 1_000_000) * (input_cost_per_1m + output_cost_per_1m)

    # Embedding cost for diversity
    embedding_cost_per_1m = 0.02
    embedding_tokens = num_problems * branches_per_problem * 500
    embedding_cost = (embedding_tokens / 1_000_000) * embedding_cost_per_1m

    return {
        "exploration_cost": exploration_cost,
        "embedding_cost": embedding_cost,
        "total_cost": exploration_cost + embedding_cost,
        "total_tokens": total_tokens,
    }
