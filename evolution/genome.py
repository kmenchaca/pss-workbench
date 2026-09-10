"""Genome operations for evolutionary code synthesis.

Provides functions for parsing, manipulating, and comparing genomes
at the AST level for structural operations rather than text-only.
"""

import ast
import hashlib
import uuid
from typing import Any

from .types import Gene, GeneCategory, Genome


def parse_to_ast(code: str) -> ast.AST | None:
    """Convert code string to AST for manipulation.

    Args:
        code: Python source code as string

    Returns:
        AST node if parsing succeeds, None if code is invalid
    """
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def ast_to_code(tree: ast.AST) -> str:
    """Convert AST back to code string.

    Args:
        tree: AST node to convert

    Returns:
        Python source code as string
    """
    return ast.unparse(tree)


def _hash_ast(node: ast.AST) -> str:
    """Create a structural hash of an AST node.

    This normalizes variable names and constants to focus on structure.
    """
    # Use ast.dump with normalized names for structural comparison
    class Normalizer(ast.NodeTransformer):
        def __init__(self) -> None:
            self.name_map: dict[str, str] = {}
            self.counter = 0

        def visit_Name(self, node: ast.Name) -> ast.Name:
            if node.id not in self.name_map:
                self.name_map[node.id] = f"var_{self.counter}"
                self.counter += 1
            node.id = self.name_map[node.id]
            return node

        def visit_Constant(self, node: ast.Constant) -> ast.Constant:
            # Normalize numeric constants but preserve type
            if isinstance(node.value, (int, float)):
                node.value = 0
            elif isinstance(node.value, str):
                node.value = ""
            return node

    normalizer = Normalizer()
    normalized = normalizer.visit(ast.parse(ast.unparse(node)))
    dump = ast.dump(normalized, annotate_fields=False)
    return hashlib.sha256(dump.encode()).hexdigest()[:16]


def genome_hash(genome: Genome) -> str:
    """Generate unique structural hash for a genome.

    Uses AST structure rather than raw text for semantic comparison.

    Args:
        genome: The genome to hash

    Returns:
        Hex string hash of the genome's structure
    """
    tree = parse_to_ast(genome.code)
    if tree is None:
        # Fall back to text hash if code is invalid
        return hashlib.sha256(genome.code.encode()).hexdigest()[:16]
    return _hash_ast(tree)


def genome_distance(g1: Genome, g2: Genome) -> float:
    """Calculate genetic distance between two genomes.

    Uses a combination of:
    - AST edit distance (structural similarity)
    - Shared lineage (common ancestors)

    Returns a value from 0 (identical) to 1 (completely different).

    Args:
        g1: First genome
        g2: Second genome

    Returns:
        Distance metric from 0 to 1
    """
    # Quick check: identical hashes mean identical structure
    if g1.ast_hash and g2.ast_hash and g1.ast_hash == g2.ast_hash:
        return 0.0

    tree1 = parse_to_ast(g1.code)
    tree2 = parse_to_ast(g2.code)

    if tree1 is None or tree2 is None:
        # Fall back to text similarity
        return _text_distance(g1.code, g2.code)

    # Calculate AST-based distance
    ast_dist = _ast_distance(tree1, tree2)

    # Factor in shared lineage
    lineage_similarity = _lineage_similarity(g1, g2)

    # Weight: 70% structural, 30% lineage
    return ast_dist * 0.7 + (1 - lineage_similarity) * 0.3


def _text_distance(s1: str, s2: str) -> float:
    """Simple text-based distance using character-level comparison."""
    if s1 == s2:
        return 0.0
    if not s1 or not s2:
        return 1.0

    # Jaccard similarity of character n-grams
    n = 3
    grams1 = set(s1[i : i + n] for i in range(len(s1) - n + 1))
    grams2 = set(s2[i : i + n] for i in range(len(s2) - n + 1))

    if not grams1 or not grams2:
        return 1.0

    intersection = len(grams1 & grams2)
    union = len(grams1 | grams2)

    return 1 - (intersection / union)


def _ast_distance(tree1: ast.AST, tree2: ast.AST) -> float:
    """Calculate structural distance between two ASTs.

    Uses a simplified tree edit distance based on node types.
    """
    nodes1 = _extract_node_types(tree1)
    nodes2 = _extract_node_types(tree2)

    if not nodes1 and not nodes2:
        return 0.0
    if not nodes1 or not nodes2:
        return 1.0

    # Jaccard similarity of node type sequences
    set1 = set(nodes1)
    set2 = set(nodes2)

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return 1 - (intersection / union)


def _extract_node_types(tree: ast.AST) -> list[str]:
    """Extract sequence of node types from AST."""
    nodes = []
    for node in ast.walk(tree):
        nodes.append(type(node).__name__)
    return nodes


def _lineage_similarity(g1: Genome, g2: Genome) -> float:
    """Calculate similarity based on shared lineage."""
    if not g1.lineage and not g2.lineage:
        return 0.0

    set1 = set(g1.lineage)
    set2 = set(g2.lineage)

    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union


def extract_genes(genome: Genome) -> list[Gene]:
    """Identify reusable patterns/genes from a genome.

    Extracts function definitions, loops, and other structural
    patterns that could be transplanted to other genomes.

    Args:
        genome: The genome to extract genes from

    Returns:
        List of extracted genes
    """
    tree = parse_to_ast(genome.code)
    if tree is None:
        return []

    genes = []

    for node in ast.walk(tree):
        gene = _node_to_gene(node, genome.id)
        if gene is not None:
            genes.append(gene)

    return genes


def _node_to_gene(node: ast.AST, origin_id: str) -> Gene | None:
    """Convert an AST node to a Gene if it's a reusable pattern."""
    gene_id = str(uuid.uuid4())[:8]

    if isinstance(node, ast.FunctionDef):
        return Gene(
            id=gene_id,
            pattern=ast.unparse(node),
            category=_categorize_function(node),
            origin=origin_id,
            ast_node_type="FunctionDef",
        )
    elif isinstance(node, ast.For):
        return Gene(
            id=gene_id,
            pattern=ast.unparse(node),
            category=GeneCategory.LOOP,
            origin=origin_id,
            ast_node_type="For",
        )
    elif isinstance(node, ast.While):
        return Gene(
            id=gene_id,
            pattern=ast.unparse(node),
            category=GeneCategory.LOOP,
            origin=origin_id,
            ast_node_type="While",
        )
    elif isinstance(node, ast.Try):
        return Gene(
            id=gene_id,
            pattern=ast.unparse(node),
            category=GeneCategory.ERROR_HANDLING,
            origin=origin_id,
            ast_node_type="Try",
        )
    elif isinstance(node, ast.ClassDef):
        return Gene(
            id=gene_id,
            pattern=ast.unparse(node),
            category=GeneCategory.DATASTRUCTURE,
            origin=origin_id,
            ast_node_type="ClassDef",
        )

    return None


def _categorize_function(node: ast.FunctionDef) -> GeneCategory:
    """Categorize a function definition by its characteristics."""
    # Check for recursion
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name) and child.func.id == node.name:
                return GeneCategory.RECURSION

    # Check for algorithm patterns
    has_loop = any(isinstance(n, (ast.For, ast.While)) for n in ast.walk(node))
    has_conditionals = any(isinstance(n, ast.If) for n in ast.walk(node))

    if has_loop and has_conditionals:
        return GeneCategory.ALGORITHM

    if has_loop:
        return GeneCategory.LOOP

    return GeneCategory.UTILITY


def create_genome(
    code: str,
    generation: int = 0,
    parent_ids: list[str] | None = None,
    lineage: list[str] | None = None,
) -> Genome:
    """Create a new genome from code.

    Automatically computes AST hash and sets up lineage.

    Args:
        code: The Python source code
        generation: Generation number
        parent_ids: IDs of parent genomes
        lineage: Full lineage history

    Returns:
        New Genome instance
    """
    genome_id = str(uuid.uuid4())[:8]

    genome = Genome(
        id=genome_id,
        code=code,
        generation=generation,
        parent_ids=parent_ids or [],
        lineage=lineage or [],
    )

    # Compute AST hash
    genome.ast_hash = genome_hash(genome)

    return genome


def is_valid_code(code: str) -> bool:
    """Check if code is syntactically valid Python.

    Args:
        code: Python source code to validate

    Returns:
        True if code parses successfully
    """
    return parse_to_ast(code) is not None


def get_functions(genome: Genome) -> list[ast.FunctionDef]:
    """Extract all function definitions from a genome.

    Args:
        genome: The genome to extract from

    Returns:
        List of FunctionDef AST nodes
    """
    tree = parse_to_ast(genome.code)
    if tree is None:
        return []

    return [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]


def get_classes(genome: Genome) -> list[ast.ClassDef]:
    """Extract all class definitions from a genome.

    Args:
        genome: The genome to extract from

    Returns:
        List of ClassDef AST nodes
    """
    tree = parse_to_ast(genome.code)
    if tree is None:
        return []

    return [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]


def merge_functions(funcs1: list[ast.FunctionDef], funcs2: list[ast.FunctionDef]) -> str:
    """Merge two lists of functions into a single code string.

    Handles name conflicts by preferring the first list.

    Args:
        funcs1: First list of functions
        funcs2: Second list of functions

    Returns:
        Merged code as string
    """
    seen_names: set[str] = set()
    merged: list[ast.FunctionDef] = []

    for func in funcs1:
        if func.name not in seen_names:
            merged.append(func)
            seen_names.add(func.name)

    for func in funcs2:
        if func.name not in seen_names:
            merged.append(func)
            seen_names.add(func.name)

    if not merged:
        return ""

    module = ast.Module(body=list(merged), type_ignores=[])
    return ast.unparse(module)
