import re
import unicodedata
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ORIGINAL = DOCS / "THEORY (org).md"
MAIN = DOCS / "THEORY.md"


def slugify(value: str) -> str:
    value = re.sub(r"\$`.*?`\$", "", value)
    value = value.replace("&", " and ")
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return re.sub(r"-+", "-", value)


def split_sections(text: str, pattern: str) -> tuple[str, list[tuple[str, str]]]:
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
    if not matches:
        raise ValueError(f"No Markdown headings matched: {pattern}")
    prefix = text[:matches[0].start()]
    sections = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append((match.group(1).strip(), text[match.start():end]))
    return prefix, sections


def count_math(text: str) -> tuple[int, int]:
    inline = len(re.findall(r"\$`.*?`\$", text, flags=re.DOTALL))
    display = len(re.findall(r"^```math\s*$", text, flags=re.MULTILINE))
    return inline, display


def write_page(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = body.rstrip("\n") + "\n"
    path.write_text("[Back to theory contents](../THEORY.md)\n\n" + body, encoding="utf-8")


if not ORIGINAL.exists():
    raise FileNotFoundError(f"Preserved source document is missing: {ORIGINAL}")
original_bytes = ORIGINAL.read_bytes()

source = original_bytes.decode("utf-8")
required_markers = ["## Table of Contents", "## Preamble:", "## Part I:", "## Part II:", "## Part III:"]
for marker in required_markers:
    if marker not in source:
        raise ValueError(f"Original document is missing expected heading: {marker}")

toc_start = source.index("## Table of Contents")
preamble_start = source.index("## Preamble:")
part_i_start = source.index("## Part I:")
part_ii_start = source.index("## Part II:")
part_iii_start = source.index("## Part III:")
header = source[:toc_start].rstrip()
preamble = source[preamble_start:part_i_start]
part_i = source[part_i_start:part_ii_start]
part_ii = source[part_ii_start:part_iii_start]
part_iii = source[part_iii_start:]

formula_intro, formulas = split_sections(part_i, r"^### Section \d+: (.+)$")
algorithm_intro, algorithms = split_sections(part_ii, r"^### Algorithm (\d+): (.+)$")
symbol_intro, symbols = split_sections(part_iii, r"^### (Geometric Manifolds, Bundles, and Sets|Indices, Dimensions, and Architectural Constants|State Variables and Dynamic Vector Fields|Scalar Fields, Potentials, and Wavefunctions|Matrices, Tensors, and Differential Operators|Hyperparameters, Coefficients, and Multipliers|Mathematical Operators and Notations)$")

if len(formulas) != 38 or len(algorithms) != 13 or len(symbols) != 7:
    raise ValueError(f"Unexpected section counts: formulas={len(formulas)}, algorithms={len(algorithms)}, symbols={len(symbols)}")

formula_rows = []
formula_bodies = []
preamble_path = DOCS / "formulas" / "00-preamble-binary-layout-and-memory-topology.md"
write_page(preamble_path, preamble)
formula_bodies.append((preamble_path, preamble))
for index, (title, body) in enumerate(formulas, start=1):
    heading = re.match(r"^### Section (\d+): (.+)$", body, flags=re.MULTILINE)
    if not heading or int(heading.group(1)) != index:
        raise ValueError(f"Formula section numbering mismatch at {index}")
    filename = f"{index:02d}-{slugify(heading.group(2))}.md"
    path = DOCS / "formulas" / filename
    write_page(path, body)
    formula_rows.append((heading.group(1), heading.group(2), f"formulas/{filename}"))
    formula_bodies.append((path, body))

algorithm_rows = []
algorithm_bodies = []
for expected, (_, body) in enumerate(algorithms, start=1):
    heading = re.match(r"^### Algorithm (\d+): (.+)$", body, flags=re.MULTILINE)
    if not heading or int(heading.group(1)) != expected:
        raise ValueError(f"Algorithm numbering mismatch at {expected}")
    filename = f"{expected:02d}-{slugify(heading.group(2))}.md"
    path = DOCS / "algorithms" / filename
    write_page(path, body)
    algorithm_rows.append((heading.group(1), heading.group(2), f"algorithms/{filename}"))
    algorithm_bodies.append((path, body))

symbol_rows = []
symbol_bodies = []
for index, (_, body) in enumerate(symbols, start=1):
    heading = re.match(r"^### (.+)$", body, flags=re.MULTILINE)
    if not heading:
        raise ValueError(f"Symbol group heading missing at {index}")
    filename = f"{index:02d}-{slugify(heading.group(1))}.md"
    path = DOCS / "symbols" / filename
    write_page(path, body)
    symbol_rows.append((f"{index:02d}", heading.group(1), f"symbols/{filename}"))
    symbol_bodies.append((path, body))

formula_index = "# Mathematical Foundations and Dynamical Equations\n\n" + formula_intro
formula_index += "\n## Preamble\n\n- [Binary Layout and Memory Topology](00-preamble-binary-layout-and-memory-topology.md)\n"
formula_index += "\n## Sections\n\n" + "\n".join(
    f"- [{number}. {title}]({Path(filename).name})" for number, title, filename in formula_rows
) + "\n"
(DOCS / "formulas" / "README.md").write_text(formula_index, encoding="utf-8")

algorithm_index = "# Algorithmic Execution Specifications\n\n" + algorithm_intro
algorithm_index += "\n## Algorithms\n\n" + "\n".join(
    f"- [Algorithm {number}: {title}]({Path(filename).name})" for number, title, filename in algorithm_rows
) + "\n"
(DOCS / "algorithms" / "README.md").write_text(algorithm_index, encoding="utf-8")

symbol_index = "# Complete Mathematical and Symbol Index\n\n" + symbol_intro
symbol_index += "\n## Symbol Groups\n\n" + "\n".join(
    f"- [Part {number}: {title}]({Path(filename).name})" for number, title, filename in symbol_rows
) + "\n"
(DOCS / "symbols" / "README.md").write_text(symbol_index, encoding="utf-8")

quick_info = """## Overview

This reference specification covers non-Euclidean geometric deep learning, active inference, symplectic mechanics, algorithmic information theory, causal discovery, and systems architecture.

**Reference profile:** ambient dimension $`d = 16`$, covariance rank $`r = 4`$, hidden dimension $`h_{\\mathrm{dim}} = 32`$, and motor dimension $`d_{\\mathrm{act}} = 8`$.

The complete source document is preserved in [THEORY (org).md](THEORY%20%28org%29.md).
"""
contents = "\n## Contents\n\n"
contents += "### Preamble\n\n- [Binary layout and memory topology](formulas/00-preamble-binary-layout-and-memory-topology.md)\n\n"
contents += "### Mathematical Foundations\n\n" + "\n".join(
    f"- [Section {number}: {title}]({filename})" for number, title, filename in formula_rows
) + "\n\n"
contents += "### Algorithms\n\n" + "\n".join(
    f"- [Algorithm {number}: {title}]({filename})" for number, title, filename in algorithm_rows
) + "\n\n"
contents += "### Symbols\n\n" + "\n".join(
    f"- [Part {number}: {title}]({filename})" for number, title, filename in symbol_rows
) + "\n\n"
main_text = (header + "\n\n" + quick_info + contents).rstrip() + "\n"
MAIN.write_text(main_text, encoding="utf-8")

# Remove only the intermediate pages generated by the previous pass.
for name in [
    "THEORY_ALGORITHMS_01.md", "THEORY_ALGORITHMS_02.md",
    "THEORY_PART_I_01.md", "THEORY_PART_I_02.md", "THEORY_PART_I_03.md",
    "THEORY_PREAMBLE.md", "THEORY_SYMBOLS_01.md", "THEORY_SYMBOLS_02.md",
    "THEORY_SYMBOLS_03.md", "THEORY_SYMBOLS_04.md",
]:
    path = DOCS / name
    if path.exists():
        path.unlink()

expected_inline, expected_display = count_math(source)
all_paths = [path for path, _ in formula_bodies + algorithm_bodies + symbol_bodies]
actual_inline = actual_display = 0
for path, body in formula_bodies + algorithm_bodies + symbol_bodies:
    generated = path.read_text(encoding="utf-8")
    if not generated.endswith(body.rstrip("\n") + "\n"):
        raise ValueError(f"Content changed during transfer: {path.relative_to(ROOT)}")
    inline, display = count_math(body)
    actual_inline += inline
    actual_display += display

if (actual_inline, actual_display) != (expected_inline, expected_display):
    raise ValueError(f"Math count mismatch: expected {(expected_inline, expected_display)}, got {(actual_inline, actual_display)}")
if ORIGINAL.read_bytes() != original_bytes:
    raise ValueError("Original copy differs from the source bytes")

all_targets = [DOCS / row[2] for row in formula_rows + algorithm_rows + symbol_rows]
all_targets += [DOCS / "formulas" / "README.md", DOCS / "algorithms" / "README.md", DOCS / "symbols" / "README.md", ORIGINAL]
missing = [str(path.relative_to(ROOT)) for path in all_targets if not path.is_file()]
if missing:
    raise ValueError(f"Missing generated files: {missing}")

markdown_files = [MAIN, ORIGINAL]
for directory in (DOCS / "formulas", DOCS / "algorithms", DOCS / "symbols"):
    markdown_files.extend(directory.glob("*.md"))
broken_links = []
for markdown_file in markdown_files:
    text = markdown_file.read_text(encoding="utf-8")
    for target in re.findall(r"\]\(([^)]+\.md(?:#[^)]*)?)\)", text):
        parsed = urlsplit(target)
        if parsed.scheme or target.startswith("#"):
            continue
        target_path = unquote(parsed.path)
        if not target_path:
            continue
        if not (markdown_file.parent / target_path).is_file():
            broken_links.append(f"{markdown_file.relative_to(ROOT)} -> {target}")
if broken_links:
    raise ValueError(f"Broken Markdown links: {broken_links}")

print(f"Preserved original: {ORIGINAL.relative_to(ROOT)}")
print(f"Generated {len(formula_rows)} formula-section files, {len(algorithm_rows)} algorithm files, and {len(symbol_rows)} symbol-group files.")
print(f"Verified unchanged math: {expected_inline} inline and {expected_display} display expressions.")
