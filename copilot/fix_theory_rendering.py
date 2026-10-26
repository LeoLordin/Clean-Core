import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "THEORY.md"
TARGET = Path(__file__).resolve().parent / "THEORY_FIXED.md"

if not SOURCE.exists():
    raise FileNotFoundError(f"Source file not found: {SOURCE}")

text = SOURCE.read_text(encoding="utf-8")

# 1) Fix inline math: $`...`$ -> $...$
text = re.sub(r"\$`(.*?)`\$", r"$\1$", text, flags=re.S)

# 2) Fix fenced math blocks: ```math ... ``` -> $$ ... $$
text = re.sub(r"```\s*math\s*\n(.*?)\n```", r"$$\n\1\n$$", text, flags=re.S)

# 3) Remove accidental leading/trailing blank lines around math blocks
text = re.sub(r"\n{3,}\$\$", "\n\n$$", text)
text = re.sub(r"\$\$\n{3,}", "$$\n\n", text)

TARGET.write_text(text, encoding="utf-8")
print(f"Fixed file written to: {TARGET}")
