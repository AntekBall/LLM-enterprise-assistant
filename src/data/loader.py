import re
from dataclasses import dataclass
from pathlib import Path

RAW_DIR = Path("data/raw")

FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
SHORTCODE = re.compile(r"\{\{[<%].*?[>%]\}\}", re.DOTALL)
TITLE = re.compile(r"^title:\s*[\"']?(.*?)[\"']?\s*$", re.MULTILINE)


@dataclass
class Document:
    text: str
    source: str
    department: str
    title: str


def parse_markdown(raw: str, fallback_title: str) -> tuple[str, str]:
    """Return (cleaned_text, title)."""
    title = fallback_title
    match = FRONT_MATTER.match(raw)
    if match:
        found = TITLE.search(match.group(1))
        if found and found.group(1):
            title = found.group(1)
        raw = raw[match.end():]
    text = SHORTCODE.sub("", raw)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, title


def load_documents(raw_dir: Path = RAW_DIR) -> list[Document]:
    documents = []
    for path in sorted(raw_dir.rglob("*.md")):
        raw = path.read_text(encoding="utf-8")
        text, title = parse_markdown(raw, fallback_title=path.stem)
        if text:
            documents.append(
                Document(
                    text=text,
                    source=path.name,
                    department=path.parent.name,
                    title=title,
                )
            )
    return documents