"""Sentence handling. Dataset notes are already numbered; unseen notes must be split here."""
import re
from pathlib import Path

import nltk

NLTK_DIR = Path(__file__).resolve().parents[2] / "data" / "nltk_data"
_LINE = re.compile(r"^(\d+)\.\s*(.*)$", re.DOTALL)
_LIST_MARK = re.compile(r"^\s*(?:[-*\u2022]+|\d+[.)])\s+")
_HEADING = re.compile(r"^[A-Za-z][A-Za-z /&-]{0,40}:$")


def parse_numbered_note(note):
    """Turn a dataset note ("0. text\\n1. text") into {sentence_id: text}, keeping original IDs."""
    sentences = {}
    for line in note.split("\n"):
        if not line.strip():
            continue
        match = _LINE.match(line.strip())
        if match is None:
            raise ValueError(f"line is not numbered: {line[:40]!r}")
        sentence_id = int(match.group(1))
        if sentence_id != len(sentences):
            raise ValueError(f"expected sentence {len(sentences)}, found {sentence_id}")
        sentences[sentence_id] = match.group(2).strip()
    if not sentences:
        raise ValueError("note has no sentences")
    return sentences


def format_numbered(sentences):
    """Render {sentence_id: text} in the same layout the dataset uses."""
    return "\n".join(f"{sentence_id}. {text}" for sentence_id, text in sorted(sentences.items()))


def split_unseen_note(text):
    """Number the sentences of a note that did not come from the dataset."""
    if str(NLTK_DIR) not in nltk.data.path:
        nltk.data.path.insert(0, str(NLTK_DIR))
    pieces = []
    for block in text.splitlines():
        block = _LIST_MARK.sub("", block).strip()
        if block and not _HEADING.match(block):
            pieces.extend(s.strip() for s in nltk.sent_tokenize(block, language="english") if s.strip())
    return dict(enumerate(pieces))
