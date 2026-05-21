import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import json
import os
import re

from config import Paths


TRANSLATOR_NOTE_RE = re.compile(r"Translator\s*:\s*\w+|Editor\s*:\s*\w+")


def extract_quoted_dialogue(text: str) -> str | None:
    """Extract the actual dialogue/thought from a Fang Yuan paragraph.

    Looks for text inside guillemets «» or curly quotes "" (common in CN webnovel translations).
    Returns the extracted dialogue, or None if no clean dialogue is found.
    """
    # Remove translator/editor notes first
    text = TRANSLATOR_NOTE_RE.sub("", text).strip()

    # Chinese-style quotes: \u201c...\u201d or \u2018...\u2019
    quotes = re.findall(r"[\u201c\u201e]([^\u201d\u201f]+)[\u201d\u201f]", text)
    # Also try normal ASCII quotes
    if not quotes:
        quotes = re.findall(r"""(?<=["\u201c])[^"\u201d]+(?=["\u201d])""", text)
    # Guillemets
    if not quotes:
        quotes = re.findall(r"\u00ab([^\u00bb]+)\u00bb", text)

    if quotes:
        return quotes[0].strip()

    # If no quotes found, check if the text is purely narrative about Fang Yuan
    # (e.g., "Because everything Fang Yuan said was the truth!")
    # Skip those — they are not Fang Yuan's own words
    return None


def is_actual_fang_yuan_output(text: str) -> bool:
    """Check if a paragraph actually contains Fang Yuan's spoken/thought content.

    Skips pure narrative sentences that describe Fang Yuan but aren't his words.
    """
    narrative_patterns = [
        r"^Because everything Fang Yuan",
        r"^Fang Yuan \w+ed (and|as|but|with)",
        r"^If Fang Yuan",
        r"^This made Fang Yuan",
        r"^All of (this|these) made Fang Yuan",
    ]
    for pat in narrative_patterns:
        if re.match(pat, text.strip()):
            return False
    return True


def extract_fang_yuan_data():
    epub_path = Paths["epub"]
    output_path = Paths["dataset"]

    if not os.path.exists(epub_path):
        print(f"Error: EPUB not found at {epub_path}")
        return

    book = epub.read_epub(epub_path)
    dataset = []

    identifiers = [
        "Fang Yuan said",
        "Fang Yuan thought",
        "Fang Yuan sneered",
        "Fang Yuan laughed",
        "Fang Yuan replied",
        "Fang Yuan asked",
        "Fang Yuan spoke",
        "Fang Yuan muttered",
        "Fang Yuan shouted",
        "Fang Yuan exclaimed",
    ]

    print(f"Processing {epub_path}...")

    seen_outputs: set[str] = set()

    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            paragraphs = [
                p.get_text().strip() for p in soup.find_all("p") if p.get_text().strip()
            ]

            for i in range(1, len(paragraphs)):
                current_p = paragraphs[i]

                if not any(id_text in current_p for id_text in identifiers):
                    continue
                if not is_actual_fang_yuan_output(current_p):
                    continue

                dialogue = extract_quoted_dialogue(current_p)
                if dialogue is None:
                    continue

                # Skip very short / meaningless dialogue
                if len(dialogue) < 10:
                    continue
                # Deduplicate
                if dialogue in seen_outputs:
                    continue
                seen_outputs.add(dialogue)

                context = paragraphs[i - 1]
                # Clean translator notes from context too
                context = TRANSLATOR_NOTE_RE.sub("", context).strip()
                # Skip if context is too short (likely a stray line)
                if len(context) < 15:
                    continue
                # Skip if context is itself a Fang Yuan identifier line
                if any(id_text in context for id_text in identifiers):
                    continue

                entry = {
                    "instruction": "Analyze the situation and respond as Fang Yuan.",
                    "input": f"Context: {context}",
                    "output": dialogue,
                }
                dataset.append(entry)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print(f"Extraction complete! Created {len(dataset)} entries in {output_path}")


if __name__ == "__main__":
    print("Starting dataset generation...")
    extract_fang_yuan_data()
