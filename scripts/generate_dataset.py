import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import json
import os

# Importing your config
from config import Paths


def extract_fang_yuan_data():
    epub_path = Paths["epub"]
    output_path = Paths["dataset"]

    if not os.path.exists(epub_path):
        print(f"Error: EPUB not found at {epub_path}")
        return

    book = epub.read_epub(epub_path)
    dataset = []

    # Core identifiers for Fang Yuan's perspective
    identifiers = [
        "Fang Yuan said",
        "Fang Yuan thought",
        "Fang Yuan sneered",
        "Fang Yuan laughed",
    ]

    print(f"Processing {epub_path}...")

    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            # Extract text and filter out empty lines
            paragraphs = [
                p.get_text().strip() for p in soup.find_all("p") if p.get_text().strip()
            ]

            for i in range(1, len(paragraphs)):
                current_p = paragraphs[i]

                # If the paragraph contains Fang Yuan's name and an action/speech
                if any(id_text in current_p for id_text in identifiers):
                    context = paragraphs[
                        i - 1
                    ]  # The situation leading up to his response

                    # Formatting for Unsloth Instruction Fine-Tuning
                    entry = {
                        "instruction": "Analyze the situation and respond as Fang Yuan.",
                        "input": f"Context: {context}",
                        "output": current_p,
                    }
                    dataset.append(entry)

    # Ensure the data directory exists before saving
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print(f"Extraction complete! Created {len(dataset)} entries in {output_path}")


if __name__ == "__main__":
    print("Starting dataset generation...")
    extract_fang_yuan_data()
