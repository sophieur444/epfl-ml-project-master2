# src/tsc/preprocess/vocab/build_vocab.py

from pathlib import Path
from collections import Counter
from tsc.utils.paths import DATA_TWITTER, DATA_INTERMEDIATE


def main() -> None:

    text_parts: list[str] = []
    for path in (
        DATA_TWITTER / "train_pos_full.txt",
        DATA_TWITTER / "train_neg_full.txt",
    ):
        text_parts.append(path.read_text(encoding="utf-8"))

    words: list[str] = []
    for chunk in text_parts:
        for token in chunk.split():
            token = token.strip()
            if token:
                words.append(token)

    counts = Counter(words)

    out_path = DATA_INTERMEDIATE / "vocab_full.txt"
    with out_path.open("w", encoding="utf-8") as f:
        for word in sorted(counts.keys()):
            count = counts[word]
            f.write(f"{count:7d} {word}\n")


if __name__ == "__main__":
    main()
