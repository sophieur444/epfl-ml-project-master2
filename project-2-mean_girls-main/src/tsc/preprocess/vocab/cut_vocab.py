# src/tsc/preprocess/vocab/cut_vocab.py

from pathlib import Path
from tsc.utils.paths import DATA_INTERMEDIATE


def main() -> None:

    in_path = DATA_INTERMEDIATE / "vocab_full.txt"
    out_path = DATA_INTERMEDIATE / "vocab_cut.txt"

    entries: list[tuple[int, str]] = []

    with in_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                count = int(parts[0])
            except ValueError:
                continue

            word = parts[1]
            if count >= 5:
                entries.append((count, word))

    entries.sort(key=lambda cw: (-cw[0], cw[1]))

    with out_path.open("w", encoding="utf-8") as f:
        for _, word in entries:
            f.write(word + "\n")


if __name__ == "__main__":
    main()
