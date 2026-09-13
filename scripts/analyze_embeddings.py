"""Offline embedding analysis; see docs/embedding_analysis.md for the protocol."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1] / "src"),
    str(Path(__file__).resolve().parents[1]),
]

from embedding_analysis import main  # noqa: E402


if __name__ == "__main__":
    main()
