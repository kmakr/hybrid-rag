#!/usr/bin/env python3
"""CLI to ingest documents into the contextual RAG system."""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline import ingest


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into contextual RAG")
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "data"),
        help="Directory containing documents (default: ./data)",
    )
    parser.add_argument(
        "--threads", type=int, default=4,
        help="Parallel threads for contextualization (default: 4)",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(data_dir):
        print(f"Error: {data_dir} is not a directory")
        sys.exit(1)

    ingest(data_dir, parallel_threads=args.threads)


if __name__ == "__main__":
    main()
