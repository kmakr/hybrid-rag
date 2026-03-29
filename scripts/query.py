#!/usr/bin/env python3
"""CLI to query the contextual RAG system."""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline import query


def main():
    parser = argparse.ArgumentParser(description="Query the contextual RAG system")
    parser.add_argument("query", help="The question to ask")
    parser.add_argument(
        "-k", type=int, default=5,
        help="Number of chunks to retrieve (default: 5)",
    )
    parser.add_argument(
        "--rerank", action="store_true",
        help="Rerank results with a cross-encoder before generating (improves accuracy)",
    )
    args = parser.parse_args()

    answer = query(args.query, k=args.k, rerank=args.rerank)
    print("\n" + "=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(answer)


if __name__ == "__main__":
    main()
