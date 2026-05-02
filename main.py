#!/usr/bin/env python3
"""
ragforge — Generic RAG Pipeline CLI

Commands:
  ingest <file> [<file> ...]   Index documents into the knowledge base
  query  "<question>"          Answer a single question
  chat   [--session ID]        Start an interactive chat session
  sessions                     List all stored sessions

Examples:
  python main.py ingest docs/manual.pdf docs/faq.docx
  python main.py query "What is the return policy?"
  python main.py chat
  python main.py chat --session abc123
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragforge",
        description="Generic Retrieval-Augmented Generation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        metavar="PATH",
        help="YAML config file (default: configs/default.yaml).",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Index documents into the knowledge base.")
    ingest.add_argument("files", nargs="+", help="Paths to documents to ingest.")

    query = sub.add_parser("query", help="Answer a single question and exit.")
    query.add_argument("question", help="The question to answer.")
    query.add_argument("--session", metavar="ID", help="Attach query to an existing session.")

    chat = sub.add_parser("chat", help="Start an interactive conversation.")
    chat.add_argument("--session", metavar="ID", help="Resume an existing session.")

    sub.add_parser("sessions", help="List all stored sessions.")

    return parser


def _load_pipeline(config_path: str):
    from src.core.config import AppConfig
    from src.core.pipeline import RAGPipeline

    config = (
        AppConfig.from_yaml(config_path)
        if Path(config_path).exists()
        else AppConfig()
    )
    return RAGPipeline.from_config(config)


def cmd_ingest(args) -> None:
    pipeline = _load_pipeline(args.config)
    count = pipeline.ingest(args.files)
    print(f"Indexed {count} chunks from {len(args.files)} file(s).")


def cmd_query(args) -> None:
    pipeline = _load_pipeline(args.config)
    result = pipeline.query(args.question, session_id=args.session)
    print(f"\nAnswer: {result.answer}")
    if result.sources:
        print(f"Sources: {', '.join(result.sources)}")
    print(f"Session: {result.session_id}  |  Chunks used: {result.context_chunks}")


def cmd_chat(args) -> None:
    pipeline = _load_pipeline(args.config)
    session_id = args.session or pipeline.new_session()

    print(f"Session: {session_id}")
    print("Type 'exit' or press Ctrl+C to end the session.\n")

    try:
        while True:
            try:
                question = input("You: ").strip()
            except EOFError:
                break

            if not question:
                continue
            if question.lower() in {"exit", "quit", "q", ":q"}:
                break

            result = pipeline.query(question, session_id=session_id)
            print(f"\nAssistant: {result.answer}")
            if result.sources:
                print(f"  [Sources: {', '.join(result.sources)}]")
            print()

    except KeyboardInterrupt:
        print("\nGoodbye.")


def cmd_sessions(args) -> None:
    pipeline = _load_pipeline(args.config)
    sessions = pipeline.list_sessions()

    if not sessions:
        print("No sessions found.")
        return

    print(f"{'ID':36}  {'Name':30}  {'Updated':19}")
    print("-" * 90)
    for s in sessions:
        print(f"{s['id']:36}  {s['name'][:30]:30}  {s['updated_at'][:19]}")


def main() -> None:
    from src.utils.logger import configure_logging

    configure_logging()

    parser = _build_parser()
    args = parser.parse_args()

    handlers = {
        "ingest": cmd_ingest,
        "query": cmd_query,
        "chat": cmd_chat,
        "sessions": cmd_sessions,
    }

    handlers[args.command](args)


if __name__ == "__main__":
    main()
