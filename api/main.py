"""CLI entry point for the SecureMail-ML API server.

Usage:
    python -m api.main
    python -m api.main --host 0.0.0.0 --port 8000
    uvicorn api.app:app --reload
"""
from __future__ import annotations

import argparse
import logging


def main(argv: list[str] | None = None) -> None:
    """Start the uvicorn server with the SecureMail-ML API."""
    parser = argparse.ArgumentParser(
        description="Start the SecureMail-ML API server.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Bind port (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging level (default: info)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    import uvicorn

    uvicorn.run(
        "api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
