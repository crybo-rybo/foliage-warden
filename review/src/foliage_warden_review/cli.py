"""Command-line entry point for serving review media or exporting labels."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .manifest import ManifestError, load_manifest
from .server import create_server
from .storage import AnnotationStore
from .validation import AnnotationError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="foliage-warden-review",
        description="Review manifest-listed media on a loopback-only web UI.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="run the loopback-only review UI")
    serve.add_argument("--manifest", type=Path, required=True)
    serve.add_argument("--annotations", type=Path, required=True)
    serve.add_argument("--port", type=int, default=8765)
    export = commands.add_parser(
        "export", help="atomically export GroundTruthEvent JSONL"
    )
    export.add_argument("--manifest", type=Path, required=True)
    export.add_argument("--annotations", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "export":
            manifest = load_manifest(args.manifest)
            store = AnnotationStore(args.annotations, manifest)
            store.write_export(args.output)
            print(f"wrote {len(store.export_records())} records to {args.output}")
            return 0
        server = create_server(args.manifest, args.annotations, args.port)
    except (AnnotationError, ManifestError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    host, port = server.server_address
    print(f"review UI: http://{host}:{port}")
    print("loopback only; no upload, inference, or actuator endpoints")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()
    return 0
