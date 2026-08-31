"""The `atlas` command.

Six subcommands, one per operation:

    atlas create <netcdf-dir> <collection>   build a collection
    atlas rm     <collection> <name>...      remove datasets
    atlas ls     <collection>                list datasets
    atlas show   <collection> <name>         one dataset, ncdump style
    atlas info   <collection>                the whole collection

Every one of them takes a local path or a URL (`s3://`, `gs://`, `az://`,
`https://`) for the collection, so the same command works against a bucket.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from typing import Any, Optional, Sequence

from . import __version__, _ops, _source


# ── formatting helpers ───────────────────────────────────────────────


def _human_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"  # pragma: no cover - unreachable


def _format_value(value: Any) -> str:
    """Render an attribute or statistic the way ncdump would."""
    if value is None:
        return "-"
    if isinstance(value, bytes):
        try:
            return f'"{value.decode()}"'
        except UnicodeDecodeError:
            return repr(value)
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ", ".join(_format_value(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value)
    return str(value)


def _format_stats(stats: Optional[dict[str, Any]]) -> str:
    if not stats:
        return ""
    parts = [f"count={stats['row_count']}"]
    if stats["null_count"]:
        parts.append(f"nulls={stats['null_count']}")
    if stats["min"] is not None:
        parts.append(f"min={_format_value(stats['min'])}")
    if stats["max"] is not None:
        parts.append(f"max={_format_value(stats['max'])}")
    return "  ".join(parts)


def _timestamp(ms: int) -> str:
    return (
        _dt.datetime.fromtimestamp(ms / 1000, tz=_dt.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _emit(payload: Any, as_json: bool, render) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, default=_json_default))
    else:
        render(payload)


def _json_default(value: Any) -> Any:
    if isinstance(value, bytes):
        try:
            return value.decode()
        except UnicodeDecodeError:
            return value.hex()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


# ── store options ────────────────────────────────────────────────────


def _store_options(args: argparse.Namespace) -> dict[str, Any]:
    """Backend settings shared by every subcommand."""
    options: dict[str, Any] = {}
    if getattr(args, "region", None):
        options["region"] = args.region
    if getattr(args, "endpoint", None):
        options["endpoint"] = args.endpoint
    if getattr(args, "anonymous", False):
        options["skip_signature"] = True
    return options


def _add_store_flags(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("remote storage")
    group.add_argument("--region", help="bucket region, for s3:// sources")
    group.add_argument("--endpoint", help="override the service endpoint")
    group.add_argument(
        "--anonymous",
        action="store_true",
        help="skip request signing, for public buckets",
    )


# ── commands ─────────────────────────────────────────────────────────


def cmd_create(args: argparse.Namespace) -> int:
    chunks = json.loads(args.chunks) if args.chunks else None

    def progress(name: str) -> None:
        if not args.quiet:
            print(f"  {name}", file=sys.stderr)

    if not args.quiet:
        print(f"Writing {args.destination}", file=sys.stderr)

    result = _ops.create(
        args.directory,
        args.destination,
        recursive=args.recursive,
        codec=args.codec,
        chunks=chunks,
        on_error="skip" if args.skip_errors else "stop",
        progress=progress,
        **_store_options(args),
    )

    def render(r: dict[str, Any]) -> None:
        print(f"{r['dataset_count']} dataset(s) written to {r['destination']}")
        for item in r["skipped"]:
            print(f"  skipped {item['file']}: {item['error']}", file=sys.stderr)

    _emit(result, args.json, render)
    return 1 if result["skipped"] and not args.json else 0


def cmd_rm(args: argparse.Namespace) -> int:
    result = _ops.remove(
        args.collection,
        args.names,
        missing_ok=args.missing_ok,
        **_store_options(args),
    )

    def render(r: dict[str, Any]) -> None:
        if r["removed"]:
            print(f"removed {len(r['removed'])}: {', '.join(r['removed'])}")
        for name in r["missing"]:
            print(f"  not present: {name}", file=sys.stderr)
        print(f"{r['remaining']} dataset(s) remain")

    _emit(result, args.json, render)
    return 0


def cmd_ls(args: argparse.Namespace) -> int:
    names = _ops.list_datasets(args.collection, **_store_options(args))
    _emit(names, args.json, lambda n: print("\n".join(n)) if n else None)
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    detail = _ops.describe_dataset(args.collection, args.name, **_store_options(args))
    _emit(detail, args.json, _render_dataset)
    return 0


def _render_dataset(d: dict[str, Any]) -> None:
    """ncdump-style, because that is what people already know how to read."""
    print(f"dataset {d['name']} {{")

    if d["dimensions"]:
        print("dimensions:")
        for dim, size in d["dimensions"].items():
            print(f"\t{dim} = {size} ;")

    if d["arrays"]:
        print("variables:")
        for array in d["arrays"]:
            dims = ", ".join(array["dimensions"])
            marker = "  // coordinate" if array["is_coordinate"] else ""
            print(f"\t{array['dtype']} {array['name']}({dims}) ;{marker}")

            if array["chunk_shape"] != array["shape"]:
                print(f"\t\t{array['name']}:_ChunkShape = {array['chunk_shape']} ;")
            if array["fill_value"] is not None:
                print(
                    f"\t\t{array['name']}:_FillValue = "
                    f"{_format_value(array['fill_value'])} ;"
                )
            for key, value in array["attributes"].items():
                print(f"\t\t{array['name']}:{key} = {_format_value(value)} ;")

            stats = _format_stats(array["stats"])
            if stats:
                print(f"\t\t// stats: {stats}")

    if d["attributes"]:
        print("\n// global attributes:")
        for key, value in d["attributes"].items():
            print(f"\t\t:{key} = {_format_value(value)} ;")

    start, end = d["segment_range"]
    print(f"\n// ordinal {d['ordinal']}, segment bytes {start}..{end}")
    print("}")


def cmd_info(args: argparse.Namespace) -> int:
    summary = _ops.info(args.collection, **_store_options(args))

    def render(s: dict[str, Any]) -> None:
        print(f"collection {s['source']}")
        print(f"  format version    {s['format_version']}")
        print(f"  created           {_timestamp(s['created_unix_ms'])}")
        print(f"  codec             {s['codec']}")
        print(f"  container size    {_human_bytes(s['container_bytes'])}")
        print(f"  datasets          {s['dataset_count']}")
        if s["deleted_count"]:
            print(
                f"  removed           {s['deleted_count']} "
                f"(of {s['total_datasets']} written; space not reclaimed)"
            )
        print(f"  interned schemas  {s['interned_schemas']}")
        arrays = s["distinct_arrays"]
        print(f"  distinct arrays   {len(arrays)}")
        for name in arrays:
            print(f"      {name}")

    _emit(summary, args.json, render)
    return 0


# ── argument parsing ─────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atlas",
        description=(
            "Build and inspect atlas collections: thousands of NetCDF datasets "
            "in one immutable file, local or on object storage."
        ),
        epilog=(
            "A collection is a local path or a URL: s3://bucket/prefix, "
            "gs://..., az://..., https://... . Remote sources need "
            '`pip install "atlas-python[cloud]"`.'
        ),
    )
    parser.add_argument("--version", action="version", version=f"atlas {__version__}")
    subs = parser.add_subparsers(dest="command", required=True)

    # create
    p = subs.add_parser(
        "create",
        help="build a collection from a directory of NetCDF files",
        description=(
            "Each NetCDF file becomes one dataset, named after the file stem. "
            "Nothing is readable at the destination until every file is "
            "written, so a failure leaves no half-built collection."
        ),
    )
    p.add_argument("directory", help="directory holding the NetCDF files")
    p.add_argument("destination", help="where to write the collection")
    p.add_argument("-r", "--recursive", action="store_true", help="descend into subdirectories")
    p.add_argument(
        "--codec",
        default="zstd",
        choices=["zstd", "lz4", "none"],
        help="block compression (default: zstd)",
    )
    p.add_argument(
        "--chunks",
        metavar="JSON",
        help='per-variable chunk shape, e.g. \'{"temperature": [64, 64]}\'',
    )
    p.add_argument(
        "--skip-errors",
        action="store_true",
        help="skip files that fail instead of abandoning the collection",
    )
    p.add_argument("-q", "--quiet", action="store_true", help="do not list files as they are written")
    p.set_defaults(func=cmd_create)

    # rm
    p = subs.add_parser(
        "rm",
        help="remove datasets from a collection",
        description=(
            "Updates the deletion mask beside the container in one call. The "
            "container is untouched, so no space is reclaimed and no ordinal "
            "moves. Names may be given as dataset names or NetCDF paths."
        ),
    )
    p.add_argument("collection")
    p.add_argument("names", nargs="+", help="dataset names, or the NetCDF files they came from")
    p.add_argument(
        "--missing-ok",
        action="store_true",
        help="report names that are absent instead of failing",
    )
    p.set_defaults(func=cmd_rm)

    # ls
    p = subs.add_parser(
        "ls", help="list the datasets in a collection", description="Removed datasets are not listed."
    )
    p.add_argument("collection")
    p.set_defaults(func=cmd_ls)

    # show
    p = subs.add_parser(
        "show",
        help="show one dataset in ncdump style",
        description=(
            "Dimensions, every array's type, shape, chunking, fill value and "
            "attributes, and the statistics recorded when it was written."
        ),
    )
    p.add_argument("collection")
    p.add_argument("name", help="dataset name, or the NetCDF file it came from")
    p.set_defaults(func=cmd_show)

    # info
    p = subs.add_parser(
        "info",
        help="summarise the whole collection",
        description="Dataset counts, size, codec, and the distinct array names.",
    )
    p.add_argument("collection")
    p.set_defaults(func=cmd_info)

    for sub in subs.choices.values():
        sub.add_argument("--json", action="store_true", help="emit JSON instead of text")
        _add_store_flags(sub)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (_ops.AtlasError, _source.SourceError) as exc:
        print(f"atlas: {exc}", file=sys.stderr)
        return 1
    except KeyError as exc:
        print(f"atlas: not found: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 1
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"atlas: {exc}", file=sys.stderr)
        return 1
    except BrokenPipeError:  # `atlas ls ... | head`
        return 0


if __name__ == "__main__":
    sys.exit(main())
