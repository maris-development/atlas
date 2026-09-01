"""The `atlas` command.

Five subcommands, one per operation:

    atlas create <netcdf-dir> <collection>   build a collection
    atlas rm     <collection> <name>...      remove datasets
    atlas ls     <collection>                list datasets
    atlas show   <collection> <name>         one dataset, ncdump style
    atlas info   <collection>                the whole collection

Every one takes a local path or a URL (`s3://`, `gs://`, `az://`, `https://`)
for the collection. The same command therefore works against a bucket.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from typing import Any, Optional, Sequence

from . import __version__, _log, _ops, _source

_LOG = _log.get_logger("cli")


# ── formatting helpers ───────────────────────────────────────────────


def _human_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"  # pragma: no cover - unreachable


def _format_value(value: Any) -> str:
    """Renders an attribute or a statistic the way ncdump does."""
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
    """Backend settings every subcommand shares."""
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


def _parse_open_chunks(value: str) -> Any:
    """`--open-chunks` takes a mode name or a JSON dict."""
    if value in ("auto", "native"):
        return value
    if value == "none":
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        raise _ops.AtlasError(
            f"--open-chunks must be auto, native, none, or a JSON dict; got {value!r}"
        ) from None
    if not isinstance(parsed, dict):
        raise _ops.AtlasError(
            f"--open-chunks JSON must be an object, e.g. '{{\"time\": 100}}'"
        )
    return parsed


def cmd_create(args: argparse.Namespace) -> int:
    chunks = json.loads(args.chunks) if args.chunks else None
    open_chunks = _parse_open_chunks(args.open_chunks)

    def progress(name: str) -> None:
        if not args.quiet:
            print(f"  {name}", file=sys.stderr)

    if not args.quiet:
        print(f"Writing {args.destination}", file=sys.stderr)

    result = _ops.create(
        args.directory,
        args.destination,
        recursive=not args.no_recursive,
        codec=args.codec,
        chunks=chunks,
        open_chunks=open_chunks,
        chunk_size=args.chunk_size,
        on_error="skip" if args.skip_errors else "stop",
        on_unsupported="skip" if args.skip_unsupported else "stop",
        progress=progress,
        **_store_options(args),
    )

    def render(r: dict[str, Any]) -> None:
        print(f"{r['dataset_count']} dataset(s) written to {r['destination']}")
        for item in r["skipped_arrays"]:
            print(
                f"  skipped array {item['dataset']}/{item['array']} "
                f"({item['dtype']})",
                file=sys.stderr,
            )
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
    """Prints in ncdump style, because people already read that format."""
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
        width = max((len(name) for name in arrays), default=0)
        for name in arrays:
            # Statistics for the whole collection, not for one dataset.
            stats = _format_stats(s["array_stats"].get(name))
            print(f"      {name:<{width}}  {stats}".rstrip())

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
            "Each NetCDF file becomes one dataset, named after the file. "
            "Nothing at the destination is readable until every file lands. A "
            "failure therefore leaves no half-built collection."
        ),
    )
    p.add_argument("directory", help="directory holding the NetCDF files")
    p.add_argument("destination", help="where to write the collection")
    p.add_argument(
        "--no-recursive",
        action="store_true",
        help="scan the top directory alone. The scan descends by default",
    )
    p.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="accepted for compatibility. The scan already descends",
    )
    p.add_argument(
        "--codec",
        default="zstd",
        choices=["zstd", "lz4", "none"],
        help="block compression (default: zstd)",
    )
    p.add_argument(
        "--chunks",
        metavar="JSON",
        help='per-variable stored chunk shape, e.g. \'{"temperature": [64, 64]}\'',
    )
    p.add_argument(
        "--open-chunks",
        metavar="MODE",
        default="auto",
        help=(
            "how to read the source files: 'auto' (the default, dask sizes "
            "blocks to --chunk-size), 'native' (the file's own chunking), "
            "'none' (read each variable whole), or a JSON dict of "
            "per-dimension sizes. This also sets the stored chunk shape, "
            "unless --chunks says otherwise"
        ),
    )
    p.add_argument(
        "--chunk-size",
        metavar="SIZE",
        default=_ops.DEFAULT_CHUNK_SIZE,
        help=(
            f"block size 'auto' aims at. It is about the memory ceiling per "
            f"variable (default: {_ops.DEFAULT_CHUNK_SIZE})"
        ),
    )
    p.add_argument(
        "--skip-errors",
        action="store_true",
        help="skip files that fail instead of abandoning the collection",
    )
    p.add_argument(
        "--skip-unsupported",
        action="store_true",
        help=(
            "leave out an array whose dtype atlas cannot store, instead of "
            "failing the file. The rest of the dataset still lands"
        ),
    )
    p.add_argument("-q", "--quiet", action="store_true", help="do not list files as they are written")
    p.set_defaults(func=cmd_create)

    # rm
    p = subs.add_parser(
        "rm",
        help="remove datasets from a collection",
        description=(
            "Updates the deletion mask beside the container, in one call. The "
            "container does not change, so this reclaims no space and moves no "
            "ordinal. A name is a dataset name or a NetCDF path."
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
            "Dimensions, and for every array its type, shape, chunking, fill "
            "value, attributes, and the statistics the write recorded."
        ),
    )
    p.add_argument("collection")
    p.add_argument("name", help="dataset name, or the NetCDF file it came from")
    p.set_defaults(func=cmd_show)

    # info
    p = subs.add_parser(
        "info",
        help="summarise the whole collection",
        description=(
            "Dataset counts, size, codec, and one set of statistics per "
            "distinct array, over the whole collection."
        ),
    )
    p.add_argument("collection")
    p.set_defaults(func=cmd_info)

    for sub in subs.choices.values():
        sub.add_argument("--json", action="store_true", help="emit JSON instead of text")
        sub.add_argument(
            "--log-file",
            metavar="PATH",
            help="append errors and warnings to this file",
        )
        _add_store_flags(sub)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.log_file:
        try:
            _log.log_to_file(args.log_file)
        except OSError as exc:
            print(f"atlas: cannot open log file: {exc}", file=sys.stderr)
            return 1
        _LOG.info("atlas %s: %s", __version__, " ".join(argv or sys.argv[1:]))
    try:
        return args.func(args)
    except (_ops.AtlasError, _source.SourceError) as exc:
        return _fail(exc)
    except KeyError as exc:
        return _fail(exc, f"not found: {exc.args[0] if exc.args else exc}")
    except (ValueError, OSError, RuntimeError) as exc:
        return _fail(exc)
    except BrokenPipeError:  # `atlas ls ... | head`
        return 0


def _fail(exc: BaseException, message: Optional[str] = None) -> int:
    """Reports one error on stderr and in the log. Returns the exit code."""
    text = message or str(exc)
    _LOG.error("%s", _log.describe_exception(exc))
    print(f"atlas: {text}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
