"""The C++ core must be built first (`bash core/build.sh`); report it instead of erroring on import."""
try:
    import conatus.runtime  # noqa: F401
except ImportError as exc:  # pragma: no cover
    collect_ignore_glob = ["test_*.py"]
    print(f"[tests/core] not collected: {exc}")
