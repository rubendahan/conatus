"""tests/core exercises the compiled C++ core, which Conatus requires (it is never optional: the
banded solves and the IMEX integrators live there). A core that cannot be imported fails the
collection loudly, with the reason and the fix (review finding 98: these tests used to
be dropped with a print, so a broken build passed as "nothing to test")."""
try:
    import conatus.runtime  # noqa: F401
except ImportError as exc:
    # the fix comes from the runtime's own error (it knows a development checkout from an
    # installed package, and the public export rewrites its advice): a hard-coded development
    # build command pointed public users at files their tree does not have (critic review of the
    # runtime lot)
    raise ImportError(f"tests/core: the C++ core cannot be imported, and the core tests are never "
                      f"skipped: {exc}") from exc
