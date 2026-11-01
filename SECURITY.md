# Security

## Reporting a vulnerability

Please report it privately through a GitHub security advisory on
[rubendahan/conatus](https://github.com/rubendahan/conatus/security/advisories/new). Do not open a
public issue. Fixes go into the next release, and only the latest release gets them.

## What Conatus trusts

Conatus runs on your own account. Treat these inputs as you would treat code.

- **A run registry carries code.** A run record names a function and its arguments, and
  `run.reproduce()` can call it. `reproduce()` needs `fn=` or an explicit `trust_record=True`; pass
  `trust_record=True` only for records you made. Loading and searching a registry run nothing.
- **A checkpoint is data, but a large one.** The loaders refuse pickled arrays and HDF5 files that reach other
  files (links, external storage, virtual datasets). A malformed file can still cost memory and time. Read files from people you trust.
- **A translated script is a program.** `from_dedalus` reads a script without executing it. The file
  `to_script()` writes is a program: read it before you run it.
- **SymPy strings are code.** Functions that take a base flow or a coefficient as an expression refuse
  a string; pass a SymPy expression, a number or a callable.

## The Studio and the hub

- `hub serve --local` listens on the loopback address only, with a token stored in a file of mode 0600
  in a directory of mode 0700. Any account on the same machine can reach the port, so the token is what
  protects it.
- Keep the spool directory private. It holds what each run reports and the control file that sends
  commands to the run.
- Agent tokens default to the read scope. A command token lets its holder confirm its own stop or fork:
  the two-step confirmation guards against a slip, and it does not prove authority.
- Text a run writes to its log reaches whatever reads it, an LLM included. Treat it as untrusted input.
- Over plain HTTP a client cannot authenticate the server, so use plain HTTP only to localhost.
  `HubClient` sends a token to its own origin and refuses redirects to anywhere else.
- The hosted mode is not part of the public release.

## The MCP server

`python -m conatus mcp` (extra `mcp`) talks to one client on stdin and stdout and listens on no port.

- It is read-only unless started with `--allow <command>`. Each opt-in adds one named tool, states what
  it permits and adds no authority: token scopes, declared bounds and rate limits still decide.
- With `--allow stop` or `--allow fork` an agent can act without a person. The two-step confirmation
  guards against a slip.
- No tool runs a script, submits a job or calls `run.reproduce()`, and no argument is evaluated.
- A run's log and a record reach the model marked as data. The marking does not make a model immune to what
  a log says, so keep write tools off when a run can log text you did not write.

## Dependencies

Run `pip-audit` on your own environment. The `studio` extra needs Tornado 6.6 or later. The package has
no install hook.
