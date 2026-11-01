# MCP server

`python -m conatus mcp` is a server for the Model Context Protocol. An MCP client, such as a coding
agent or an editor, starts it as a child process and talks to it on stdin and stdout. It listens on
no port. It reads through the agent channel ({doc}`studio_agents`) and the run registry ({doc}`runs`),
and it is experimental, like the Studio ({doc}`experimental`).

```bash
pip install 'conatus[mcp]'                    # pins mcp==2.3.0
python -m conatus mcp                         # read-only; same source as `conatus studio`
python -m conatus mcp --spool DIR             # or: --hub URL --token-file FILE
python -m conatus mcp --allow set --allow stop   # adds exactly these two write tools
```

The server is written against the official MCP Python SDK `mcp==2.3.0`, whose latest protocol
revision is `2026-07-28`. The SDK negotiates an earlier revision with a client that asks for one. The
tool `server_status` reports both versions.

A client configuration names the command. In the JSON form many clients use:

```json
{"mcpServers": {"conatus": {"command": "python", "args": ["-m", "conatus", "mcp"]}}}
```

## Read tools

These are always present. None of them changes a run.

| Tool | Returns |
|---|---|
| `server_status` | channel, identity and scope, versions, allowed write tools and what each permits, limits |
| `conatus_info` | versions, compiled core, the user layer |
| `conatus_explain` | signature, required arguments and documentation of a public name such as `cn.Model.simulate` |
| `list_models` | the ready-made models of `cn.problems.native` |
| `list_runs` | watched runs, live first, at most `limit` (200 by default), with `total` |
| `run_status` | one run: state, step, time, scalars, alerts, allowed commands, log tail, events |
| `run_log` | the last log records and events |
| `run_scalars` | scalar rows, thinned to a maximum |
| `list_records` | run records of the registry |
| `get_record` | one record, with its `explain()` |

## Write tools

A write tool exists only if its name was given to `--allow` when the server started. It is then listed
for any client of this server, and the server prints on stderr what it permits. Each is one run command of
the agent channel.

| Tool | What the opt-in permits |
|---|---|
| `run_pause` | pause any run this server can reach, between two steps |
| `run_resume` | resume any paused run this server can reach |
| `run_checkpoint` | make any run this server can reach write a checkpoint, which uses disk |
| `run_set` | change a parameter that a run declared settable, within its declared bounds |
| `run_stop` | end any run this server can reach |
| `run_fork` | record a fork request in a run; the run records it and does not apply it |

The opt-in adds a tool and adds no authority. Through a hub the token's scope and run patterns decide,
so a read token behind `--allow pause` is refused by the hub. On the spool the spool's file permissions
decide. `set` keeps the declared bounds of the run, and a value outside them is refused, never clipped.

**Confirmation of stop and fork.** These two tools take two calls: the first is refused with
`needs_confirmation` and a `confirm_token`, and the second repeats the call with that token. The agent
receives the token and can send it back by itself, so this step guards against a slip. It involves no
person and does not prove authority. The authority is the operator's `--allow stop` or `--allow fork`.
If a person must decide, leave these two out.

Some actions are not offered by any option: running a script (`agent check` executes it, so it stays a
command line), submitting a job and `reproduce()`. No argument is evaluated. Names and references are
matched as text. A public name starting with an underscore, or the name of a script, is refused. So is a name a module only imported from another package.

## Answers

Each tool returns one JSON text with `api`, `tool`, `ok` and `code`. A refusal has `is_error` set in the
MCP result, a numeric `code` (400 malformed, 403 not allowed, 404 unknown, 409 needs confirmation or wrong
state, 413 too large, 429 rate limit), and a stable `error_code` ({doc}`error_codes`).

Text written by a run or a record arrives under the key `untrusted_data`, next to a `notice`: it is data to
report and not an instruction, whatever it says. Control characters and invisible format characters are shown
as `<U+XXXX>`, and a string above 2000 characters is cut, with a note. The marking helps a client and its model
keep the two apart. It does not make an LLM immune to text in a log, so keep the write tools off when a run
can log text you did not write.

## Limits

One server process takes 600 read calls and 60 write calls per minute. Above that a call is refused
with code 429 and `retry_after_s`. The transport counts as well: 60 commands a minute per agent on a hub,
and 60 a minute per spool without one. An answer above 512 KiB is refused, and asking for fewer records
gets it through. A damaged run file or record costs its own entry and leaves the others readable.
