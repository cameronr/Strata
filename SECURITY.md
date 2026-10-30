# Security

## Reporting a vulnerability

Report what should stay private until it is fixed through GitHub's private vulnerability reporting: the repository's
**Security** tab, **Report a vulnerability**. Say what you found, how to reproduce it (the request, the config keys
involved, the commit) and what an attacker gains. Findings that are safe to discuss in public can go to an
[issue](https://github.com/eddoursul/Lamina/issues). Fixes go into `main`.

## What the server exposes

The server (`lamina-server`) serves the model over HTTP. Its settings: [USING.md](docs/USING.md#from-other-devices).

- **Where it listens.** `127.0.0.1` by default: this PC only. `"host": "0.0.0.0"` in the config (or `--host`) opens it
  to the network, and the server warns when no API key is set.
- **API key.** `"api_key"` in the config (or `STRATA_API_KEY`) is required on `/v1/*`, on every endpoint that shows the
  model's state, requests or answers, and on every `POST`; it is compared in constant time. Set one before the server
  is reachable from other machines.
- **Host check.** Without an API key the server answers only requests whose `Host` is a name it knows (`localhost`, an
  IP address, the address it listens on and, when it listens beyond this PC, this PC's name), against DNS rebinding.
  `"allowed_hosts"` adds names.
- **Origin check.** Without an API key, a browser's `POST` to `/v1/*` from another site's page is refused unless that
  origin is the server's own page, `localhost`, an allowed host, or listed in `"trusted_origins"` or `"cors_origins"`.
  The settings (`/settings`, `/config`: a few keys of the config, never the network, key, MCP or program keys),
  `/load`, `/unload` and the MCP tools take JSON from the server's own page (or `"trusted_origins"`) only.
- **CORS** is off unless `"cors_origins"` lists origins, and then only for `/v1/*`.
- **MCP tools** run only the servers the config names, for requests from the server's own page that ask for them. They
  run with your user's rights, and the model decides when to call them.
- **The request monitor** (`/api-monitor`, the last prompts and answers in memory) is off unless `"api_monitor": true`.
  Apart from it, the server keeps the text and tokens of its last few prompts in memory to continue the next turn; no
  endpoint shows them, and they are gone when it stops.

## What runs and what is downloaded

Setup downloads llama.cpp at a pinned commit and the model files at pinned Hugging Face revisions, each checked by
size and SHA-256, and installs the Python packages of `requirements.txt` with pip. The engine and the image encoder are
built on your PC from this repository. Running a model downloads nothing; the server fetches only image URLs a request
sends (`--offline` refuses them), and the MCP servers and image generation the config sets up.
