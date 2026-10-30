# Contributing to Lamina

Issues and pull requests are welcome, AI-assisted ones too. Whoever submits a change answers for it: be able to explain
every line of it, and show what you ran.

## Reporting a problem

Search the [issues](https://github.com/eddoursul/Lamina/issues) first, closed ones too. A report says:

- what you did (the request, the command), what you expected and what happened, with the exact error;
- the commit (`git rev-parse --short HEAD`), the output of `setup --check`, and the config's `args` if you changed them;
- the engine's log (`lamina-<model>.log`) and the server's output, with keys and private text removed;
- whether you saw it happen or found it by reading the code (then say so: "not reproduced").

[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) has the common problems.

## Pull requests

Against `main`, one problem per pull request. Say what it fixes or adds, why this way, and what it does not do; link
the issue when there is one. For a large change or a new default, open an issue first.

- **The model stays the model.** Lamina computes every routed expert and every row; changes that approximate the model
  to gain speed (skipping experts, cutting rows) are not taken. Changes that round differently are, with their
  agreement measured (below).
- **Options for one kind of hardware are off by default.** A feature that pays on some PCs only (a RAM layout, a pair
  of GPUs, a link) goes behind an option, measured on the hardware it is for.
- **The engine's output.** Show what a change does to it. A change meant to keep the arithmetic is checked bitwise:
  the verify windows' hashes (`--window-hashes`) on a fixed text (`--spec-follow`, the token ids of a continuation;
  `--adapt-every 0` and a fixed `--expert-cache N`) before and after. A change of rounding is measured against
  llama.cpp: next-token distributions (`--window-logits`) along llama.cpp's own greedy text, mean KL and argmax
  agreement, beside the distance between llama.cpp's CPU and CUDA backends ([COMPARISON.md](docs/COMPARISON.md)).
- **Speed.** Before and after on the same PC and the same fixed text, several runs in both orders, with every changed
  setting named; the runs themselves, not only the best. Long prompts and long conversations come first: a gain on
  short texts alone does not decide a default.
- **Tests.** The engine builds without warnings; the test suite builds with `-DSTRATA_BUILD_TESTS=ON` and runs with
  `ctest`. The server's and the tools' tests run from the clone's folder:
  `python -m unittest discover -s serve -p "test_*.py"` and `python -m unittest discover -s tools -p "test_*.py"`.
- **Docs** describe what is, not what changed: update the ones your change makes wrong.

Report what you ran as passed, failed or not run, with the commands and the hardware. Never submit measurements,
logs or test results you did not get yourself.
