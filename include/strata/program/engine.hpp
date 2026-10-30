// include/strata/program/engine.hpp - the engine as a function: the `lamina` binary's main, and the engine that
// lamina-server runs on a thread of its own (with in-process channels: serve_io.hpp).
#pragma once

namespace strata::program {

/// The engine's command line (argv[0] is the program's name). Returns its exit code.
int run(int argc, char** argv);

}  // namespace strata::program
