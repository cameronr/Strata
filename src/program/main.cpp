// src/program/main.cpp - the `lamina` binary: the engine on the command line, --serve on stdin and stdout.
#include "strata/program/engine.hpp"

#include <cstdio>

int main(int argc, char** argv) {
    // UNBUFFERED, BECAUSE THE INTERESTING OUTPUT IS THE OUTPUT BEFORE A CRASH. `stdout` redirected to a pipe or a
    // file is block-buffered, so a program that dies loses every line it had already printed.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    return strata::program::run(argc, argv);
}
