// src/qwen35/q36_main.cpp - lamina-q36: the Qwen3.6 engine path on its own (the same entry point `lamina` switches
// to for a qwen35moe model).
#include "lamina/qwen35/engine.hpp"

int main(int argc, char** argv) { return lamina::qwen35::run(argc, argv); }
