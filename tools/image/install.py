"""tools/image/install.py - image generation (Qwen-Image-2.1-Turbo through stable-diffusion.cpp's sd-cli) for a Lamina
install: it compiles sd-cli for CUDA (or the CPU), downloads and checks the model's three files, and writes the
"images" section into a run config, or prints it. The server then answers POST /v1/images/generations. Run it with
the Python setup made, after setup (it uses setup.py's toolchain and downloads):

    .venv/Scripts/python.exe tools/image/install.py --config lamina-iq3_s.json      (the quant by the GPU's VRAM)
    .venv/bin/python tools/image/install.py --backend cpu --print

--gpu-policy auto|keep|unload: what the text model does during an image job (serve/images.py).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import qimage as QI  # noqa: E402
import setup as S  # noqa: E402


def sd_source() -> Path:
    """stable-diffusion.cpp and its ggml at the pinned commits, from GitHub's zips (no git)."""
    src = S.ROOT / "third_party" / "stable-diffusion.cpp"
    if QI.source_ok(src):
        return src
    zips = [S.ROOT / "third_party" / f"stable-diffusion.cpp-{QI.SD_CPP_COMMIT[:8]}.zip",
            S.ROOT / "third_party" / f"sd-ggml-{QI.SD_GGML_COMMIT[:8]}.zip"]
    S.get(QI.SD_CPP_ZIP, zips[0], "stable-diffusion.cpp")
    S.get(QI.SD_GGML_ZIP, zips[1], "stable-diffusion.cpp's ggml")
    src = QI.unpack_source(S.ROOT / "third_party", *zips)
    for z in zips:
        z.unlink()
    return src


def choose_quant(backend: str, gpu_gb: float, asked: str | None) -> str:
    """The quant for the GPU's memory; one asked for above it is kept, with a note (stable-diffusion.cpp then keeps
    part of the model in RAM, which is slower)."""
    rec = QI.pick_quant(gpu_gb) if backend != "cpu" else "Q4_K_M"
    if asked and QI.QUANTS.index(asked) > QI.QUANTS.index(rec):
        S.warn(f"image model {asked}: {rec} fits {gpu_gb:.0f} GB of GPU memory; {asked} kept as asked")
    return asked or rec


def build(tc: S.Toolchain, backend: str, archs: str | None) -> tuple[Path, list[str]]:
    """sd-cli for the backend, in build-image-<backend> (again only when the pins, the backend or the CUDA
    architectures changed) -> its path and the library folders it runs with."""
    b = S.ROOT / f"build-image-{backend}"
    exe, stamp = b / "bin" / QI.SD_EXE, b / "lamina-image.json"
    want = QI.build_stamp(backend, archs)
    if exe.exists() and stamp.exists() and json.loads(stamp.read_text(encoding="utf-8")) == want:
        S.ok(f"the image generator ({backend}) is built")
    else:
        src = sd_source()
        defs = QI.cmake_defs(backend, arch=archs)
        if backend == "cuda":
            spec = importlib.util.spec_from_file_location("sdcpp_native_apply", QI.NATIVE_DIR / "apply.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.apply(src)                             # the native DiT runner, used when the config's "native" is on
            defs += S.cuda_defs(tc.nvcc)
        S.say(f"  compiling the image generator (stable-diffusion.cpp {QI.SD_CPP_COMMIT[:8]}, {backend}) ...")
        S.cmake_build(tc, src, b, "sd-cli", defs, S.vs_environment(tc.vcvars) if S.WIN else S.clean_env())
        stamp.write_text(json.dumps(want), encoding="utf-8")
        S.ok(f"the image generator: {exe}")
    if backend != "cuda":
        return exe, []
    if S.WIN:
        return exe, sorted({str(d.parent) for d in S.cuda_dlls(tc.nvcc)})
    lib = S.cuda_lib_dir(tc.nvcc)
    return exe, [lib] if lib else []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=("cuda", "cpu"), help="default: cuda with an NVIDIA GPU, else cpu")
    ap.add_argument("--arch", help="the CUDA architectures to compile for, e.g. 86;120 (default: this PC's GPUs)")
    ap.add_argument("--quant", choices=QI.QUANTS, help="default: by the largest GPU's VRAM")
    ap.add_argument("--models-dir", default=str(S.ROOT.parent / "Lamina-data" / "models"),
                    help="where the model files go (default: setup's data folder's models)")
    ap.add_argument("--config", help="a run config (lamina-<model>.json) to write the \"images\" section into")
    ap.add_argument("--gpu-policy", choices=["auto", "keep", "unload"], default="auto")
    ap.add_argument("--gpu", type=int, help="the GPU's number for the image job (default: CUDA's first)")
    ap.add_argument("--native", choices=["on", "off"], default="off",
                    help="EXPERIMENTAL: the native DiT runner on CUDA (the config's \"native\")")
    ap.add_argument("--build-only", action="store_true", help="compile sd-cli only")
    ap.add_argument("--print", action="store_true", help="print the images section instead of writing it")
    a = ap.parse_args(argv)
    usable = [g for g in S.gpus() if g.cc >= 80]
    backend = a.backend or ("cuda" if usable else "cpu")
    tc = S.toolchain()
    if not tc.cmake or not tc.ninja:
        S.fail("CMake and Ninja are not in this Python environment", "run setup first (setup.bat or setup.sh)")
    if S.WIN and not tc.vcvars:
        S.fail("Visual Studio's C++ tools were not found", "setup.bat --check names what the builds need")
    if backend == "cuda" and not tc.nvcc:
        S.fail("the CUDA build needs the CUDA toolkit (nvcc)", "or --backend cpu")
    archs = a.arch or (S.cuda_archs(usable) if backend == "cuda" and usable else None)
    if backend == "cuda" and not archs:
        S.fail("no NVIDIA GPU of compute capability 8.0 or newer was found", "pass --arch (e.g. 86) or --backend cpu")
    gpu_gb = S.pick_main(usable).mem_mib / 1024 if usable and backend == "cuda" else 0.0
    quant = choose_quant(backend, gpu_gb, a.quant)
    S.ok(f"image generation: {quant} on {backend}" + (f" ({gpu_gb:.0f} GB of GPU memory)" if gpu_gb else ""))
    exe, lib_dirs = build(tc, backend, archs)
    if a.build_only:
        return 0
    models = Path(a.models_dir).expanduser().resolve()
    S.say("  " + QI.LICENSE_NOTE)
    for name in (quant, "vae", "llm"):
        _, _, size, sha = QI.FILES[name]
        S.download(QI.file_url(name, S.hf_endpoint()), QI.local_path(models, name), size, sha)
    section = QI.image_config(backend, quant, exe, models, gpu=a.gpu, lib_dirs=lib_dirs, policy=a.gpu_policy,
                              native=a.native == "on")
    if a.print or not a.config:
        print(json.dumps({"images": section}, indent=1))
        return 0
    path = Path(a.config)
    cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    cfg["images"] = section
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8", newline="\n")
    S.ok(f"{path.name}: image generation added (POST /v1/images/generations once the server starts again)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
