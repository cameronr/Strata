"""tools/mmproj_q8.py - an image projector GGUF in BF16 as the Q8_0 file llama.cpp's llama-quantize makes of it: each
BF16 matrix whose rows are whole 32-value blocks in Q8_0, every other BF16 tensor in F16, the F32 tensors as they are,
general.file_type 7 (MOSTLY_Q8_0).  The CPU image encoder reads a Q8_0 projector ~20% faster than the BF16 one.
gguf-py's Q8_0 is ggml's reference rounding: every tensor comes out byte for byte llama-quantize's, the metadata
the same (in the source's order; llama-quantize moves the file type keys to the end and orders the tensors otherwise).

    python tools/mmproj_q8.py <mmproj-BF16.gguf> <out-Q8_0.gguf>   (LAMINA_GGUF_PY: gguf-py's folder, tools/_paths.py)
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("LAMINA_GGUF_PY") or str(Path(__file__).resolve().parents[1] / "third_party/llama.cpp/gguf-py"))
import gguf  # noqa: E402

FILE_TYPE_Q8_0 = 7
SKIP = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}


def convert(src: Path, dst: Path) -> None:
    r = gguf.GGUFReader(src)
    arch = r.fields["general.architecture"].contents()
    tmp = dst.with_name(dst.name + ".part")
    w = gguf.GGUFWriter(tmp, arch)
    for f in r.fields.values():
        if f.name in SKIP:
            continue
        if f.name == "general.file_type":
            w.add_file_type(FILE_TYPE_Q8_0)
            continue
        vtype = f.types[0]
        sub = f.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(f.name, f.contents(), vtype, sub_type=sub)
    BF16, F32, Q8_0 = gguf.GGMLQuantizationType.BF16, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.Q8_0
    for t in r.tensors:
        if t.tensor_type == F32:
            w.add_tensor(t.name, np.asarray(t.data))
        elif t.tensor_type == BF16:
            x = gguf.quants.dequantize(t.data, BF16)       # float32, the BF16 values exactly
            if x.ndim == 2 and x.shape[-1] % 32 == 0:
                w.add_tensor(t.name, gguf.quants.quantize(x, Q8_0), raw_dtype=Q8_0)
            else:
                w.add_tensor(t.name, x.astype(np.float16))
        else:
            raise SystemExit(f"{src}: tensor {t.name} is {t.tensor_type.name}, not BF16 or F32")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    os.replace(tmp, dst)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"{sys.argv[2]}: written")
