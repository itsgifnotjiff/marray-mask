"""Auto-build `build/libcmc_marray_bridge.so` when `MARRAY_CMC_IO` (or sibling `../cmc-io`) is present.

Invoked automatically by `pixi run build-ext` (or standalone via `pixi run build-cmc-bridge`).
  1. Locates `<cmc_root>/src/mojo/cmc_io_fst/xdf/unpack_batch.mojo` via `MARRAY_CMC_IO`
     or relative `../cmc-io` / `../boreal-io` siblings (no hardcoded host paths).
  2. Compiles `bridges/cmc_io_cabi_shim.mojo` -> `build/libcmc_marray_bridge.so`.
  3. If `unpack_one_f32` in `unpack_batch.mojo` has a non-void return type, `raises`, or
     reordered scalar parameters, parses the exact `def/fn unpack_one_f32(...)` signature
     from `unpack_batch.mojo` and compiles an adapted shim into `build/libcmc_marray_bridge.so`.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIM_PATH = REPO_ROOT / "bridges" / "cmc_io_cabi_shim.mojo"
OUT_SO = REPO_ROOT / "build" / "libcmc_marray_bridge.so"


def find_cmc_mojo_dir(override: str | Path | None = None) -> Path | None:
    cands: list[Path] = []
    if override is not None:
        cands.append(Path(override).expanduser())
    env_c = os.environ.get("MARRAY_CMC_IO", "").strip()
    if env_c:
        cands.append(Path(env_c).expanduser())
    for rel in ("../cmc-io", "../boreal-io"):
        cands.append((REPO_ROOT / rel).resolve())

    for root in cands:
        for sub in (root / "src" / "mojo", root / "mojo", root):
            if (sub / "cmc_io_fst" / "xdf" / "unpack_batch.mojo").is_file():
                return sub.resolve()
    return None


def _run_mojo_build(mojo_bin: str, include_dir: Path, src_file: Path, out_so: Path) -> subprocess.CompletedProcess[str]:
    out_so.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        mojo_bin,
        "build",
        "-O3",
        "--emit",
        "shared-lib",
        "-I",
        str(include_dir),
        str(src_file),
        "-o",
        str(out_so),
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def _generate_adapted_shim(unpack_batch_path: Path) -> str | None:
    text = unpack_batch_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(
        r"(?:def|fn)\s+unpack_one_f32\s*\((.*?)\)\s*(raises\b)?\s*(?:->\s*([^\s:]+))?\s*:",
        text,
        re.DOTALL,
    )
    if not m:
        return None
    params_raw, is_raises, ret_type = m.group(1), bool(m.group(2)), (m.group(3) or "").strip()
    has_mutptr = bool(re.search(r"\bMutPtr\b", text))

    arg_exprs: list[str] = []
    for part in params_raw.split(","):
        p = part.strip()
        if not p:
            continue
        pname = p.split(":")[0].strip().split()[-1]
        ptype = p.split(":")[1].strip().split("=")[0].strip() if ":" in p else "Int"
        pl = pname.lower()
        if "dst" in pl and ("ptr" in ptype.lower() or "pointer" in ptype.lower() or "f32" in pl or pl == "dst" or pl == "out"):
            arg_exprs.append("dst_p")
        elif ("raw" in pl or "src" in pl or "buf" in pl) and ("len" not in pl and "nbyte" not in pl and "size" not in pl):
            arg_exprs.append("raw_p")
        elif "raw_len" in pl or "buf_len" in pl or "src_len" in pl:
            arg_exprs.append("Int(raw_byte_len)")
        elif "byte_off" in pl:
            arg_exprs.append("0")
        elif "nbyte" in pl:
            arg_exprs.append("Int(raw_byte_len)")
        elif "addr" in pl or "swa" in pl:
            arg_exprs.append("1")
        elif "lng" in pl:
            arg_exprs.append("rpn_lng")
        elif pl == "ni":
            arg_exprs.append("Int(ni)")
        elif pl == "nj":
            arg_exprs.append("Int(nj)")
        elif pl == "nk":
            arg_exprs.append("max(1, Int(nk))")
        elif "datyp" in pl or pl == "dt":
            arg_exprs.append("dt")
        elif "nbit" in pl or pl == "nb":
            arg_exprs.append("Int(nbits)")
        elif "off" in pl or "start" in pl:
            arg_exprs.append("0")
        else:
            return None

    call_prefix = "_ = " if (ret_type and ret_type != "None") else ""
    call_stmt = f"{call_prefix}unpack_one_f32({', '.join(arg_exprs)})"
    if is_raises:
        body_call = f"    try:\n        {call_stmt}\n    except:\n        return Int32(-2)"
    else:
        body_call = f"    {call_stmt}"

    if has_mutptr:
        imp_line = "from cmc_io_fst.xdf.unpack_batch import MutPtr, unpack_one_f32"
        ptr_setup = (
            "    var raw_p = MutPtr[UInt8](unsafe_from_address=Int(raw_ptr))\n"
            "    var dst_p = MutPtr[Float32](unsafe_from_address=Int(dst_f32))"
        )
    else:
        imp_line = "from std.memory import UnsafePointer, MutExternalOrigin\nfrom cmc_io_fst.xdf.unpack_batch import unpack_one_f32"
        ptr_setup = (
            "    var raw_p = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(raw_ptr))\n"
            "    var dst_p = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(dst_f32))"
        )

    return f"""from std.math import max
from std.memory import ImmPointer, MutPointer
{imp_line}

@export("cmc_fst_chunk_decode_cabi")
def cmc_fst_chunk_decode_cabi(
    ctx_addr: Int64,
    raw_ptr: ImmPointer[UInt8, ImmUntrackedOrigin],
    raw_byte_len: Int64,
    datyp: Int32,
    nbits: Int32,
    ni: Int32,
    nj: Int32,
    nk: Int32,
    dst_f32: MutPointer[Float32, MutUntrackedOrigin],
) abi("C") -> Int32:
    var raw_dt = Int(datyp)
    var dt = (raw_dt - 64) if (raw_dt >= 64 and raw_dt < 128) else raw_dt
{ptr_setup}
    var rpn_lng = max(1, Int(raw_byte_len) >> 3)
{body_call}
    return Int32(0)
"""


def build_bridge(cmc_override: str | Path | None = None, out_so: Path = OUT_SO) -> Path | None:
    mojo_dir = find_cmc_mojo_dir(cmc_override)
    if mojo_dir is None:
        return None
    mojo_bin = shutil.which("mojo")
    if mojo_bin is None:
        return None

    res = _run_mojo_build(mojo_bin, mojo_dir, SHIM_PATH, out_so)
    if res.returncode == 0 and out_so.is_file():
        print(f"[marray-mask] Built Tier-A C-ABI bridge: {out_so.name} (against {mojo_dir.name})")
        return out_so

    unpack_batch = mojo_dir / "cmc_io_fst" / "xdf" / "unpack_batch.mojo"
    adapted = _generate_adapted_shim(unpack_batch)
    if adapted is not None:
        gen_path = out_so.parent / "_cmc_cabi_adapted.mojo"
        gen_path.write_text(adapted, encoding="utf-8")
        res2 = _run_mojo_build(mojo_bin, mojo_dir, gen_path, out_so)
        if res2.returncode == 0 and out_so.is_file():
            print(f"[marray-mask] Built adapted Tier-A C-ABI bridge: {out_so.name}")
            return out_so
        print(f"[marray-mask] WARN: Tier-A bridge build failed:\n{res2.stderr}", file=sys.stderr)
    else:
        print(f"[marray-mask] WARN: Tier-A bridge build failed:\n{res.stderr}", file=sys.stderr)
    return None


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    override = args[0] if args else None
    build_bridge(override)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
