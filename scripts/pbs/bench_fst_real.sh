#!/usr/bin/env bash
# Real-FST hybrid bench for marray-mask (boreal-io decode + M1 catalog / B2–B4).
#
# Required env (no absolute suite/checkout defaults in-tree):
#   MARRAY_CMC_IO       — boreal-io / cmc-io repo root (cmc_io_fst_*.so + boreal.io)
#   MARRAY_FST_PRES_ROOT — GDPS-style prog/pres directory of FST leads
#
# Submit example:
#   export MARRAY_CMC_IO=... MARRAY_FST_PRES_ROOT=...
#   ord_soumet scripts/pbs/bench_fst_real.sh -mach ppp8 -cpus 16 -mpi 1 \
#     -t 3600 -m 64G -w $(pwd) -jn marray-fst-real \
#     -args '--leads 4 --mode both'
#
# Login/vis smoke (explicit opt-in):
#   MARRAY_ALLOW_LOGIN=1 bash scripts/pbs/bench_fst_real.sh --smoke
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}"

if [[ -z "${PBS_JOBID:-}" && "${PBS_ENVIRONMENT:-}" != "PBS_BATCH" && "${MARRAY_ALLOW_LOGIN:-}" != "1" ]]; then
  echo "Refuse: not in PBS and MARRAY_ALLOW_LOGIN!=1 (use ord_soumet -mach ppp8)" >&2
  exit 1
fi

if [[ -z "${MARRAY_CMC_IO:-}" ]]; then
  echo "ERROR: set MARRAY_CMC_IO to the boreal-io / cmc-io checkout root" >&2
  exit 2
fi
if [[ -z "${MARRAY_FST_PRES_ROOT:-}" ]]; then
  echo "ERROR: set MARRAY_FST_PRES_ROOT to a prog/pres FST directory" >&2
  exit 2
fi

# Optional: PIXI_SSM path for r.load.dot when pixi is not already on PATH.
if ! command -v pixi >/dev/null 2>&1 && [[ -n "${PIXI_SSM:-}" ]] && command -v r.load.dot >/dev/null 2>&1; then
  set +eu
  # shellcheck disable=SC1091
  . r.load.dot "${PIXI_SSM}" || true
  set -eu
fi

export MARRAY_CMC_IO MARRAY_FST_PRES_ROOT
export PYTHONPATH="${MARRAY_CMC_IO}:${MARRAY_CMC_IO}/src:${REPO}/python:${PYTHONPATH:-}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

echo "bench_fst_real: host=$(hostname) PBS_JOBID=${PBS_JOBID:-}"
echo "bench_fst_real: MARRAY_CMC_IO=${MARRAY_CMC_IO}"
echo "bench_fst_real: MARRAY_FST_PRES_ROOT=${MARRAY_FST_PRES_ROOT}"

if [[ -x "${REPO}/.pixi/envs/default/bin/python" ]]; then
  PY="${REPO}/.pixi/envs/default/bin/python"
elif command -v pixi >/dev/null 2>&1; then
  (cd "${REPO}" && pixi install && pixi run build-ext)
  PY="$(cd "${REPO}" && pixi run python -c 'import sys; print(sys.executable)')"
else
  echo "ERROR: pixi / .pixi env not available in ${REPO} (load PIXI_SSM or install pixi)" >&2
  exit 2
fi

if [[ ! -f "${REPO}/build/marray_mojo.so" ]]; then
  (cd "${REPO}" && pixi run build-ext)
fi

echo "bench_fst_real: python=${PY}"
exec "${PY}" -u "${REPO}/benchmarks/bench_fst_real.py" "$@"
