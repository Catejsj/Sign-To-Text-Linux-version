"""Make the CUDA libraries visible to onnxruntime-gpu before it builds a session.

Why this is needed
------------------
onnxruntime-gpu's CUDA provider (libonnxruntime_providers_cuda.so) links against
plain sonames — libcublas.so.12, libcudnn.so.9, libcurand.so.10, ... — and leaves
it to the dynamic loader to find them. On Linux we do not install a system CUDA
toolkit; the CUDA libraries we have are the ones bundled with the torch wheel,
under site-packages/nvidia/*/lib. That directory is not on the loader's search
path, so the provider fails to load.

The dangerous part is what happens next: onnxruntime does NOT raise. It logs a
warning, quietly drops to CPUExecutionProvider, and keeps returning correct
results — just far slower. Even an explicit
    InferenceSession(model, providers=['CUDAExecutionProvider'])
succeeds on CPU. So a silent CPU fallback is easy to ship without noticing.

Setting LD_LIBRARY_PATH from inside Python does not help, because glibc reads it
once at process start. Instead we dlopen each library with RTLD_GLOBAL: once the
soname is loaded into the process, the loader resolves the provider's NEEDED
entries against the already-open handles.

Call preload_cuda_libs() before the first InferenceSession is created.
"""

import ctypes
import glob
import logging
import os
import sys
import sysconfig

logger = logging.getLogger(__name__)

# The CUDA sonames libonnxruntime_providers_cuda.so declares as NEEDED.
# Keep in sync with:  objdump -p libonnxruntime_providers_cuda.so | grep NEEDED
_CUDA_SONAMES = [
    "libcudart.so.12",
    "libcublasLt.so.12",
    "libcublas.so.12",
    "libcurand.so.10",
    "libcufft.so.11",
    "libcudnn.so.9",
]

_done = False


def preload_cuda_libs() -> bool:
    """Load the CUDA libs bundled with torch so onnxruntime's CUDA provider
    can resolve them. Returns True if every library was loaded.

    Safe to call more than once; only the first call does any work.
    """
    global _done
    if _done:
        return True

    if sys.platform == "win32":
        _add_nvidia_dlls_windows()
        _done = True
        return True

    site_packages = sysconfig.get_paths()["purelib"]
    missing = []
    for soname in _CUDA_SONAMES:
        hits = glob.glob(os.path.join(site_packages, "nvidia", "*", "lib", soname))
        if not hits:
            missing.append(soname)
            continue
        try:
            ctypes.CDLL(hits[0], mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            missing.append(f"{soname} ({exc})")

    if missing:
        logger.warning(
            "CUDA libs not preloaded: %s. onnxruntime will fall back to CPU "
            "silently. Is torch installed from the cu12 index?",
            ", ".join(missing),
        )
        return False

    _done = True
    return True


def _add_nvidia_dlls_windows() -> None:
    venv_path = os.path.dirname(sys.executable)
    site_packages = os.path.join(os.path.dirname(venv_path), "Lib", "site-packages")
    for sub in ("cublas", "cudnn", "cuda_runtime"):
        path = os.path.join(site_packages, "nvidia", sub, "bin")
        if os.path.exists(path):
            try:
                os.add_dll_directory(path)
            except Exception:
                pass


def assert_onnx_gpu() -> None:
    """Raise unless onnxruntime can really put a session on the GPU.

    get_available_providers() is not evidence — it lists what the build supports,
    not what can actually load. The only honest check is to build a session and
    read back the provider it bound.
    """
    preload_cuda_libs()

    import numpy as np
    import onnxruntime as ort
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("Relu", ["x"], ["y"])],
        "probe",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10

    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CUDAExecutionProvider"]
    )
    bound = sess.get_providers()
    if "CUDAExecutionProvider" not in bound:
        raise RuntimeError(
            "onnxruntime fell back to CPU (bound providers: %s). RTMPose would "
            "run on CPU. Check that torch is the cu12 build and that the CUDA "
            "libs live in site-packages/nvidia/*/lib." % bound
        )
    sess.run(None, {"x": np.zeros(1, dtype=np.float32)})
