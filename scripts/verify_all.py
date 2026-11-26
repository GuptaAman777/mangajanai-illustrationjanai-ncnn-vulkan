import glob
import os
import sys
import numpy as np
import onnxruntime as ort
import ncnn


def get_test_shapes(onnx_path: str):
    norm = onnx_path.replace("\\", "/")
    if norm.startswith("SPAN/") or norm.startswith("ESRGAN_4x/"):
        return [(1, 3, 32, 32), (1, 3, 47, 53)]
    if norm.startswith("ESRGAN_2x/"):
        return [(1, 3, 32, 32), (1, 3, 40, 56)]
    if norm.startswith("FDAT/"):
        if "unshuffle" in norm:
            return [(1, 3, 32, 32), (1, 3, 32, 48)]
        return [(1, 3, 32, 32), (1, 3, 24, 40)]
    raise ValueError(onnx_path)


def verify_model(onnx_path: str, param_path: str, bin_path: str, large_tile=False):
    assert os.path.exists(param_path), f"Missing {param_path}"
    assert os.path.exists(bin_path) and os.path.getsize(bin_path) > 0, f"Missing/empty {bin_path}"

    shapes = get_test_shapes(onnx_path)
    if large_tile:
        shapes = shapes + [(1, 3, 128, 128)]

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_refs = {}
    inputs = {}
    rng = np.random.default_rng(42)
    for s in shapes:
        x = rng.random(s, dtype=np.float32).astype(np.float16)
        inputs[s] = x
        ort_refs[s] = sess.run(None, {"input": x})[0].astype(np.float32)

    non_vk = None
    for vk, fp16 in [(False, False), (True, False), (True, True)]:
        net = ncnn.Net()
        net.opt.use_vulkan_compute = vk
        net.opt.use_fp16_packed = fp16
        net.opt.use_fp16_storage = fp16
        net.opt.use_fp16_arithmetic = fp16
        assert net.load_param(param_path) == 0, f"load_param failed: {param_path}"
        assert net.load_model(bin_path) == 0, f"load_model failed: {bin_path}"

        if vk and not fp16:
            non_vk = [
                (l.type, l.name)
                for l in net.layers()
                if l.type not in ("Input", "MemoryData", "Split") and not l.support_vulkan
            ]
            assert len(non_vk) == 0, f"{param_path} has non-Vulkan layers: {non_vk}"

        for s in shapes:
            x32 = inputs[s].astype(np.float32)
            with net.create_extractor() as ex:
                ex.input("in0", ncnn.Mat(x32, batch_index=0).clone())
                ret, out_m = ex.extract("out0")
            assert ret == 0, f"extract failed ({ret}) for {param_path} shape={s} vk={vk} fp16={fp16}"
            out = out_m.numpy(batch_index=0)
            ref = ort_refs[s]
            assert out.shape == ref.shape, f"shape mismatch {out.shape} != {ref.shape}"
            assert np.isfinite(out).all(), f"NaN/Inf in {param_path} shape={s} vk={vk} fp16={fp16}"
            diff = np.abs(out - ref)
            mean_err = float(diff.mean())
            max_err = float(diff.max())
            limit = 0.03 if fp16 else 0.005
            assert mean_err < limit, (
                f"Accuracy failure {param_path} s={s} vk={vk} fp16={fp16}: "
                f"mean_err={mean_err:.6f} max_err={max_err:.6f}"
            )
            print(
                f"  PASS {os.path.basename(param_path)} s={s} vk={vk} fp16={fp16} "
                f"mean_err={mean_err:.6f} max_err={max_err:.6f}",
                flush=True,
            )


def main():
    onnx_files = sorted(
        f for d in ("ESRGAN_2x", "ESRGAN_4x", "FDAT", "SPAN") for f in glob.glob(f"{d}/*.onnx")
    )
    print(f"Verifying {len(onnx_files)} models...", flush=True)
    large_done = set()
    for onnx_path in onnx_files:
        folder = os.path.dirname(onnx_path)
        stem = os.path.splitext(os.path.basename(onnx_path))[0]
        param_path = os.path.join("ncnn", folder, f"{stem}.ncnn.param")
        bin_path = os.path.join("ncnn", folder, f"{stem}.ncnn.bin")
        do_large = folder not in large_done
        large_done.add(folder)
        print(f"[{onnx_path}]", flush=True)
        verify_model(onnx_path, param_path, bin_path, large_tile=do_large)
    print(f"\nALL {len(onnx_files)} MODELS VERIFIED SUCCESSFULLY!", flush=True)


if __name__ == "__main__":
    main()
