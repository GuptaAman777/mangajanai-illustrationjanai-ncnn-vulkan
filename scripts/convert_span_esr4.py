import glob
import os
import subprocess
import sys
import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


def onnx_fp16_to_fp32(src_path: str, dst_path: str) -> None:
    m = onnx.load(src_path)
    for init in m.graph.initializer:
        if init.data_type == TensorProto.FLOAT16:
            arr = numpy_helper.to_array(init).astype(np.float32)
            init.CopyFrom(numpy_helper.from_array(arr, name=init.name))
    for n in m.graph.node:
        for a in n.attribute:
            if a.type == onnx.AttributeProto.TENSOR and a.t.data_type == TensorProto.FLOAT16:
                arr = numpy_helper.to_array(a.t).astype(np.float32)
                a.t.CopyFrom(numpy_helper.from_array(arr, name=a.t.name))
            if a.type == onnx.AttributeProto.INT and n.op_type == "Cast" and a.i == TensorProto.FLOAT16:
                a.i = TensorProto.FLOAT
    for vi in list(m.graph.input) + list(m.graph.output) + list(m.graph.value_info):
        if vi.type.tensor_type.elem_type == TensorProto.FLOAT16:
            vi.type.tensor_type.elem_type = TensorProto.FLOAT
    onnx.save(m, dst_path)


def convert_via_fp32_onnx(onnx_path: str, out_dir: str, tmp_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(onnx_path))[0]
    fp32_onnx = os.path.join(tmp_dir, f"{stem}_fp32.onnx")
    in1 = os.path.join(tmp_dir, "in1.npy")
    in2 = os.path.join(tmp_dir, "in2.npy")
    if not os.path.exists(in1):
        np.save(in1, np.random.rand(1, 3, 32, 32).astype(np.float32))
    if not os.path.exists(in2):
        np.save(in2, np.random.rand(1, 3, 48, 48).astype(np.float32))

    onnx_fp16_to_fp32(onnx_path, fp32_onnx)

    ncnn_param = os.path.join(out_dir, f"{stem}.ncnn.param")
    ncnn_bin = os.path.join(out_dir, f"{stem}.ncnn.bin")
    cmd = [
        "pnnx",
        fp32_onnx,
        "fp16=1",
        f"input={in1}",
        f"input2={in2}",
        f"ncnnparam={ncnn_param}",
        f"ncnnbin={ncnn_bin}",
        f"pnnxparam={os.path.join(tmp_dir, stem + '.pnnx.param')}",
        f"pnnxbin={os.path.join(tmp_dir, stem + '.pnnx.bin')}",
        f"pnnxpy={os.path.join(tmp_dir, stem + '_pnnx.py')}",
        f"ncnnpy={os.path.join(tmp_dir, stem + '_ncnn.py')}",
        f"pnnxonnx={os.path.join(tmp_dir, stem + '.pnnx.onnx')}",
    ]
    print(f"Converting {onnx_path} -> {ncnn_param}", flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout[-1500:])
        print(res.stderr[-1500:])
        raise RuntimeError(f"pnnx failed for {onnx_path} with code {res.returncode}")
    # Remove any .pnnxsim.onnx generated alongside fp32_onnx
    sim_path = os.path.splitext(fp32_onnx)[0] + ".pnnxsim.onnx"
    if os.path.exists(sim_path):
        os.remove(sim_path)


if __name__ == "__main__":
    tmp = "_test_pnnx/work_span_esr4"
    targets = sorted(glob.glob("SPAN/*.onnx")) + sorted(glob.glob("ESRGAN_4x/*.onnx"))
    for p in targets:
        folder = os.path.basename(os.path.dirname(p))
        convert_via_fp32_onnx(p, os.path.join("ncnn", folder), tmp)
    print(f"Successfully converted {len(targets)} models.")
