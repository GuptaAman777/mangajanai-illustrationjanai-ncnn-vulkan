import glob
import os
import subprocess
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock(nn.Module):
    def __init__(self, nf: int = 64, gc: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Note: 0.2 scaling is already fused into ConvMulFusion conv5 weights/bias
        return x5 + x


class RRDB(nn.Module):
    def __init__(self, nf: int = 64, gc: int = 32):
        super().__init__()
        self.RDB1 = ResidualDenseBlock(nf, gc)
        self.RDB2 = ResidualDenseBlock(nf, gc)
        self.RDB3 = ResidualDenseBlock(nf, gc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet2x(nn.Module):
    def __init__(self, nf: int = 64, nb: int = 23, gc: int = 32):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(2)
        self.conv_first = nn.Conv2d(12, nf, 3, 1, 1)
        self.body = nn.ModuleList([RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.unshuffle(x)
        feat = self.conv_first(feat)
        body_feat = feat
        for block in self.body:
            body_feat = block(body_feat)
        feat = feat + self.conv_body(body_feat)
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2.0, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2.0, mode="nearest")))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


def load_esr2_from_onnx(onnx_path: str) -> RRDBNet2x:
    m = onnx.load(onnx_path)
    inits = {i.name: numpy_helper.to_array(i).astype(np.float32) for i in m.graph.initializer}

    model = RRDBNet2x(nf=64, nb=23, gc=32).eval()
    with torch.no_grad():
        model.conv_first.weight.copy_(torch.from_numpy(inits["model.0.weight"]))
        model.conv_first.bias.copy_(torch.from_numpy(inits["model.0.bias"]))
        for b in range(23):
            for r in range(1, 4):
                rdb = getattr(model.body[b], f"RDB{r}")
                for c in range(1, 5):
                    conv = getattr(rdb, f"conv{c}")
                    conv.weight.copy_(torch.from_numpy(inits[f"model.1.sub.{b}.RDB{r}.conv{c}.0.weight"]))
                    conv.bias.copy_(torch.from_numpy(inits[f"model.1.sub.{b}.RDB{r}.conv{c}.0.bias"]))
                w_key = f"ConvMulFusion_W_model.1.sub.{b}.RDB{r}.conv5.0.weight"
                b_key = f"ConvMulFusion_Mul_B_/model/model.1/sub/sub.{b}/RDB{r}/Constant_output_0"
                rdb.conv5.weight.copy_(torch.from_numpy(inits[w_key]))
                rdb.conv5.bias.copy_(torch.from_numpy(inits[b_key]))
        model.conv_body.weight.copy_(torch.from_numpy(inits["model.1.sub.23.weight"]))
        model.conv_body.bias.copy_(torch.from_numpy(inits["model.1.sub.23.bias"]))
        model.conv_up1.weight.copy_(torch.from_numpy(inits["model.3.weight"]))
        model.conv_up1.bias.copy_(torch.from_numpy(inits["model.3.bias"]))
        model.conv_up2.weight.copy_(torch.from_numpy(inits["model.6.weight"]))
        model.conv_up2.bias.copy_(torch.from_numpy(inits["model.6.bias"]))
        model.conv_hr.weight.copy_(torch.from_numpy(inits["model.8.weight"]))
        model.conv_hr.bias.copy_(torch.from_numpy(inits["model.8.bias"]))
        model.conv_last.weight.copy_(torch.from_numpy(inits["model.10.weight"]))
        model.conv_last.bias.copy_(torch.from_numpy(inits["model.10.bias"]))
    return model


def convert_esr2(onnx_path: str, out_dir: str, tmp_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(onnx_path))[0]

    model = load_esr2_from_onnx(onnx_path)
    x_test = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        pt_out = model(x_test).numpy()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"input": x_test.numpy().astype(np.float16)})[0].astype(np.float32)
    diff = np.abs(pt_out - ort_out)
    print(f"{stem}: PT vs ORT max_err={diff.max():.6f} mean_err={diff.mean():.6f}", flush=True)
    if diff.mean() > 0.01:
        raise RuntimeError(f"PyTorch reconstruction mismatch for {onnx_path}: mean_err={diff.mean()}")

    pt_path = os.path.join(tmp_dir, f"{stem}.pt")
    traced = torch.jit.trace(model, x_test)
    traced.save(pt_path)

    ncnn_param = os.path.join(out_dir, f"{stem}.ncnn.param")
    ncnn_bin = os.path.join(out_dir, f"{stem}.ncnn.bin")
    cmd = [
        "pnnx",
        pt_path,
        "fp16=1",
        "inputshape=[1,3,32,32]",
        "inputshape2=[1,3,48,48]",
        f"ncnnparam={ncnn_param}",
        f"ncnnbin={ncnn_bin}",
        f"pnnxparam={os.path.join(tmp_dir, stem + '.pnnx.param')}",
        f"pnnxbin={os.path.join(tmp_dir, stem + '.pnnx.bin')}",
        f"pnnxpy={os.path.join(tmp_dir, stem + '_pnnx.py')}",
        f"ncnnpy={os.path.join(tmp_dir, stem + '_ncnn.py')}",
        f"pnnxonnx={os.path.join(tmp_dir, stem + '.pnnx.onnx')}",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout[-1500:])
        print(res.stderr[-1500:])
        raise RuntimeError(f"pnnx failed for {onnx_path} with code {res.returncode}")


if __name__ == "__main__":
    for p in sorted(glob.glob("ESRGAN_2x/*.onnx")):
        stem = os.path.splitext(os.path.basename(p))[0]
        if os.path.exists(os.path.join("ncnn", "ESRGAN_2x", f"{stem}.ncnn.bin")):
            print(f"Skipping existing {stem}", flush=True)
            continue
        convert_esr2(p, os.path.join("ncnn", "ESRGAN_2x"), "_test_pnnx/work_esr2")
    print("All ESRGAN_2x models converted.")
