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


class SpatialAttn(nn.Module):
    """Even-block window self-attention using <=4D tensors only."""

    def __init__(self, dim: int, num_heads: int, window_size: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.bias = nn.Parameter(torch.zeros(1, num_heads, window_size * window_size, window_size * window_size))

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # x: (1, H*W, C)
        ws = self.window_size
        nh = H // ws
        nw = W // ws
        nW = nh * nw
        C = self.dim

        # (1, H*W, C) -> (nh, ws, nw, ws*C) -> (nh, nw, ws, ws*C) -> (nW, ws*ws, C)
        xw = x.reshape(nh, ws, nw, ws * C).permute(0, 2, 1, 3).reshape(nW, ws * ws, C)
        q, k, v = self.qkv(xw).chunk(3, dim=-1)
        q = q.reshape(nW, ws * ws, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(nW, ws * ws, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(nW, ws * ws, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=self.bias, scale=self.scale)
        out = out.permute(0, 2, 1, 3).reshape(nW, ws * ws, C)
        out = out.reshape(nh, nw, ws, ws * C).permute(0, 2, 1, 3).reshape(1, H * W, C)
        return self.proj(out)


class ChannelAttn(nn.Module):
    """Odd-block channel attention using SDPA and Vulkan-compatible L2 norm."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.temp = nn.Parameter(torch.ones(1, num_heads, 1, 1))

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # x: (1, HW, C)
        HW = H * W
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(1, HW, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        k = k.reshape(1, HW, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        v = v.reshape(1, HW, self.num_heads, self.head_dim).permute(0, 2, 3, 1)

        q = q / torch.clamp(torch.sqrt(torch.sum(q * q, dim=-1, keepdim=True)), min=1.1920928955078125e-07, max=1e9)
        k = k / torch.clamp(torch.sqrt(torch.sum(k * k, dim=-1, keepdim=True)), min=1.1920928955078125e-07, max=1e9)

        out = F.scaled_dot_product_attention(q * self.temp, k, v, scale=1.0)
        out = out.permute(0, 3, 1, 2).reshape(1, HW, self.dim)
        return self.proj(out)


class FDATBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int, cg_mid: int, is_channel: bool, window_size: int = 8):
        super().__init__()
        self.dim = dim
        self.is_channel = is_channel
        self.n1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = ChannelAttn(dim, num_heads) if is_channel else SpatialAttn(dim, num_heads, window_size)
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.GELU(),
        )
        if is_channel:
            self.sg = nn.Sequential(
                nn.Conv2d(dim, 1, 1, 1, 0, bias=False),
                nn.Sigmoid(),
            )
        else:
            self.cg = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, cg_mid, 1, 1, 0, bias=False),
                nn.GELU(),
                nn.Conv2d(cg_mid, dim, 1, 1, 0, bias=False),
                nn.Sigmoid(),
            )
        self.n2 = nn.LayerNorm(dim, eps=1e-5)
        self.fc1 = nn.Linear(dim, ffn_dim, bias=False)
        self.smix = nn.Conv2d(ffn_dim, ffn_dim, 3, 1, 1, groups=ffn_dim, bias=False)
        self.fc2 = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # x: (1, H*W, C)
        x_norm = self.n1(x)
        attn_out = self.attn(x_norm, H, W)  # (1, H*W, C)

        x_2d = x_norm.permute(0, 2, 1).reshape(1, self.dim, H, W)
        conv_2d = self.conv(x_2d)  # (1, C, H, W)

        if self.is_channel:
            attn_2d = attn_out.permute(0, 2, 1).reshape(1, self.dim, H, W)
            gate = self.sg(attn_2d)  # (1, 1, H, W)
            inter_2d = attn_2d + conv_2d * gate
        else:
            gate = self.cg(conv_2d)  # (1, C, 1, 1)
            attn_2d = attn_out.permute(0, 2, 1).reshape(1, self.dim, H, W)
            inter_2d = attn_2d * gate + conv_2d

        inter_seq = inter_2d.reshape(1, self.dim, H * W).permute(0, 2, 1)
        x = x + inter_seq

        h = F.gelu(self.fc1(self.n2(x)))
        h_2d = h.permute(0, 2, 1).reshape(1, -1, H, W)
        h_2d = self.smix(h_2d)
        h = h_2d.reshape(1, -1, H * W).permute(0, 2, 1)
        x = x + self.fc2(h)
        return x


class FDATGroup(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_blocks: int, ffn_dim: int, cg_mid: int, window_size: int = 8):
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList([
            FDATBlock(dim, num_heads, ffn_dim, cg_mid, is_channel=(i % 2 == 1), window_size=window_size)
            for i in range(num_blocks)
        ])
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)

    def forward(self, x_2d: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x_2d.shape
        x = x_2d.reshape(1, self.dim, H * W).permute(0, 2, 1)
        for blk in self.blocks:
            x = blk(x, H, W)
        out_2d = x.permute(0, 2, 1).reshape(1, self.dim, H, W)
        return x_2d + self.conv(out_2d)


class FDATNet(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_groups: int, num_blocks: int, ffn_dim: int, cg_mid: int, unshuffle: bool):
        super().__init__()
        self.unshuffle = unshuffle
        if unshuffle:
            self.pu = nn.PixelUnshuffle(2)
            self.conv_first = nn.Conv2d(12, dim, 3, 1, 1, bias=True)
        else:
            self.conv_first = nn.Conv2d(3, dim, 3, 1, 1, bias=True)
        self.groups = nn.ModuleList([
            FDATGroup(dim, num_heads, num_blocks, ffn_dim, cg_mid, window_size=8)
            for _ in range(num_groups)
        ])
        self.conv_after = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, 4, 2, 1, bias=True),
            nn.GELU(),
            nn.ConvTranspose2d(dim, 3, 4, 2, 1, bias=True),
            nn.Conv2d(3, 3, 3, 1, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.unshuffle:
            x = self.pu(x)
        x0 = self.conv_first(x)
        feat = x0
        for grp in self.groups:
            feat = grp(feat)
        feat = x0 + self.conv_after(feat)
        return self.upsampler(feat)


def load_fdat_from_onnx(onnx_path: str) -> FDATNet:
    m = onnx.load(onnx_path)
    inits = {i.name: numpy_helper.to_array(i).astype(np.float32) for i in m.graph.initializer}

    # Map MatMul node names to their weight tensors
    matmul_weights = {}
    for n in m.graph.node:
        if n.op_type == "MatMul" and n.input[1] in inits:
            matmul_weights[n.name] = inits[n.input[1]]

    unshuffle = "conv_first.1.weight" in inits
    cf_w = inits["conv_first.1.weight"] if unshuffle else inits["conv_first.weight"]
    dim = cf_w.shape[0]
    num_heads = inits["groups.0.blocks.0.attn.bias"].shape[0]
    cg_mid = inits["groups.0.blocks.0.inter.cg.1.weight"].shape[0]
    ffn_dim = inits["groups.0.blocks.0.ffn.smix.weight"].shape[0]
    num_groups = max(int(k.split(".")[1]) for k in inits if k.startswith("groups.")) + 1
    num_blocks = max(int(k.split(".")[3]) for k in inits if k.startswith("groups.0.blocks.")) + 1

    model = FDATNet(dim, num_heads, num_groups, num_blocks, ffn_dim, cg_mid, unshuffle).eval()
    with torch.no_grad():
        if unshuffle:
            model.conv_first.weight.copy_(torch.from_numpy(inits["conv_first.1.weight"]))
            model.conv_first.bias.copy_(torch.from_numpy(inits["conv_first.1.bias"]))
        else:
            model.conv_first.weight.copy_(torch.from_numpy(inits["conv_first.weight"]))
            model.conv_first.bias.copy_(torch.from_numpy(inits["conv_first.bias"]))

        for g in range(num_groups):
            grp = model.groups[g]
            for b in range(num_blocks):
                blk = grp.blocks[b]
                prefix = f"groups.{g}.blocks.{b}"
                nprefix = f"/groups/groups.{g}/blocks.{b}"
                blk.n1.weight.copy_(torch.from_numpy(inits[f"{prefix}.n1.weight"]))
                blk.n1.bias.copy_(torch.from_numpy(inits[f"{prefix}.n1.bias"]))
                blk.n2.weight.copy_(torch.from_numpy(inits[f"{prefix}.n2.weight"]))
                blk.n2.bias.copy_(torch.from_numpy(inits[f"{prefix}.n2.bias"]))

                blk.attn.qkv.weight.copy_(torch.from_numpy(matmul_weights[f"{nprefix}/attn/qkv/MatMul"].T))
                blk.attn.proj.weight.copy_(torch.from_numpy(matmul_weights[f"{nprefix}/attn/proj/MatMul"].T))
                blk.attn.proj.bias.copy_(torch.from_numpy(inits[f"{prefix}.attn.proj.bias"]))
                if blk.is_channel:
                    blk.attn.temp.data.copy_(torch.from_numpy(inits[f"{prefix}.attn.temp"]).unsqueeze(0))
                    blk.sg[0].weight.copy_(torch.from_numpy(inits[f"{prefix}.inter.sg.0.weight"]))
                else:
                    blk.attn.bias.data.copy_(torch.from_numpy(inits[f"{prefix}.attn.bias"]).unsqueeze(0))
                    blk.cg[1].weight.copy_(torch.from_numpy(inits[f"{prefix}.inter.cg.1.weight"]))
                    blk.cg[3].weight.copy_(torch.from_numpy(inits[f"{prefix}.inter.cg.3.weight"]))

                blk.conv[0].weight.copy_(torch.from_numpy(inits[f"{prefix}.conv.0.weight"]))
                blk.fc1.weight.copy_(torch.from_numpy(matmul_weights[f"{nprefix}/ffn/fc1/MatMul"].T))
                blk.smix.weight.copy_(torch.from_numpy(inits[f"{prefix}.ffn.smix.weight"]))
                blk.fc2.weight.copy_(torch.from_numpy(matmul_weights[f"{nprefix}/ffn/fc2/MatMul"].T))

            grp.conv.weight.copy_(torch.from_numpy(inits[f"groups.{g}.conv.weight"]))

        model.conv_after.weight.copy_(torch.from_numpy(inits["conv_after.weight"]))
        model.upsampler[0].weight.copy_(torch.from_numpy(inits["upsampler.0.weight"]))
        model.upsampler[0].bias.copy_(torch.from_numpy(inits["upsampler.0.bias"]))
        model.upsampler[2].weight.copy_(torch.from_numpy(inits["upsampler.2.weight"]))
        model.upsampler[2].bias.copy_(torch.from_numpy(inits["upsampler.2.bias"]))
        model.upsampler[3].weight.copy_(torch.from_numpy(inits["upsampler.3.weight"]))
        model.upsampler[3].bias.copy_(torch.from_numpy(inits["upsampler.3.bias"]))
    return model


def convert_fdat(onnx_path: str, out_dir: str, tmp_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(onnx_path))[0]

    model = load_fdat_from_onnx(onnx_path)
    x_test = torch.rand(1, 3, 16, 16)
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
        "inputshape=[1,3,16,16]",
        "inputshape2=[1,3,32,48]",
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
    for p in sorted(glob.glob("FDAT/*.onnx")):
        stem = os.path.splitext(os.path.basename(p))[0]
        if os.path.exists(os.path.join("ncnn", "FDAT", f"{stem}.ncnn.bin")):
            print(f"Skipping existing {stem}", flush=True)
            continue
        convert_fdat(p, os.path.join("ncnn", "FDAT"), "_test_pnnx/work_fdat")
    print("All FDAT models converted.")
