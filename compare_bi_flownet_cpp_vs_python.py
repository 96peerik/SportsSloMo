#!/usr/bin/env python
"""Run a single forward pass in Python BiFlowNet and compare against C++ run outputs previously dumped.

Usage:
  python compare_bi_flownet_cpp_vs_python.py --ckpt path/to/bi-flownet-SportsSloMo.pkl \
      --cpp-flow cpp_flow.bin --width 256 --height 256 [--pyr_level 3 --warp_type middle-forward]

Expect cpp_flow.bin to contain raw float32 tensor shaped [1,4,H,W].
Prints L1, L2, and cosine similarity metrics per-channel and overall.
"""
import argparse, os, sys, struct, math
from pathlib import Path
import numpy as np
import torch

def load_cpp_flow(path, h, w):
    data = np.fromfile(path, dtype=np.float32)
    expected = 1*4*h*w
    if data.size != expected:
        raise ValueError(f"Size mismatch: got {data.size}, expected {expected}")
    return torch.from_numpy(data.reshape(1,4,h,w))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--cpp-flow', required=True)
    ap.add_argument('--height', type=int, required=True)
    ap.add_argument('--width', type=int, required=True)
    ap.add_argument('--pyr_level', type=int, default=3)
    ap.add_argument('--warp_type', type=str, default='middle-forward', choices=['middle-forward','backward','forward','none'])
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).parent / 'SportsSloMo_EBME'))
    from core.modules.bi_flownet import BiFlowNet  # type: ignore
    import types
    margs = types.SimpleNamespace()
    margs.pyr_level = args.pyr_level
    margs.warp_type = None if args.warp_type == 'none' else args.warp_type
    state_obj = torch.load(args.ckpt, map_location='cpu')
    if isinstance(state_obj, dict) and 'state_dict' in state_obj:
        state_dict = state_obj['state_dict']
    else:
        state_dict = state_obj
    model = BiFlowNet(margs).eval()
    model.load_state_dict(state_dict, strict=False)
    with torch.no_grad():
        x0 = torch.randn(1,3,args.height,args.width)
        x1 = torch.randn(1,3,args.height,args.width)
        py_flow = model(x0, x1).float()
    cpp_flow = load_cpp_flow(args.cpp_flow, args.height, args.width)
    if cpp_flow.shape != py_flow.shape:
        print('Shape mismatch:', cpp_flow.shape, py_flow.shape)
    diff = (cpp_flow - py_flow)
    l1 = diff.abs().mean().item()
    l2 = torch.sqrt((diff**2).mean()).item()
    # Cosine similarity per-pixel per direction pair aggregated
    def cos(a,b):
        return (a*b).sum() / (a.norm()*b.norm() + 1e-8)
    cos_overall = cos(cpp_flow.view(-1), py_flow.view(-1)).item()
    print(f'L1 mean: {l1:.6f}\nRMSE: {l2:.6f}\nCosine: {cos_overall:.6f}')
    for c in range(4):
        d = (cpp_flow[:,c]-py_flow[:,c])
        l1c = d.abs().mean().item(); l2c = torch.sqrt((d**2).mean()).item(); cc = cos(cpp_flow[:,c].view(-1), py_flow[:,c].view(-1)).item()
        print(f'Channel {c}: L1 {l1c:.6f} RMSE {l2c:.6f} Cos {cc:.6f}')

if __name__ == '__main__':
    main()
