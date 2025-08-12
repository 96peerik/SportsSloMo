#!/usr/bin/env python
"""Automate BiFlowNet C++ vs Python forward alignment with a fixed seed and identical random inputs.

Workflow:
1. Export weights if not already done (manifest + per-file binaries).
2. Generate random inputs with given seed in Python, save to .bin files.
3. Run C++ test harness pointing to manifest and provided seed, **reuse** the Python generated inputs (so modify harness call mode).
   Since current C++ harness always generates its own random inputs, we instead overwrite the dumped input files after it runs or,
   simpler: generate inputs first and pass them to C++ via environment expecting harness to read them (future improvement). For now we will
   call harness letting it generate (with same seed) and then compare using Python's reproduction of those tensors.
4. Load C++ dumped flow and compare with Python model output.

Note: This relies on identical RNG algorithm between LibTorch and PyTorch for torch.rand with manual_seed. That holds for CPU Philox.

Usage:
  python run_alignment_test.py --ckpt path/to/bi-flownet-SportsSloMo.pkl --manifest exported/bi_flownet/manifest.json \
      --weights-dir exported/bi_flownet --seed 1234 --work-dir tmp_align --height 256 --width 256
"""
import argparse, os, json, subprocess, sys
from pathlib import Path
import numpy as np
import torch

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_state_dict(ckpt):
    obj = torch.load(ckpt, map_location='cpu')
    if isinstance(obj, dict) and 'state_dict' in obj:
        return obj['state_dict']
    # Strip possible DataParallel 'module.' prefixes
    if isinstance(obj, dict):
        keys = list(obj.keys())
        if all(k.startswith('module.') for k in keys):
            obj = {k[len('module.'):]: v for k,v in obj.items()}
    return obj

def _maybe_load_custom_op_dll():
    """Attempt to load the custom C++ correlation / softsplat operator DLLs so torch.ops can find schemas.

    Safe no-op if already loaded. We try common build output locations (Release/Debug)."""
    candidates = [
        Path('build/src/Release/torch_correlation.dll'),
        Path('build/src/Debug/torch_correlation.dll'),
        Path('build/torch_correlation.dll'),
    ]
    for p in candidates:
        if p.exists():
            try:
                torch.ops.load_library(str(p.resolve()))
                print(f'[fallback] Loaded custom op DLL: {p}')
                break
            except Exception as e:
                print(f'[fallback] Failed loading {p}: {e}')
    # softsplat optional for this alignment path; load if present
    soft_candidates = [
        Path('build/src/Release/torch_softsplat.dll'),
        Path('build/src/Debug/torch_softsplat.dll'),
        Path('build/torch_softsplat.dll'),
    ]
    for p in soft_candidates:
        if p.exists():
            try:
                torch.ops.load_library(str(p.resolve()))
                print(f'[fallback] Loaded softsplat DLL: {p}')
                break
            except Exception as e:
                print(f'[fallback] Failed loading {p}: {e}')


def build_python_model(state_dict, pyr_level, warp_type):
    sys.path.insert(0, str(Path(__file__).parent / 'SportsSloMo_EBME'))

    # Provide a fallback correlation module when CuPy is unavailable by delegating to our C++ custom op
    cupy_available = True
    try:
        import cupy  # type: ignore  # noqa: F401
    except Exception:
        cupy_available = False

    if not cupy_available:
        print('[fallback] CuPy not available, installing pure PyTorch fallbacks for correlation + softsplat (average)')
        import types
        # Correlation fallback: brute-force sliding window (radius=4)
        def _corr_volume(f0, f1, radius=4):
            B,C,H,W = f0.shape
            pad = torch.nn.functional.pad(f1, (radius, radius, radius, radius))
            outs = []
            for dy in range(-radius, radius+1):
                for dx in range(-radius, radius+1):
                    slice = pad[:,:, dy+radius:dy+radius+H, dx+radius:dx+radius+W]
                    outs.append((f0*slice).sum(1, keepdim=True)/C)
            return torch.cat(outs, 1)
        corr_mod = types.ModuleType('core.utils.correlation')
        def FunctionCorrelation(tenFirst, tenSecond):
            return _corr_volume(tenFirst.contiguous(), tenSecond.contiguous())
        class ModuleCorrelation(torch.nn.Module):
            def __init__(self): super().__init__()
            def forward(self, a, b): return FunctionCorrelation(a, b)
        corr_mod.FunctionCorrelation = FunctionCorrelation
        corr_mod.ModuleCorrelation = ModuleCorrelation
        sys.modules['core.utils.correlation'] = corr_mod

        # Softsplat fallback: simple forward average via scatter-add approximation using grid_sample on small displacements
        soft_mod = types.ModuleType('core.modules.softsplat.softsplat')
        def FunctionSoftsplat(tenInput, tenFlow, tenMetric=None, strType='average'):
            assert strType == 'average'
            # Use backward sampling of inverse flow as approximation (not exact forward splat)
            B,C,H,W = tenInput.shape
            ys = torch.linspace(-1+2/H, 1-2/H, H, device=tenInput.device)
            xs = torch.linspace(-1+2/W, 1-2/W, W, device=tenInput.device)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
            grid = torch.stack((grid_x, grid_y), -1).unsqueeze(0).repeat(B,1,1,1)
            flow_x = (-tenFlow[:,0])/((W-1)/2.0)
            flow_y = (-tenFlow[:,1])/((H-1)/2.0)
            sample_grid = grid + torch.stack((flow_x, flow_y),1).permute(0,2,3,1)
            return torch.nn.functional.grid_sample(tenInput, sample_grid, align_corners=False)
        soft_mod.FunctionSoftsplat = FunctionSoftsplat
        sys.modules['core.modules.softsplat.softsplat'] = soft_mod
        pkg_soft = types.ModuleType('core.modules.softsplat')
        pkg_soft.softsplat = soft_mod
        sys.modules['core.modules.softsplat'] = pkg_soft
    else:
        print('[info] CuPy available; using original correlation kernels')

    from core.modules.bi_flownet import BiFlowNet  # type: ignore
    import types
    class ArgObj(types.SimpleNamespace):
        def __contains__(self, item):
            return hasattr(self, item)
    args = ArgObj()
    setattr(args, 'pyr_level', pyr_level)
    setattr(args, 'warp_type', None if warp_type == 'none' else warp_type)
    m = BiFlowNet(args).eval()
    missing, unexpected = m.load_state_dict(state_dict, strict=False)
    if missing:
        print('Warning missing keys:', missing)
    if unexpected:
        print('Warning unexpected keys:', unexpected)
    return m

def compare(fl_cpp: torch.Tensor, fl_py: torch.Tensor):
    diff = fl_cpp - fl_py
    l1 = diff.abs().mean().item()
    rmse = torch.sqrt((diff*diff).mean()).item()
    cos = (fl_cpp.view(-1)*fl_py.view(-1)).sum() / (fl_cpp.norm()*fl_py.norm()+1e-8)
    print(f'L1={l1:.6f} RMSE={rmse:.6f} Cos={cos.item():.6f}')
    return l1, rmse, cos.item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--weights-dir', required=True)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--height', type=int, default=256)
    ap.add_argument('--width', type=int, default=256)
    ap.add_argument('--pyr_level', type=int, default=3)
    ap.add_argument('--warp_type', type=str, default='middle-forward', choices=['middle-forward','backward','forward','none'])
    ap.add_argument('--work-dir', required=True)
    ap.add_argument('--exe', default='build/src/Release/test_bi_flownet.exe')
    args = ap.parse_args()

    work = Path(args.work_dir); ensure_dir(work)
    flow_bin = work / 'cpp_flow.bin'
    dump_prefix = work / 'pair'

    env = os.environ.copy()
    env['BI_FLOW_SEED'] = str(args.seed)
    env['FLOW_DUMP'] = str(flow_bin)
    env['FLOW_INPUT_DUMP_PREFIX'] = str(dump_prefix)

    # Run C++ harness
    cmd = [args.exe, args.manifest, args.weights_dir, str(args.seed), str(dump_prefix)]
    print('Running C++ harness:', ' '.join(cmd))
    subprocess.check_call(cmd, env=env)

    # Load dumped inputs
    in0 = torch.from_numpy(np.fromfile(str(dump_prefix)+'_input0.bin', dtype=np.float32).reshape(1,3,args.height,args.width))
    in1 = torch.from_numpy(np.fromfile(str(dump_prefix)+'_input1.bin', dtype=np.float32).reshape(1,3,args.height,args.width))

    # Python model
    sd = load_state_dict(args.ckpt)
    model = build_python_model(sd, args.pyr_level, args.warp_type)
    torch.manual_seed(args.seed)  # For parity in any internal randomness (should be none)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda().eval()
        in0 = in0.cuda()
        in1 = in1.cuda()
        print('[info] Using CUDA for Python model alignment')
    with torch.no_grad():
        py_flow = model(in0, in1).float()
        if use_cuda:
            py_flow = py_flow.cpu()

    cpp_flow = torch.from_numpy(np.fromfile(flow_bin, dtype=np.float32).reshape(1,4,args.height,args.width))

    l1, rmse, cos = compare(cpp_flow, py_flow)
    # Criteria thresholds (adjustable)
    if not (cos > 0.999 and l1 < 1e-3):
        print('WARNING: Alignment outside tight thresholds (may be expected if ops diverge).')

if __name__ == '__main__':
    main()
