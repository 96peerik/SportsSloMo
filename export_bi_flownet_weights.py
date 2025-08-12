#!/usr/bin/env python
"""
Export raw weights for BiFlowNet (bi-flownet-SportsSloMo.pkl) into a C++/LibTorch friendly format.

Outputs (in specified --out dir):
  manifest.json : metadata with list of parameters (name, shape, dtype, file, offset)
  weights/<sanitized_name>.bin : one float32 binary file per tensor (row-major contiguous as in PyTorch)
  (optional) consolidated_weights.bin + manifest_consolidated.json if --consolidated flag passed

Binary format (per-file mode): raw little-endian float32 values in PyTorch's contiguous memory order.
Shapes use NCHW for convolutional weights: [out_channels, in_channels, kH, kW]. Biases are [out_channels].

Example usage:
  python export_bi_flownet_weights.py --ckpt path/to/bi-flownet-SportsSloMo.pkl --out exported/bi_flownet
  python export_bi_flownet_weights.py --ckpt bi-flownet.pkl --out exported/bi_flownet --consolidated

You can then load in C++ by reading manifest.json, mmap / read each .bin and memcpy into torch::Tensor created with the listed shape & dtype.
"""
import argparse
import json
import os
import sys
from pathlib import Path
import struct

import torch


def sanitize(name: str) -> str:
    return name.replace('.', '__').replace('/', '_')


def export_state_dict(state_dict, out_dir: Path, consolidated: bool):
    weights_dir = out_dir / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        'format_version': 1,
        'framework': 'pytorch',
        'model': 'BiFlowNet',
        'consolidated': consolidated,
        'tensors': []
    }

    if consolidated:
        consolidated_path = out_dir / 'consolidated_weights.bin'
        offset = 0
        with open(consolidated_path, 'wb') as f_con:
            for name, tensor in state_dict.items():
                if not isinstance(tensor, torch.Tensor):
                    continue
                t = tensor.detach().cpu().contiguous().to(torch.float32)
                data = t.numpy().tobytes(order='C')
                f_con.write(data)
                size_bytes = len(data)
                manifest['tensors'].append({
                    'name': name,
                    'sanitized_name': sanitize(name),
                    'shape': list(t.shape),
                    'dtype': 'float32',
                    'storage': 'consolidated',
                    'offset': offset,
                    'nbytes': size_bytes
                })
                offset += size_bytes
        manifest['total_nbytes'] = offset
    else:
        for name, tensor in state_dict.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            t = tensor.detach().cpu().contiguous().to(torch.float32)
            bin_name = sanitize(name) + '.bin'
            bin_path = weights_dir / bin_name
            with open(bin_path, 'wb') as f:
                f.write(t.numpy().tobytes(order='C'))
            manifest['tensors'].append({
                'name': name,
                'sanitized_name': sanitize(name),
                'shape': list(t.shape),
                'dtype': 'float32',
                'storage': 'per-file',
                'file': f'weights/{bin_name}',
                'nbytes': t.numel() * 4
            })

    with open(out_dir / ('manifest_consolidated.json' if consolidated else 'manifest.json'), 'w') as mf:
        json.dump(manifest, mf, indent=2)

    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, help='Path to bi-flownet-SportsSloMo.pkl (state_dict)')
    parser.add_argument('--out', required=True, help='Output directory')
    parser.add_argument('--pyr_level', type=int, default=3, help='Pyramid levels (needed only to build model if you later validate)')
    parser.add_argument('--warp_type', type=str, default='middle-forward', choices=['middle-forward','backward','forward','none'], help='Warp type for validation run (optional)')
    parser.add_argument('--validate', action='store_true', help='Run a dummy forward to ensure weights load')
    parser.add_argument('--consolidated', action='store_true', help='Also produce single consolidated binary file')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load raw state_dict (training saved torch.save(model.state_dict()))
    state_obj = torch.load(args.ckpt, map_location='cpu')
    if isinstance(state_obj, dict) and 'state_dict' in state_obj:
        state_dict = state_obj['state_dict']
    else:
        state_dict = state_obj

    # Export weights only (no model instantiation required)
    manifest = export_state_dict(state_dict, out_dir, consolidated=False)
    if args.consolidated:
        export_state_dict(state_dict, out_dir, consolidated=True)

    print(f'Exported {len(manifest["tensors"])} tensors to {out_dir}')

    if args.validate:
        # Optional: instantiate and load to verify shapes match
        sys.path.insert(0, os.path.join(Path(__file__).parent, 'SportsSloMo_EBME'))
        from core.modules.bi_flownet import BiFlowNet  # type: ignore
        # Build args namespace similar to training scripts
        import types
        model_args = types.SimpleNamespace()
        model_args.pyr_level = args.pyr_level
        model_args.warp_type = None if args.warp_type == 'none' else args.warp_type
        model = BiFlowNet(model_args)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print('Warning: missing keys:', missing)
        if unexpected:
            print('Warning: unexpected keys:', unexpected)
        model.eval()
        # Dummy inference to ensure forward works
        with torch.no_grad():
            x0 = torch.randn(1,3,256,256)
            x1 = torch.randn(1,3,256,256)
            flow = model(x0, x1)
        print('Validation forward ok. Output flow shape:', tuple(flow.shape))

if __name__ == '__main__':
    main()
