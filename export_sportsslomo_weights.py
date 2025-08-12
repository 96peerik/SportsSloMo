#!/usr/bin/env python
"""
Export raw weights for complete SportsSloMo pipeline (BiFlowNet + FusionNet) into C++/LibTorch friendly format.

Outputs (in specified --out dir):
  bi_flownet/manifest.json : BiFlowNet metadata
  bi_flownet/weights/<name>.bin : BiFlowNet weights  
  fusionnet/manifest.json : FusionNet metadata
  fusionnet/weights/<name>.bin : FusionNet weights

Binary format: raw little-endian float32 values in PyTorch's contiguous memory order.
Shapes use NCHW for convolutional weights: [out_channels, in_channels, kH, kW]. Biases are [out_channels].

Example usage:
  python export_sportsslomo_weights.py --bi_flownet path/to/bi-flownet-SportsSloMo.pkl --fusionnet path/to/fusionnet-SportsSloMo.pkl --out exported/sportsslomo

You can then load in C++ by reading each manifest.json and loading the corresponding weights.
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


def export_state_dict(state_dict, out_dir: Path, model_name: str):
    weights_dir = out_dir / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        'format_version': 1,
        'framework': 'pytorch',
        'model': model_name,
        'consolidated': False,
        'tensors': []
    }

    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            print(f'Warning: skipping non-tensor {name} (type: {type(tensor)})')
            continue

        # Ensure contiguous float32
        tensor = tensor.contiguous().float()
        
        # Sanitize filename
        safe_name = sanitize(name)
        bin_path = weights_dir / f'{safe_name}.bin'
        
        # Write binary data
        with open(bin_path, 'wb') as f:
            f.write(tensor.numpy().tobytes())
        
        # Add to manifest
        manifest['tensors'].append({
            'name': name,
            'shape': list(tensor.shape),
            'dtype': 'float32',
            'file': f'weights/{safe_name}.bin',
            'size_bytes': tensor.numel() * 4
        })
        
        print(f'  {name}: {list(tensor.shape)} -> {safe_name}.bin')

    # Write manifest
    manifest_path = out_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    return manifest


def main():
    parser = argparse.ArgumentParser(description='Export SportsSloMo (BiFlowNet + FusionNet) weights for C++')
    parser.add_argument('--bi_flownet', type=str, required=True, help='Path to bi-flownet checkpoint')
    parser.add_argument('--fusionnet', type=str, required=True, help='Path to fusionnet checkpoint')
    parser.add_argument('--out', type=str, required=True, help='Output directory')
    parser.add_argument('--validate', action='store_true', help='Validate by loading into PyTorch models')
    parser.add_argument('--pyr_level', type=int, default=3, help='BiFlowNet pyramid level for validation')
    parser.add_argument('--warp_type', type=str, default='middle-forward', choices=['middle-forward','backward','forward','none'], help='BiFlowNet warp type for validation')
    
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Export BiFlowNet
    print("Exporting BiFlowNet...")
    bi_flownet_state = torch.load(args.bi_flownet, map_location='cpu')
    if isinstance(bi_flownet_state, dict) and 'state_dict' in bi_flownet_state:
        bi_flownet_state = bi_flownet_state['state_dict']
    
    bi_flownet_dir = out_dir / 'bi_flownet'
    bi_flownet_manifest = export_state_dict(bi_flownet_state, bi_flownet_dir, 'BiFlowNet')
    
    # Export FusionNet
    print("Exporting FusionNet...")
    fusionnet_state = torch.load(args.fusionnet, map_location='cpu')
    if isinstance(fusionnet_state, dict) and 'state_dict' in fusionnet_state:
        fusionnet_state = fusionnet_state['state_dict']
    
    fusionnet_dir = out_dir / 'fusionnet'
    fusionnet_manifest = export_state_dict(fusionnet_state, fusionnet_dir, 'FusionNet')

    print(f'Exported BiFlowNet: {len(bi_flownet_manifest["tensors"])} tensors to {bi_flownet_dir}')
    print(f'Exported FusionNet: {len(fusionnet_manifest["tensors"])} tensors to {fusionnet_dir}')

    if args.validate:
        print("Validating exports...")
        sys.path.insert(0, os.path.join(Path(__file__).parent, 'SportsSloMo_EBME'))
        
        # Validate BiFlowNet
        from core.modules.bi_flownet import BiFlowNet  # type: ignore
        import types
        bi_args = types.SimpleNamespace()
        bi_args.pyr_level = args.pyr_level
        bi_args.warp_type = None if args.warp_type == 'none' else args.warp_type
        bi_model = BiFlowNet(bi_args)
        missing, unexpected = bi_model.load_state_dict(bi_flownet_state, strict=False)
        if missing:
            print('BiFlowNet missing keys:', missing)
        if unexpected:
            print('BiFlowNet unexpected keys:', unexpected)
        
        # Validate FusionNet
        from core.modules.fusionnet import FusionNet  # type: ignore
        fusion_args = types.SimpleNamespace()
        fusion_args.high_synthesis = False  # Default setting
        fusion_model = FusionNet(fusion_args)
        missing, unexpected = fusion_model.load_state_dict(fusionnet_state, strict=False)
        if missing:
            print('FusionNet missing keys:', missing)
        if unexpected:
            print('FusionNet unexpected keys:', unexpected)
            
        # Test forward pass
        bi_model.eval()
        fusion_model.eval()
        with torch.no_grad():
            x0 = torch.randn(1,3,256,256)
            x1 = torch.randn(1,3,256,256)
            bi_flow = bi_model(x0, x1)
            interp_img = fusion_model(x0, x1, bi_flow, time_period=0.5)
        
        print('Validation forward ok.')
        print('BiFlowNet output shape:', tuple(bi_flow.shape))
        print('FusionNet output shape:', tuple(interp_img.shape))


if __name__ == '__main__':
    main()
