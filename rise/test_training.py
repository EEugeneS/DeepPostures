"""CPU integration smoke test: temporary synthetic data, train/resume/predict."""
import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile

import h5py
import numpy as np
import torch

import run


def main():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for split, chap_split in [('train', 'train'), ('validation', 'val'), ('test', 'test_complete')]:
            with h5py.File(root / f'10s_{chap_split}.h5', 'w') as f:
                f['x'] = np.zeros((1, 42, 100, 3), np.float32)
                f['y'] = np.arange(42).reshape(1, 42) % 2
                f['subject_id'] = np.array(['001'], dtype=h5py.string_dtype())
                f['timestamp'] = np.arange(42).reshape(1, 42) * 10.
            torch.save(torch.zeros(2, 1, 300, 3), root / f'X_{split}.pt')
            torch.save(torch.tensor([0, 1]), root / f'y_{split}.pt')
        # Fake six-axis SSL weights exercise row selection and time adaptation.
        spec = importlib.util.spec_from_file_location('test_vendor', run.ROOT / 'rise/MoCA/vendor/models_vit.py')
        vendor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vendor)
        source = vendor.vit_tiny_patch16(img_size=(6, 200), patch_size=(1, 20), in_chans=1, num_classes=2)
        torch.save({'model': source.state_dict()}, root / 'ssl.pt')
        for family, mode in [('chap', 'scratch'), ('moca', 'scratch'), ('moca', 'lp'), ('moca', 'ft')]:
            output = root / (family + mode)
            cmd = [sys.executable, str(run.ROOT / 'rise/run.py'), '--family', family,
                   '--mode', mode, '--data-dir', str(root), '--output-dir', str(output),
                   '--device', 'cpu', '--epochs', '1', '--batch-size', '2', '--arch', 'tiny']
            if mode in ('lp', 'ft'):
                cmd += ['--checkpoint', str(root/'ssl.pt'), '--acc-rows', '0', '1', '2']
            subprocess.run(cmd, check=True)
            cp = run.load(output/'last.pt')
            if mode == 'lp':
                assert torch.equal(cp['model']['encoder.patch_embed.proj.weight'], source.patch_embed.proj.weight)
            subprocess.run([*cmd, '--resume', str(output/'last.pt'), '--epochs', '2'], check=True)
            subprocess.run([sys.executable, str(run.ROOT/'rise/run.py'), '--family', family,
                            '--mode', 'predict', '--data-dir', str(root), '--checkpoint', str(output/'best.pt'),
                            '--output-dir', str(output), '--device', 'cpu'], check=True)
            assert (output/'test_predictions.csv').is_file()
        print('PASS: CHAP/MoCA train, LP freezing, resume and prediction')


if __name__ == '__main__':
    main()
