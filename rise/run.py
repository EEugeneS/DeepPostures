"""Rise supervised transfer entry point. Run from any directory; see TRAINING.md."""
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import random
import sys

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]


def load(path, mmap=False):
    # Only load trusted project checkpoints (optimizer/config can contain Python objects).
    with open(path, 'rb') as f:
        if f.read(80).startswith(b'version https://git-lfs.github.com/spec/v1'):
            raise ValueError(f'{path} is a Git LFS pointer, not downloaded weights; fetch the actual checkpoint first')
    return torch.load(path, map_location='cpu', weights_only=False, mmap=mmap)


class RiseData(Dataset):
    def __init__(self, root, family, split):
        self.family, self.handle = family, None
        self.path = Path(root)
        if family == 'chap':
            suffix = {'train': 'train', 'validation': 'val', 'test': 'test_complete'}[split]
            self.path /= f'10s_{suffix}.h5'
            with h5py.File(self.path, 'r') as f:
                self.n = len(f['y'])
                if f['x'].shape[1:] != (42, 100, 3) or f['y'].shape != (self.n, 42):
                    raise ValueError('CHAP requires x=(N,42,100,3), y=(N,42)')
        else:
            self.x = load(self.path / f'X_{split}.pt', mmap=True)
            self.y = load(self.path / f'y_{split}.pt', mmap=True)
            self.n = len(self.y)
            if self.x.shape != (self.n, 1, 300, 3) or self.y.shape != (self.n,):
                raise ValueError('MoCA requires X=(N,1,300,3), y=(N,)')
        if not self.n:
            raise ValueError(f'Empty {split} dataset')

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.family == 'chap':
            if self.handle is None:
                self.handle = h5py.File(self.path, 'r')
            f = self.handle
            x = torch.tensor(f['x'][i], dtype=torch.float32)
            y = torch.tensor(f['y'][i], dtype=torch.long)
            subject = f['subject_id'][i]
            subject = subject.decode() if isinstance(subject, bytes) else str(subject)
            times = torch.tensor(f['timestamp'][i], dtype=torch.float64)
        else:
            x = self.x[i].permute(0, 2, 1).float()
            y = self.y[i].long()
            # Current builder has no per-window metadata. Do not invent an ID/time.
            subject, times = '', torch.tensor(float('nan'), dtype=torch.float64)
        if not torch.isfinite(x).all() or not ((y == 0) | (y == 1)).all():
            raise ValueError(f'Invalid signal or nonbinary label at index {i}')
        return x, y, i, subject, times


def state_dict(checkpoint):
    state = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    return {k.removeprefix('module.'): v for k, v in state.items()}


class MoCAClassifier(nn.Module):
    def __init__(self, arch, patch):
        super().__init__()
        spec = importlib.util.spec_from_file_location('rise_moca_vit', ROOT / 'rise/MoCA/vendor/models_vit.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.encoder = getattr(module, f'vit_{arch}_patch16')(
            img_size=(3, 300), patch_size=(1, patch), in_chans=1,
            num_classes=2, global_pool=False)

    def forward(self, x):
        return self.encoder.head(self.encoder.forward_features(x))


def build(args, checkpoint):
    state = state_dict(checkpoint) if checkpoint else None
    if args.family == 'chap':
        sys.path.insert(0, str(ROOT / 'CHAP2'))
        from chap_model import CHAP
        model = CHAP(2, 42, 2)
        if state is not None:
            model.load_state_dict(state, strict=True)
        return model
    if checkpoint and 'rise_config' in checkpoint:
        cfg = checkpoint['rise_config']
        args.arch, args.patch = cfg['arch'], cfg['patch']
        model = MoCAClassifier(args.arch, args.patch)
        model.load_state_dict(state, strict=True)
        return model
    if state is not None:
        weight = state['patch_embed.proj.weight']
        args.arch = {192: 'tiny', 768: 'base'}.get(weight.shape[0])
        if args.arch is None or weight.shape[1:3] != (1, 1):
            raise ValueError('Expected MoCA tiny/base checkpoint with axes as patch-grid rows')
        args.patch = weight.shape[-1]
    if 300 % args.patch:
        raise ValueError('Patch width must divide 300; preserve source patch width')
    model = MoCAClassifier(args.arch, args.patch)
    if state is not None:
        if args.acc_rows is None:
            raise ValueError('Specify --acc-rows after verifying the source checkpoint axis order')
        pos = state['pos_embed']
        if (pos.shape[1] - 1) % 6:
            raise ValueError('Expected a six-axis source position grid')
        grid = pos[:, 1:].reshape(1, 6, -1, pos.shape[-1])[:, args.acc_rows]
        # Select ACC rows, interpolate only along time, retain the CLS position.
        grid = grid.permute(0, 1, 3, 2).reshape(3, pos.shape[-1], -1)
        grid = nn.functional.interpolate(grid, size=300 // args.patch, mode='linear', align_corners=False)
        grid = grid.reshape(1, 3, pos.shape[-1], -1).permute(0, 1, 3, 2).reshape(1, -1, pos.shape[-1])
        target = model.encoder.state_dict()
        filtered = {k: v for k, v in state.items() if k in target and not k.startswith('head.')}
        filtered['pos_embed'] = torch.cat([pos[:, :1], grid], dim=1)
        report = model.encoder.load_state_dict(filtered, strict=False)
        if set(report.missing_keys) != {'head.weight', 'head.bias'} or report.unexpected_keys:
            raise ValueError(f'Incomplete encoder load: {report}')
    return model


def forward(model, x, family):
    return model(x.reshape(-1, 1, 100, 3)) if family == 'chap' else model(x)


def metrics(cm):
    tn, fp, fn, tp = cm.ravel().tolist()
    ratio = lambda a, b: a / b if b else None
    recalls = [ratio(tn, tn + fp), ratio(tp, tp + fn)]
    f1s = [ratio(2*tn, 2*tn+fp+fn), ratio(2*tp, 2*tp+fp+fn)]
    return dict(confusion_matrix=cm.tolist(), n=tn+fp+fn+tp,
                balanced_accuracy=sum(recalls)/2 if None not in recalls else None,
                macro_f1=sum(f1s)/2 if None not in f1s else None,
                accuracy=ratio(tn+tp, tn+fp+fn+tp),
                non_sitting_recall=recalls[1], sitting_recall=recalls[0])


@torch.no_grad()
def evaluate(model, loader, args, csv_path=None):
    model.eval()
    cm = np.zeros((2, 2), dtype=np.int64)
    handle = open(csv_path, 'w', newline='') if csv_path else None
    writer = csv.writer(handle) if handle else None
    if writer:
        writer.writerow(['sample_index', 'window_index', 'subject_id', 'timestamp', 'label', 'prediction', 'p_non_sitting'])
    try:
        for x, y, indices, subjects, times in loader:
            logits = forward(model, x.to(args.device), args.family)
            prob = (logits.sigmoid() if args.family == 'chap' else logits.softmax(-1)[:, 1]).cpu().reshape(len(y), -1)
            labels = y.reshape(len(y), -1)
            pred = (prob >= .5).long()
            cm += np.bincount((labels*2+pred).numpy().ravel(), minlength=4).reshape(2, 2)
            if writer:
                for b in range(len(y)):
                    for j in range(labels.shape[1]):
                        t = times.reshape(len(y), -1)[b, j].item()
                        writer.writerow([indices[b].item(), j, subjects[b], t if np.isfinite(t) else '', labels[b,j].item(), pred[b,j].item(), prob[b,j].item()])
    finally:
        if handle:
            handle.close()
    return metrics(cm)


def atomic_save(value, path):
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family', choices=['chap', 'moca'], required=True)
    p.add_argument('--mode', choices=['scratch', 'ft', 'lp', 'predict'], required=True)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--resume', help='Rise last.pt, resumes at the next epoch')
    p.add_argument('--split', choices=['validation', 'test'], default='test')
    p.add_argument('--arch', choices=['tiny', 'base'], default='base')
    p.add_argument('--patch', type=int, default=20)
    p.add_argument('--acc-rows', nargs=3, type=int)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.mode in ('ft', 'lp', 'predict') and not (args.checkpoint or args.resume):
        p.error('This mode requires --checkpoint or --resume')
    if args.mode == 'scratch' and args.checkpoint:
        p.error('Scratch must not load a pretrained checkpoint')
    if args.acc_rows and (len(set(args.acc_rows)) != 3 or any(i < 0 or i > 5 for i in args.acc_rows)):
        p.error('--acc-rows must be three distinct indices in 0..5')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = load(args.resume or args.checkpoint) if (args.resume or args.checkpoint) else None
    if args.mode == 'predict' and args.family == 'moca' and 'rise_config' not in checkpoint:
        p.error('MoCA prediction requires a trained Rise checkpoint, not an SSL checkpoint')
    if checkpoint and 'rise_config' in checkpoint and checkpoint['rise_config']['family'] != args.family:
        p.error('Checkpoint model family does not match --family')
    model = build(args, checkpoint).to(args.device)
    head = model.fc_bilstm if args.family == 'chap' else model.encoder.head
    if args.mode == 'lp':
        model.requires_grad_(False)
        head.requires_grad_(True)
    def loader(split, shuffle=False):
        return DataLoader(RiseData(args.data_dir, args.family, split), batch_size=args.batch_size,
                          shuffle=shuffle, num_workers=args.workers, pin_memory=args.device.startswith('cuda'))
    if args.mode == 'predict':
        result = evaluate(model, loader(args.split), args, out / f'{args.split}_predictions.csv')
        (out / f'{args.split}_metrics.json').write_text(json.dumps(result, indent=2))
        print(result)
        return
    if (out / 'last.pt').exists() and not args.resume:
        p.error('Output contains last.pt; use a new run directory or --resume')
    train, val = loader('train', True), loader('validation')
    optimizer = torch.optim.AdamW((v for v in model.parameters() if v.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    start, best = 0, -1.
    if args.resume:
        for key in ('rise_config', 'optimizer', 'epoch', 'best', 'rng'):
            if key not in checkpoint:
                p.error(f'Resume checkpoint missing {key}')
        if checkpoint['rise_config']['mode'] != args.mode:
            p.error('Resume mode must match the original training mode')
        optimizer.load_state_dict(checkpoint['optimizer'])
        start, best = checkpoint['epoch']+1, checkpoint['best']
        torch.set_rng_state(checkpoint['rng'])
        random.setstate(checkpoint['python_rng'])
        np.random.set_state(checkpoint['numpy_rng'])
        if torch.cuda.is_available() and checkpoint.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    (out / 'config.json').write_text(json.dumps(vars(args), indent=2))
    for epoch in range(start, args.epochs):
        model.train()
        if args.mode == 'lp':
            model.eval()
            head.train()
        total = 0.
        for x, y, *_ in train:
            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = forward(model, x, args.family)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y.float()) if args.family == 'chap' else nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total += loss.item()
        result = evaluate(model, val, args)
        score = result['balanced_accuracy']
        if score is None:
            raise ValueError('Validation requires both classes to select by balanced accuracy')
        improved = score > best
        best = max(best, score)
        saved = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch, best=best,
                     rise_config=vars(args), rng=torch.get_rng_state(), python_rng=random.getstate(),
                     numpy_rng=np.random.get_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        atomic_save(saved, out / 'last.pt')
        if improved:
            atomic_save(saved, out / 'best.pt')
        record = dict(epoch=epoch, train_loss=total/len(train), validation=result)
        with (out / 'log.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        print(record, flush=True)


if __name__ == '__main__':
    main()
