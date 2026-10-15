"""Causal history3/next-frame windows over the validated 5 m VAE cache."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_cache_entries(train_root, cache_root, vae_checkpoint):
    metadata = json.loads((cache_root / 'metadata.json').read_text())
    if metadata['vae_sha256'] != sha256(vae_checkpoint):
        raise ValueError('latent cache does not match the selected VAE weights')
    config = json.loads((vae_checkpoint.parent / 'config.json').read_text())
    if not np.isclose(float(config['scaling_factor']), metadata['scaling_factor'], rtol=0, atol=1e-12):
        raise ValueError('latent cache/VAE scaling factor mismatch')
    if metadata.get('shape') != [4, 27, 5] or not metadata.get('vae_gate', {}).get('passed'):
        raise ValueError('invalid or unvalidated latent cache')
    manifest = train_root.parent / 'manifest.json'
    if not manifest.is_file() or sha256(manifest) != metadata['dataset_manifest_sha256']:
        raise ValueError('dataset manifest does not match the latent cache')
    entries = []
    for item in metadata['shards']:
        relative = Path(item['dataset'])
        if relative.parts[0] != 'train':
            raise ValueError('cache dataset entry must start with train/')
        source = train_root / Path(*relative.parts[1:])
        cached = cache_root / item['file']
        if not source.is_file() or not cached.is_file():
            raise FileNotFoundError(f'missing source/cache: {source}, {cached}')
        entries.append({'source': str(source), 'cache': str(cached), 'frames': int(item['frames'])})
    if not entries:
        raise ValueError('empty latent cache')
    return entries, metadata


def split_entries(entries, fraction, seed):
    if not 0 < fraction < 1:
        raise ValueError('validation fraction must be in (0,1)')
    groups = {}
    for index, entry in enumerate(entries):
        group = Path(entry['source']).parent.parent.name
        groups.setdefault(group, []).append(index)
    rng = np.random.default_rng(seed)
    held_out = set()
    for group, indices in sorted(groups.items()):
        if len(indices) < 2:
            raise ValueError(f'need >=2 shards for group {group}')
        count = min(len(indices)-1, max(1, round(len(indices)*fraction)))
        held_out.update(int(x) for x in rng.choice(indices, count, replace=False))
    return ([entry for i, entry in enumerate(entries) if i not in held_out],
            [entry for i, entry in enumerate(entries) if i in held_out])


class CachedFutureDataset(Dataset):
    def __init__(self, entries, target_offset=1, return_pixels=False):
        self.entries = entries
        self.return_pixels = bool(return_pixels)
        self._arrays = {}
        self._handles = {}
        self.trajectory_count = 0
        chunks = []
        for shard, entry in enumerate(entries):
            a = np.load(entry['cache'], mmap_mode='r')
            if a.shape != (entry['frames'], 4, 27, 5) or a.dtype != np.float16:
                raise ValueError(f'invalid cache shape/dtype: {entry["cache"]}')
            with h5py.File(entry['source'], 'r') as handle:
                if handle.attrs.get('format') != 'navrl-isaac-trajectory-hdf5-v2':
                    raise ValueError('unsupported future dataset format')
                frames = handle['frames']
                if len(frames['sim_step']) != len(a):
                    raise ValueError('source frame count/cache alignment mismatch')
                env = np.asarray(frames['env_id'][:], dtype=np.int32)
                step = np.asarray(frames['sim_step'][:], dtype=np.int32)
                index = np.asarray(frames['frame_index'][:], dtype=np.int32)
            self.trajectory_count += len(np.unique(env))
            order = np.lexsort((index, env)).astype(np.int32)
            span = 3 + int(target_offset)
            if len(order) < span:
                continue
            windows = np.lib.stride_tricks.sliding_window_view(order, span)
            valid = np.all(env[windows] == env[windows[:, :1]], axis=1)
            valid &= np.all(np.diff(index[windows], axis=1) == 1, axis=1)
            valid &= np.all(np.diff(step[windows], axis=1) == 10, axis=1)
            selected = windows[valid][:, [0, 1, 2, span-1]]
            chunks.append(np.column_stack((np.full(len(selected), shard, dtype=np.int32), selected)))
        if not chunks:
            raise ValueError('no valid cached future windows')
        self.windows = np.concatenate(chunks).astype(np.int32)

    def __len__(self):
        return len(self.windows)

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_arrays'] = {}
        state['_handles'] = {}
        return state

    def __getitem__(self, index):
        shard, *rows = self.windows[index].tolist()
        if shard not in self._arrays:
            self._arrays[shard] = np.load(self.entries[shard]['cache'], mmap_mode='r')
        values = np.array(self._arrays[shard][rows], dtype=np.float32, copy=True)
        if self.return_pixels:
            if shard not in self._handles:
                self._handles[shard] = h5py.File(self.entries[shard]['source'], 'r')
            # Read the actual recorded target frame, not a VAE reconstruction.
            pixels = np.array(self._handles[shard]['frames/range_values'][rows[-1]],
                              dtype=np.float32, copy=True)
            return torch.from_numpy(values[:3]), torch.from_numpy(values[3]), torch.from_numpy(pixels)
        return torch.from_numpy(values[:3]), torch.from_numpy(values[3])
