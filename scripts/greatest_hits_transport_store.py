"""Bounded read-only native event transport; preserves saved physical positions."""

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from scripts.greatest_hits_transport import validate_positions
from scripts.native_video_cache import sha
from tiny_omni_decision.native_candidate import NativeTokenBlock


class ExplicitEventStore:
    def __init__(self, binding, directories, hashes, *, device='cpu'):
        self.binding, self.directories, self.hashes = binding, directories, hashes
        self.device = device
        self.arrays, self.cache, self.verified = OrderedDict(), OrderedDict(), {}
        self.cache_bytes = 0
        values = self.array('audio', 'queries.npy')
        if (values.shape != (len(binding['query_keys']), 384) or values.dtype != np.float32
                or not np.isfinite(values).all()):
            raise ValueError('finite original query bank required')
        self.query_indices = {key: i for i, key in enumerate(binding['query_keys'])}
        self.queries = torch.from_numpy(np.array(values, copy=True)).to(device)

    def _path(self, kind, name):
        if kind not in self.directories or name != Path(name).name or name not in self.hashes[kind]:
            raise ValueError('registered local bank required')
        return Path(self.directories[kind])/name

    @staticmethod
    def _stamp(path):
        value = path.stat()
        return value.st_size, value.st_mtime_ns

    def _unchanged(self, keys=None):
        for kind, name in self.verified if keys is None else keys:
            if (kind, name) in self.verified and self._stamp(
                    self._path(kind, name)) != self.verified[kind, name]:
                raise ValueError('cached feature file changed')

    def array(self, kind, name):
        key, path = (kind, name), self._path(kind, name)
        stamp = self._stamp(path)
        if key not in self.verified:
            if sha(path) != self.hashes[kind][name] or self._stamp(path) != stamp:
                raise ValueError('feature bank hash differs')
            self.verified[key] = stamp
        elif self.verified[key] != stamp:
            raise ValueError('cached feature file changed')
        if key not in self.arrays:
            self.arrays[key] = np.load(path, mmap_mode='r', allow_pickle=False)
            while len(self.arrays) > 32:
                self.arrays.popitem(last=False)
        self.arrays.move_to_end(key)
        return self.arrays[key]

    def verify_files(self):
        self._unchanged()
        for kind, name in self.verified:
            if sha(self._path(kind, name)) != self.hashes[kind][name]:
                raise ValueError('feature bank hash changed')

    def get(self, question_id, *, order=None, mode='joint'):
        if question_id not in self.binding['questions'] or mode not in ('joint', 'audio', 'video'):
            raise ValueError('registered question and explicit available-input view required')
        row = self.binding['questions'][question_id]
        count = len(row['query_keys'])
        order = list(range(count)) if order is None else order
        if (not isinstance(order, (list, tuple)) or any(type(i) is not int for i in order)
                or sorted(order) != list(range(count))):
            raise ValueError('complete candidate permutation required')
        ident = row['event_id']
        event = self.binding['events'][ident]
        parent = event['parent']
        self._unchanged([('audio', 'queries.npy'), ('audio', 'speech-positions.npy'),
            ('audio', 'event_audio-positions.npy'),
            *[('audio', parent+'-'+k+'.npy') for k in ('speech', 'event_audio')],
            *[('video', parent+'-'+k+'.npy') for k in ('video', 'positions', 'pts')]])
        if ident not in self.cache:
            parent, index, n = event['parent'], event['bank_index'], event['bank_events']
            def selected(name, shape):
                kind = 'audio' if name in ('speech', 'event_audio') else 'video'
                bank = self.array(kind, parent+'-'+name+'.npy')
                if bank.shape != (n, *shape):
                    raise ValueError('native event bank shape differs')
                return np.array(bank[index], copy=True)
            values = dict(speech=selected('speech', (25,384)),
                          event_audio=selected('event_audio', (1,384)),
                          video=selected('video', (2048,768)))
            positions = {k: np.array(self.array('audio', k+'-positions.npy'), copy=True)
                         for k in ('speech', 'event_audio')}
            positions['video'] = selected('positions', (2048,3))
            pts = selected('pts', (8,))
            validate_positions(positions, pts, event['timing'])
            blocks, size = [], 0
            for kind, value in values.items():
                if value.dtype != np.float32 or not np.isfinite(value).all():
                    raise ValueError('finite saved FP32 features required')
                x = torch.from_numpy(value).to(self.device)
                p = torch.from_numpy(positions[kind]).to(self.device)
                valid = torch.ones(len(value), dtype=torch.bool, device=self.device)
                size += sum(v.numel()*v.element_size() for v in (x,p,valid))
                blocks.append(NativeTokenBlock(kind, x, p, valid))
            self.cache[ident] = blocks, size
            self.cache_bytes += size
            while self.cache_bytes > 64*2**20 and len(self.cache) > 1:
                _, (_, removed) = self.cache.popitem(last=False)
                self.cache_bytes -= removed
        self.cache.move_to_end(ident)
        blocks = [b for b in self.cache[ident][0]
                  if mode == 'joint' or (b.kind == 'video') == (mode == 'video')]
        indices = torch.tensor([self.query_indices[row['query_keys'][i]] for i in order],
                               dtype=torch.long, device=self.device)
        return blocks, self.queries[indices]
