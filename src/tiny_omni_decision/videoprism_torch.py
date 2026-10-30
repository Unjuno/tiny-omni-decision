"""Experimental frozen FP32 port of the native VideoPrism factorized encoder.

Operator conventions derive from google-deepmind/videoprism at
c6eba0ece3dd49a24ffc44582966997a4a59f1d8 (Apache-2.0). This backend is
unpadded inference at the configured native resolution; it does not interpolate.
Full native-token parity FAILED on real videos (see the R1 VideoPrism Torch
2026-10-14 report). This research backend is not the qualified extraction path.
"""

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

CHECKPOINT_SHA256 = '61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29'


@dataclass(frozen=True)
class Config:
    width: int = 768
    patch: int = 18
    frames: int = 16
    grid: int = 16
    heads: int = 12
    spatial_layers: int = 12
    temporal_layers: int = 4
    mlp: int = 3072

    def __post_init__(self):
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1
               for v in vars(self).values()) or self.width % self.heads:
            raise ValueError('positive integer dimensions and divisible attention width required')


NATIVE_CONFIG = Config()


def expected_shapes(config=NATIVE_CONFIG):
    c = config
    shapes = {
        'params/patch_projection/linear/kernel': (c.patch * c.patch * 3, c.width),
        'params/patch_projection/linear/bias': (c.width,),
        'params/spatial_pos_emb/emb_var': (c.grid * c.grid, c.width),
        'params/temporal_pos_emb/emb_var': (c.frames, c.width),
    }
    for side, layers in [('spatial', c.spatial_layers), ('temporal', c.temporal_layers)]:
        for name in ['bias', 'scale']:
            shapes[f'params/{side}_ln/{name}'] = (c.width,)
        prefix = f'params/{side}_encoder/transformers_stack/x_layers/'
        for norm in ['layer_norm', 'ff_layer/layer_norm']:
            for name in ['bias', 'scale']:
                shapes[prefix + norm + '/' + name] = (layers, c.width)
        for name, inputs, outputs in [('ffn_layer1', c.width, c.mlp),
                                     ('ffn_layer2', c.mlp, c.width)]:
            shapes[prefix + f'ff_layer/{name}/linear/kernel'] = (layers, inputs, outputs)
            shapes[prefix + f'ff_layer/{name}/linear/bias'] = (layers, outputs)
        for projection in ['query', 'key', 'value', 'post']:
            shapes[prefix + f'self_attention/{projection}/w'] = (
                layers, c.width, c.heads, c.width // c.heads
            )
            shapes[prefix + f'self_attention/{projection}/b'] = (
                (layers, c.width) if projection == 'post'
                else (layers, c.heads, c.width // c.heads)
            )
    return shapes


def patchify(video, patch_size):
    b, t, h, w, channels = video.shape
    return (video.reshape(b, t, h // patch_size, patch_size,
                          w // patch_size, patch_size, channels)
            .permute(0, 1, 2, 4, 3, 5, 6)
            .reshape(b * t, (h // patch_size) * (w // patch_size), -1))


def source_layer_norm(x, scale, bias):
    centered = x - x.mean(-1, keepdim=True)
    return centered * torch.rsqrt(centered.square().mean(-1, keepdim=True) + 1e-6) * (
        scale + 1
    ) + bias


def capped_attention(q, k, v):
    logits = torch.einsum('btnh,bsnh->bnts', q * (q.shape[-1] ** -0.5), k)
    logits = 50 * torch.tanh(logits / 50)
    probabilities = logits.float().softmax(-1)
    return torch.einsum('bnts,bsnh->btnh', probabilities, v)


class FrozenVideoPrism(nn.Module):
    def __init__(self, weights, config=NATIVE_CONFIG):
        super().__init__()
        self.config = config
        shapes = expected_shapes(config)
        if set(weights) != set(shapes):
            raise ValueError('checkpoint tensor names differ from exact reviewed schema')
        for key, shape in shapes.items():
            value = np.asarray(weights[key])
            if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f'checkpoint tensor shape/dtype/values invalid: {key}')
        self.weights = nn.ParameterDict({
            key.replace('/', '__'): nn.Parameter(
                torch.from_numpy(np.array(weights[key], copy=True)), requires_grad=False
            )
            for key in shapes
        })
        self.eval()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Preflight the complete FP32 copy before any parameter can change."""
        if not strict or assign:
            raise ValueError('state reload requires strict copy; partial/assign unsupported')
        current = self.state_dict()
        if set(state_dict) != set(current):
            raise ValueError('state tensor names differ from exact reviewed schema')
        for key, target in current.items():
            value = state_dict[key]
            if (not isinstance(value, torch.Tensor) or value.shape != target.shape
                    or value.dtype != torch.float32 or target.dtype != torch.float32
                    or value.layout != torch.strided or value.device.type == 'meta'
                    or not torch.isfinite(value).all()):
                raise ValueError(f'state tensor shape/dtype/values invalid: {key}')
        return super().load_state_dict(state_dict, strict=True, assign=False)

    def weight(self, key):
        return self.weights['params__' + key.replace('/', '__')]

    def norm(self, x, prefix, layer=None):
        scale, bias = self.weight(prefix + '/scale'), self.weight(prefix + '/bias')
        return source_layer_norm(x, scale if layer is None else scale[layer],
                                 bias if layer is None else bias[layer])

    def stack(self, x, side, count):
        prefix = side + '_encoder/transformers_stack/x_layers/'
        c = self.config
        for i in range(count):
            normalized = self.norm(x, prefix + 'layer_norm', i)
            projections = []
            for name in ['query', 'key', 'value']:
                weight = self.weight(prefix + f'self_attention/{name}/w')[i]
                bias = self.weight(prefix + f'self_attention/{name}/b')[i]
                projected = normalized @ weight.reshape(c.width, c.width)
                projections.append(projected.reshape(*x.shape[:-1], c.heads, -1) + bias)
            attended = capped_attention(*projections).reshape(*x.shape)
            post = self.weight(prefix + 'self_attention/post/w')[i].reshape(c.width, c.width)
            x = x + (attended @ post.T + self.weight(prefix + 'self_attention/post/b')[i])
            normalized = self.norm(x, prefix + 'ff_layer/layer_norm', i)
            hidden = normalized @ self.weight(prefix + 'ff_layer/ffn_layer1/linear/kernel')[i]
            hidden = F.gelu(hidden + self.weight(prefix + 'ff_layer/ffn_layer1/linear/bias')[i],
                            approximate='none')
            x = x + (hidden @ self.weight(prefix + 'ff_layer/ffn_layer2/linear/kernel')[i]
                     + self.weight(prefix + 'ff_layer/ffn_layer2/linear/bias')[i])
        return x

    def forward(self, video):
        c = self.config
        reference = next(self.parameters())
        if self.training:
            raise ValueError('frozen inference only; training mode unsupported')
        if (video.ndim != 5 or video.shape[0] < 1
                or tuple(video.shape[1:]) != (c.frames, c.grid*c.patch, c.grid*c.patch, 3)
                or video.dtype != torch.float32 or reference.dtype != torch.float32
                or torch.is_autocast_enabled(video.device.type)
                or (video.device.type == 'cuda' and torch.backends.cuda.matmul.allow_tf32)
                or video.device != reference.device or not torch.isfinite(video).all()
                or video.min() < 0 or video.max() > 1):
            raise ValueError('native unpadded FP32 RGB input on the model device required')
        b = video.shape[0]
        x = patchify(video, c.patch) @ self.weight('patch_projection/linear/kernel')
        x = x + self.weight('patch_projection/linear/bias')
        x = x + self.weight('spatial_pos_emb/emb_var')
        x = self.norm(self.stack(x, 'spatial', c.spatial_layers), 'spatial_ln')
        x = x.reshape(b, c.frames, c.grid*c.grid, c.width).permute(0, 2, 1, 3)
        x = x.reshape(b*c.grid*c.grid, c.frames, c.width)
        x = x + self.weight('temporal_pos_emb/emb_var')
        x = self.norm(self.stack(x, 'temporal', c.temporal_layers), 'temporal_ln')
        x = x.reshape(b, c.grid*c.grid, c.frames, c.width).permute(0, 2, 1, 3)
        x = x.reshape(b, c.frames*c.grid*c.grid, c.width)
        if not torch.isfinite(x).all():
            raise ValueError('nonfinite VideoPrism output')
        return x


def load_checkpoint(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    if digest.hexdigest() != CHECKPOINT_SHA256:
        raise ValueError('pinned native VideoPrism checkpoint hash differs')
    with np.load(path, allow_pickle=False) as archive:
        return FrozenVideoPrism({key: archive[key] for key in archive.files})
