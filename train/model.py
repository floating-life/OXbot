"""Small candidate scorer shared by BC, DMC and the C++ export contract.

The forward path contains only embedding, Linear, LayerNorm, causal attention,
ReLU and softmax. The fixed operations keep a dependency-free C++17 evaluator
practical; PyTorch/CUDA are used exclusively for training.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    vocabulary: int = 128
    history_length: int = 256
    width: int = 64
    layers: int = 2
    heads: int = 4
    feedforward: int = 256
    state_features: int = 128
    action_features: int = 128
    layer_norm_epsilon: float = 1e-5
    architecture: str = "oxbot-causal-candidate-v1"


class CausalBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.heads
        self.width = config.width
        self.norm1 = nn.LayerNorm(config.width, eps=config.layer_norm_epsilon)
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.projection = nn.Linear(config.width, config.width)
        self.norm2 = nn.LayerNorm(config.width, eps=config.layer_norm_epsilon)
        self.ff1 = nn.Linear(config.width, config.feedforward)
        self.ff2 = nn.Linear(config.feedforward, config.width)

    def forward(self, x, valid):
        batch, length, width = x.shape
        y = self.norm1(x)
        q, k, v = self.qkv(y).chunk(3, dim=-1)
        reshape = lambda z: z.view(batch, length, self.heads, width // self.heads).transpose(1, 2)
        q, k, v = map(reshape, (q, k, v))
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(width // self.heads)
        causal = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
        allowed = causal[None, None, :, :] & valid[:, None, None, :]
        scores = scores.masked_fill(~allowed, float("-inf"))
        y = (scores.softmax(dim=-1) @ v).transpose(1, 2).contiguous().view(batch, length, width)
        x = x + self.projection(y)
        return x + self.ff2(torch.relu(self.ff1(self.norm2(x))))


class CandidateModel(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        assert c.width % c.heads == 0
        self.embedding = nn.Embedding(c.vocabulary, c.width)
        self.position = nn.Embedding(c.history_length, c.width)
        self.blocks = nn.ModuleList([CausalBlock(c) for _ in range(c.layers)])
        self.final_norm = nn.LayerNorm(c.width, eps=c.layer_norm_epsilon)
        self.state_projection = nn.Linear(c.state_features, c.width)
        self.action_projection = nn.Linear(c.action_features, c.width)
        self.head1 = nn.Linear(3 * c.width, c.width)
        self.head2 = nn.Linear(c.width, 1)

    def encode(self, tokens, lengths, state):
        # Right padding with at least one BOS token. Token 0 is exclusively PAD.
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        valid = positions[None, :] < lengths[:, None]
        x = self.embedding(tokens) + self.position(positions)[None, :, :]
        for block in self.blocks:
            x = block(x, valid)
        history = self.final_norm(x)[torch.arange(x.shape[0], device=x.device), lengths - 1]
        return history, torch.relu(self.state_projection(state))

    def forward(self, tokens, lengths, state, actions, mask=None):
        history, state_hidden = self.encode(tokens, lengths, state)
        candidate = torch.relu(self.action_projection(actions))
        count = actions.shape[1]
        context = torch.cat((history[:, None].expand(-1, count, -1),
                             state_hidden[:, None].expand(-1, count, -1), candidate), dim=-1)
        scores = self.head2(torch.relu(self.head1(context))).squeeze(-1)
        return scores if mask is None else scores.masked_fill(~mask, float("-inf"))

    def architecture(self):
        return asdict(self.config)
