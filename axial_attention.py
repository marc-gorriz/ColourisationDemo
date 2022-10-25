# XCNET - Attention-based Stylisation for Exemplar Image Colourisation
# Copyright (C) 2021  BBC Research & Development
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import torch
import torch.nn.functional as F
from torch import nn


class _LinearWithBias(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)


class MultiheadAxialAttention(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, num_heads=8, dropout=0., bias=True, width=False):
        super(MultiheadAxialAttention, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.width = width
        self.dropout = dropout
        self.head_dim = out_dim // num_heads
        assert self.head_dim * num_heads == self.out_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = nn.Parameter(torch.empty(2 * out_dim, in_dim))

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(2 * out_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        self.out_proj = _LinearWithBias(self.out_dim, self.out_dim)

        self.relative = nn.Parameter(torch.empty(self.head_dim * 2, kernel_size * 2 - 1))
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))

        self._reset_parameters()

    def _reset_parameters(self):
        scaling = float(self.head_dim) ** -0.5
        nn.init.normal_(self.in_proj_weight, mean=0, std=scaling)
        nn.init.normal_(self.relative, mean=0, std=scaling)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, need_weights=False):
        bsz, in_dim, width, height = query.size()
        assert key.size(1) == self.in_dim and value.size(1) == self.in_dim
        assert in_dim == self.in_dim

        if torch.equal(query, key) and torch.equal(key, value):

            if self.width:
                query = query.permute(0, 2, 3, 1)
            else:
                query = query.permute(0, 3, 2, 1)
            query = query.contiguous().view(bsz * width, height, in_dim)
            qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
            q, k, v = torch.split(qkv, [self.out_dim // 2, self.out_dim // 2, self.out_dim], dim=2)

        elif torch.equal(key, value):

            if self.width:
                query = query.permute(0, 2, 3, 1)
            else:
                query = query.permute(0, 3, 2, 1)
            query = query.contiguous().view(bsz * width, height, in_dim)
            _b = self.in_proj_bias
            _start = 0
            _end = self.out_dim // 2
            _w = self.in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            if self.width:
                key = key.permute(0, 2, 3, 1)
            else:
                key = key.permute(0, 3, 2, 1)
            key = key.contiguous().view(bsz * width, height, in_dim)
            _b = self.in_proj_bias
            _start = self.out_dim // 2
            _w = self.in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            kv = F.linear(key, _w, _b)
            k, v = torch.split(kv, [self.out_dim // 2, self.out_dim], dim=2)

        else:
            q = None
            k = None
            v = None
            ValueError('Incompatible mode')

        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.head_dim * 2,
                                                                                       self.kernel_size,
                                                                                       self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings,
                                                            [self.head_dim // 2, self.head_dim // 2,
                                                             self.head_dim], dim=0)

        q = q.contiguous().view(bsz * width, height, self.num_heads, self.head_dim // 2).permute(0, 2, 3, 1)
        k = k.contiguous().view(bsz * width, height, self.num_heads, self.head_dim // 2).permute(0, 2, 3, 1)
        v = v.contiguous().view(bsz * width, height, self.num_heads, self.head_dim).permute(0, 2, 3, 1)

        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = stacked_similarity.view(bsz * width, 3, self.num_heads, height, height).sum(dim=1)

        similarity = F.softmax(stacked_similarity, dim=3)

        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
        stacked_output = torch.cat([sv, sve], dim=-1).view(bsz * width, self.out_dim * 2, height)
        output = stacked_output.view(bsz * width, self.out_dim, 2, height).sum(dim=-2).transpose(1, 2)
        output = F.linear(output, self.out_proj.weight, self.out_proj.bias)
        output = output.view(bsz, width, height, self.out_dim).transpose(2, 3)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if need_weights:
            similarity = similarity.view(bsz, width, self.num_heads, height, height)
            return output, similarity.sum(dim=2) / self.num_heads  # check
        else:
            return output
