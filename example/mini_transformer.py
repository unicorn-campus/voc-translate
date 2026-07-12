# -*- coding: utf-8 -*-
"""미니 트랜스포머 (인코더-디코더) — 교육용 축소판.

Colab 번역 과제(`transformer_translate_colab_answers.ipynb`)에서 E2E로 검증한
아키텍처와 동일한 구조를 독립 모듈로 정리한 것. "질문 → 답" seq2seq 학습에 사용한다.

구성 요소
    - PositionalEncoding : 순서 정보를 sin/cos 파도무늬로 주입
    - scaled_dot_product_attention : 어텐션 핵심 공식
    - MultiHeadAttention : 여러 관점(헤드)으로 동시에 주목
    - PositionwiseFeedForward : 각 단어를 독립적으로 더 깊이 변환
    - EncoderLayer / DecoderLayer : 잔차+정규화로 감싼 블록
    - Encoder / Decoder / Transformer : 부품 조립
    - make_*_mask : 패딩·미래 가림(causal) 마스크
"""
import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """단어 순서를 sin/cos 파도무늬로 임베딩에 더해 주는 층."""

    def __init__(self, d_model: int, max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # 짝수 인덱스 = sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 홀수 인덱스 = cos
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    """어텐션 = softmax(QKᵀ/√dₖ)·V. 관련도 점수로 V를 가중합한다.

    학습용으로 수식을 직접 구현한 것. 실무에서는 PyTorch 2.0+ 의 융합·고속 구현
    ``torch.nn.functional.scaled_dot_product_attention`` 을 쓴다(마스크는 True=주목 관례로 동일).
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))  # 가릴 위치는 -무한대 → softmax 후 0
    attn_weights = torch.softmax(scores, dim=-1)
    if dropout is not None:
        attn_weights = dropout(attn_weights)
    context = torch.matmul(attn_weights, v)
    return context, attn_weights


class MultiHeadAttention(nn.Module):
    """d_model 차원을 여러 헤드로 나눠 서로 다른 관점으로 동시에 주목."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 은 num_heads 로 나눠떨어져야 함"
        self.d_model, self.num_heads, self.head_dim = d_model, num_heads, d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.last_attn_weights = None  # 시각화용: 마지막 forward의 어텐션 가중치 보관

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        return x.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, _, s, _ = x.shape
        return x.transpose(1, 2).contiguous().view(b, s, self.d_model)

    def forward(self, query, key, value, mask=None) -> torch.Tensor:
        q = self._split_heads(self.w_q(query))
        k = self._split_heads(self.w_k(key))
        v = self._split_heads(self.w_v(value))
        context, attn = scaled_dot_product_attention(q, k, v, mask, self.dropout)
        self.last_attn_weights = attn.detach()  # (B, heads, tgt_len, src_len)
        return self.w_o(self._merge_heads(context))


class PositionwiseFeedForward(nn.Module):
    """각 단어를 독립적으로 더 깊이 변환하는 작은 신경망 (d_model → d_ff → d_model)."""

    def __init__(self, d_model: int, d_ff=None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    """인코더 블록: Self-Attention → Add&Norm → FFN → Add&Norm."""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))            # 잔차(원본 x) + 정규화
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    """디코더 블록: Masked Self-Attn → Cross-Attn(원문 참고) → FFN, 각 뒤에 Add&Norm."""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask, src_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))               # 미래 가림
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, enc_out, src_mask)))  # 질문 참고
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


class Encoder(nn.Module):
    """질문을 읽고 문맥 표현으로 인코딩."""

    def __init__(self, vocab, d_model, num_heads, num_layers, d_ff, dropout, max_len):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab, d_model)
        self.pos = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, src, src_mask):
        x = self.pos(self.embedding(src) * math.sqrt(self.d_model))
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class Decoder(nn.Module):
    """인코딩된 질문을 참고해 답을 한 단어씩 만들 표현으로 디코딩."""

    def __init__(self, vocab, d_model, num_heads, num_layers, d_ff, dropout, max_len):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab, d_model)
        self.pos = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, tgt, enc_out, tgt_mask, src_mask):
        x = self.pos(self.embedding(tgt) * math.sqrt(self.d_model))
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, src_mask)
        return x


class Transformer(nn.Module):
    """인코더-디코더 미니 트랜스포머. 출력층은 디코더 임베딩과 가중치 공유(weight tying)."""

    def __init__(self, src_vocab, tgt_vocab, d_model=64, num_heads=4, num_layers=2,
                 d_ff=256, dropout=0.1, max_len=32, tie=True):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, num_heads, num_layers, d_ff, dropout, max_len)
        self.decoder = Decoder(tgt_vocab, d_model, num_heads, num_layers, d_ff, dropout, max_len)
        self.output_proj = nn.Linear(d_model, tgt_vocab, bias=False)
        if tie:
            self.output_proj.weight = self.decoder.embedding.weight

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc = self.encoder(src, src_mask)
        return self.output_proj(self.decoder(tgt, enc, tgt_mask, src_mask))

    def encode(self, src, src_mask):
        return self.encoder(src, src_mask)

    def decode_step(self, tgt, enc, tgt_mask, src_mask):
        return self.output_proj(self.decoder(tgt, enc, tgt_mask, src_mask))


def make_padding_mask(seq, pad_id):
    """패딩 토큰 위치를 가리는 마스크 (B, 1, 1, seq_len)."""
    return (seq != pad_id).unsqueeze(1).unsqueeze(2)


def make_causal_mask(seq_len, device):
    """미래를 못 보게 하는 하삼각(causal) 마스크 (1, 1, seq_len, seq_len)."""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    return mask.unsqueeze(0).unsqueeze(0)


def make_decoder_mask(tgt, pad_id):
    """디코더 마스크 = 패딩 마스크 AND 미래 가림 마스크."""
    return make_padding_mask(tgt, pad_id) & make_causal_mask(tgt.size(1), tgt.device)
