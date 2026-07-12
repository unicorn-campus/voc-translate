# -*- coding: utf-8 -*-
"""워밍업 데모: "하늘에 먹구름이 많아지면 뭐가 생각나?" → 답 생성.

미니 트랜스포머(인코더-디코더)를 아주 작은 질문→답 데이터로 학습시켜,
트랜스포머의 3가지 직관을 눈으로 확인한다.

    1) 어텐션   : 답을 만들 때 질문의 어느 단어에 주목하는가 (히트맵)
    2) 순차 생성: 답을 <sos>부터 한 단어씩, 앞말만 보고(마스킹) 생성
    3) 크로스 어텐션 : 답을 쓰는 내내 질문(원문)을 곁눈질

교육용 토이임 — 데이터가 6쌍뿐이라 '학습'보다 '구조 이해'가 목적이다.
Colab 번역 과제(한국어→영어)로 가면 질문→답이 원문→번역으로 바뀔 뿐 구조는 같다.
"""
import logging
import os
import random

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

import matplotlib
matplotlib.use("Agg")  # 창을 띄우지 않고 파일로 저장
import matplotlib.pyplot as plt
from matplotlib import font_manager

from mini_transformer import (
    Transformer,
    make_causal_mask,
    make_decoder_mask,
    make_padding_mask,
)

# ----------------------------------------------------------------- 특수 토큰
PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"
SPECIALS = [PAD, SOS, EOS, UNK]

# ----------------------------------------------------------------- 학습 데이터
# "하늘에 {대상} 보이면 뭐가 생각나" → 답.
# 문장 틀을 완전히 동일하게 두고 오직 '대상' 단어만 바꾼다. 그러면 답을 가르는 유일한
# 판별 토큰이 '대상'이 되어, 모델이 반드시 그 단어에 주목하게 된다(어텐션 시연이 선명해짐).
DATA = [
    ("하늘에 먹구름이 보이면 뭐가 생각나", "비가 올 것 같아"),
    ("하늘에 별이 보이면 뭐가 생각나", "밤이 깊었나 봐"),
    ("하늘에 해가 보이면 뭐가 생각나", "아침이 밝았구나"),
    ("하늘에 무지개가 보이면 뭐가 생각나", "비가 그쳤나 봐"),
    ("하늘에 눈송이가 보이면 뭐가 생각나", "겨울이 왔구나"),
    ("하늘에 노을이 보이면 뭐가 생각나", "저녁이 되었네"),
]


def tokenize(sentence: str):
    """공백 단위 토큰화."""
    return sentence.strip().split()


class Vocab:
    """단어 ↔ 정수 id 사전."""

    def __init__(self, sentences):
        toks = sorted({t for s in sentences for t in tokenize(s)})
        self.itos = list(SPECIALS) + toks
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    @property
    def pad_id(self):
        return self.stoi[PAD]

    @property
    def sos_id(self):
        return self.stoi[SOS]

    @property
    def eos_id(self):
        return self.stoi[EOS]

    def encode(self, sentence, add_special=True):
        ids = [self.stoi.get(t, self.stoi[UNK]) for t in tokenize(sentence)]
        return [self.sos_id] + ids + [self.eos_id] if add_special else ids

    def decode(self, ids):
        return " ".join(self.itos[i] for i in ids if self.itos[i] not in (PAD, SOS, EOS))


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)


def set_korean_font():
    """그래프의 한글이 깨지지 않도록 시스템의 한글 폰트를 탐지해 적용."""
    logging.getLogger("matplotlib.mathtext").setLevel(logging.ERROR)  # U+2212 글리프 경고 숨김
    plt.rcParams["axes.unicode_minus"] = False
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"]:
        if name in installed:
            plt.rcParams["font.family"] = name
            return name
    return None  # 한글 폰트를 못 찾으면 라벨이 □로 보일 수 있음(실행에는 지장 없음)


def build_batches(pairs, src_vocab, tgt_vocab):
    """질문/답을 id 텐서로 바꾸고 패딩해 배치로 만든다."""
    src = [torch.tensor(src_vocab.encode(q, add_special=False)) for q, _ in pairs]
    tgt = [torch.tensor(tgt_vocab.encode(a, add_special=True)) for _, a in pairs]
    src_ids = pad_sequence(src, batch_first=True, padding_value=src_vocab.pad_id)
    tgt_ids = pad_sequence(tgt, batch_first=True, padding_value=tgt_vocab.pad_id)
    return src_ids, tgt_ids


def train(pairs, src_vocab, tgt_vocab, epochs=400, lr=3e-4, seed=42):
    """teacher forcing 으로 질문→답 매핑을 학습. loss 이력을 반환."""
    set_seed(seed)
    src_ids, tgt_ids = build_batches(pairs, src_vocab, tgt_vocab)
    # 위치 인코딩 용량은 생성 길이(answer()의 max_len=20)까지 넉넉히 확보 — 짧은 답에서도 안전
    max_len = max(src_ids.size(1), tgt_ids.size(1), 20) + 2
    model = Transformer(len(src_vocab), len(tgt_vocab), max_len=max_len)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_id)

    dec_in, dec_tgt = tgt_ids[:, :-1], tgt_ids[:, 1:]
    src_mask = make_padding_mask(src_ids, src_vocab.pad_id)
    history = []
    model.train()
    for ep in range(1, epochs + 1):
        tgt_mask = make_decoder_mask(dec_in, tgt_vocab.pad_id)  # 패딩 + 미래 가림
        logits = model(src_ids, dec_in, src_mask, tgt_mask)
        loss = crit(logits.reshape(-1, logits.size(-1)), dec_tgt.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(loss.item())
        if ep == 1 or ep % 100 == 0:
            print(f"  epoch {ep:4d} | loss {loss.item():.4f}")
    return model, history


@torch.no_grad()
def answer(model, src_vocab, tgt_vocab, question, max_len=20):
    """질문에 대한 답을 greedy 로 한 단어씩 생성. (생성 스텝 로그, 답, 크로스 어텐션) 반환."""
    model.eval()
    src_ids = torch.tensor([src_vocab.encode(question, add_special=False)])
    src_mask = make_padding_mask(src_ids, src_vocab.pad_id)
    enc = model.encode(src_ids, src_mask)

    gen = [tgt_vocab.sos_id]
    steps = []  # (지금까지 입력, 새로 고른 단어) — 마스킹 시연용
    for _ in range(max_len):
        tgt = torch.tensor([gen])
        tgt_mask = make_causal_mask(tgt.size(1), tgt.device)
        logits = model.decode_step(tgt, enc, tgt_mask, src_mask)
        nxt = logits[0, -1].argmax().item()  # 마지막 위치의 최고점 단어
        steps.append((tgt_vocab.decode(gen), tgt_vocab.itos[nxt]))
        gen.append(nxt)
        if nxt == tgt_vocab.eos_id:
            break

    # 완성된 답 전체로 다시 한 번 통과시켜 크로스 어텐션을 안정적으로 확보
    tgt = torch.tensor([gen])
    tgt_mask = make_causal_mask(tgt.size(1), tgt.device)
    model.decode_step(tgt, enc, tgt_mask, src_mask)
    cross = model.decoder.layers[-1].cross_attn.last_attn_weights[0].mean(0)  # (tgt_len, src_len)
    return steps, tgt_vocab.decode(gen), cross


def plot_attention(question, answer_text, cross, out_path):
    """답의 각 단어가 질문의 어느 단어에 주목했는지 히트맵으로 저장."""
    q_tokens = tokenize(question)
    a_tokens = tokenize(answer_text)
    if not a_tokens or not q_tokens:  # 답이 비면 그릴 것이 없음(방어)
        return False
    # row i (입력 위치 i) 가 답의 i번째 단어를 예측 → 앞에서부터 len(a_tokens)개 행을 사용
    weights = cross[: len(a_tokens), : len(q_tokens)].numpy()

    fig, ax = plt.subplots(figsize=(1.1 * len(q_tokens) + 1, 0.7 * len(a_tokens) + 1))
    im = ax.imshow(weights, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(q_tokens)))
    ax.set_xticklabels(q_tokens, rotation=30, ha="right")
    ax.set_yticks(range(len(a_tokens)))
    ax.set_yticklabels(a_tokens)
    ax.set_xlabel("질문 토큰 (원문)")
    ax.set_ylabel("생성한 답 토큰")
    ax.set_title(f"크로스 어텐션 — '{answer_text}' 이(가) 주목한 곳")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def main():
    print("=" * 60)
    print("워밍업: '하늘에 먹구름이 많아지면 뭐가 생각나?' 미니 트랜스포머")
    print("=" * 60)

    font = set_korean_font()
    print(f"[폰트] 그래프 한글 폰트: {font or '못 찾음(라벨이 □로 보일 수 있음)'}")

    src_vocab = Vocab([q for q, _ in DATA])
    tgt_vocab = Vocab([a for _, a in DATA])
    print(f"[사전] 질문 단어 {len(src_vocab)}개, 답 단어 {len(tgt_vocab)}개, 학습 쌍 {len(DATA)}개\n")

    print("[학습] teacher forcing 으로 질문→답 매핑 학습")
    model, history = train(DATA, src_vocab, tgt_vocab)
    print(f"[학습] 최종 loss = {history[-1]:.4f}\n")

    # (1) 주인공 질문 — 순차 생성(마스킹) 시연 + 어텐션 히트맵
    main_q = "하늘에 먹구름이 보이면 뭐가 생각나"
    steps, ans, cross = answer(model, src_vocab, tgt_vocab, main_q)
    print("[생성] 한 단어씩(마스킹): 앞말만 보고 다음 단어를 고른다")
    print(f"  질문: {main_q}")
    for seen, nxt in steps:
        shown = seen if seen else "<sos>"
        print(f"    '{shown}'  →  다음 단어: {nxt}")
    print(f"  답  : {ans}\n")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attention_heatmap.png")
    plot_attention(main_q, ans, cross, out_path)
    print(f"[어텐션] 히트맵 저장 → {out_path}")
    print("  (답의 첫 단어가 질문의 '먹구름이'에 가장 밝게 주목하면 성공)\n")

    # (2) 대조 — 키워드가 바뀌면 주목 대상과 답이 함께 바뀐다
    print("[대조] 문장 구조는 같아도 키워드에 따라 답이 달라진다")
    for q in [
        "하늘에 먹구름이 보이면 뭐가 생각나",
        "하늘에 별이 보이면 뭐가 생각나",
        "하늘에 해가 보이면 뭐가 생각나",
    ]:
        _, a, _ = answer(model, src_vocab, tgt_vocab, q)
        keyword = tokenize(q)[1]  # 두 번째 단어 = 하늘의 대상
        print(f"    {keyword:<6} → {a}")

    print("\n완료! 히트맵(attention_heatmap.png)을 열어 어텐션이 어디에 쏠렸는지 확인하세요.")


if __name__ == "__main__":
    main()
