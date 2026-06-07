# ---
# chapter: 25
# topic: Speculative Decoding
# section: 25.4
# difficulty: ⭐⭐⭐⭐⭐
# tier: gpu
# deps: none
# run: python 06_speculative_decoding.py
# expected_runtime: <1s
# expected_output: 模拟 draft model 出 K 个候选 token，target model 一次 verify
# ---
# See: ../tutorial/25_推理引擎与高性能服务.md §25.4
# Interview hooks:
#   1. Speculative decoding 加速原理？(答: 一次 verify 多个 draft token，接受率 α 时 ~1/(1-αα) 加速)
#   2. Draft model 如何选？(答: 小同族模型、Medusa、n-gram、EAGLE)
#   3. 接受率 0 时会发生什么？(答: 回退到单 token 步，无收益)

"""Speculative decoding: draft-then-verify simulation.

The target model verifies K draft tokens in *one* forward pass (cheap,
parallel). Each draft token is accepted with probability
    p(x) = min(1, p_target(x) / p_draft(x))
so the output distribution is preserved exactly.
"""
from __future__ import annotations
import random
from dataclasses import dataclass


@dataclass
class Vocab:
    """Mock vocab with two probability distributions (target & draft)."""
    target_probs: list[float]
    draft_probs: list[float]
    tokens: list[str]

    def sample(self, dist: list[float], k: int) -> list[int]:
        return random.choices(range(len(dist)), weights=dist, k=k)

    def accept_prob(self, x: int) -> float:
        return min(1.0, self.target_probs[x] / max(self.draft_probs[x], 1e-9))


def speculative_step(vocab: Vocab, gamma: int = 4) -> tuple[list[str], int, int]:
    """One round of draft + verify.

    Returns:
        accepted_tokens: tokens that will be emitted this step
        n_accepted: number accepted
        n_verified: number verified (forward calls = 1 for verify, plus gamma for draft)
    """
    # 1) Draft model autoregressively proposes gamma tokens (cheap)
    draft = vocab.sample(vocab.draft_probs, gamma)
    # 2) Target model verifies all gamma in one forward (in real impl, single fwd)
    accepted: list[int] = []
    for x in draft:
        if random.random() < vocab.accept_prob(x):
            accepted.append(x)
        else:
            # reject: sample a correction from (target - draft)+
            correction_dist = [
                max(0.0, vocab.target_probs[i] - vocab.draft_probs[i])
                for i in range(len(vocab.tokens))
            ]
            s = sum(correction_dist) or 1.0
            correction_dist = [p / s for p in correction_dist]
            accepted.append(vocab.sample(correction_dist, 1)[0])
            break
    return [vocab.tokens[x] for x in accepted], len(accepted), 1 + gamma


def normal_step(vocab: Vocab) -> tuple[list[str], int]:
    x = vocab.sample(vocab.target_probs, 1)[0]
    return [vocab.tokens[x]], 1


def main() -> None:
    random.seed(7)
    # Draft and target agree strongly → high acceptance → big speedup
    target = [0.50, 0.20, 0.15, 0.10, 0.05]
    draft  = [0.45, 0.22, 0.18, 0.10, 0.05]
    tokens = ["A", "B", "C", "D", "E"]
    vocab = Vocab(target_probs=target, draft_probs=draft, tokens=tokens)

    gamma = 4
    trials = 5000
    spec_emitted = 0
    spec_calls = 0
    norm_emitted = 0
    norm_calls = 0

    for _ in range(trials):
        out, n_acc, n_calls = speculative_step(vocab, gamma=gamma)
        spec_emitted += n_acc
        spec_calls += n_calls
        out, n_acc = normal_step(vocab)
        norm_emitted += n_acc
        norm_calls += 1

    print(f"speculative: emitted={spec_emitted}  forward_calls={spec_calls}  "
          f"tokens/call={spec_emitted/spec_calls:.2f}")
    print(f"normal     : emitted={norm_emitted}  forward_calls={norm_calls}  "
          f"tokens/call={norm_emitted/norm_calls:.2f}")
    print(f"effective speedup (tokens per fwd): {spec_emitted/spec_calls:.2f}x")
    # Theoretical: gamma * (1 - (1-alpha)^gamma)... rough intuition only


if __name__ == "__main__":
    main()
