"""Generate a domain-specific KO -> EN toy corpus from ``data/glossary.json``.

Domain: e-commerce customer-service inquiries (이커머스 상품 문의). Two sentence
templates combine the glossary's subjects/objects/predicates:

  A. transitive : "{주체}{이/가} {대상}{을/를} {predicate}"   e.g. 고객이 배송을 문의한다
  B. status     : "{대상}{이/가} {predicate}"                 e.g. 배송이 지연된다

Korean particles (이/가, 을/를) are picked automatically from whether the
preceding syllable has a trailing consonant (받침), so glossary entries only
store the dictionary form -- no per-entry particle bookkeeping needed.

Usage:
    python data/build_dataset.py

Writes data/ko_en_pairs.txt (train) and data/ko_en_pairs_heldout.txt
(combinations withheld from training, for testing generalization to unseen
subject/object/predicate combos -- see translate.py).
"""

import json
import random
from pathlib import Path

GLOSSARY_PATH = Path(__file__).parent / "glossary.json"
TRAIN_PATH = Path(__file__).parent / "ko_en_pairs.txt"
HELDOUT_PATH = Path(__file__).parent / "ko_en_pairs_heldout.txt"

TRAIN_COUNT = 100
HELDOUT_COUNT = 20
SEED = 42


def has_final_consonant(word: str) -> bool:
    """Hangul syllable code = (initial*21 + medial)*28 + final + 0xAC00.

    final == 0 means the syllable has no trailing consonant (받침).
    """
    code = ord(word[-1]) - 0xAC00
    if not (0 <= code < 11172):
        return False  # non-Hangul character; not expected in this glossary
    return code % 28 != 0


def attach_subject_particle(word: str) -> str:
    return word + ("이" if has_final_consonant(word) else "가")


def attach_object_particle(word: str) -> str:
    return word + ("을" if has_final_consonant(word) else "를")


def conjugate_present(dict_form: str) -> str:
    """'~하다' -> '~한다', '~되다' -> '~된다' (both stems end in a vowel, so
    the plain-present ending '-ㄴ다' attaches directly)."""
    if dict_form.endswith("하다"):
        return dict_form[:-2] + "한다"
    if dict_form.endswith("되다"):
        return dict_form[:-2] + "된다"
    raise ValueError(f"unsupported predicate ending: {dict_form}")


def build_transitive_pairs(glossary: dict) -> list[tuple[str, str]]:
    pairs = []
    for subj in glossary["subjects"]:
        for obj in glossary["objects"]:
            for pred in glossary["transitive_predicates"]:
                ko = (
                    f"{attach_subject_particle(subj['ko'])} "
                    f"{attach_object_particle(obj['ko'])} "
                    f"{conjugate_present(pred['ko'])}"
                )
                en = f"{subj['en']} {pred['en']} {obj['en']}"
                pairs.append((ko, en))
    return pairs


def build_status_pairs(glossary: dict) -> list[tuple[str, str]]:
    pairs = []
    for obj in glossary["objects"]:
        for pred in glossary["status_predicates"]:
            ko = f"{attach_subject_particle(obj['ko'])} {conjugate_present(pred['ko'])}"
            en = f"{obj['en']} {pred['en']}"
            pairs.append((ko, en))
    return pairs


def main() -> None:
    glossary = json.loads(GLOSSARY_PATH.read_text(encoding="utf-8"))

    all_pairs = build_transitive_pairs(glossary) + build_status_pairs(glossary)
    total_pool = len(all_pairs)
    assert len(set(all_pairs)) == total_pool, "glossary produced duplicate sentence pairs"

    random.seed(SEED)
    random.shuffle(all_pairs)

    train_pairs = all_pairs[:TRAIN_COUNT]
    heldout_pairs = all_pairs[TRAIN_COUNT : TRAIN_COUNT + HELDOUT_COUNT]
    unused = total_pool - len(train_pairs) - len(heldout_pairs)

    TRAIN_PATH.write_text("\n".join(f"{ko}\t{en}" for ko, en in train_pairs) + "\n", encoding="utf-8")
    HELDOUT_PATH.write_text("\n".join(f"{ko}\t{en}" for ko, en in heldout_pairs) + "\n", encoding="utf-8")

    print(f"glossary combinations available: {total_pool}")
    print(f"  train   : {len(train_pairs):3d} -> {TRAIN_PATH.relative_to(Path.cwd())}")
    print(f"  heldout : {len(heldout_pairs):3d} -> {HELDOUT_PATH.relative_to(Path.cwd())}")
    print(f"  unused  : {unused:3d} (left in the glossary's combinatorial pool, not written out)")


if __name__ == "__main__":
    main()
