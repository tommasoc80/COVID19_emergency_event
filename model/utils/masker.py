from covid19_exceptius.types import *

from transformers import AutoTokenizer
from random import choice, random

Token = TypeVar('Token')


def mask_token(x: Token, mask: Token, mask_chance: float, keep_chance: float, replacements: Sequence[Token]) -> Token:
    return mask if random() < mask_chance else x if random() < keep_chance else choice(replacements)


def mask_sequence(seq: Sequence[Token], indices: Sequence[int], masker: Map[Token, Token]) -> Sequence[Token]:
    return [masker(t) if indices[i] == 1 else t for i, t in enumerate(seq)]


def mask_indices(seq: Sequence[Token], random_chance: float, unmaskable: Set[Token],
                 subwords: Set[Token], force: Maybe[int]) -> Sequence[int]:
    if not seq:
        return []
    t = seq[0]
    if t in unmaskable:
        mask = 0
    elif force is not None:
        mask = (0 if random() > random_chance else 1) if t not in subwords else force
    else:
        print(seq)
        print(t)
        raise ValueError('Trying to mask subword but no force assigned.')
    force = mask
    return [mask] + list(mask_indices(seq[1:], random_chance, unmaskable, subwords, force))


class RandomReplacer(Generic[Token]):
    def __init__(self, token_mask: Token, mask_chance: float, keep_chance: float, replacements: Set[Token]):
        self.token_mask = token_mask
        self.mask_chance = mask_chance
        self.keep_chance = keep_chance
        self.replacements = tuple(replacements)

    def __call__(self, x: Token) -> Token:
        return mask_token(x, self.token_mask, self.mask_chance, self.keep_chance, self.replacements)


class Masker(Generic[Token]):
    def __init__(self, outer_chance: float, mask_chance: float, keep_chance: float, replacements: Set[Token],
                 token_mask: Token, token_pad: Token, unmaskable: Set[Token], subwords: Set[Token]):
        """
            Masker class that handles masking and replacement functions.

        :param outer_chance:    Chance that a token is hidden from the input.
        :param mask_chance:     Chance that a hidden token is replaced with the [MASK] token.
        :param keep_chance:     Chance that a hidden token is retained.
        :param replacements:    A set of tokens that can be used as replacements.
        :param token_mask:      The [MASK] token.
        :param token_pad:       The [PAD] token.
        :param unmaskable:      A set of tokens that can never be masked.
        :param subwords:        A set of tokens that indicate word spans.
        """
        self.outer_chance = outer_chance
        self.unmaskable = unmaskable
        self.subwords = subwords
        self.token_pad = token_pad
        self.token_mask = token_mask
        self.replacer = RandomReplacer(token_mask, mask_chance, keep_chance, replacements)

    def __call__(self, sent: Sequence[Token]) -> Tuple[Sequence[int], Sequence[Token]]:
        indices = mask_indices(sent, self.outer_chance, self.unmaskable, self.subwords, None)
        return indices, mask_sequence(sent, indices, self.replacer.__call__)


def get_masker(_tokenizer: AutoTokenizer) -> Masker:
    unmaskable = set(_tokenizer.all_special_ids)
    replacements = set(_tokenizer.get_vocab().values()).difference(unmaskable)
    subwords = set({k: v for k, v in _tokenizer.get_vocab().items() if not k.startswith('▁')}.values())
    pad = _tokenizer.pad_token_id
    mask = _tokenizer.mask_token_id
    return Masker(outer_chance=0.15, mask_chance=0.8, keep_chance=0.5, replacements=replacements, token_mask=mask,
                  token_pad=pad, unmaskable=unmaskable, subwords=subwords)