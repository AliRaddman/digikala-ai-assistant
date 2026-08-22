"""Shared Persian text normalizer.

Owner: Ali. Locked module — every part of the system that touches text
(index, classifier, chains) imports from here, so the same string always
produces the same tokens. Change only by team agreement, with the reason
recorded in docs/DECISIONS.md.
"""

from __future__ import annotations

import re
import unicodedata

ZWNJ = "\u200c"

_CHAR_MAP: dict[str, str] = {
    "ي": "ی",
    "ى": "ی",
    "ك": "ک",
    "أ": "ا",
    "إ": "ا",
    "ٱ": "ا",
    "ة": "ه",
    "ۀ": "ه",
    "ؤ": "و",
    "٫": ".",
    "٬": ",",
    "،": ",",
    "؛": ";",
    "؟": "?",
    "٪": "%",
    "«": '"',
    "»": '"',
}

for _i, (_fa, _ar) in enumerate(zip("۰۱۲۳۴۵۶۷۸۹", "٠١٢٣٤٥٦٧٨٩")):
    _CHAR_MAP[_fa] = str(_i)
    _CHAR_MAP[_ar] = str(_i)

_DROP_CHARS = "".join(
    chr(c)
    for c in [
        *range(0x064B, 0x0653),
        0x0654,
        0x0655,
        0x0670,
        0x0640,
        0x200B,
        0x200D,
        0x200E,
        0x200F,
        0x202A,
        0x202B,
        0x202C,
        0x202D,
        0x202E,
        0x2066,
        0x2067,
        0x2068,
        0x2069,
        0xFEFF,
    ]
)

_SPACE_CHARS = "".join(
    chr(c) for c in [0x00A0, *range(0x2000, 0x200B), 0x202F, 0x205F, 0x3000]
)

_TRANSLATE: dict[int, str | None] = {ord(k): v for k, v in _CHAR_MAP.items()}
_TRANSLATE.update({ord(c): None for c in _DROP_CHARS})
_TRANSLATE.update({ord(c): " " for c in _SPACE_CHARS})

_EMOJI_RE = re.compile(
    "["
    "\U0001f000-\U0001faff"
    "\u2190-\u21ff"
    "\u2600-\u27bf"
    "\u2b00-\u2bff"
    "\ufe00-\ufe0f"
    "\u2049\u203c\u2122"
    "]+"
)
_REPEAT_FA_RE = re.compile(r"([\u0600-\u06ff])\1{2,}")
_REPEAT_LATIN_RE = re.compile(r"([A-Za-z])\1{2,}")
_SPACE_RE = re.compile(r"\s+")
_ZWNJ_RE = re.compile(r"[ ]*\u200c[ \u200c]*")
_NONWORD_RE = re.compile(r"[\W_]+")


def normalize(text: object) -> str:
    """Return the canonical form of a Persian/mixed string.

    Unifies Arabic letter variants, converts Persian/Arabic digits to ASCII,
    drops diacritics, emoji and bidi control characters, collapses stretched
    letters, and tidies spacing around ZWNJ. Case, punctuation and ZWNJ are
    preserved, so the output is still displayable to a user.

    Non-string input (including NaN) returns an empty string.
    """
    if not isinstance(text, str):
        return ""
    s = unicodedata.normalize("NFKC", text)
    s = s.translate(_TRANSLATE)
    s = _EMOJI_RE.sub(" ", s)
    s = _REPEAT_FA_RE.sub(r"\1", s)
    s = _REPEAT_LATIN_RE.sub(r"\1\1", s)
    s = _SPACE_RE.sub(" ", s)
    s = _ZWNJ_RE.sub(ZWNJ, s)
    return s.strip().strip(ZWNJ).strip()


def to_search_text(text: object) -> str:
    """Return the indexing form: normalized, ZWNJ split to space, punctuation
    removed, lowercased.

    This is the only form BM25 and the dense encoder should see, so that
    "کتاب‌ها" and "کتاب ها" produce identical tokens.
    """
    s = normalize(text).replace(ZWNJ, " ")
    s = _NONWORD_RE.sub(" ", s)
    return s.lower().strip()


def tokenize(text: object) -> list[str]:
    """Whitespace tokens of the search form. Used by the BM25 index."""
    return to_search_text(text).split()


def build_search_text(*parts: object, dedupe_tokens: bool = True) -> str:
    """Join several fields into one searchable string.

    By default repeated tokens are dropped while keeping first-occurrence
    order, because brand and category names usually already appear inside the
    product title and the extra term frequency only distorts BM25 scores.
    """
    tokens: list[str] = []
    for part in parts:
        tokens.extend(tokenize(part))
    if not dedupe_tokens:
        return " ".join(tokens)
    return " ".join(dict.fromkeys(tokens))