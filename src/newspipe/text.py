"""Headline text cleaning.

The archive mixes two sources. The 2008+ part was scraped with Python 2-style byte
literals left in the text, e.g. `b"Georgia 'downs two Russian warplanes'"` or
`b'Why wont America and Nato help us?'`, sometimes with escaped UTF-8
(`\\xe2\\x80\\x99`). These artifacts would otherwise become features ("b", "xe2"...).
"""

from __future__ import annotations

import codecs
import hashlib
import html
import re

_BYTES_LITERAL = re.compile(r"""^b(['"])(.*)\1$""", re.DOTALL)
_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")


def clean_headline(raw: str | None) -> str:
    if raw is None:
        return ""
    text = raw.strip()
    m = _BYTES_LITERAL.match(text)
    if m:
        inner = m.group(2)
        try:
            # interpret \x.., \', \" escapes, then decode the resulting bytes as UTF-8
            text = codecs.escape_decode(inner.encode("latin-1", "backslashreplace"))[0].decode(
                "utf-8", "replace"
            )
        except (UnicodeError, ValueError):
            text = inner
    text = html.unescape(text)
    return _WS.sub(" ", text).strip()


def normalize(text: str) -> str:
    """Lower-case alphanumeric form used for text features and de-duplication."""
    return _WS.sub(" ", _NON_ALNUM.sub(" ", text.lower())).strip()


def headline_hash(text: str) -> str:
    return hashlib.sha1(normalize(text).encode()).hexdigest()
