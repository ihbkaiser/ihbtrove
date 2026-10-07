"""Small offline tokenizer for language-neutral cross-document deduplication."""

from __future__ import annotations

import regex

from datatrove.utils.text import TERMINAL_PUNCTUATION
from datatrove.utils.word_tokenizers import WordTokenizer

_WORD = regex.compile(r"[\p{L}\p{M}\p{N}]+", regex.UNICODE)
_GRAPHEME = regex.compile(r"\X", regex.UNICODE)
# Scripts commonly written without spaces between lexical units are segmented
# into grapheme clusters so MinHash can compare local spans without a language
# detector or a script-specific model.
_NO_SPACE_SCRIPT = regex.compile(
    r"[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}"
    r"\p{Script=Thai}\p{Script=Lao}\p{Script=Khmer}\p{Script=Myanmar}]",
    regex.UNICODE,
)
_TERMINALS = frozenset(TERMINAL_PUNCTUATION)
_ASCII_TERMINALS = frozenset(".!?")
_CLOSING_PUNCTUATION = frozenset('"\'”’»)]}')


class UnicodeTokenizer(WordTokenizer):
    """Deterministic tokenization that does not assume a document language.

    Whitespace-delimited scripts use Unicode letter/number runs. Han, kana,
    Thai, Lao, Khmer, and Myanmar runs use grapheme clusters, which provides
    useful MinHash shingles without downloading a model.
    """

    def __init__(self) -> None:
        super().__init__(language=None)

    def word_tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        for match in _WORD.finditer(text):
            word = match.group()
            if _NO_SPACE_SCRIPT.search(word):
                tokens.extend(_GRAPHEME.findall(word))
            else:
                tokens.append(word)
        return tokens

    def sent_tokenize(self, text: str) -> list[str]:
        sentences: list[str] = []
        start = 0
        index = 0
        while index < len(text):
            char = text[index]
            if char == "\n":
                self._append_sentence(sentences, text[start:index])
                start = index + 1
                index += 1
                continue

            if char in _TERMINALS:
                end = index + 1
                while end < len(text) and text[end] in _CLOSING_PUNCTUATION:
                    end += 1
                # CJK and other full Unicode terminators commonly have no
                # following space. Keep ASCII dot/question/exclamation marks
                # conservative around abbreviations and inline punctuation.
                if end == len(text) or text[end].isspace() or char not in _ASCII_TERMINALS:
                    self._append_sentence(sentences, text[start:end])
                    start = end
                    while start < len(text) and text[start].isspace():
                        start += 1
                    index = start
                    continue
            index += 1

        self._append_sentence(sentences, text[start:])
        return sentences

    @staticmethod
    def _append_sentence(sentences: list[str], sentence: str) -> None:
        normalized = sentence.strip()
        if normalized:
            sentences.append(normalized)

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        cursor = 0
        for sentence in self.sent_tokenize(text):
            start = text.find(sentence, cursor)
            if start < 0:
                continue
            end = start + len(sentence)
            spans.append((start, end))
            cursor = end
        return spans
