"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.

Exercise:
- Added support to handle regular expression splitting pattern.
"""
from typing import List
import regex as re
from .base import Tokenizer, get_stats, merge


GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class BasicTokenizer(Tokenizer):
    def __init__(self, regex: bool = False):
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if regex else None
        self.compiled_pattern = re.compile(self.pattern) if self.pattern else None

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text preprocessing
        if self.pattern:
            # split text into matched chunks
            text_chunks = re.findall(self.compiled_pattern, text)
            # list of list of integers in range 0..255
            ids = [list(chunk.encode("utf-8")) for chunk in text_chunks]
        else:
            text_bytes = text.encode("utf-8")  # raw bytes
            ids = list(text_bytes)  # list of integers in range 0..255

        # iteratively merge the most common pairs to create new tokens
        merges = {}  # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)}  # int -> bytes
        for i in range(num_merges):
            stats = {}
            # count up the number of times every consecutive pair appears
            if self.pattern:
                for chunked_ids in ids: get_stats(chunked_ids, stats)
            else:
                stats = get_stats(ids, stats)
                
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            if self.pattern:
                ids = [merge(chunked_ids, pair, idx) for chunked_ids in ids]
            else:
                ids = merge(ids, pair, idx)
                
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(
                    f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences"
                )

        # save class variables
        self.merges = merges  # used in encode()
        self.vocab = vocab  # used in decode()

    def decode(self, ids):
        # given ids (list of integers), return Python string
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text) -> List[int]:
        if self.pattern:
            text_chunks = re.findall(self.compiled_pattern, text)
        else:
            text_chunks = [text]
        ids = []
        for chunk in text_chunks:
            ids.extend(self._encode_chunk(chunk))
        return ids

    def _encode_chunk(self, text) -> List[int]:
        # given a string text, return the token ids
        text_bytes = text.encode("utf-8")  # raw bytes
        ids = list(text_bytes)  # list of integers in range 0..255
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break  # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids
