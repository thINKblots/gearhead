"""
Custom tokenizer for Gearhead with specialized vocabulary for diagnostics.
"""

import json
from pathlib import Path
from typing import List, Optional, Union

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from tokenizers.processors import TemplateProcessing


class GearheadTokenizer:
    """
    Tokenizer for equipment diagnostic text with specialized tokens
    for error codes, equipment types, and diagnostic procedures.
    """

    # Special tokens
    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    UNK_TOKEN = "<unk>"

    # Diagnostic-specific special tokens
    ERROR_CODE_TOKEN = "<error>"
    SYMPTOM_TOKEN = "<symptom>"
    CAUSE_TOKEN = "<cause>"
    SOLUTION_TOKEN = "<solution>"
    EQUIPMENT_TOKEN = "<equipment>"

    def __init__(self, tokenizer_path: Optional[str] = None):
        """
        Initialize tokenizer.

        Args:
            tokenizer_path: Path to saved tokenizer JSON file
        """
        if tokenizer_path and Path(tokenizer_path).exists():
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
        else:
            # Initialize with BPE model
            self.tokenizer = Tokenizer(models.BPE(unk_token=self.UNK_TOKEN))
            self._setup_tokenizer()

    def _setup_tokenizer(self):
        """Configure tokenizer components."""
        # Normalization
        self.tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFD(),
            normalizers.Lowercase(),
            normalizers.StripAccents(),
        ])

        # Pre-tokenization
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        # Decoder
        self.tokenizer.decoder = decoders.ByteLevel()

        # Post-processing (add BOS/EOS)
        self.tokenizer.post_processor = TemplateProcessing(
            single=f"{self.BOS_TOKEN} $A {self.EOS_TOKEN}",
            special_tokens=[
                (self.BOS_TOKEN, 1),
                (self.EOS_TOKEN, 2),
            ],
        )

    def train(
        self,
        files: List[str],
        vocab_size: int = 32000,
        min_frequency: int = 2,
    ):
        """
        Train tokenizer on text files.

        Args:
            files: List of text file paths to train on
            vocab_size: Target vocabulary size
            min_frequency: Minimum frequency for token inclusion
        """
        # Special tokens for diagnostics
        special_tokens = [
            self.PAD_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
            self.UNK_TOKEN,
            self.ERROR_CODE_TOKEN,
            self.SYMPTOM_TOKEN,
            self.CAUSE_TOKEN,
            self.SOLUTION_TOKEN,
            self.EQUIPMENT_TOKEN,
        ]

        # Add common error code prefixes as special tokens
        error_prefixes = [
            f"<P{i:04d}>" for i in range(100)  # OBD-II codes
        ] + [
            f"<SPN{i}>" for i in range(100)  # J1939 SPNs
        ] + [
            f"<FMI{i}>" for i in range(32)  # Failure Mode Identifiers
        ]

        special_tokens.extend(error_prefixes[:200])  # Limit to 200 error code tokens

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            show_progress=True,
        )

        self.tokenizer.train(files, trainer)

    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
    ) -> Union[List[int], List[List[int]]]:
        """
        Encode text to token IDs.

        Args:
            text: Input text or list of texts
            add_special_tokens: Whether to add BOS/EOS tokens

        Returns:
            Token IDs or list of token IDs
        """
        if isinstance(text, str):
            encoding = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
            return encoding.ids
        else:
            encodings = self.tokenizer.encode_batch(text, add_special_tokens=add_special_tokens)
            return [enc.ids for enc in encodings]

    def decode(
        self,
        token_ids: Union[List[int], List[List[int]]],
        skip_special_tokens: bool = True,
    ) -> Union[str, List[str]]:
        """
        Decode token IDs to text.

        Args:
            token_ids: Token IDs or list of token ID sequences
            skip_special_tokens: Whether to remove special tokens

        Returns:
            Decoded text or list of texts
        """
        if isinstance(token_ids[0], int):
            return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        else:
            return self.tokenizer.decode_batch(token_ids, skip_special_tokens=skip_special_tokens)

    def save(self, path: str):
        """Save tokenizer to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(path)

    def __len__(self) -> int:
        """Get vocabulary size."""
        return self.tokenizer.get_vocab_size()

    def get_vocab(self) -> dict:
        """Get vocabulary as dict."""
        return self.tokenizer.get_vocab()

    @property
    def pad_token_id(self) -> int:
        """Get pad token ID."""
        return self.tokenizer.token_to_id(self.PAD_TOKEN)

    @property
    def bos_token_id(self) -> int:
        """Get BOS token ID."""
        return self.tokenizer.token_to_id(self.BOS_TOKEN)

    @property
    def eos_token_id(self) -> int:
        """Get EOS token ID."""
        return self.tokenizer.token_to_id(self.EOS_TOKEN)

    @property
    def unk_token_id(self) -> int:
        """Get UNK token ID."""
        return self.tokenizer.token_to_id(self.UNK_TOKEN)
