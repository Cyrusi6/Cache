"""
Token Aligner for handling different tokenizers between SLM and LLM models.

This module provides functionality to align tokens between two different tokenizers,
handling cases where the same text is tokenized differently.
"""

import hashlib
import math
from typing import Any, List, Tuple, Optional, Dict, Literal, Union
import torch
from transformers import PreTrainedTokenizerBase
from enum import Enum

from rosetta.utils.prompt_identity import render_chat_template_text


class AlignmentStrategy(Enum):
    """Strategies for handling 1-to-many token alignments"""

    FIRST = "first"  # Always take the first LLM token
    LONGEST = "longest"  # Take the LLM token with the longest string
    SPAN_OVERLAP = "span_overlap"  # Choose the LLM token with max char-span overlap
    SOFT_SPAN_OVERLAP = "soft_span_overlap"  # Keep top-k overlap weights
    SOFT_SPAN_OVERLAP_V2 = "soft_span_overlap_v2"  # Configurable top-k overlap weights
    LEARNED_SPAN_ALIGNMENT = "learned_span_alignment"  # Route-3 learned KV router


class TokenAligner:
    """
    Aligns tokens between SLM (Small Language Model) and LLM (Large Language Model) tokenizers.

    This class handles the case where the same text sequence is tokenized differently
    by different tokenizers, using the SLM tokenization as the base and finding
    corresponding LLM tokens for each SLM token.
    """

    def __init__(
        self,
        slm_tokenizer: PreTrainedTokenizerBase,
        llm_tokenizer: PreTrainedTokenizerBase,
        strategy: Union[AlignmentStrategy, str] = AlignmentStrategy.FIRST,
        soft_alignment_score_mode: str = "overlap",
        soft_alignment_boundary_bonus: float = 0.0,
        soft_alignment_boundary_tolerance: int = 1,
        soft_alignment_min_weight: float = 0.0,
        soft_alignment_confidence_mode: str = "none",
        soft_alignment_confidence_alpha: float = 0.5,
        soft_alignment_confidence_floor: float = 0.0,
        soft_alignment_fallback_confidence: float = 1.0,
        soft_alignment_confidence_control_mode: str = "native",
        soft_alignment_confidence_constant_value: Optional[float] = None,
        soft_alignment_confidence_shuffle_seed: int = 0,
        soft_alignment_reweight_mode: str = "none",
        soft_alignment_reweight_strength: float = 1.0,
        soft_alignment_reweight_power: float = 2.0,
        soft_alignment_candidate_window: int = 0,
        learned_alignment_prior_mode: str = "anchor",
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ):
        """
        Initialize the TokenAligner.

        Args:
            slm_tokenizer: The tokenizer for the Small Language Model (base)
            llm_tokenizer: The tokenizer for the Large Language Model
            strategy: Strategy for handling 1-to-many token mappings
                     Either AlignmentStrategy enum or string
                     ('first', 'longest', 'span_overlap', 'soft_span_overlap',
                     or 'soft_span_overlap_v2')
            soft_alignment_score_mode: Candidate weighting mode for soft span
                     alignment. Supported values are 'overlap', 'uniform',
                     'overlap_power2', and 'boundary_power2'.
            soft_alignment_boundary_bonus: Bonus multiplier per matched boundary
                     for 'boundary_power2'.
            soft_alignment_boundary_tolerance: Character tolerance for boundary
                     matches in 'boundary_power2'.
            soft_alignment_min_weight: Optional pruning threshold after top-k
                     normalization.
            soft_alignment_confidence_mode: Optional source confidence policy.
                     Supported values are 'none' and 'entropy'.
            soft_alignment_confidence_alpha: Entropy penalty strength for
                     'entropy' confidence.
            soft_alignment_confidence_floor: Lower bound for non-fallback
                     'entropy' confidence.
            soft_alignment_fallback_confidence: Confidence assigned to fallback
                     rows in 'entropy' mode.
            soft_alignment_confidence_control_mode: Identifiability control for
                     confidence/entropy signals. Supported values are 'native',
                     'constant', and 'shuffle'. The default preserves the native
                     confidence path.
            soft_alignment_confidence_constant_value: Required per-token source
                     confidence for 'constant' control. Projector entropy is set
                     to zero in this mode.
            soft_alignment_confidence_shuffle_seed: Base seed for deterministic
                     within-sequence confidence/entropy shuffling.
            soft_alignment_reweight_mode: Optional post-normalization top-k
                     weight calibration. Supported values are 'none' and
                     'adaptive_overlap'.
            soft_alignment_reweight_strength: Maximum interpolation strength for
                     adaptive reweighting.
            soft_alignment_reweight_power: Overlap exponent used by
                     'adaptive_overlap'.
            soft_alignment_candidate_window: Extra neighbor tokens to include around
                     span candidates. Used by 'learned_span_alignment'.
            learned_alignment_prior_mode: Initial Route-3 source weights.
                     Supported values are 'anchor' and 'soft_span'. The default
                     preserves the original one-hot anchor behavior.
            chat_template_kwargs: Explicit variables passed identically to both
                     tokenizer chat templates, such as a frozen date_string.
            verbose: Whether to print debug information during alignment
        """
        self.slm_tokenizer = slm_tokenizer
        self.llm_tokenizer = llm_tokenizer

        if self.slm_tokenizer.pad_token is None:
            self.slm_tokenizer.pad_token = self.slm_tokenizer.eos_token
            self.slm_tokenizer.pad_token_id = self.slm_tokenizer.eos_token_id
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
            self.llm_tokenizer.pad_token_id = self.llm_tokenizer.eos_token_id

        # Handle string strategy input
        if isinstance(strategy, str):
            strategy = AlignmentStrategy(strategy.lower())
        self.strategy = strategy
        valid_score_modes = {
            "overlap",
            "uniform",
            "overlap_power2",
            "boundary_power2",
        }
        if soft_alignment_score_mode not in valid_score_modes:
            raise ValueError(
                f"Unsupported soft_alignment_score_mode: {soft_alignment_score_mode}. "
                f"Expected one of {sorted(valid_score_modes)}"
            )
        valid_confidence_modes = {"none", "entropy"}
        if soft_alignment_confidence_mode not in valid_confidence_modes:
            raise ValueError(
                f"Unsupported soft_alignment_confidence_mode: "
                f"{soft_alignment_confidence_mode}. "
                f"Expected one of {sorted(valid_confidence_modes)}"
            )
        valid_confidence_control_modes = {"native", "constant", "shuffle"}
        if soft_alignment_confidence_control_mode not in (
            valid_confidence_control_modes
        ):
            raise ValueError(
                "Unsupported soft_alignment_confidence_control_mode: "
                f"{soft_alignment_confidence_control_mode}. Expected one of "
                f"{sorted(valid_confidence_control_modes)}"
            )
        if (
            soft_alignment_confidence_control_mode == "constant"
            and soft_alignment_confidence_constant_value is None
        ):
            raise ValueError(
                "soft_alignment_confidence_constant_value is required when "
                "soft_alignment_confidence_control_mode='constant'"
            )
        valid_reweight_modes = {"none", "adaptive_overlap"}
        if soft_alignment_reweight_mode not in valid_reweight_modes:
            raise ValueError(
                f"Unsupported soft_alignment_reweight_mode: "
                f"{soft_alignment_reweight_mode}. "
                f"Expected one of {sorted(valid_reweight_modes)}"
            )
        valid_learned_prior_modes = {"anchor", "soft_span"}
        if learned_alignment_prior_mode not in valid_learned_prior_modes:
            raise ValueError(
                f"Unsupported learned_alignment_prior_mode: "
                f"{learned_alignment_prior_mode}. "
                f"Expected one of {sorted(valid_learned_prior_modes)}"
            )
        self.soft_alignment_score_mode = soft_alignment_score_mode
        self.soft_alignment_boundary_bonus = float(soft_alignment_boundary_bonus)
        self.soft_alignment_boundary_tolerance = max(
            0, int(soft_alignment_boundary_tolerance)
        )
        self.soft_alignment_min_weight = max(0.0, float(soft_alignment_min_weight))
        self.soft_alignment_confidence_mode = soft_alignment_confidence_mode
        self.soft_alignment_confidence_alpha = float(soft_alignment_confidence_alpha)
        self.soft_alignment_confidence_floor = min(
            1.0, max(0.0, float(soft_alignment_confidence_floor))
        )
        self.soft_alignment_fallback_confidence = min(
            1.0, max(0.0, float(soft_alignment_fallback_confidence))
        )
        self.soft_alignment_confidence_control_mode = (
            soft_alignment_confidence_control_mode
        )
        self.soft_alignment_confidence_constant_value = (
            None
            if soft_alignment_confidence_constant_value is None
            else float(soft_alignment_confidence_constant_value)
        )
        if self.soft_alignment_confidence_constant_value is not None and not (
            0.0 <= self.soft_alignment_confidence_constant_value <= 1.0
        ):
            raise ValueError(
                "soft_alignment_confidence_constant_value must be in [0, 1], "
                f"got {soft_alignment_confidence_constant_value}"
            )
        self.soft_alignment_confidence_shuffle_seed = int(
            soft_alignment_confidence_shuffle_seed
        )
        self.soft_alignment_reweight_mode = soft_alignment_reweight_mode
        self.soft_alignment_reweight_strength = max(
            0.0, float(soft_alignment_reweight_strength)
        )
        self.soft_alignment_reweight_power = max(
            1e-6, float(soft_alignment_reweight_power)
        )
        self.soft_alignment_candidate_window = max(
            0, int(soft_alignment_candidate_window)
        )
        self.learned_alignment_prior_mode = learned_alignment_prior_mode
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.verbose = verbose

        # Cache for token mappings to improve performance
        self._alignment_cache: Dict[Tuple[int, ...], List[int]] = {}

    def align_tokens(
        self,
        slm_token_ids: Union[List[int], torch.Tensor],
        return_mapping: bool = False,
    ) -> Union[List[int], Tuple[List[int], List[Tuple[int, List[int]]]]]:
        """
        Align SLM tokens to LLM tokens.

        Args:
            slm_token_ids: Token IDs from the SLM tokenizer
            return_mapping: If True, also return the detailed mapping

        Returns:
            If return_mapping is False: List of aligned LLM token IDs
            If return_mapping is True: Tuple of (aligned_llm_token_ids, mapping_details)
                where mapping_details is a list of (slm_token_id, [candidate_llm_token_ids])
        """
        # Convert to list if tensor
        if isinstance(slm_token_ids, torch.Tensor):
            slm_token_ids = slm_token_ids.tolist()

        # Check cache
        cache_key = tuple(slm_token_ids)
        if cache_key in self._alignment_cache and not return_mapping:
            return self._alignment_cache[cache_key]

        aligned_llm_tokens = []
        mapping_details = []

        for slm_token_id in slm_token_ids:
            # Decode SLM token to string (without special token processing)
            slm_token_str = self.slm_tokenizer.decode(
                [slm_token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

            # Handle special tokens
            if slm_token_id in self.slm_tokenizer.all_special_ids:
                # Try to find corresponding special token in LLM tokenizer
                llm_token_id = self._map_special_token(slm_token_id, slm_token_str)
                aligned_llm_tokens.append(llm_token_id)
                mapping_details.append((slm_token_id, [llm_token_id]))
                continue

            # Tokenize the string with LLM tokenizer
            llm_token_ids = self.llm_tokenizer.encode(
                slm_token_str, add_special_tokens=False, return_tensors=None
            )

            if len(llm_token_ids) == 0:
                # Handle empty tokenization (shouldn't normally happen)
                if self.verbose:
                    print(
                        f"Warning: SLM token {slm_token_id} ('{slm_token_str}') "
                        f"resulted in empty LLM tokenization"
                    )
                # Use unknown token as fallback
                llm_token_id = self.llm_tokenizer.unk_token_id or 0
                aligned_llm_tokens.append(llm_token_id)
                mapping_details.append((slm_token_id, [llm_token_id]))

            elif len(llm_token_ids) == 1:
                # Perfect 1-to-1 mapping
                aligned_llm_tokens.append(llm_token_ids[0])
                mapping_details.append((slm_token_id, llm_token_ids))

            else:
                # 1-to-many mapping, apply strategy
                selected_token = self._apply_strategy(llm_token_ids, slm_token_str)
                aligned_llm_tokens.append(selected_token)
                mapping_details.append((slm_token_id, llm_token_ids))

                if self.verbose:
                    selected_str = self.llm_tokenizer.decode(
                        [selected_token],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    print(
                        f"SLM token {slm_token_id} ('{slm_token_str}') -> "
                        f"LLM tokens {llm_token_ids}, selected {selected_token} ('{selected_str}')"
                    )

        # Cache the result
        self._alignment_cache[cache_key] = aligned_llm_tokens

        if return_mapping:
            return aligned_llm_tokens, mapping_details
        return aligned_llm_tokens

    def _map_special_token(self, slm_token_id: int, slm_token_str: str) -> int:
        """
        Map special tokens between tokenizers.

        Args:
            slm_token_id: The SLM special token ID
            slm_token_str: The string representation of the special token

        Returns:
            The corresponding LLM token ID
        """
        # Common special token mappings
        special_token_map = {
            self.slm_tokenizer.pad_token_id: self.llm_tokenizer.pad_token_id,
            self.slm_tokenizer.eos_token_id: self.llm_tokenizer.eos_token_id,
            self.slm_tokenizer.bos_token_id: self.llm_tokenizer.bos_token_id,
            self.slm_tokenizer.unk_token_id: self.llm_tokenizer.unk_token_id,
        }

        # Direct mapping if available
        if (
            slm_token_id in special_token_map
            and special_token_map[slm_token_id] is not None
        ):
            return special_token_map[slm_token_id]

        # Try to find by string representation
        try:
            llm_token_id = self.llm_tokenizer.convert_tokens_to_ids(slm_token_str)
            if llm_token_id != self.llm_tokenizer.unk_token_id:
                return llm_token_id
        except:
            pass

        # Fallback to unknown token
        return self.llm_tokenizer.unk_token_id or 0

    def _apply_strategy(self, llm_token_ids: List[int], original_str: str) -> int:
        """
        Apply the selected strategy to choose one LLM token from multiple candidates.

        Args:
            llm_token_ids: List of candidate LLM token IDs
            original_str: The original string from SLM token

        Returns:
            The selected LLM token ID
        """
        if self.strategy == AlignmentStrategy.FIRST:
            return llm_token_ids[0]

        elif self.strategy == AlignmentStrategy.LONGEST:
            # Find the token with the longest string representation
            longest_token = llm_token_ids[0]
            longest_length = 0

            for token_id in llm_token_ids:
                token_str = self.llm_tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                if len(token_str) > longest_length:
                    longest_length = len(token_str)
                    longest_token = token_id

            return longest_token

        elif self.strategy in {
            AlignmentStrategy.SPAN_OVERLAP,
            AlignmentStrategy.SOFT_SPAN_OVERLAP,
            AlignmentStrategy.SOFT_SPAN_OVERLAP_V2,
            AlignmentStrategy.LEARNED_SPAN_ALIGNMENT,
        }:
            # Token-local alignment has no full-context offsets, so preserve
            # LONGEST behavior outside chat message section alignment.
            longest_token = llm_token_ids[0]
            longest_length = 0

            for token_id in llm_token_ids:
                token_str = self.llm_tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                if len(token_str) > longest_length:
                    longest_length = len(token_str)
                    longest_token = token_id

            return longest_token

        else:
            # Default to first token if unknown strategy
            return llm_token_ids[0]

    def align_sequence(
        self, text: str, return_details: bool = False
    ) -> Union[Tuple[List[int], List[int]], Dict[str, any]]:
        """
        Tokenize text with both tokenizers and return aligned sequences.

        Args:
            text: The input text to tokenize and align
            return_details: If True, return detailed alignment information

        Returns:
            If return_details is False: Tuple of (slm_token_ids, aligned_llm_token_ids)
            If return_details is True: Dictionary with detailed alignment information
        """
        # Tokenize with SLM
        slm_tokens = self.slm_tokenizer.encode(
            text, add_special_tokens=True, return_tensors=None
        )

        # Get aligned LLM tokens
        if return_details:
            aligned_llm_tokens, mapping = self.align_tokens(
                slm_tokens, return_mapping=True
            )

            # Decode tokens for inspection
            slm_decoded = [
                self.slm_tokenizer.decode(
                    [tid], skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                for tid in slm_tokens
            ]
            llm_decoded = [
                self.llm_tokenizer.decode(
                    [tid], skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                for tid in aligned_llm_tokens
            ]

            # Original LLM tokenization for comparison
            original_llm_tokens = self.llm_tokenizer.encode(
                text, add_special_tokens=True, return_tensors=None
            )

            # One-to-one mapping statistics
            num_tokens = len(slm_tokens)
            one_to_one_count = sum(
                1 for _slm_id, candidates in mapping if len(candidates) == 1
            )
            one_to_one_rate = (one_to_one_count / num_tokens) if num_tokens > 0 else 0.0

            return {
                "text": text,
                "slm_token_ids": slm_tokens,
                "slm_decoded": slm_decoded,
                "aligned_llm_token_ids": aligned_llm_tokens,
                "aligned_llm_decoded": llm_decoded,
                "original_llm_token_ids": original_llm_tokens,
                "mapping": mapping,
                "strategy": self.strategy.value,
                "num_tokens": num_tokens,
                "one_to_one_count": one_to_one_count,
                "one_to_one_rate": one_to_one_rate,
            }
        else:
            aligned_llm_tokens = self.align_tokens(slm_tokens)
            return slm_tokens, aligned_llm_tokens

    def visualize_alignment(self, text: str):
        """
        Print a visual representation of the token alignment.

        Args:
            text: The text to analyze
        """
        details = self.align_sequence(text, return_details=True)

        print("=" * 80)
        print(f"Text: {text}")
        print(f"Strategy: {details['strategy']}")
        print("=" * 80)
        print(
            f"SLM tokens ({len(details['slm_token_ids'])}): {details['slm_token_ids']}"
        )
        print(
            f"Aligned LLM tokens ({len(details['aligned_llm_token_ids'])}): {details['aligned_llm_token_ids']}"
        )
        print(
            f"Original LLM tokens ({len(details['original_llm_token_ids'])}): {details['original_llm_token_ids']}"
        )
        print("-" * 80)
        print("Token-by-token alignment:")

        for i, (slm_id, llm_id) in enumerate(
            zip(details["slm_token_ids"], details["aligned_llm_token_ids"])
        ):
            slm_str = details["slm_decoded"][i]
            llm_str = details["aligned_llm_decoded"][i]
            mapping_info = details["mapping"][i]

            if len(mapping_info[1]) > 1:
                candidates_str = ", ".join(
                    [
                        f"{tid}:'{self.llm_tokenizer.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)}'"
                        for tid in mapping_info[1]
                    ]
                )
                print(
                    f"  [{i:3d}] SLM {slm_id:6d} ('{slm_str}') -> "
                    f"LLM {llm_id:6d} ('{llm_str}') "
                    f"[candidates: {candidates_str}]"
                )
            else:
                print(
                    f"  [{i:3d}] SLM {slm_id:6d} ('{slm_str}') -> "
                    f"LLM {llm_id:6d} ('{llm_str}')"
                )
        print("=" * 80)

    def clear_cache(self):
        """Clear the alignment cache."""
        self._alignment_cache.clear()

    # ========================
    # Chat messages alignment
    # ========================
    def _apply_chat_template_to_ids(
        self,
        tokenizer: PreTrainedTokenizerBase,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool,
        enable_thinking: bool,
        remove_last_surfix: bool,
    ) -> Tuple[str, List[int], Optional[List[Tuple[int, int]]]]:
        """
        Apply chat template (no tokenization) then tokenize to ids with optional offsets.
        If remove_last_surfix is True, remove the last suffix from the LLM text
        Returns (templated_text, input_ids, offsets) where offsets may be None.
        """
        if remove_last_surfix:
            assert (
                messages[-1]["role"] == "assistant"
            ), "Last message must be an assistant message"
            _canonical, templated_text = render_chat_template_text(
                tokenizer,
                messages,
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
                remove_last_suffix=True,
                template_kwargs=self.chat_template_kwargs,
            )
        else:
            _canonical, templated_text = render_chat_template_text(
                tokenizer,
                messages,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
                remove_last_suffix=False,
                template_kwargs=self.chat_template_kwargs,
            )
        encoded = tokenizer(
            templated_text, add_special_tokens=False, return_offsets_mapping=True
        )
        input_ids: List[int] = encoded["input_ids"]
        offsets = encoded.get("offset_mapping")
        return templated_text, input_ids, offsets

    @staticmethod
    def _first_non_empty_content(messages: List[Dict[str, str]]) -> Optional[str]:
        for m in messages:
            content = m.get("content")
            if isinstance(content, str) and len(content.strip()) > 0:
                return content
        return None

    def _find_boundary_token_index(
        self,
        tokenizer: PreTrainedTokenizerBase,
        templated_text: str,
        offsets: Optional[List[Tuple[int, int]]],
        content_text: Optional[str],
    ) -> int:
        """
        Find token index where the first non-empty message content starts.
        Falls back to 0 if not found.
        """
        if not content_text:
            return 0
        char_idx = templated_text.find(content_text)
        if char_idx < 0:
            # Try a shorter probe to improve chances
            probe = content_text[: min(32, len(content_text))]
            if len(probe) > 0:
                char_idx = templated_text.find(probe)
        if char_idx < 0:
            return 0

        if offsets:
            for idx, (start, _end) in enumerate(offsets):
                if start >= char_idx:
                    return idx
            return len(offsets)

        # Fallback without offsets: tokenize prefix and count tokens
        prefix = templated_text[:char_idx]
        prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        return len(prefix_ids)

    @staticmethod
    def _compute_content_spans(
        templated_text: str, messages: List[Dict[str, str]]
    ) -> List[Tuple[int, int]]:
        """
        Compute character spans in templated_text that correspond to message contents.
        Searches sequentially to reduce ambiguity when contents repeat.
        Enhanced matching: ensures the found content is followed by '<' (special token start)
        to avoid matching content inside special tokens like <begin_of_text>.
        """
        spans: List[Tuple[int, int]] = []
        search_from = 0
        for m in messages:
            content = m.get("content")
            if not isinstance(content, str) or len(content) == 0:
                continue

            # Find all possible matches starting from search_from
            idx = search_from
            found_valid_match = False

            while idx < len(templated_text):
                idx = templated_text.find(content, idx)
                if idx < 0:
                    break

                # Check if this match is valid (followed by '<' indicating a special token)
                end_pos = idx + len(content)
                if end_pos < len(templated_text) and templated_text[end_pos] == "<":
                    # Valid match: content is followed by a special token
                    spans.append((idx, end_pos))
                    search_from = end_pos
                    found_valid_match = True
                    break
                else:
                    # Check if this is the end of the text (also valid for last message)
                    if end_pos == len(templated_text):
                        spans.append((idx, end_pos))
                        search_from = end_pos
                        found_valid_match = True
                        break

                # Invalid match, try next occurrence
                idx += 1

            # Fallback: if no valid match found with '<' requirement, use the old method
            # but only as a last resort and with additional validation
            if not found_valid_match:
                idx = templated_text.find(content, search_from)
                if idx < 0:
                    # Try searching from start as last resort
                    idx = templated_text.find(content)

                if idx >= 0:
                    end_pos = idx + len(content)
                    # Additional check: avoid matching inside obvious special tokens
                    # Check if we're inside a special token (preceded by '<' and not followed by '>')
                    start_context = templated_text[max(0, idx - 10) : idx]
                    end_context = templated_text[
                        end_pos : min(len(templated_text), end_pos + 10)
                    ]

                    # Skip if we're clearly inside a special token
                    if (
                        "<" in start_context
                        and ">" not in start_context
                        and "begin_of_text"
                        in templated_text[max(0, idx - 20) : idx + 20]
                    ):
                        # This looks like we're matching inside <begin_of_text> or similar
                        continue

                    spans.append((idx, end_pos))
                    search_from = end_pos

        return spans

    @staticmethod
    def _build_token_mask_from_spans(
        offsets: Optional[List[Tuple[int, int]]],
        num_tokens: int,
        spans: List[Tuple[int, int]],
    ) -> List[bool]:
        """
        Build a boolean mask for tokens whose offset range overlaps any span.
        If offsets are missing, default to all False.
        """
        if not offsets or len(offsets) != num_tokens:
            return [False] * num_tokens
        mask: List[bool] = []
        for start, end in offsets:
            if end <= start:
                mask.append(False)
                continue
            is_msg = False
            for s, e in spans:
                # overlap check
                if start < e and end > s:
                    is_msg = True
                    break
            mask.append(is_msg)
        return mask

    @staticmethod
    def _spans_to_token_ranges(
        offsets: List[Tuple[int, int]], spans: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Convert character spans to token index ranges using offsets.
        start token = first token with end > span_start
        end token = first token with start >= span_end
        """
        ranges: List[Tuple[int, int]] = []
        n = len(offsets)
        for s, e in spans:
            # find start index
            start_idx = 0
            while start_idx < n and offsets[start_idx][1] <= s:
                start_idx += 1
            # find end index
            end_idx = start_idx
            while end_idx < n and offsets[end_idx][0] < e:
                end_idx += 1
            ranges.append((start_idx, end_idx))
        return ranges

    @staticmethod
    def _overlap_len(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
        return max(0, min(a_end, b_end) - max(a_start, b_start))

    @staticmethod
    def _boundary_hit_count(
        target_start: int,
        target_end: int,
        token_start: int,
        token_end: int,
        tolerance: int,
    ) -> int:
        """Count aligned left/right boundaries within a character tolerance."""
        hits = 0
        if abs(token_start - target_start) <= tolerance:
            hits += 1
        if abs(token_end - target_end) <= tolerance:
            hits += 1
        return hits

    def _soft_candidate_weight_score(
        self,
        overlap: int,
        target_start: int,
        target_end: int,
        token_start: int,
        token_end: int,
    ) -> float:
        """Convert a positive overlap into a configurable soft-alignment score."""
        if self.soft_alignment_score_mode == "uniform":
            return 1.0

        if self.soft_alignment_score_mode in {"overlap_power2", "boundary_power2"}:
            score = float(overlap * overlap)
        else:
            score = float(overlap)

        if self.soft_alignment_score_mode == "boundary_power2":
            hits = self._boundary_hit_count(
                target_start,
                target_end,
                token_start,
                token_end,
                self.soft_alignment_boundary_tolerance,
            )
            if hits > 0:
                score *= 1.0 + self.soft_alignment_boundary_bonus * hits

        return score

    def _normalize_soft_scores(self, scores: List[float]) -> List[float]:
        """Normalize selected candidate scores and optionally prune tiny weights."""
        if not scores:
            return []
        total = float(sum(scores))
        if total <= 0:
            weights = [0.0 for _ in scores]
            weights[0] = 1.0
            return weights

        weights = [float(score) / total for score in scores]
        if self.soft_alignment_min_weight <= 0:
            return weights

        pruned = [
            weight if weight >= self.soft_alignment_min_weight else 0.0
            for weight in weights
        ]
        pruned_total = float(sum(pruned))
        if pruned_total <= 0:
            return [1.0] + [0.0 for _ in weights[1:]]
        return [weight / pruned_total for weight in pruned]

    @staticmethod
    def _normalized_weight_entropy(weights: List[float]) -> float:
        positive_weights = [float(weight) for weight in weights if weight > 0]
        if len(positive_weights) <= 1:
            return 0.0

        total = float(sum(positive_weights))
        if total <= 0:
            return 0.0
        probs = [weight / total for weight in positive_weights]
        entropy = -sum(prob * math.log(max(prob, 1e-12)) for prob in probs)
        return entropy / math.log(len(probs))

    def _reweight_soft_alignment_weights(
        self,
        base_weights: List[float],
        overlaps: List[int],
    ) -> List[float]:
        """Optionally calibrate normalized top-k weights using span evidence."""
        if self.soft_alignment_reweight_mode == "none" or len(base_weights) <= 1:
            return base_weights

        if self.soft_alignment_reweight_mode == "adaptive_overlap":
            overlap_scores = [
                float(max(overlap, 0)) ** self.soft_alignment_reweight_power
                for overlap in overlaps
            ]
            overlap_total = float(sum(overlap_scores))
            if overlap_total <= 0:
                return base_weights

            overlap_weights = [score / overlap_total for score in overlap_scores]
            evidence_strength = 1.0 - self._normalized_weight_entropy(overlap_weights)
            mix = min(
                1.0,
                max(0.0, self.soft_alignment_reweight_strength * evidence_strength),
            )
            if mix <= 0:
                return base_weights

            mixed = [
                (1.0 - mix) * base_weight + mix * overlap_weight
                for base_weight, overlap_weight in zip(base_weights, overlap_weights)
            ]
            return self._normalize_soft_scores(mixed)

        return base_weights

    def _soft_alignment_confidence(
        self,
        weights: List[float],
        used_fallback: bool,
    ) -> float:
        """Compute optional confidence for how strongly to inject source KV."""
        if self.soft_alignment_confidence_mode == "none":
            return 1.0
        if used_fallback:
            return self.soft_alignment_fallback_confidence

        positive_weights = [float(weight) for weight in weights if weight > 0]
        if len(positive_weights) <= 1:
            return 1.0

        entropy = -sum(
            weight * math.log(max(weight, 1e-12)) for weight in positive_weights
        )
        entropy_norm = entropy / math.log(len(positive_weights))
        confidence = 1.0 - self.soft_alignment_confidence_alpha * entropy_norm
        return min(1.0, max(self.soft_alignment_confidence_floor, confidence))

    @staticmethod
    def _stable_confidence_shuffle_seed(
        base_seed: int,
        token_ids: List[int],
    ) -> int:
        """Derive a process-independent shuffle seed from config and sequence."""
        digest = hashlib.blake2b(digest_size=8)
        digest.update(int(base_seed).to_bytes(8, byteorder="little", signed=True))
        for token_id in token_ids:
            digest.update(int(token_id).to_bytes(8, byteorder="little", signed=True))
        return int.from_bytes(digest.digest(), byteorder="little") & ((1 << 63) - 1)

    def _apply_confidence_control(
        self,
        source_confidence: List[float],
        source_entropy: List[float],
        message_mask: List[bool],
        token_ids: List[int],
        shuffle_active_mask: Optional[List[bool]] = None,
    ) -> Tuple[List[float], List[float], List[bool]]:
        """Apply an identifiability control without changing source weights."""
        override = [False] * len(source_confidence)
        mode = self.soft_alignment_confidence_control_mode
        if mode == "native":
            return source_confidence, source_entropy, override

        if mode == "constant":
            controlled_indices = [
                idx for idx, is_message in enumerate(message_mask) if is_message
            ]
            for idx in controlled_indices:
                override[idx] = True
            constant_value = self.soft_alignment_confidence_constant_value
            if constant_value is None:
                raise RuntimeError("constant confidence control requires a value")
            for idx in controlled_indices:
                source_confidence[idx] = constant_value
                source_entropy[idx] = 0.0
            return source_confidence, source_entropy, override

        active_mask = shuffle_active_mask if shuffle_active_mask is not None else message_mask
        if len(active_mask) != len(message_mask):
            raise ValueError("shuffle_active_mask must match message_mask length")
        controlled_indices = [
            idx
            for idx, (is_message, is_active) in enumerate(
                zip(message_mask, active_mask)
            )
            if is_message and is_active
        ]
        for idx in controlled_indices:
            override[idx] = True

        if len(controlled_indices) <= 1:
            return source_confidence, source_entropy, override

        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            self._stable_confidence_shuffle_seed(
                self.soft_alignment_confidence_shuffle_seed,
                [token_ids[idx] for idx in controlled_indices],
            )
        )
        order = torch.randperm(len(controlled_indices), generator=generator).tolist()
        if order == list(range(len(controlled_indices))):
            order = order[1:] + order[:1]
        shuffled_pairs = [
            (
                source_confidence[controlled_indices[src_idx]],
                source_entropy[controlled_indices[src_idx]],
            )
            for src_idx in order
        ]
        for dst_idx, (confidence, entropy) in zip(controlled_indices, shuffled_pairs):
            source_confidence[dst_idx] = confidence
            source_entropy[dst_idx] = entropy
        return source_confidence, source_entropy, override

    def _fallback_llm_index_for_span(
        self,
        target_start: int,
        llm_offsets: List[Tuple[int, int]],
        llm_range: Tuple[int, int],
    ) -> int:
        """Pick the nearest LLM token index when no positive span overlap is found."""
        start_idx, end_idx = llm_range
        if start_idx >= end_idx:
            return -1

        best_idx = start_idx
        best_distance: Optional[int] = None
        for idx in range(start_idx, end_idx):
            start, end = llm_offsets[idx]
            if end <= start:
                continue
            if start <= target_start < end:
                return idx
            distance = min(abs(target_start - start), abs(target_start - end))
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_idx = idx
        return best_idx

    def _fallback_llm_token_for_span(
        self,
        target_start: int,
        llm_ids: List[int],
        llm_offsets: List[Tuple[int, int]],
        llm_range: Tuple[int, int],
    ) -> int:
        """Pick the nearest LLM token when no positive span overlap is found."""
        fallback_idx = self._fallback_llm_index_for_span(
            target_start, llm_offsets, llm_range
        )
        if fallback_idx < 0:
            return self.llm_tokenizer.unk_token_id or 0
        return llm_ids[fallback_idx]

    @staticmethod
    def _append_unique_limited(values: List[int], value: int, limit: int) -> None:
        if value < 0 or len(values) >= limit or value in values:
            return
        values.append(value)

    def _learned_alignment_row_from_candidates(
        self,
        candidates: List[Tuple[float, int, int, int, bool]],
        fallback_idx: int,
        llm_range: Tuple[int, int],
        top_k: int,
    ) -> Tuple[List[int], List[float], bool, bool]:
        """
        Build Route-3 candidate rows.

        Slot 0 is the native C2C/span anchor. Extra overlap and neighbor tokens
        are only enumerated; the projector learns their KV mixture.
        """
        selected_indices: List[int] = []
        used_fallback = False
        top1_boundary_hit = False

        if candidates:
            candidates.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
            anchor_idx = candidates[0][3]
            top1_boundary_hit = bool(candidates[0][4])
            self._append_unique_limited(selected_indices, anchor_idx, top_k)
            for _weight_score, _overlap, _token_len, llm_idx, _hit in candidates:
                self._append_unique_limited(selected_indices, llm_idx, top_k)
        elif fallback_idx >= 0:
            used_fallback = True
            self._append_unique_limited(selected_indices, fallback_idx, top_k)
        else:
            used_fallback = True

        start_idx, end_idx = llm_range
        window = self.soft_alignment_candidate_window
        if window > 0 and selected_indices:
            anchors = list(selected_indices)
            for idx in anchors:
                for neighbor_idx in range(
                    max(start_idx, idx - window), min(end_idx, idx + window + 1)
                ):
                    self._append_unique_limited(selected_indices, neighbor_idx, top_k)

        row_indices = [-1] * top_k
        row_weights = [0.0] * top_k
        for out_idx, llm_idx in enumerate(selected_indices[:top_k]):
            row_indices[out_idx] = llm_idx
        if (
            self.learned_alignment_prior_mode == "soft_span"
            and candidates
            and selected_indices
        ):
            score_by_idx: Dict[int, float] = {}
            overlap_by_idx: Dict[int, int] = {}
            for weight_score, overlap, _token_len, llm_idx, _hit in candidates:
                if llm_idx not in score_by_idx:
                    score_by_idx[llm_idx] = float(weight_score)
                    overlap_by_idx[llm_idx] = int(overlap)
            selected_scores = [
                score_by_idx.get(llm_idx, 0.0) for llm_idx in selected_indices[:top_k]
            ]
            selected_overlaps = [
                overlap_by_idx.get(llm_idx, 0) for llm_idx in selected_indices[:top_k]
            ]
            weights = self._normalize_soft_scores(selected_scores)
            weights = self._reweight_soft_alignment_weights(weights, selected_overlaps)
            for out_idx, weight in enumerate(weights):
                row_weights[out_idx] = weight
        elif selected_indices:
            row_weights[0] = 1.0

        return row_indices, row_weights, used_fallback, top1_boundary_hit

    def _align_message_by_span_overlap(
        self,
        slm_ids: List[int],
        slm_offsets: List[Tuple[int, int]],
        slm_range: Tuple[int, int],
        slm_span: Tuple[int, int],
        llm_ids: List[int],
        llm_offsets: List[Tuple[int, int]],
        llm_range: Tuple[int, int],
        llm_span: Tuple[int, int],
    ) -> List[int]:
        """
        Align message tokens by projecting each SLM token's character span inside
        the message content onto the LLM-rendered message content and selecting
        the LLM token with the largest overlap.
        """
        aligned: List[int] = []
        slm_start_idx, slm_end_idx = slm_range
        llm_start_idx, llm_end_idx = llm_range
        slm_span_start, slm_span_end = slm_span
        llm_span_start, llm_span_end = llm_span

        if llm_start_idx >= llm_end_idx:
            fallback_id = self.llm_tokenizer.unk_token_id or 0
            return [fallback_id] * max(0, slm_end_idx - slm_start_idx)

        for slm_idx in range(slm_start_idx, slm_end_idx):
            slm_token_id = slm_ids[slm_idx]
            slm_token_start, slm_token_end = slm_offsets[slm_idx]

            if slm_token_id in self.slm_tokenizer.all_special_ids:
                aligned.append(
                    self._map_special_token(
                        slm_token_id,
                        self.slm_tokenizer.decode(
                            [slm_token_id],
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        ),
                    )
                )
                continue

            clipped_start = max(slm_token_start, slm_span_start)
            clipped_end = min(slm_token_end, slm_span_end)
            if clipped_end <= clipped_start:
                aligned.append(self.align_tokens([slm_token_id])[0])
                continue

            rel_start = clipped_start - slm_span_start
            rel_end = clipped_end - slm_span_start
            target_start = min(llm_span_end, llm_span_start + rel_start)
            target_end = min(llm_span_end, llm_span_start + rel_end)
            if target_end <= target_start:
                aligned.append(
                    self._fallback_llm_token_for_span(
                        target_start, llm_ids, llm_offsets, llm_range
                    )
                )
                continue

            best_idx: Optional[int] = None
            best_score = 0
            best_token_len = -1
            for llm_idx in range(llm_start_idx, llm_end_idx):
                llm_token_start, llm_token_end = llm_offsets[llm_idx]
                if llm_token_end <= llm_token_start:
                    continue
                score = self._overlap_len(
                    target_start, target_end, llm_token_start, llm_token_end
                )
                token_len = llm_token_end - llm_token_start
                if score > best_score or (
                    score == best_score and score > 0 and token_len > best_token_len
                ):
                    best_idx = llm_idx
                    best_score = score
                    best_token_len = token_len

            if best_idx is None:
                aligned.append(
                    self._fallback_llm_token_for_span(
                        target_start, llm_ids, llm_offsets, llm_range
                    )
                )
            else:
                aligned.append(llm_ids[best_idx])

        return aligned

    def _soft_align_message_by_span_overlap(
        self,
        slm_ids: List[int],
        slm_offsets: List[Tuple[int, int]],
        slm_range: Tuple[int, int],
        slm_span: Tuple[int, int],
        llm_offsets: List[Tuple[int, int]],
        llm_range: Tuple[int, int],
        llm_span: Tuple[int, int],
        top_k: int,
    ) -> Dict[str, List[Any]]:
        """
        Build top-k soft source-token weights for each SLM token in a message.
        Indices are global positions in the original LLM token sequence.
        """
        top_k = max(1, int(top_k))
        rows_indices: List[List[int]] = []
        rows_weights: List[List[float]] = []
        rows_confidence: List[float] = []
        rows_entropy: List[float] = []
        fallback_mask: List[bool] = []
        positive_counts: List[int] = []
        top1_boundary_hit_mask: List[bool] = []

        slm_start_idx, slm_end_idx = slm_range
        llm_start_idx, llm_end_idx = llm_range
        slm_span_start, slm_span_end = slm_span
        llm_span_start, llm_span_end = llm_span
        learned_alignment = self.strategy == AlignmentStrategy.LEARNED_SPAN_ALIGNMENT

        for slm_idx in range(slm_start_idx, slm_end_idx):
            row_indices = [-1] * top_k
            row_weights = [0.0] * top_k
            used_fallback = False
            positive_count = 0

            if llm_start_idx >= llm_end_idx:
                rows_indices.append(row_indices)
                rows_weights.append(row_weights)
                rows_entropy.append(self._normalized_weight_entropy(row_weights))
                rows_confidence.append(
                    self._soft_alignment_confidence(row_weights, used_fallback=True)
                    if not learned_alignment
                    or self.soft_alignment_confidence_mode != "none"
                    else 1.0
                )
                fallback_mask.append(True)
                positive_counts.append(0)
                top1_boundary_hit_mask.append(False)
                continue

            slm_token_start, slm_token_end = slm_offsets[slm_idx]
            clipped_start = max(slm_token_start, slm_span_start)
            clipped_end = min(slm_token_end, slm_span_end)
            if clipped_end <= clipped_start:
                fallback_idx = self._fallback_llm_index_for_span(
                    slm_token_start, llm_offsets, llm_range
                )
                if fallback_idx >= 0:
                    row_indices[0] = fallback_idx
                    row_weights[0] = 1.0
                rows_indices.append(row_indices)
                rows_weights.append(row_weights)
                rows_entropy.append(self._normalized_weight_entropy(row_weights))
                rows_confidence.append(
                    self._soft_alignment_confidence(row_weights, used_fallback=True)
                    if not learned_alignment
                    or self.soft_alignment_confidence_mode != "none"
                    else 1.0
                )
                fallback_mask.append(True)
                positive_counts.append(0)
                top1_boundary_hit_mask.append(False)
                continue

            rel_start = clipped_start - slm_span_start
            rel_end = clipped_end - slm_span_start
            target_start = min(llm_span_end, llm_span_start + rel_start)
            target_end = min(llm_span_end, llm_span_start + rel_end)
            if target_end <= target_start:
                fallback_idx = self._fallback_llm_index_for_span(
                    target_start, llm_offsets, llm_range
                )
                if fallback_idx >= 0:
                    row_indices[0] = fallback_idx
                    row_weights[0] = 1.0
                rows_indices.append(row_indices)
                rows_weights.append(row_weights)
                rows_entropy.append(self._normalized_weight_entropy(row_weights))
                rows_confidence.append(
                    self._soft_alignment_confidence(row_weights, used_fallback=True)
                    if not learned_alignment
                    or self.soft_alignment_confidence_mode != "none"
                    else 1.0
                )
                fallback_mask.append(True)
                positive_counts.append(0)
                top1_boundary_hit_mask.append(False)
                continue

            candidates: List[Tuple[float, int, int, int, bool]] = []
            for llm_idx in range(llm_start_idx, llm_end_idx):
                llm_token_start, llm_token_end = llm_offsets[llm_idx]
                if llm_token_end <= llm_token_start:
                    continue
                score = self._overlap_len(
                    target_start, target_end, llm_token_start, llm_token_end
                )
                if score > 0:
                    token_len = llm_token_end - llm_token_start
                    weight_score = self._soft_candidate_weight_score(
                        score,
                        target_start,
                        target_end,
                        llm_token_start,
                        llm_token_end,
                    )
                    boundary_hit = (
                        self._boundary_hit_count(
                            target_start,
                            target_end,
                            llm_token_start,
                            llm_token_end,
                            self.soft_alignment_boundary_tolerance,
                        )
                        > 0
                    )
                    candidates.append(
                        (weight_score, score, token_len, llm_idx, boundary_hit)
                    )

            positive_count = len(candidates)
            top1_boundary_hit = False
            if learned_alignment:
                fallback_idx = -1
                if not candidates:
                    fallback_idx = self._fallback_llm_index_for_span(
                        target_start, llm_offsets, llm_range
                    )
                row_indices, row_weights, used_fallback, top1_boundary_hit = (
                    self._learned_alignment_row_from_candidates(
                        candidates=candidates,
                        fallback_idx=fallback_idx,
                        llm_range=llm_range,
                        top_k=top_k,
                    )
                )
            elif not candidates:
                fallback_idx = self._fallback_llm_index_for_span(
                    target_start, llm_offsets, llm_range
                )
                if fallback_idx >= 0:
                    row_indices[0] = fallback_idx
                    row_weights[0] = 1.0
                used_fallback = True
            else:
                candidates.sort(
                    key=lambda item: (-item[0], -item[1], -item[2], item[3])
                )
                selected = candidates[:top_k]
                weights = self._normalize_soft_scores(
                    [
                        weight_score
                        for weight_score, _overlap, _token_len, _idx, _hit in selected
                    ]
                )
                weights = self._reweight_soft_alignment_weights(
                    weights,
                    [
                        _overlap
                        for _weight_score, _overlap, _token_len, _idx, _hit in selected
                    ],
                )
                top1_boundary_hit = bool(selected[0][4])
                for out_idx, (
                    _weight_score,
                    _overlap,
                    _token_len,
                    llm_idx,
                    _boundary_hit,
                ) in enumerate(selected):
                    row_indices[out_idx] = llm_idx
                    row_weights[out_idx] = weights[out_idx]

            rows_indices.append(row_indices)
            rows_weights.append(row_weights)
            rows_entropy.append(self._normalized_weight_entropy(row_weights))
            rows_confidence.append(
                self._soft_alignment_confidence(row_weights, used_fallback)
                if not learned_alignment
                or self.soft_alignment_confidence_mode != "none"
                else 1.0
            )
            fallback_mask.append(used_fallback)
            positive_counts.append(positive_count)
            top1_boundary_hit_mask.append(top1_boundary_hit)

        return {
            "source_indices": rows_indices,
            "source_weights": rows_weights,
            "source_confidence": rows_confidence,
            "source_entropy": rows_entropy,
            "fallback_mask": fallback_mask,
            "positive_overlap_counts": positive_counts,
            "top1_boundary_hit_mask": top1_boundary_hit_mask,
        }

    def align_chat_messages_soft(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
        remove_last_surfix: bool = False,
        top_k: int = 4,
        return_details: bool = False,
        apply_confidence_control: bool = True,
    ) -> Dict[str, Any]:
        """
        Return unpadded SLM/LLM chat-template token sequences plus top-k soft
        source-token alignment from SLM positions to LLM positions.
        """
        assert not (
            add_generation_prompt and remove_last_surfix
        ), "add_generation_prompt and remove_last_surfix cannot be True at the same time"
        top_k = max(1, int(top_k))

        slm_text, slm_ids, slm_offsets = self._apply_chat_template_to_ids(
            self.slm_tokenizer,
            messages,
            add_generation_prompt,
            enable_thinking,
            remove_last_surfix,
        )
        llm_text, llm_ids, llm_offsets = self._apply_chat_template_to_ids(
            self.llm_tokenizer,
            messages,
            add_generation_prompt,
            enable_thinking,
            remove_last_surfix,
        )

        content_spans_slm = self._compute_content_spans(slm_text, messages)
        content_spans_llm = self._compute_content_spans(llm_text, messages)
        assert (
            slm_offsets is not None and llm_offsets is not None
        ), "offset_mapping required"
        assert len(content_spans_slm) == len(
            content_spans_llm
        ), "Content span count mismatch"

        slm_msg_ranges = self._spans_to_token_ranges(slm_offsets, content_spans_slm)
        llm_msg_ranges = self._spans_to_token_ranges(llm_offsets, content_spans_llm)

        def build_sections(total_len: int, msg_ranges: List[Tuple[int, int]]):
            sections: List[Tuple[str, int, int]] = []
            prev = 0
            for s, e in msg_ranges:
                if prev < s:
                    sections.append(("template", prev, s))
                sections.append(("message", s, e))
                prev = e
            if prev < total_len:
                sections.append(("template", prev, total_len))
            return sections

        slm_sections = build_sections(len(slm_ids), slm_msg_ranges)
        llm_sections = build_sections(len(llm_ids), llm_msg_ranges)
        assert len(slm_sections) == len(llm_sections), "Section count mismatch"

        message_mask = [False] * len(slm_ids)
        source_indices = [[-1] * top_k for _ in slm_ids]
        source_weights = [[0.0] * top_k for _ in slm_ids]
        source_confidence = [1.0] * len(slm_ids)
        source_entropy = [0.0] * len(slm_ids)
        fallback_mask = [False] * len(slm_ids)
        positive_overlap_counts = [0] * len(slm_ids)
        top1_boundary_hit_mask = [False] * len(slm_ids)
        detailed_sections: List[Dict[str, Union[str, Tuple[int, int]]]] = []

        message_section_idx = 0
        for (stype_s, s_s, e_s), (stype_l, s_l, e_l) in zip(slm_sections, llm_sections):
            assert stype_s == stype_l, "Section type mismatch"
            detailed_sections.append(
                {
                    "type": stype_s,
                    "slm_range": (s_s, e_s),
                    "llm_range": (s_l, e_l),
                }
            )
            if stype_s != "message":
                continue

            rows = self._soft_align_message_by_span_overlap(
                slm_ids=slm_ids,
                slm_offsets=slm_offsets,
                slm_range=(s_s, e_s),
                slm_span=content_spans_slm[message_section_idx],
                llm_offsets=llm_offsets,
                llm_range=(s_l, e_l),
                llm_span=content_spans_llm[message_section_idx],
                top_k=top_k,
            )
            for local_idx, slm_idx in enumerate(range(s_s, e_s)):
                message_mask[slm_idx] = True
                source_indices[slm_idx] = rows["source_indices"][local_idx]
                source_weights[slm_idx] = rows["source_weights"][local_idx]
                source_confidence[slm_idx] = rows["source_confidence"][local_idx]
                source_entropy[slm_idx] = rows["source_entropy"][local_idx]
                fallback_mask[slm_idx] = rows["fallback_mask"][local_idx]
                positive_overlap_counts[slm_idx] = rows["positive_overlap_counts"][
                    local_idx
                ]
                top1_boundary_hit_mask[slm_idx] = rows["top1_boundary_hit_mask"][
                    local_idx
                ]
            message_section_idx += 1

        shuffle_active_mask = message_mask.copy()
        if messages and messages[-1].get("role") == "assistant" and slm_msg_ranges:
            last_assistant_start = slm_msg_ranges[-1][0]
            shuffle_active_mask = [
                is_message and idx < last_assistant_start
                for idx, is_message in enumerate(message_mask)
            ]

        if apply_confidence_control:
            source_confidence, source_entropy, source_entropy_override = (
                self._apply_confidence_control(
                    source_confidence=source_confidence,
                    source_entropy=source_entropy,
                    message_mask=message_mask,
                    token_ids=slm_ids,
                    shuffle_active_mask=shuffle_active_mask,
                )
            )
        else:
            source_entropy_override = [False] * len(source_confidence)

        result: Dict[str, Any] = {
            "slm_ids": slm_ids,
            "llm_ids": llm_ids,
            "slm_ids_padded": slm_ids,
            "llm_ids_padded": llm_ids,
            "message_mask": message_mask,
            "slm_padding_mask": [False] * len(slm_ids),
            "llm_padding_mask": [False] * len(llm_ids),
            "soft_alignment": {
                "source_indices": source_indices,
                "source_weights": source_weights,
                "source_confidence": source_confidence,
                "source_entropy": source_entropy,
                "source_entropy_override": source_entropy_override,
                "fallback_mask": fallback_mask,
                "positive_overlap_counts": positive_overlap_counts,
                "top1_boundary_hit_mask": top1_boundary_hit_mask,
                "top_k": top_k,
                "score_mode": self.soft_alignment_score_mode,
                "boundary_bonus": self.soft_alignment_boundary_bonus,
                "boundary_tolerance": self.soft_alignment_boundary_tolerance,
                "min_weight": self.soft_alignment_min_weight,
                "confidence_mode": self.soft_alignment_confidence_mode,
                "confidence_alpha": self.soft_alignment_confidence_alpha,
                "confidence_floor": self.soft_alignment_confidence_floor,
                "fallback_confidence": self.soft_alignment_fallback_confidence,
                "confidence_control_mode": (
                    self.soft_alignment_confidence_control_mode
                ),
                "confidence_constant_value": (
                    self.soft_alignment_confidence_constant_value
                ),
                "confidence_shuffle_seed": (
                    self.soft_alignment_confidence_shuffle_seed
                ),
                "confidence_control_applied": apply_confidence_control,
                "reweight_mode": self.soft_alignment_reweight_mode,
                "reweight_strength": self.soft_alignment_reweight_strength,
                "reweight_power": self.soft_alignment_reweight_power,
                "candidate_window": self.soft_alignment_candidate_window,
            },
            "sections": detailed_sections,
            "slm_text": slm_text,
            "llm_text": llm_text,
        }

        if return_details:
            result["content_spans_slm"] = content_spans_slm
            result["content_spans_llm"] = content_spans_llm
            result["slm_offsets"] = slm_offsets
            result["llm_offsets"] = llm_offsets
        return result

    def align_chat_messages(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
        return_details: bool = False,
        remove_last_surfix: bool = False,
    ) -> Dict[str, any]:
        """
        Align chat-templated sequences by sections (template/message/template...):
        - Preserve all template tokens (pad the shorter template section)
        - For each message section, map SLM tokens to LLM tokens 1:1 via strategy
        - If remove_last_surfix is True, remove the last suffix from the LLM text
        Returns essentials: slm_ids_padded, llm_ids_padded, message_mask (shared),
        slm_padding_mask, llm_padding_mask (True where token is padding inserted).
        When return_details=True, also returns 'sections' with aligned ranges.
        """
        if self.strategy in {
            AlignmentStrategy.SOFT_SPAN_OVERLAP,
            AlignmentStrategy.SOFT_SPAN_OVERLAP_V2,
            AlignmentStrategy.LEARNED_SPAN_ALIGNMENT,
        }:
            return self.align_chat_messages_soft(
                messages=messages,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
                remove_last_surfix=remove_last_surfix,
                return_details=return_details,
            )

        assert not (
            add_generation_prompt and remove_last_surfix
        ), "add_generation_prompt and remove_last_surfix cannot be True at the same time"

        # Build templated sequences with offsets
        slm_text, slm_ids, slm_offsets = self._apply_chat_template_to_ids(
            self.slm_tokenizer,
            messages,
            add_generation_prompt,
            enable_thinking,
            remove_last_surfix,
        )
        llm_text, llm_ids, llm_offsets = self._apply_chat_template_to_ids(
            self.llm_tokenizer,
            messages,
            add_generation_prompt,
            enable_thinking,
            remove_last_surfix,
        )

        # Required pad tokens
        assert self.slm_tokenizer.pad_token_id is not None, "SLM pad_token_id required"
        assert self.llm_tokenizer.pad_token_id is not None, "LLM pad_token_id required"
        slm_pad_id = self.slm_tokenizer.pad_token_id
        llm_pad_id = self.llm_tokenizer.pad_token_id

        # Content spans (char) and token ranges
        content_spans_slm = self._compute_content_spans(slm_text, messages)
        content_spans_llm = self._compute_content_spans(llm_text, messages)
        assert (
            slm_offsets is not None and llm_offsets is not None
        ), "offset_mapping required"
        assert len(content_spans_slm) == len(
            content_spans_llm
        ), "Content span count mismatch"
        slm_msg_ranges = self._spans_to_token_ranges(slm_offsets, content_spans_slm)
        llm_msg_ranges = self._spans_to_token_ranges(llm_offsets, content_spans_llm)

        # Build section ranges (template/message alternating)
        def build_sections(total_len: int, msg_ranges: List[Tuple[int, int]]):
            sections: List[Tuple[str, int, int]] = []
            prev = 0
            for s, e in msg_ranges:
                if prev < s:
                    sections.append(("template", prev, s))
                sections.append(("message", s, e))
                prev = e
            if prev < total_len:
                sections.append(("template", prev, total_len))
            return sections

        slm_sections = build_sections(len(slm_ids), slm_msg_ranges)
        llm_sections = build_sections(len(llm_ids), llm_msg_ranges)
        assert len(slm_sections) == len(llm_sections), "Section count mismatch"

        slm_out: List[int] = []
        llm_out: List[int] = []
        mask_out: List[bool] = []
        slm_pad_mask_out: List[bool] = []
        llm_pad_mask_out: List[bool] = []
        detailed_sections: List[Dict[str, Union[str, Tuple[int, int]]]] = []
        message_section_idx = 0

        for (stype_s, s_s, e_s), (stype_l, s_l, e_l) in zip(slm_sections, llm_sections):
            assert stype_s == stype_l, "Section type mismatch"
            slm_start_out = len(slm_out)
            llm_start_out = len(llm_out)
            if stype_s == "template":
                slm_seg_len = e_s - s_s
                llm_seg_len = e_l - s_l
                target_len = slm_seg_len if slm_seg_len >= llm_seg_len else llm_seg_len
                slm_pad_needed = target_len - slm_seg_len
                llm_pad_needed = target_len - llm_seg_len
                slm_seg = slm_ids[s_s:e_s] + [slm_pad_id] * slm_pad_needed
                llm_seg = llm_ids[s_l:e_l] + [llm_pad_id] * llm_pad_needed
                slm_out.extend(slm_seg)
                llm_out.extend(llm_seg)
                mask_out.extend([False] * target_len)
                slm_pad_mask_out.extend([False] * slm_seg_len + [True] * slm_pad_needed)
                llm_pad_mask_out.extend([False] * llm_seg_len + [True] * llm_pad_needed)
            else:  # message
                slm_msg = slm_ids[s_s:e_s]
                if self.strategy == AlignmentStrategy.SPAN_OVERLAP:
                    llm_msg = self._align_message_by_span_overlap(
                        slm_ids=slm_ids,
                        slm_offsets=slm_offsets,
                        slm_range=(s_s, e_s),
                        slm_span=content_spans_slm[message_section_idx],
                        llm_ids=llm_ids,
                        llm_offsets=llm_offsets,
                        llm_range=(s_l, e_l),
                        llm_span=content_spans_llm[message_section_idx],
                    )
                else:
                    llm_msg = self.align_tokens(slm_msg)
                message_section_idx += 1
                assert len(llm_msg) == len(slm_msg)
                slm_out.extend(slm_msg)
                llm_out.extend(llm_msg)
                mask_out.extend([True] * len(slm_msg))
                # no padding in message sections
                slm_pad_mask_out.extend([False] * len(slm_msg))
                llm_pad_mask_out.extend([False] * len(slm_msg))
            slm_end_out = len(slm_out)
            llm_end_out = len(llm_out)
            detailed_sections.append(
                {
                    "type": stype_s,
                    "slm_range": (slm_start_out, slm_end_out),
                    "llm_range": (llm_start_out, llm_end_out),
                }
            )

        result_min = {
            "slm_ids_padded": slm_out,
            "llm_ids_padded": llm_out,
            "message_mask": mask_out,
            "slm_padding_mask": slm_pad_mask_out,
            "llm_padding_mask": llm_pad_mask_out,
        }
        if return_details:
            result_min["sections"] = detailed_sections
            result_min["slm_text"] = slm_text
            result_min["llm_text"] = llm_text
        return result_min
