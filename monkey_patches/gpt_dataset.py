# File modified by Ingus:
#   Hardcoded create_attention_mask to True.
#   Changed default pad token index from None to 0 like in TildeOpen.
#   Changed ltor function to work with the full sequence rather rather than the input sequence/input tokens.
#   Hardcoded out caching of attention masks, loss masks and position ids.
#   Fixed _query_document_sample_shuffle_indices to not erronously append empty document slices.
#   Rewrite packing to always be k=16 kbins packing.
#   Throw out support for reset_position_ids.
#   Rewrite ltor function to generate loss masks as in TildeOpen.
#     Hardcoded tokens to mask for untieing the knots.
#     Hardcoded eod_mask_loss as in GPT NeoX.
#     Hardcoded eod token to 48 like in TildeOpen.
#   Rewrite ltor function to generate attention masks as boolean tensors indicating EOD in input sequence.

# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import warnings

import numpy
import torch

from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.datasets.megatron_dataset import MegatronDataset
#from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer
MegatronTokenizer = None #Was just used for typechecking anyway
from megatron.core.datasets.object_storage_utils import ObjectStorageConfig, is_object_storage_path
from megatron.core.datasets.utils import Split
from megatron.core.utils import log_single_rank

logger = logging.getLogger(__name__)

_PAD_TOKEN_ID = -1


@dataclass
class GPTDatasetConfig(BlendedMegatronDatasetConfig):
    """Configuration object for Megatron Core GPT datasets"""

    reset_position_ids: Optional[bool] = None
    """Option to reset the position IDs in the dataset at an interval"""

    reset_attention_mask: Optional[bool] = None
    """Option to reset the attention mask from the dataset"""

    eod_mask_loss: Optional[bool] = None
    """Option to enable the EOD mask loss"""

    create_attention_mask: bool = True
    """Option to enable the attention masks generation. Can be disabled if attention kernel
       generates masks by itself.
    """

    drop_last_partial_validation_sequence: bool = True
    """Option to drop the last partial validation sequence"""

    add_extra_token_to_sequence: bool = True
    """Option to draw sequences with one extra token to ensure the sample input tokens and sample
       output tokens are both of the desired sequence length
    """

    object_storage_cache_path: Optional[str] = None
    """Path for caching indices for s3 or msc dataloading."""

    def __post_init__(self) -> None:
        """Do asserts and set fields post init"""
        super().__post_init__()

        assert self.tokenizer is not None

        assert self.reset_position_ids is not None
        assert self.reset_attention_mask is not None
        assert self.eod_mask_loss is not None


class GPTDataset(MegatronDataset):
    """The base GPT dataset

    Args:
        indexed_dataset (IndexedDataset): The IndexedDataset around which to build the GPTDataset

        dataset_path (Optional[str]): The real path on disk to the dataset, for bookkeeping

        indexed_indices (numpy.ndarray): The set of the documents indices to expose

        num_samples (Optional[int]): The number of samples to draw from the indexed dataset. When
            None, build as many samples as correspond to one epoch.

        index_split (Split): The indexed_indices Split

        config (GPTDatasetConfig): The config
    """

    def __init__(
        self,
        indexed_dataset: IndexedDataset,
        dataset_path: Optional[str],
        indexed_indices: numpy.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: GPTDatasetConfig,
    ) -> None:
        config.create_attention_mask=True
        super().__init__(
            indexed_dataset, dataset_path, indexed_indices, num_samples, index_split, config
        )
        self.masks_and_position_ids_are_cacheable = not any(
            [
                self.config.reset_position_ids,
                self.config.reset_attention_mask,
                self.config.eod_mask_loss,
            ]
        )
        self.masks_and_position_ids_are_cached = False
        self.cached_attention_mask = None
        self.cached_loss_mask = None
        self.cached_position_ids = None

        try:
            self._pad_token_id = self.config.tokenizer.pad
        except Exception:
            self._pad_token_id = _PAD_TOKEN_ID
        
        # Override to be the same as in gpt neox
        if self._pad_token_id is None:
          self._pad_token_id = 0

        (self.document_index, self.sample_index, self.shuffle_index) = (
            self._build_document_sample_shuffle_indices()
        )

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: IndexedDataset) -> int:
        """Abstract method implementation

        For GPT, the underlying IndexedDataset should be split by sequence, as opposed to, say,
        BERT, which should be split by document

        Args:
            low_level_dataset (IndexedDataset): The underlying IndexedDataset

        Returns:
            int: The number of unique elements in the underlying IndexedDataset
        """
        return low_level_dataset.sequence_lengths.shape[0]

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: GPTDatasetConfig) -> IndexedDataset:
        """Abstract method implementation

        Args:
            dataset_path (str): The real path prefix to the IndexedDataset .bin and .idx files

            config (GPTDatasetConfig): The config

        Returns:
            IndexedDataset: The underlying IndexedDataset
        """
        if is_object_storage_path(dataset_path):
            assert config.object_storage_cache_path is not None
            return IndexedDataset(
                dataset_path,
                multimodal=False,
                mmap=config.mmap_bin_files,
                object_storage_config=ObjectStorageConfig(
                    path_to_idx_cache=config.object_storage_cache_path
                ),
            )
        return IndexedDataset(dataset_path, multimodal=False, mmap=config.mmap_bin_files)

    def __len__(self) -> int:
        """Abstract method implementation

        Returns:
            int: The length of the dataset
        """
        return self.sample_index.shape[0] - 1

    def __getitem__(self, idx: Optional[int]) -> Dict[str, torch.Tensor]:
        """Abstract method implementation

        Args:
            idx (Optioal[int]): The index into the dataset

        Returns:
            Dict[str, torch.Tensor]: The sample information wrapped in a dictionary
        """
        if idx is None:
            # Batch padding sequence so the index does not matter
            text, _ = self._query_document_sample_shuffle_indices(0)
        else:
            text, _ = self._query_document_sample_shuffle_indices(idx)

        text = torch.from_numpy(text).long()
        if self.config.add_extra_token_to_sequence:
            tokens = text[:-1].contiguous()
            labels = text[1:].contiguous()
        else:
            tokens = text
            labels = torch.roll(text, shifts=-1, dims=0)
            labels[-1] = self._pad_token_id

        if (
            not self.masks_and_position_ids_are_cacheable
            or not self.masks_and_position_ids_are_cached
        ):
            attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
                text,
                self.config.tokenizer.eod,
                self.config.reset_position_ids,
                self.config.reset_attention_mask,
                self.config.eod_mask_loss,
                self.config.create_attention_mask,
            )
            # We are not caching this.
            if self.masks_and_position_ids_are_cacheable and False:
                self.cached_attention_mask = attention_mask
                self.cached_loss_mask = loss_mask
                self.cached_position_ids = position_ids
                self.masks_and_position_ids_are_cached = True
        else:
            attention_mask = self.cached_attention_mask
            loss_mask = self.cached_loss_mask.clone()
            position_ids = self.cached_position_ids

        # For padded sequences, mask the loss
        loss_mask[labels == self._pad_token_id] = 0.0

        # For padded sequences, ensure the embedding layer can map the token ID
        tokens[tokens == self._pad_token_id] = 0
        labels[labels == self._pad_token_id] = 0

        # Batch padding sequence so we mask the loss
        if idx is None:
            loss_mask = torch.zeros_like(loss_mask)

        if self.config.create_attention_mask:
            return {
                "tokens": tokens,
                "labels": labels,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
            }
        else:
            return {
                "tokens": tokens,
                "labels": labels,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
            }

    def _query_document_sample_shuffle_indices(
        self, idx: int
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Get the text (token ids) and document ids for a given index

        Args:
            idx (int): The index into the dataset

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: The text ids and document ids
        """
        # Do the shuffle mapping
        idx = self.shuffle_index[idx]

        # Get the beginning and end documents and offsets
        doc_index_beg, doc_index_beg_offset = self.sample_index[idx]
        doc_index_end, doc_index_end_offset = self.sample_index[idx + 1]

        document_ids = []
        sample_parts = []

        # Sample spans a single document
        if doc_index_beg == doc_index_end:
            # Add the document id
            document_ids.append(self.document_index[doc_index_beg])

            # Add the entire sample
            sample_parts.append(
                self.dataset.get(
                    self.document_index[doc_index_beg],
                    offset=doc_index_beg_offset,
                    length=doc_index_end_offset
                    - doc_index_beg_offset
                    + self.config.add_extra_token_to_sequence,
                )
            )

        # Sample spans multiple documents
        else:
            for i in range(doc_index_beg, doc_index_end + 1):
                # Add the document id
                #document_ids.append(self.document_index[i])

                # Add the sample part
                offset = 0 if i > doc_index_beg else doc_index_beg_offset
                length = (
                    None
                    if i < doc_index_end
                    else doc_index_end_offset + self.config.add_extra_token_to_sequence
                )
                if i == doc_index_end and doc_index_end_offset == 0:
                  continue
                sample_parts.append(
                    self.dataset.get(self.document_index[i], offset=offset, length=length)
                )
                document_ids.append(self.document_index[i])
        #assert len(document_ids) == len(
        #    sample_parts
        #), f"len(document_ids) ({len(document_ids)}) != len(sample_parts) ({len(sample_parts)})"

        length = sum(map(len, sample_parts))

        # Pad the sample if necessary
        if length < (self.config.sequence_length + self.config.add_extra_token_to_sequence):
            sample_parts.append(
                [self._pad_token_id]
                * (self.config.sequence_length + self.config.add_extra_token_to_sequence - length)
            )

        return (
            numpy.concatenate(sample_parts, dtype=numpy.int64),
            numpy.array(document_ids, dtype=numpy.int64),
        )

    def _build_document_sample_shuffle_indices(
        self,
    ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """Build the document index, the sample index, and the shuffle index

        The document index:
            -- 1-D
            -- An ordered array of document ids

        The sample index:
            -- 2-D
            -- The document indices and offsets which mark the start of every sample

        The shuffle index:
            -- 1-D
            -- A random permutation of index range of the sample index

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: The document index, the sample
            index, and the shuffle index
        """
        path_to_cache = self.config.path_to_cache
        if path_to_cache is None and not self.config.mock:
            path_to_cache = os.path.join(
                self.dataset.path_prefix, "cache", f"{type(self).__name__}_indices"
            )

        if path_to_cache:
            base = f"{self.unique_description_hash}-{type(self).__name__}-{self.index_split.name}"
            get_path_to = lambda affix: os.path.join(path_to_cache, f"{base}-{affix}")
            path_to_description = get_path_to("description.txt")
            path_to_document_index = get_path_to("document_index.npy")
            path_to_sample_index = get_path_to("sample_index.npy")
            path_to_shuffle_index = get_path_to("shuffle_index.npy")
            cache_hit = all(
                map(
                    os.path.isfile,
                    [
                        path_to_description,
                        path_to_document_index,
                        path_to_sample_index,
                        path_to_shuffle_index,
                    ],
                )
            )
        else:
            cache_hit = False

        # IP: This is where actual construction is done.
        if not path_to_cache or (
            not cache_hit
            and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0)
        ):
            log_single_rank(
                logger,
                logging.INFO,
                f"Build and save the {type(self).__name__} {self.index_split.name} indices",
            )
            t_beg = time.time()

            full_sequence_length = self.config.sequence_length + 1
            # Amount of tokens per sample.
            # Used for both construction of input token sample and labels.

            numpy_random_state = numpy.random.RandomState(self.config.random_seed)

            assert self.dataset.sequence_lengths.dtype == numpy.int32

            sequence_lengths = self.dataset.sequence_lengths

            document_index = []
            sample_index = [(0, 0)]

            samples_constructed = 0
            num_epochs = 0
            usable_document_count = len(self.indices)
            assert usable_document_count > 0, "No documents available to build indices"
            # The Megatron-LM devs desired that we only use these indices.

            available_documents = None

            bin_count = 16  # Same as Martin used in GPT NeoX.
            bins = [[] for _ in range(bin_count)]
            slack = [full_sequence_length for _ in range(bin_count)]  # Space left in the bin.
            # We will try to keep bins sorted from least full to most full.

            current_available_document_pointer = usable_document_count  # To trigger automatic index reshuffle.

            while samples_constructed < self.num_samples:
                # Check for running out of documents in this epoch.
                if current_available_document_pointer == usable_document_count:
                    available_documents = self.indices.copy()
                    numpy_random_state.shuffle(available_documents)
                    current_available_document_pointer = 0
                    num_epochs+= 1

                current_document = available_documents[current_available_document_pointer]  # Index in self.dataset
                current_sample_length = sequence_lengths[current_document]

                if current_sample_length > full_sequence_length:
                    print("Currently can't handle samples larger than sequence length", current_sample_length, " > ", full_sequence_length)

                # Finding tightest bin we can fit the sample in.
                best_bin = None
                for bin_idx in range(bin_count - 1, - 1, - 1):
                    if slack[bin_idx] >= current_sample_length:
                        best_bin = bin_idx
                        break

                if best_bin is None:
                    # Have to flush a bin and create a new one.
                    document_index.extend(bins[-1])
                    sample_index.append((len(document_index), 0))
                    bins = [[]] + bins[:-1]
                    slack = [full_sequence_length] + slack[:-1]
                    samples_constructed+= 1

                else:
                    # Add sample to best bin and sort.
                    bins[best_bin].append(current_document)
                    slack[best_bin]-= current_sample_length

                    # Just shift bin rightwards until it's in the right position
                    while best_bin < bin_count - 1:
                        if slack[best_bin + 1] < slack[best_bin]:
                            break

                        # Swap bin with one to the right.
                        tmp_bin, tmp_slack = bins[best_bin], slack[best_bin]
                        bins[best_bin], slack[best_bin] = bins[best_bin + 1], slack[best_bin + 1]
                        bins[best_bin + 1], slack[best_bin + 1] = tmp_bin, tmp_slack
                        best_bin+= 1

                    current_available_document_pointer+= 1

            document_index = numpy.array(document_index)
            sample_index = numpy.array(sample_index)
            shuffle_index = numpy.arange(0, samples_constructed, dtype=numpy.int64)
            numpy_random_state.shuffle(shuffle_index)

            if path_to_cache:
                os.makedirs(path_to_cache, exist_ok=True)
                # Write the description
                with open(path_to_description, "wt") as writer:
                    writer.write(self.unique_description)
                numpy.save(path_to_document_index, document_index, allow_pickle=True)
                numpy.save(path_to_sample_index, sample_index, allow_pickle=True)
                numpy.save(path_to_shuffle_index, shuffle_index, allow_pickle=True)
            else:
                log_single_rank(
                    logger,
                    logging.WARNING,
                    f"Unable to save {type(self).__name__} indexes because path_to_cache is None",
                )

            t_end = time.time()
            log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

            log_single_rank(
                logger, logging.INFO, f"> total number of samples: {sample_index.shape[0] - 1}"
            )
            log_single_rank(logger, logging.INFO, f"> total number of epochs: {num_epochs}")

            return document_index, sample_index, shuffle_index

        log_single_rank(
            logger, logging.INFO, f"Load the {type(self).__name__} {self.index_split.name} indices"
        )

        log_single_rank(
            logger,
            logging.INFO,
            f"\tLoad the document index from {os.path.basename(path_to_document_index)}",
        )
        t_beg = time.time()
        document_index = numpy.load(path_to_document_index, allow_pickle=True, mmap_mode="r")
        t_end = time.time()
        log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_single_rank(
            logger,
            logging.INFO,
            f"\tLoad the sample index from {os.path.basename(path_to_sample_index)}",
        )
        t_beg = time.time()
        sample_index = numpy.load(path_to_sample_index, allow_pickle=True, mmap_mode="r")
        t_end = time.time()
        log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_single_rank(
            logger,
            logging.INFO,
            f"\tLoad the shuffle index from {os.path.basename(path_to_shuffle_index)}",
        )
        t_beg = time.time()
        shuffle_index = numpy.load(path_to_shuffle_index, allow_pickle=True, mmap_mode="r")
        t_end = time.time()
        log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_single_rank(
            logger, logging.INFO, f"> total number of samples: {sample_index.shape[0] - 1}"
        )

        return document_index, sample_index, shuffle_index

    def _get_num_tokens_per_epoch(self) -> int:
        """Calculate the number of tokens in a single epoch

        Returns:
            int: The number of tokens in a single epoch
        """
        return int(numpy.sum(self.dataset.sequence_lengths[self.indices]))

    def _get_num_epochs(self, num_tokens_per_epoch: int) -> int:
        """Calculate the number of epochs

        Args:
            num_tokens_per_epoch (int): The number of tokens in a single epoch

        Returns:
            int: The number of epochs
        """
        num_epochs = 1
        num_tokens = num_tokens_per_epoch
        if self.num_samples is None:
            return num_epochs
        else:
            num_tokens_requested = (
                self.num_samples * self.config.sequence_length
            ) + self.config.add_extra_token_to_sequence
            while num_tokens < num_tokens_requested:
                num_epochs += 1
                num_tokens += num_tokens_per_epoch
        return num_epochs


def _build_document_index(
    documents: numpy.ndarray,
    num_epochs: int,
    numpy_random_state: numpy.random.RandomState,
    separate_final_epoch: bool,
) -> numpy.ndarray:
    """Build an array with length = num epochs * num documents

    Args:
        documents (numpy.ndarray): the subset of exposed document indices

        num_epochs (int): The number of epochs

        numpy_random_state (numpy.random.RandomState): The NumPy random state

        separate_final_epoch (bool): Whether to exclude the last epoch from the global shuffle

    Returns:
        numpy.ndarray: The document index
    """

    if not separate_final_epoch or num_epochs == 1:
        document_index = numpy.mgrid[0:num_epochs, 0 : len(documents)][1]
        document_index[:] = documents
        document_index = document_index.reshape(-1)
        document_index = document_index.astype(numpy.int32)
        numpy_random_state.shuffle(document_index)
        return document_index

    doc_idx_first = _build_document_index(documents, num_epochs - 1, numpy_random_state, False)
    doc_idx_last = _build_document_index(documents, 1, numpy_random_state, False)
    return numpy.concatenate((doc_idx_first, doc_idx_last))


def _build_shuffle_index(
    num_samples: int, total_size: int, numpy_random_state: numpy.random.RandomState
) -> numpy.ndarray:
    """Build the range [0, size) and shuffle

    Args:
        num_samples (int): The size of the first shuffle range [0, num_samples)

        total_size (int): The size of the entire index. If larger than 'num_samples', it defines
            the second shuffle range [num_samples, total_size)

        numpy_random_state (numpy.random.RandomState): The NumPy random state

    Returns:
        numpy.ndarray: The shuffle index
    """

    dtype_ = numpy.uint32
    if total_size >= (numpy.iinfo(numpy.uint32).max - 1):
        dtype_ = numpy.int64

    shuffle_idx_first = numpy.arange(start=0, stop=num_samples, step=1, dtype=dtype_)
    numpy_random_state.shuffle(shuffle_idx_first)
    if num_samples == total_size:
        return shuffle_idx_first

    shuffle_idx_last = numpy.arange(start=num_samples, stop=total_size, step=1, dtype=dtype_)
    numpy_random_state.shuffle(shuffle_idx_last)

    return numpy.concatenate((shuffle_idx_first, shuffle_idx_last))


def _get_ltor_masks_and_position_ids(
    data: torch.Tensor,
    eod_token: int,
    reset_position_ids: bool,
    reset_attention_mask: bool,
    eod_mask_loss: bool,
    create_attention_mask: bool,
):
    """Build masks and position id for left to right model.
    Function calculates stuff for one sample, not a batch.

    Args:
        data (torch.Tensor): The data tenor that holds the tokens from the dataset

        eod_token (int): ID of the token to that is considered the EOD

        reset_position_ids (bool): Flag doesn't do anything.

        reset_attention_mask (bool): Switch to reset the attention mask

        eod_mask_loss (bool): Switch to enable the EOD mask loss

        create_attention_mask (bool): Switch to enable the attention masks generation. Can be
            disabled if attention kernel generates masks by itself.

    Returns:
        torch.Tensor: Attention mask needed to be used for Attention

        torch.Tensor: The mask used for loss value during training

        torch.Tensor: The position ID's of the token
    """
    loss_mask_pairs = [61, 62]
    loss_mask_individual = [51, 52, 53, 54, 56, 57, 58, 59, 60]
    loss_mask_alternating_individual = []

    sample_end_masks = True
    create_attention_mask = True
    reset_attention_mask = True
    eod_mask_loss = True
    eod_token = 48

    if reset_position_ids:
        warnings.warn("WARNING: the reset_position_ids flag in _get_ltor_masks_and_position_ids was removed by Ingus "
                      "and does nothing now.",
                      category=DeprecationWarning,
                      stacklevel=1)

    if loss_mask_pairs is None:
        loss_mask_pairs = []
    if loss_mask_individual is None:
        loss_mask_individual = []
    if loss_mask_alternating_individual is None:
        loss_mask_alternating_individual = []

    full_seq_length = data.numel()
    seq_length = full_seq_length - 1

    in_seq = data[:-1]
    out_seq = data[1:]

    # Creating attention mask.
    if create_attention_mask:
        if not reset_attention_mask:
            # Just create a triangular matrix.

            attention_mask = torch.tril(
                torch.ones((seq_length, seq_length), device=data.device)
            ).unsqueeze(0)

            # convert to bool and flip. (other code wants this)
            attention_mask = attention_mask < 0.5

        elif sample_end_masks:
            # Attention mask is a binary representation of whether each output token is eod.

            attention_mask = torch.zeros(seq_length, device=data.device, dtype=torch.bool)
            eod_positions = (in_seq == eod_token).nonzero(as_tuple=False).flatten()
            for i in eod_positions:
                attention_mask[i] = True

            attention_mask = attention_mask.unsqueeze(0)  # I'm hope the rest of the code will stack along this dim.

        else:
            # A reversed lower triangular attention matrix with resets.
            # This is as expected by code outside this function.

            attention_mask = torch.tril(
                torch.ones((seq_length, seq_length), device=data.device)
            ).unsqueeze(0)

            # Use EOD token to zero out cross-document attention
            eod_positions = (in_seq == eod_token).nonzero(as_tuple=False).flatten()
            prev_idx = 0
            for i in eod_positions:
                attention_mask[0, (i + 1):, (prev_idx + 1):(i + 1)] = 0.0  # Remove cross-document attention
                prev_idx = i

            # convert to bool and flip (other code wants this)
            attention_mask = attention_mask < 0.5

    else:
        # Don't create an attention mask at all.

        attention_mask = None

    # Loss mask creation code starts here.
    # In the result, the loss mask should be 1.0 where loss is backpropped and 0.0 where it's not.

    block_loss = torch.zeros_like(out_seq, dtype=torch.bool) # Boolean loss mask. bools [seqlen]

    unclosed_pair = False

    # Perform pair loss masking.
    # Guess I'll just reuse and change up Martin's code for this.
    if len(loss_mask_pairs) > 0:
        if len(loss_mask_pairs) % 2 == 1:
            raise ValueError("--loss-mask-pairs should have an even length, not size.")
        for pair_start in range(0, len(loss_mask_pairs), 2):
            b_id = loss_mask_pairs[pair_start]  # int
            e_id = loss_mask_pairs[pair_start + 1]  # int
            if b_id == e_id:
                cs = (data == b_id).cumsum(dim=0)  # ints [full_seqlen]
                pair_mask = (cs & 1).bool()  # Masked where pair is not closed yet (excluding closer token). bools [full_seqlen]
                pair_mask[1:] = pair_mask[1:] | pair_mask[:-1]  # Mask on pair closing too.
                block_loss|= pair_mask[1:]
                # Checking whether we have a dangling unclosed pair.
                if (cs[-1] & 1).bool():
                    warnings.warn(f"WARNING: Found unclosed pair {(b_id, e_id)} in one or more samples")
                    unclosed_pair = True
            else:
                cs_open = (data == b_id).cumsum(dim=0)  # bool [full_seqlen]
                cs_close = (data == e_id).cumsum(dim=0)  # bool [full_seqlen]
                pair_mask = (cs_open - cs_close) > 0  # Masked where opens are not closed yet. (excluding closer token)
                if pair_mask[-1]:
                    warnings.warn(f"WARNING: Found unclosed pair {(b_id, e_id)} in one or more samples")
                    unclosed_pair = True
                pair_mask[1:] = pair_mask[1:] | pair_mask[:-1]  # Mask on pair closing too.
                block_loss|= pair_mask[1:]

    # Perform individual token masking.
    if len(loss_mask_individual) > 0:
        for id in loss_mask_individual:
            block_loss|= out_seq == id

    # Perform post eod masking.
    # Explainer - Neox code used to mask the loss on the output tokens predicted from EOD as input tokens.
    # And Martin wants this.
    if eod_mask_loss:
        block_loss|= in_seq == eod_token

    # Perform masking of the odd-numbered token recurrances.
    # Martin wants this, ask him or read the neox_args docstring for the variable.
    if len(loss_mask_alternating_individual) > 0:
        for id in loss_mask_alternating_individual:
            eq = data == id  # bools [full_seqlen]
            cs = eq.cumsum(dim=0)  # ints [full_seqlen]
            odd_pos = eq & (cs % 2 == 1)  # Odd positions of occurance in sequence. bools [full_seqlen]
            block_loss|= odd_pos[1:]
            if (cs[-1] % 2 == 1).any():
                warnings.warn(f"WARNING: Found unclosed alternating individual token {id} in one or more samples.")
                unclosed_pair = True

    loss_mask = (~block_loss).to(torch.float)  # Code that uses this expect floats.

    # Samples or packing is malformed.
    if unclosed_pair:
        warnings.warn(
            f"WARNING: sequence(s) contained invalid loss_mask_* tokens. " +
            "Either your samples are too long, you're using the wrong packing method, or your loss_mask_* arguments " +
            "are incorrectly configured."
        )

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)

    return attention_mask, loss_mask, position_ids


class MockGPTLowLevelDataset:
    """The mock GPT low level dataset

    This class is meant to generate tokenized data in the classic "Megatron-LM" GPT style. Notably,
    we add the end of document token to each element indexed in __getitem__

    Args:
        tokenizer (MegatronTokenizerBase): The tokenizer the special token information of which
        we use to augment the mock data.
    """

    seed: int = 0
    """The hard-coded random seed to use to set the NumPy RNG"""

    size: int = 100000
    """The hard-coded number of samples to generate"""

    max_sequence_length: int = 4096
    """The hard-coded max sequence length to generate"""

    def __init__(self, tokenizer: MegatronTokenizer) -> None:
        self.tokenizer = tokenizer
        rng = numpy.random.default_rng(seed=self.seed)
        self.sequence_lengths = rng.integers(
            low=1, high=self.max_sequence_length, size=self.size, dtype=numpy.int32
        )

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> numpy.number:
        length = self.sequence_lengths[idx]
        sample = numpy.int64(
            numpy.concatenate([numpy.arange(length - 1) + 1, [self.tokenizer.eod]])
        )
        return sample

    def get(self, idx: int, offset: int = 0, length: Optional[int] = None) -> numpy.ndarray:
        """This function is n abstraction over __getitem__ with support for slicing

        Args:
            idx (int): The index into the dataset

            offset (int): The integer token offset in the sequence

            length (Optional[int]): The number of tokens to grab from the sequence

        Returns:
            numpy.ndarray: The sequence tokens at the index
        """
        if length is None:
            length = self.sequence_lengths[idx] - offset
        return self[idx][offset : offset + length]


class MockGPTDataset(GPTDataset):
    """The mock GPT dataset

    Args:
        indexed_dataset (MockGPTLowLevelDataset): The MockGPTLowLevelDataset around which to build
            the MockGPTDataset

        dataset_path (Optional[str]): This argument is of no consequence for the MockGPTDataset

        indices (numpy.ndarray): The set of the dataset indices to expose

        num_samples (int): The number of samples to draw from the dataset

        index_split (Split): The indices Split

        config (GPTDatasetConfig): The config
    """

    def __init__(
        self,
        dataset: MockGPTLowLevelDataset,
        dataset_path: Optional[str],
        indices: numpy.ndarray,
        num_samples: int,
        index_split: Split,
        config: GPTDatasetConfig,
    ) -> None:
        assert config.mock

        super().__init__(
            dataset,  # type: ignore[arg-type]
            dataset_path,
            indices,
            num_samples,
            index_split,
            config,
        )

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: MockGPTLowLevelDataset) -> int:
        """Abstract method implementation

        Args:
            low_level_dataset (MockGPTLowLevelDataset): The underlying MockGPTLowLevelDataset

        Returns:
            int: The number of unique elements in the underlying MockGPTLowLevelDataset
        """
        return len(low_level_dataset)

    @staticmethod
    def build_low_level_dataset(  # type: ignore[override]
        dataset_path: Optional[str], config: GPTDatasetConfig
    ) -> MockGPTLowLevelDataset:
        """Abstract method implementation

        Args:
            dataset_path (Optional[str]): This argument is of no consequence for the
                MockGPTLowLevelDataset

            config (GPTDatasetConfig): The config

        Returns:
            MockGPTLowLevelDataset: The underlying MockGPTLowLevelDataset
        """
        return MockGPTLowLevelDataset(config.tokenizer)
