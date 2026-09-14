# tests/fixture_types.py
# typed dictionaries for fixture annotation


from typing import TypedDict

import torch

from gmex.utils.datasets import MultiSeqDataset


class MarkovCountData(TypedDict):
    """Data from a homogeneous Markov chain."""

    transition_matrix: torch.Tensor
    dataset: MultiSeqDataset
    count_matrices: torch.Tensor


class InhomogeneousMarkovCountData(TypedDict):
    """Data from an inhomogeneous Markov chain."""

    transition_stack: torch.Tensor
    dataset: MultiSeqDataset
    count_matrices: torch.Tensor
