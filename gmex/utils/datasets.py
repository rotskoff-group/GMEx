# gmex/utils/datasets.py
# Contains helper functions and classes for data preprocessing.


import json
import os
from math import ceil
from pathlib import Path
from random import choices
from typing import Final

import numpy as np
import torch
from scipy.interpolate import interp1d
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from .core import coarsen_states

RESULTS_DIR: Final[str] = "results"
METADATA_FN: Final[str] = "metadata.json"
RATES_FN: Final[str] = "rates.pt"
GROUPS_FN: Final[str] = "groups.json"
TIMES_FN: Final[str] = "times.pt"
STATE_TENSOR_FN: Final[str] = "microstates.pt"
STATE_LIST_FN: Final[str] = "macrostates.json"
LIM_DIST_FN: Final[str] = "lim_dist.pt"


class StateSeqDataset(Dataset):
    data: torch.Tensor | list[torch.Tensor]
    n_transitions_raw: int
    n_transitions_resampled: int

    def __init__(
        self,
        timestep: float,
        context_length: int,
        times: Tensor | list | None = None,
        states: Tensor | list | None = None,
        data: Tensor | list | None = None,
    ) -> None:
        """Dataset for sequence of states.
        Either data xor both times and states must be passed.

        Parameters
        ----------
        timestep : float
            Timestep at which records are sampled.
        context_length : int
            Number of consecutive states to process at a time.
        states : torch.Tensor | None
            Observed states.
        times : torch.Tensor | None
            Observation times.
        data : torch.Tensor | list | None
            Record or list of records.
        """
        self.dt = timestep
        if context_length < 1:
            raise ValueError("context_length must be a positive integer")
        self.context_length = context_length
        if data is not None:
            if not isinstance(data, list):
                raise ValueError(f"data must be a list, got type {type(data)}")
            if times is not None or states is not None:
                raise ValueError("If data is passed, times and states must be None.")
            self.resampled = False
        elif times is not None and states is not None:
            if not isinstance(times, Tensor):
                raise ValueError(
                    f"times must be a torch.Tensor, got type {type(times)}"
                )
            if not isinstance(states, Tensor):
                raise ValueError(
                    f"states must be a torch.Tensor, got type {type(states)}"
                )
            if states.shape != times.shape:
                raise ValueError(
                    f"Got {tuple(states.shape)} states and {tuple(times.shape)} times, "
                    "but states and times must have same shape."
                )
            self.resampled = True
        else:
            raise ValueError("Either data xor both times and states must be passed.")

    def __len__(self) -> int:
        """Return number of contexts in dataset."""
        raise NotImplementedError("Method not implemented by base class.")

    def _get_metadata(self) -> None:
        """Get metadata."""
        # get length of longest record
        if isinstance(self.data, list):
            self.max_rec_len = max([rec.shape[0] for rec in self.data])
            self.n_records = len(self.data)
        else:
            self.max_rec_len = len(self.data)
            self.n_records = 1

        # get total observations
        records = self.data if isinstance(self.data, list) else [self.data]
        concatenated_records = torch.cat(records).to(torch.long).cpu().numpy()
        unique_states, total_cts = np.unique(concatenated_records, return_counts=True)
        del concatenated_records
        observed_states = set(unique_states.tolist())
        self.n_states = max(observed_states) + 1
        expected_states = set(range(self.n_states))
        missing_states = expected_states - observed_states
        if missing_states:
            raise ValueError(f"Missing states: {sorted(missing_states)}.")
        self.n_obs_by_state = total_cts.astype(int).tolist()
        self.n_obs = sum(self.n_obs_by_state)
        self.lim_dist = [n / self.n_obs for n in self.n_obs_by_state]

        # get fraction of contexts containing only one state
        self.n_contexts = len(self)
        if self.context_length == 1:
            self.n_transitionless_contexts = self.n_contexts
        else:
            self.n_transitionless_contexts = 0
            for idx in range(self.n_contexts):
                x, _ = self[idx]  # get context
                self.n_transitionless_contexts += torch.all(x == x[0])
        self.n_transitionless_contexts = int(self.n_transitionless_contexts)
        self.frac_transitionless = float(
            self.n_transitionless_contexts / self.n_contexts
        )

        # get fraction of transitions lost to resampling
        if self.resampled == True and self.n_transitions_raw != 0:
            self.frac_transitions_lost = (
                1.0 - self.n_transitions_resampled / self.n_transitions_raw
            )
        else:
            self.frac_transitions_lost = 0.0

        # assign and return metadata dictionary
        self.metadata = {
            "n_states": self.n_states,
            "n_records": self.n_records,
            "max_rec_len": self.max_rec_len,
            "n_obs": self.n_obs,
            "n_obs_by_state": self.n_obs_by_state,
            "lim_dist": self.lim_dist,
            "resampled": self.resampled,
            "n_contexts": self.n_contexts,
            "n_transitionless_contexts": self.n_transitionless_contexts,
            "frac_transitionless": self.frac_transitionless,
            "frac_transitions_lost": self.frac_transitions_lost,
        }

    @torch.no_grad()
    def get_count_matrices(
        self,
        sample_interval: int = 1,
        max_lag: int | None = None,
        bootstrap: bool = False,
        verbose: bool = True,
    ) -> torch.Tensor:
        """Assemble count matrices.

        Parameters
        ----------
        sample_interval : int, optional
            Sampled lags are integer multiples of sampling_interval steps.
        max_lag : int, optional
            Maximum number of lags to sample excluding zeroth, which is always sampled.
        bootstrap : bool, optional
            If True, performs a random bootstrap resampling of records.
            This makes no difference if data are a single record.
        verbose : bool, optional
            Whether to display progress bar.

        Returns
        -------
        Cs : torch.Tensor
            Count matrices. Channels for lags (actually lag / sample_interval),
            columns for initial states, rows for states after lag.
        """
        if sample_interval <= 0:
            raise ValueError("sample_interval must be a positive integer")
        if max_lag is not None and max_lag < 0:
            raise ValueError("max_lag must be a nonnegative integer")

        # create buffer
        max_save = ceil(self.max_rec_len / sample_interval)
        if max_lag is not None:
            max_save = min(max_save, max_lag + 1)
        Cs = torch.zeros((max_save, self.n_states, self.n_states), dtype=torch.float64)

        # prepare records as list of 1D long tensors on CPU for vectorized counting
        records = self.data if isinstance(self.data, list) else [self.data]
        records = [rec.to(torch.long).cpu() for rec in records]
        if bootstrap:
            records = choices(records, k=len(records))

        # For each lag, compute transition-count matrix using bincount on flattened indices.
        for i in tqdm(range(max_save), desc="Lags", disable=not verbose, leave=False):
            lag_steps = i * sample_interval  # get lag timesteps from index
            counts_acc = torch.zeros(
                self.n_states**2, dtype=torch.long
            )  # accumulated counts

            for rec in records:
                if lag_steps >= len(rec):
                    continue  # record too short to contribute
                if lag_steps == 0:
                    b = rec  # before states
                else:
                    b = rec[:-lag_steps]
                a = rec[lag_steps:]  # after states
                # flatten pair indices: idx = before * n_states + after
                # notice that b[t] * n_states + a[t] uniquely codes every (before, after) pair
                idx = (b * self.n_states + a).to(torch.long)
                counts_rec = torch.bincount(idx, minlength=self.n_states**2)
                counts_acc += counts_rec

            # reshape and transpose for column-stochasticity of transfer operators
            counts_mat = counts_acc.reshape(self.n_states, self.n_states).T
            # convert to float and write into buffer
            Cs[i] += counts_mat.to(dtype=Cs.dtype)

        return Cs


class MultiSeqDataset(StateSeqDataset):
    """Dataset from multiple sequences of states.

    Parameters
    ----------
    timestep : float
        Resampling timestep.
    context_length : int, optional
        Number of consecutive states to process at a time.
    times : torch.Tensor | None, optional
        2D tensor of observation times.
        Either data xor both times and states must be provided.
    states : torch.Tensor | None, optional
        2D tensor of observed states.
        Either data xor both times and states must be provided.
    data : list[list[int]] | None, optional
        list of lists of observed states.
        Either data xor both times and states must be provided.
    verbose: optional, bool
        Whether to print metadata.
    """

    data: list[torch.Tensor]

    def __init__(
        self,
        timestep: float,
        context_length: int,
        times: torch.Tensor | None = None,
        states: torch.Tensor | None = None,
        data: list[list[int]] | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            timestep, context_length, times=times, states=states, data=data
        )

        if data is not None:
            self.data = [
                torch.tensor(rec, dtype=torch.long)
                for rec in data
                if len(rec) > self.context_length
            ]
        elif times is not None and states is not None:
            self.data = []
            per_rec_transitions_raw = []
            per_rec_transitions_resampled = []
            # resample all simulations at fixed timestep
            for i in range(times.shape[0]):
                _, states_resampled, n_transitions = fixed_timestep_resample(
                    times[i], states[i], self.dt
                )
                if len(states_resampled) > self.context_length:
                    self.data += [states_resampled]
                    per_rec_transitions_raw += [n_transitions[0]]
                    per_rec_transitions_resampled += [n_transitions[1]]
                self.n_transitions_raw = sum(per_rec_transitions_raw)
                self.n_transitions_resampled = sum(per_rec_transitions_resampled)
        if len(self.data) < 1:
            raise ValueError(
                f"No records longer than context_length {self.context_length}."
            )

        # get metadata
        self.n_iters_cumsum = (
            torch.Tensor([len(s) - self.context_length for s in self.data])
            .to(torch.long)
            .cumsum(dim=0)
        )
        self._get_metadata()
        if verbose:
            print(self.metadata)

    def __len__(self) -> int:
        return int(self.n_iters_cumsum[-1])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get context and target.

        Parameters
        ----------
        index : int
            Index of context.

        Returns
        -------
        x : torch.Tensor
            Context of shape (context_length,) containing states.
        y : torch.Tensor
            Target of shape (context_length,) containing states.
        """
        # find which record to slice
        i = torch.searchsorted(self.n_iters_cumsum, index + 1)
        # get start index of context slice
        j = index - (self.n_iters_cumsum[i - 1] if i > 0 else 0)

        # slice record to get context and target
        x = self.data[i][j : j + self.context_length]
        y = self.data[i][j + 1 : j + self.context_length + 1]
        return x, y


class SingleSeqDataset(StateSeqDataset):
    """Dataset for a single long sequence of states.

    Parameters
    ----------
    timestep : float
        Resampling timestep.
    context_length : int, optional
        Number of consecutive states to process at a time.
    times : torch.Tensor | None, optional
        1D tensor of observation times.
        Either data xor both times and states must be provided.
    states : torch.Tensor | None, optional
        1D tensor of observed states.
        Either data xor both times and states must be provided.
    data : list[int] | None, optional
        list of observed states.
        Either data xor both times and states must be provided.
    verbose : bool, optional
        Whether to print metadata.
    """

    data: torch.Tensor

    def __init__(
        self,
        timestep: float,
        context_length: int,
        times: Tensor | None = None,
        states: Tensor | None = None,
        data: list[int] | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            timestep, context_length, data=data, times=times, states=states
        )

        if data is not None:
            self.data = torch.tensor(data, dtype=torch.long)
        elif times is not None and states is not None:
            _, self.data, n_transitions = fixed_timestep_resample(
                times, states, self.dt
            )
            self.n_transitions_raw = n_transitions[0]
            self.n_transitions_resampled = n_transitions[1]
        if len(self.data) <= self.context_length:
            raise ValueError(
                f"Record of length {len(self.data)} shorter than context length {self.context_length}."
            )

        self._get_metadata()
        if verbose:
            print(self.metadata)

    def __len__(self) -> int:
        return len(self.data) - self.context_length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get context and target.

        Returns
        -------
        x : torch.Tensor
            Context of shape (context_length,) containing states.
        y : torch.Tensor
            Target of shape (context_length,) containing states.
        """
        x = self.data[index : index + self.context_length]
        y = self.data[index + 1 : index + self.context_length + 1]
        return x, y


### HELPERS ###


@torch.no_grad()
def get_split(
    timestep: float,
    context_length: int,
    times: torch.Tensor | None = None,
    states: torch.Tensor | None = None,
    data: list[int] | list[list[int]] | None = None,
    train_frac: float = 0.8,
    verbose: bool = False,
) -> StateSeqDataset | tuple[StateSeqDataset, StateSeqDataset]:
    """Split data into training and validation datasets.

    Parameters
    ----------
    timestep : float
        Resampling timestep.
    context_length : int, optional
        Number of consecutive states to process at a time.
    times : torch.Tensor | None, optional
        1D or 2D tensor of observation times.
        Either data xor both times and states must be provided.
    states : torch.Tensor | None, optional
        1D or 2D tensor of observed states.
        Either data xor both times and states must be provided.
    data : list[int | list] | None, optional
        list of (lists of) observed states.
        Either data xor both times and states must be provided.
    train_frac : float, optional
        Fraction of data to use for training, default 0.8.
    verbose : bool, optional
        Whether to print metadata.

    Returns
    -------
    StateSeqDataset | tuple[StateSeqDataset, StateSeqDataset]
        Dataset, or (training dataset, validation dataset) if train_frac < 1.
    """
    if data is not None and (times is not None or states is not None):
        raise ValueError("Either data xor both times and states must be provided.")

    ds1 = lambda dt, cl, t, s, d, v: SingleSeqDataset(
        dt, cl, times=t, states=s, data=d, verbose=v
    )
    ds2 = lambda dt, cl, t, s, d, v: MultiSeqDataset(
        dt, cl, times=t, states=s, data=d, verbose=v
    )
    if data is not None:
        if type(data[0]) == int:
            ds = ds1
        elif type(data[0]) == list:
            ds = ds2
        else:
            raise ValueError(
                f"data must be a list of int or list, got a list of {type(data[0])}"
            )
    elif times is not None and states is not None:
        if states.ndim == 1:
            ds = ds1
        elif states.ndim == 2:
            ds = ds2
        else:
            raise ValueError(f"states must be 1D or 2D, got {states.ndim}D")
    else:
        raise ValueError("Either data xor both times and states must be provided.")

    if train_frac == 1:
        return ds(timestep, context_length, times, states, data, verbose)
    elif train_frac > 0 and train_frac < 1:
        if data is not None:
            ntrain = int(train_frac * len(data))
            if verbose:
                print("Training data:")
            train_data = ds(
                timestep, context_length, None, None, data[:ntrain], verbose
            )
            if verbose:
                print("Validation data:")
            valid_data = ds(
                timestep, train_data.context_length, None, None, data[ntrain:], verbose
            )
        elif times is not None and states is not None:
            ntrain = int(train_frac * states.shape[0])
            if verbose:
                print("Training data:")
            train_data = ds(
                timestep, context_length, times[:ntrain], states[:ntrain], None, verbose
            )
            if verbose:
                print("Validation data:")
            valid_data = ds(
                timestep,
                train_data.context_length,
                times[ntrain:],
                states[ntrain:],
                None,
                verbose,
            )
        return train_data, valid_data
    else:
        raise ValueError("train_frac must be in range (0, 1]")


@torch.no_grad()
def fixed_timestep_resample(
    times: torch.Tensor,
    states: torch.Tensor,
    timestep: float,
    return_transition_cts: bool = True,
) -> tuple:
    """From times and states sampled at possibly irregular intervals,
    produce a fixed timestep trajectory on the timestep grid from 0
    through the largest multiple of timestep not exceeding the final time.
    If the final time is on-grid within floating-point tolerance, include it.

    Parameters
    ----------
    times : torch.Tensor
        Series of jump times of length 1 greater than number of steps to include 0.
    states : torch.Tensor
        Series of states of length 1 greater than number of steps to include initial state.
    timestep : float
        Resampling timestep.
    return_transition_cts : bool, optional
        Whether to return raw and resampled transition counts.

    Returns
    -------
    new_times : torch.Tensor
        Resampled times.
    new_states : torch.Tensor
        Resampled states.
    n_transitions : tuple
        Raw and resampled transition counts.
    """
    # make sure states and times match
    assert len(states) == len(times), "states and times must have the same length!"

    times_np = times.cpu().numpy()
    states_np = states.cpu().numpy()

    # perform interpolation
    interpolated_trajectory = interp1d(
        times_np,
        states_np,
        kind="previous",
        bounds_error=False,
        fill_value="extrapolate",
    )

    # include the final sample when it lies on the timestep grid up to floating-point tolerance
    ratio = float(times_np[-1]) / float(timestep)
    nearest = round(ratio)
    tol = 10 * np.finfo(np.float64).eps * max(1.0, abs(ratio))
    if np.isclose(ratio, nearest, rtol=0.0, atol=tol):
        last_step = int(nearest)
    else:
        last_step = int(np.floor(ratio))
    new_times = float(timestep) * torch.arange(
        last_step + 1, dtype=torch.float64, device=times.device
    )
    new_states = torch.tensor(
        interpolated_trajectory(new_times.cpu().numpy()), dtype=torch.int64
    )

    n_transitions_raw = (states[:-1] != states[1:]).sum().item()
    n_transitions_resampled = (new_states[:-1] != new_states[1:]).sum().item()

    if return_transition_cts:
        return new_times, new_states, (n_transitions_raw, n_transitions_resampled)
    return new_times, new_states


def save_metadata(argvars: dict, outfile: Path):
    """Save metadata or hyperparameters.

    Parameters
    ----------
    argvars : dict
        Arguments copied to dictionary.
    outfile : pathlib.Path
        Directory in which to save metadata.
    """
    for k in ("_run", "_section", "config", "cmd"):
        argvars.pop(k, None)
    with open(outfile, "w") as f:
        json.dump(argvars, f, default=str, indent=2)


def get_data_dir() -> Path:
    """Get a writable data directory.

    Returns
    -------
    datadir : pathlib.Path
        Absolute path to the data directory. Defaults to CWD/'data',
        overridable with the environment variable 'GMEX_DATA_DIR'.
    """
    return Path(os.getenv("GMEX_DATA_DIR", Path.cwd() / "data")).resolve()


def get_results_dir() -> Path:
    """Get a writable results directory.

    Returns
    -------
    resdir : pathlib.Path
        Absolute path to the results directory. Defaults to CWD/'results',
        overridable with the environment variable 'GMEX_RESULTS_DIR'.
    """
    return Path(os.getenv("GMEX_RESULTS_DIR", Path.cwd() / "results")).resolve()


def load_times_macrostates(ds: str | Path, device: str | torch.device) -> tuple:
    """Load times and macrostates from saved dataset.

    Parameter
    ---------
    ds : pathlib.Path
        Subdirectory of data in which dataset is saved.
    device : str | torch.device
        Device on which to load tensors.

    Returns
    -------
    dt : float | None
        Fixed observation timestep.
    times : torch.Tensor | None
        Observed times.
    macrostates : torch.Tensor | None
        Observed macrostates.
    data : list | None
        Observed macrostates.
    n_macrostates : int
        Number of distinct observed macrostates.
    """
    ds = Path(ds)
    datadir = ds if Path(ds).is_absolute() else get_data_dir() / ds
    datafns = os.listdir(datadir)

    with open(datadir / METADATA_FN, "r") as f:
        dt = json.load(f)["dt"]  # this can be None

    if GROUPS_FN in datafns:
        with open(datadir / GROUPS_FN, "r") as f:
            groups = json.load(f)

    if STATE_LIST_FN in datafns:
        with open(datadir / STATE_LIST_FN, "r") as f:
            data = json.load(f)
        if GROUPS_FN in datafns:
            data = coarsen_states(data, groups)
        times, macrostates = None, None
        if isinstance(data[0], list):
            observed_macrostates = {x for sublist in data for x in sublist}
        elif isinstance(data[0], int):
            observed_macrostates = set(data)
        else:
            raise ValueError(
                f"data must be a list of int or list, got list of {type(data[0])}"
            )
        expected_macrostates = {i for i in range(max(observed_macrostates) + 1)}
    elif STATE_TENSOR_FN in datafns:
        microstates = torch.load(datadir / STATE_TENSOR_FN, weights_only=True).to(
            device
        )
        if GROUPS_FN in datafns:
            macrostates = coarsen_states(microstates, groups)
        else:
            macrostates = microstates
        times = torch.load(datadir / TIMES_FN, weights_only=True).to(device)
        data = None
        unique = torch.unique(macrostates)
        expected_macrostates = set(torch.arange(unique.max() + 1).tolist())
        observed_macrostates = set(unique.tolist())
    else:
        raise ValueError(
            f"Dataset {datadir} must contain {STATE_LIST_FN} xor {STATE_TENSOR_FN} and {TIMES_FN}."
        )

    assert expected_macrostates == observed_macrostates, (
        f"Expected macrostates {expected_macrostates}, but missing {expected_macrostates - observed_macrostates}."
    )

    return dt, times, macrostates, data, len(expected_macrostates)
