"""scripts/preprocess/CFTR.py

Preprocess CFTR FRET traces from
J. Levring, D.S. Terry, Z. Kilic, G. Fitzgerald, S.C. Blanchard, and J. Chen,
CFTR function, pathology, and pharmacology at single-molecule resolution,
Nature 616, 606 (2023).
Traces are coarse-grained to the paper's idealized efficiencies.
This script takes about 5 min to run.
"""

import json
import os
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from gmex.utils import METADATA_FN, STATE_LIST_FN, get_data_dir, save_metadata

### CONFIGURATIONS ###
WT_IDL_REF_LEVELS = [0.0, 0.25, 0.49]  # idealized efficiencies found in WT
G551D_IDL_REF_LEVELS = [0.0, 0.25, 0.37, 0.49]  # idealized efficiencies found in G551D
L927P_IDL_REF_LEVELS = [0.0, 0.25, 0.31, 0.49]  # idealized efficiencies found in L927P
IDL_FMT = r"^Idl_(\d+)$"  # format for idealization columns
REF_FMT = r"^Ref_(\d+)$"  # format for reference-efficiency columns
STATE_FMT = r"^State_(\d+)$"  # format for state-assignment column
FRET_FMT = r"^Fret_(\d+)$"
IDL_FMT_FN = lambda suffix: f"Idl_{suffix}"  # get format for idealization columns
REF_FMT_FN = lambda suffix: (
    f"Ref_{suffix}"
)  # get format for reference-efficiency columns
STATE_FMT_FN = lambda suffix: (
    f"State_{suffix}"
)  # get format for state-assignment columns
FRET_FMT_FN = lambda suffix: (
    f"Fret_{suffix}"
)  # get format for raw FRET-efficiency columns
DONOR_FMT_FN = lambda suffix: (
    f"Donor_{suffix}"
)  # get format for donor-fluorescence columns
ACCEPTOR_FMT_FN = lambda suffix: (
    f"Acceptor_{suffix}"
)  # get format for acceptor-fluorescence columns
DT = 0.1  # FRET traces are sampled at 10 Hz

OUTDIR = get_data_dir() / "CFTR"
INDIR = OUTDIR / "raw"
EXPT_FMT = r"^expt_(\d+)\.txt$"
EXPT_FMT_FN = lambda suffix: f"expt_{suffix}.txt"
TRACE_FMT_FN = lambda suffix: f"expt_{suffix}.csv"
METADATA: dict[str, None | int | float | str] = {
    "outdir": None,
    "seed": None,
    "nmicro": None,
    "nmacro": None,
    "ntraj": None,
    "nsteps": None,
    "dt": DT,
    "rates": None,
    "probs": None,
    "type": "Markov chain",
    "comment": "CWJL: I filled this in manually. Timestep dt is in units of s.",
}
EXPTS = {
    "WT_3mM_ATP": {"levels": WT_IDL_REF_LEVELS},
    "WT_3mM_ATP_10uM_GLPG1837": {"levels": WT_IDL_REF_LEVELS},
    "G551D_3mM_ATP": {"levels": G551D_IDL_REF_LEVELS},
    "G551D_3mM_ATP_10uM_GLPG1837": {"levels": G551D_IDL_REF_LEVELS},
    "L927P_3mM_ATP": {"levels": L927P_IDL_REF_LEVELS},
    "L927P_3mM_ATP_10uM_GLPG1837": {"levels": L927P_IDL_REF_LEVELS},
}


### HELPERS ###
def _load_CFTR_FRET_raw(fn: str | Path, levels: list[float]) -> pd.DataFrame:
    """Load and validate CFTR FRET efficiencies and idealizations.

    Parameters
    ----------
    fn : str | pathlib.Path
        Path to CFTR FRET TXT file.
    levels : list
        Levels to which to match idealized efficiencies.

    Returns
    -------
    df : pd.DataFrame
        Validated CFTR FRET efficiencies and idealizations.
    """
    fn = Path(fn)
    if not fn.is_absolute():
        fn = get_data_dir() / fn

    df = pd.read_csv(fn, sep="\t", index_col=False)
    pattern = re.compile(IDL_FMT)
    idl_matches = [
        (col, match) for col in df.columns if (match := pattern.match(col)) is not None
    ]
    if not idl_matches:
        raise ValueError(f"No Idl_* columns found in {fn!s}.")

    retained_cols = ["Time (s)"]
    for idl_col, match in idl_matches:
        suffix = match.group(1)
        fret_col = FRET_FMT_FN(suffix)
        donor_col = DONOR_FMT_FN(suffix)
        acceptor_col = ACCEPTOR_FMT_FN(suffix)
        missing_cols = [
            col for col in (fret_col, donor_col, acceptor_col) if col not in df.columns
        ]
        if missing_cols:
            missing_str = ", ".join(missing_cols)
            raise ValueError(f"{idl_col} is missing matching columns: {missing_str}.")
        cols_to_mask = [fret_col, idl_col]
        retained_cols.extend(cols_to_mask)

    df = df[retained_cols]

    max_levels = len(levels)
    degen_cols = [
        col
        for col in df.filter(regex=IDL_FMT).columns
        if df[col].nunique(dropna=True) > max_levels
    ]
    if degen_cols:
        raise ValueError(
            f"Some traces have more unique idealized FRET efficiences than the "
            f"{len(levels)} levels passed: {degen_cols}"
        )

    return df


def _match_CFTR_FRET_levels(col: pd.Series, levels: list[float]) -> pd.Series:
    """Match idealized CFTR FRET efficiencies to reference levels.

    Parameters
    ----------
    col : pd.Series
        Idealized CFTR FRET efficiencies of a single trace indexed by timestep.
    levels : list
        Levels to which to match idealized efficiencies.

    Returns
    -------
    out : pd.Series
        Idealized CFTR FRET efficiencies of col mapped to reference levels.
    """
    pattern = re.compile(IDL_FMT)
    if (
        not isinstance(col.name, str)
        or (colname_match := pattern.match(col.name)) is None
    ):
        raise ValueError(f"{col.name!r} does not match {IDL_FMT_FN('*')}.")
    suffix = colname_match.group(1)

    refs = np.array(levels)
    refs.sort()
    if refs[0] != 0.0:
        raise ValueError(
            f"Reference-value global variables should have 0.0 as least value, got {refs[0]}."
        )
    obs = np.unique(col)
    obs.sort()  # these should be sorted already but let's be safe

    if np.any(obs > 0.8):  # this doesn't seem to occur
        raise ValueError(
            "Trace has idealized efficiencies greater than 0.8, suggesting denaturation."
        )
    if len(obs) == 1:  # immediately assume initial photobleaching
        return pd.Series(0.0, index=col.index, name=f"{REF_FMT_FN(suffix)}")
    if len(obs) > len(refs):
        raise ValueError(
            f"Trace has {len(obs)} unique idealized efficiences, "
            f"more than the {len(refs)} passed."
        )
    if np.all(obs < 0.0):
        raise ValueError("Trace cannot have only nonpositive idealized efficiencies.")

    obs_match = np.empty_like(obs)
    obs_match[:] = np.nan  # use nan for initial match buffer
    obs_match[obs <= col.iloc[-1]] = (
        0.0  # efficiencies less than or equal to the last assumed photobleached
    )
    obs_match[obs < 0.05] = (
        0.0  # remaining efficiencies in z-test rejection region assumed photobleached
    )
    unmatched_idcs = np.isnan(obs_match)  # the positions where obs_match is still nan
    if not unmatched_idcs.any():  # immediately assume initial photobleaching
        return pd.Series(0.0, index=col.index, name=f"{REF_FMT_FN(suffix)}")

    # remaining efficiences assumed photoactive
    # candidate assignments are ordered combinations of unique nonzero reference values
    # since noise is i.i.d. Gaussian, assignment with minimal Euclidean distance is likeliest
    candidates = np.array(list(combinations(refs[1:], sum(unmatched_idcs))))
    squared_distances = np.sum((candidates - obs[unmatched_idcs]) ** 2, axis=1)
    obs_match[unmatched_idcs] = candidates[np.argmin(squared_distances)]

    mapper = dict(zip(obs, [float(i) for i in obs_match]))
    return col.copy().map(mapper).round(2).rename(REF_FMT_FN(suffix))


def _repair_CFTR_FRET_blinks(col: pd.Series) -> pd.Series:
    """Compact leading zeros and backfill internal FRET blinks.

    Parameter
    ---------
    col : pd.Series
        Idealized CFTR FRET efficiencies set to reference values.

    Returns
    -------
    out : pd.Series
        Same object but repaired.
    """
    pattern = re.compile(REF_FMT)
    if not isinstance(col.name, str) or (_ := pattern.match(col.name)) is None:
        raise ValueError(f"{col.name!r} does not match {REF_FMT_FN('*')}.")

    nonzero_idcs = np.flatnonzero(col.to_numpy(copy=False) != 0)
    if len(nonzero_idcs) == 0:
        return col.copy()

    # leading zeros are compacted away
    # remaining internal zero runs are treated as blinks;
    # they inherit the next observed positive efficiency
    active = col.iloc[nonzero_idcs[0] :]
    active_repaired = active.mask(active.eq(0), pd.NA).bfill().fillna(0.0)

    repaired = pd.Series(0.0, index=col.index, name=col.name)
    repaired.iloc[: len(active_repaired)] = active_repaired.to_numpy()
    return repaired.round(2).astype(col.dtype, copy=False).rename(col.name)


def _assign_CFTR_FRET_states(col: pd.Series, levels: list[float]) -> pd.Series:
    """Assign states to CFTR FRET traces after mapping to reference efficiencies.

    Parameters
    ----------
    col : pd.Series
        Reference-mapped CFTR FRET efficiencies of a single trace indexed by timestep.
    levels : list
        Levels to which to match idealized efficiencies.

    Returns
    -------
    out : pd.Series
        Integer state assignments of CFTR FRET efficiencies in col.
    """
    pattern = re.compile(REF_FMT)
    if (
        not isinstance(col.name, str)
        or (colname_match := pattern.match(col.name)) is None
    ):
        raise ValueError(f"{col.name!r} does not match {REF_FMT_FN('*')}.")
    suffix = colname_match.group(1)

    refs = np.array(levels)
    refs.sort()
    refs_set = set(refs.tolist())
    obs = np.unique(col)
    obs.sort()  # again, both this and refs should already be sorted, but let's be safe
    obs_set = set(obs.tolist())
    if not obs_set.issubset(refs_set):
        raise ValueError(
            f"Column {col.name} has efficiencies {obs_set - refs_set} not in the reference set."
        )

    mapper = {ref: pd.NA if i == 0 else i - 1 for i, ref in enumerate(refs.tolist())}
    return col.copy().map(mapper).rename(STATE_FMT_FN(suffix))


def _get_processed_CFTR_FRET_df(fn: str | Path, levels: list[float]) -> pd.DataFrame:
    """Load CFTR FRET traces and get preprocessed DataFrame.

    Parameters
    ----------
    fn : str | pathlib.Path
        Path to CFTR FRET TXT file.
    levels : list
        Levels to which to match idealized efficiencies.

    Returns
    -------
    df : pd.DataFrame
        FRET traces with raw efficiencies, idealizations,
        mappings to reference efficiences, and integer state assignments.
    """
    df = _load_CFTR_FRET_raw(fn, levels)  # load and validate raw file

    # get idealization columns and map to reference efficiencies
    idl_colnames = [colname for colname in df.filter(regex=IDL_FMT).columns]
    derived_cols = []
    for colname in idl_colnames:
        refcol = _match_CFTR_FRET_levels(df[colname], levels)
        repaired_ref = _repair_CFTR_FRET_blinks(refcol)
        statecol = _assign_CFTR_FRET_states(repaired_ref, levels)
        derived_cols.extend([repaired_ref, statecol])

    derived_df = (
        pd.concat(derived_cols, axis=1)
        if derived_cols
        else pd.DataFrame(index=df.index)
    )
    return pd.concat([df.filter(regex=FRET_FMT), derived_df], axis=1)


def _extract_CFTR_FRET_df(df: pd.DataFrame) -> list[list[int]]:
    """Extract macrostate records from processed CFTR FRET DataFrame.

    Parameter
    ---------
    df : pd.DataFrame
        Output of _get_processed_CFTR_FRET_df

    Returns
    -------
    macrostates : list[list[int]]
        list of macrostate records from CFTR FRET experiment.
    """
    colnames = df.filter(regex=STATE_FMT).columns
    if len(colnames) == 0:
        raise ValueError(f"df lacks {STATE_FMT_FN('*')} columns")

    macrostates = []
    for colname in colnames:
        s = df[colname]
        na = s.isna()
        still_has_blinks = na.cummax().ne(na).any()
        if still_has_blinks:
            raise ValueError(f"Column {colname} still has blinks.")

        rec = s.dropna().astype(int).tolist()
        if len(rec) > 1:
            macrostates.append(rec)

    return macrostates


if __name__ == "__main__":
    pattern = re.compile(EXPT_FMT)
    for expt in EXPTS:
        print(f"Saving {expt.replace('_', ' ')} CFTR data...")
        indir = INDIR / expt
        outdir = OUTDIR / expt
        outdir.mkdir(parents=True, exist_ok=True)
        levels = EXPTS[expt]["levels"]

        macrostates = []
        for fn in os.listdir(indir):
            match = pattern.match(fn)
            if match is None:
                continue
            else:
                df = _get_processed_CFTR_FRET_df(indir / fn, levels)
                df.to_csv(outdir / TRACE_FMT_FN(match.group(1)), index=False)
                macrostates += _extract_CFTR_FRET_df(df)

        with open(outdir / STATE_LIST_FN, "w") as f:
            json.dump(macrostates, f)

        metadata = dict(METADATA)
        metadata["outdir"] = str(outdir)
        metadata["nmacro"] = len(levels)
        metadata["ntraj"] = len(macrostates)
        save_metadata(metadata, outdir / METADATA_FN)

        print(f"{expt.replace('_', ' ')} CFTR data saved to {outdir!s}.")
