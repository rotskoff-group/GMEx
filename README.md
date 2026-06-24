# GMEx
Code associated with C.-W.J. Liu, J. Klinger, and G.M. Rotskoff, "Optimal parameterization of nonequilibrium generalized master equations from discrete-time experimental data."

## Setup

Install `uv`:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone repository:
```bash
git clone https://github.com/rotskoff-group/GMEx.git GMEx
```

Enter repository and sync Python build:
```bash
cd GMEx # or wherever you have cloned the project directory
uv sync
```

This creates a virtual environment in the directory `GMEx/.venv/` that has the required Python build.
You can run `uv python` or activate the environment from the command line using
```bash
source .venv/bin/activate
```

Check that `git-lfs` is installed:
```bash
git lfs version # this should print something like `git-lfs/3.x.x`
```

If it is not, install `git-lfs` on Linux:
```bash
sudo apt update
sudo apt install git-lfs
```

This repository is untested on other operating systems.

Finally, set data and results directories (`"$GMEX_DATA_DIR"` and `"$GMEX_RESULTS_DIR"`):
```bash
export GMEX_DATA_DIR="$PWD/data" # or something else
export GMEX_RESULTS_DIR="$PWD/results" # or something else
mkdir -p "$GMEX_DATA_DIR" "$GMEX_RESULTS_DIR"
```

## Usage
Run the following scripts to reproduce results figures without ribbon models:
```bash
cd GMEx # or wherever the project directory is
chmod +x bin/toys.sh; bash bin/toys.sh # takes about 8 hrs
chmod +x bin/CFTR.sh; bash bin/CFTR.sh # takes about 6 hrs
chmod +x bin/BMF1.sh; bash bin/BMF1.sh # takes about 16 hrs
chmod +x bin/HP35.sh; bash bin/HP35.sh # takes about 3 days
```

Toy data will be generated and protein data will be downloaded from other repositories.
Results and figures will be saved in `"$GMEX_RESULTS_DIR"`.
