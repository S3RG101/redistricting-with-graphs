# Preventing Gerrymandering with Community-Similarity Clustered Graphs

CS 4701 project by Davey Seeman (djs467), Jiasheng Shi (js3672), and Sergio Lopez (sal292).

We build a similarity graph of individuals from ACS-PUMS census microdata and
cluster it into district-sized communities with different dimensionality reduction and clustering algorithms.

## Project layout

```
.
├── check_environment.py   # verifies the libraries work
├── requirements.txt       # exact versions for Python packages
├── src/
│   ├── __init__.py
│   └── pums_loader.py     # downloads + loads ACS-PUMS person records
└── data/                  # created by the data loader; NOT tracked by git
    └── raw/pums/2024_5-Year/
```

## 1. Set up the Python environment (one time)

Requires **Python 3.12**.

**Windows (PowerShell):**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell refuses to run `Activate.ps1`, run this once and retry:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**macOS / Linux:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Then check that everything works:

```
python check_environment.py
```

> If pip fails with `CERTIFICATE_VERIFY_FAILED` (common behind antivirus or campus
> proxies), add `--use-feature=truststore` to the pip commands so pip uses your
> operating system's certificate store.

## 2. Load the ACS-PUMS data

We use the **2024 ACS 5-Year PUMS person records** for **New York**.
The first run downloads about 140 MB from census.gov into `data/raw/pums/`
and extracts the CSV; later runs reuse the local files.

```
python -m src.pums_loader --state NY
```

Options: `--period 1-Year` (smaller sample), `--all-columns` (load all ~290
columns instead of the default subset), `--state CA` (any other state).

From Python / a notebook:

```python
from src.pums_loader import load_pums_persons

people = load_pums_persons(state="NY")   # pandas DataFrame, one row per person
```

Notes on the data:

- `PWGTP` is the person weight: how many real people each row represents.
- `PUMA` (with `STATE`) is the finest geography available (~100k+ people each).
- Code columns (e.g. `OCCP`, `SCHL`, `LANP`) are text codes stored as pandas `category`
  (leading zeros kept, ~170 MB for NY); look the codes up in the
  [PUMS data dictionary](https://www.census.gov/programs-surveys/acs/microdata/documentation.html).
- `*_ADJ` income columns are in constant 2024 dollars (adjusted with `ADJINC`).
