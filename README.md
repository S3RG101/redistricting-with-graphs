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
│   ├── pums_loader.py     # downloads + loads ACS-PUMS person records
│   ├── features.py        # turns each person into a numeric feature vector
│   ├── run_pca.py         # PCA experiment: how much each field drives the result
│   ├── puma_profiles.py   # step 2: one community profile + population per PUMA
│   ├── puma_graph.py      # step 3: neighboring PUMAs + similarity weights
│   └── cluster_districts.py  # step 4: PUMAs -> 26 connected, balanced districts
├── results/               # small CSV outputs of steps 2-4
└── data/                  # created by the loaders; NOT tracked by git
    ├── raw/pums/2024_5-Year/
    └── raw/tiger/         # PUMA boundary shapefile
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

## 3. Build feature vectors and run PCA

`src/features.py` keeps adults (18+) and turns each one into a vector of
178 numbers (NY). Each field is converted according to its type:

| Type | Fields | Conversion |
|---|---|---|
| Continuous | `AGEP`, `WKHP`, `JWMNP`, `PINCP_ADJ`, `PERNP_ADJ` | blank → 0, signed log on income, standardized |
| Ordered codes | `SCHL`, `ENG` | code → number (blank `ENG` = English only → 0), standardized |
| Yes/no | `SEX`, `NATIVITY`, `DIS` | 0/1 |
| Categories | `MAR`, `CIT`, `MIL`, `MIG`, `ESR`, `COW`, `JWTRNS`, `RAC1P`, `HISP`, `SOCP`, `INDP`, `FOD1P`, `POBP`, `LANP`, `ANC1P` | one-hot; blank = its own "NA" category; categories under 0.5% of people merged |

Codes with hundreds of values are grouped first: occupation → 23 SOC major
groups, industry → ~19 sectors, field of degree → broad field, birthplace → regions.

Not used as features: IDs (`SERIALNO`, `SPORDER`), geography (`STATE`, `PUMA`),
the weight `PWGTP`, `OCCP` (same as `SOCP`) and `WAGP_ADJ` (almost the same as `PERNP_ADJ`).

How much each field counts is set in `FIELD_WEIGHTS` in `src/features.py` (all 1.0 by default).

```
python src\features.py                 # build the matrix and list columns per field
python src\run_pca.py                  # PCA on all NY adults (~1 min)
python src\run_pca.py --sample 200000  # faster, on a random subset
```

(On macOS/Linux use `/` instead of `\` in the paths.)

## 4. From people to districts

PUMS only tells us each person's PUMA (an area of ~100k-250k people), not
their address, so districts are built from whole PUMAs (144 in NY).

Run the three steps in order; each saves a CSV in `results/` that the next one reads:

```
python src\puma_profiles.py      # step 2 (~1 min)
python src\puma_graph.py         # step 3 (downloads the 1.6 MB boundary file once)
python src\cluster_districts.py  # step 4 (--tolerance 0.10 to allow +-10%)
```

| Step | What it does | Output |
|---|---|---|
| 2. Profiles | PCA (26 components, ~80% of variance) on adults; average each PUMA's adults weighted by `PWGTP`; population = sum of `PWGTP` over everyone, children included | `results/puma_profiles.csv` |
| 3. Graph | Neighbors = PUMAs whose TIGER/Line borders touch (water borders count, so Staten Island links to Brooklyn); weight = `exp(-distance² / σ²)` between profiles, σ = median neighbor distance | `results/puma_edges.csv` |
| 4. Districts | Agglomerative (Ward) clustering into 26 districts that may only merge neighbors, so every district is connected; then balance population by moving border PUMAs (or passing them along a chain of districts) | `results/districts_agglomerative.csv` |

"Profile variance explained" measures how similar the PUMAs inside each district
are (0-100%, higher is more similar). The district CSV keeps both the raw and the
balanced district of every PUMA, so both stages can be mapped.
