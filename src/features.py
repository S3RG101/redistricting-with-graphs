"""
features.py

Turns the PUMS person table (one row per person) into a numeric feature
matrix (one vector per person) that can be fed to PCA and other
dimensionality reduction / clustering algorithms.

The work happens in two stages:

  1. prepare_fields(people)
     Plain pandas cleaning. Fills "not applicable" blanks, applies a log to
     incomes, turns ordered codes into numbers, and collapses codes with
     hundreds of values (occupation, industry, birthplace, ...) into groups.

  2. build_feature_transformer() + fit_transform
     A scikit-learn ColumnTransformer with ONE step per field:
       * continuous and ordered fields -> StandardScaler (mean 0, variance 1)
       * yes/no fields                  -> kept as 0/1
       * category fields                -> one-hot columns (rare values grouped)
     Each step is named after its field, so FIELD_WEIGHTS can scale how much
     every field counts (see the "transformer_weights" argument).

Usage from the command line (run from the project root folder):
    python -m src.features

Usage from other Python code:
    from src.pums_loader import load_pums_persons
    from src.features import filter_adults, build_feature_matrix
    people = filter_adults(load_pums_persons(state="NY"))
    features, feature_names, transformer = build_feature_matrix(people)
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Which fields are used, grouped by how they are converted to numbers
# ---------------------------------------------------------------------------

# Real numbers: standardized. Blank (not applicable) becomes 0 first.
CONTINUOUS_FIELDS = [
    "AGEP",       # Age
    "WKHP",       # Usual hours worked per week (blank = did not work -> 0)
    "JWMNP",      # Travel time to work in minutes (blank = no commute -> 0)
    "PINCP_ADJ",  # Total income, 2024 dollars (signed log applied)
    "PERNP_ADJ",  # Total earnings, 2024 dollars (signed log applied)
]

# Subset of the continuous fields that get the signed log (they are very skewed)
LOG_FIELDS = ["PINCP_ADJ", "PERNP_ADJ"]

# Codes whose order means something: converted to a single number, then standardized
ORDINAL_FIELDS = [
    "SCHL",  # Educational attainment: 1 (no schooling) ... 24 (doctorate)
    "ENG",   # English difficulty: 0 (speaks only English) ... 4 (does not speak English)
]

# Yes/no fields: converted to 0/1 and left unscaled
BINARY_FIELDS = [
    "SEX",       # 1 = female, 0 = male
    "NATIVITY",  # 1 = foreign born, 0 = native
    "DIS",       # 1 = has a disability, 0 = no disability
]

# Category fields with no order: one-hot encoded.
# Blank values become their own "NA" category (e.g. "not in labor force").
CATEGORY_FIELDS = [
    "MAR",     # Marital status
    "CIT",     # Citizenship status
    "MIL",     # Military service
    "MIG",     # Moved in the last year?
    "ESR",     # Employment status
    "COW",     # Class of worker
    "JWTRNS",  # Means of transportation to work
    "RAC1P",   # Race
    "HISP",    # Hispanic origin
    "SOCP",    # Occupation        -> grouped to 23 SOC major groups
    "INDP",    # Industry          -> grouped to ~19 sectors
    "FOD1P",   # Field of degree   -> grouped to ~38 broad fields
    "POBP",    # Place of birth    -> grouped to ~12 regions
    "LANP",    # Language spoken at home (only common languages kept)
    "ANC1P",   # Ancestry (only common ancestries kept)
]

# In one-hot encoding, categories held by less than this fraction of people
# are merged into a single "infrequent" column (0.005 = 0.5% of people).
MIN_CATEGORY_FREQUENCY = 0.005

# How much each field counts. 1.0 = default. The PCA experiment
# (src/run_pca.py) shows how much each field drives the result; change these
# numbers afterwards to rebalance. Every output column of a field is
# multiplied by its weight.
FIELD_WEIGHTS = {}
for field_name in CONTINUOUS_FIELDS + ORDINAL_FIELDS + BINARY_FIELDS + CATEGORY_FIELDS:
    FIELD_WEIGHTS[field_name] = 1.0


# ---------------------------------------------------------------------------
# Code groupings (from the PUMS 2020-2024 data dictionary)
# ---------------------------------------------------------------------------

# Industry (INDP) code ranges. Each entry is (first code of the range, sector).
# A code belongs to the last sector whose first code is <= the code.
INDUSTRY_SECTOR_STARTS = [
    (170, "Agriculture"),
    (370, "Mining"),
    (570, "Utilities"),
    (770, "Construction"),
    (1070, "Manufacturing"),
    (4070, "Wholesale"),
    (4670, "Retail"),
    (6070, "Transportation"),
    (6471, "Information"),
    (6871, "Finance"),
    (7270, "Professional"),
    (7860, "Education"),
    (7970, "Medical"),
    (8370, "SocialAssistance"),
    (8561, "Entertainment"),
    (8770, "OtherServices"),
    (9370, "PublicAdministration"),
    (9670, "Military"),
    (9920, "Unemployed"),
]

# Place of birth (POBP) code ranges. Same rule as above.
# New York (036) and Puerto Rico (072) are handled separately in group_birthplace().
BIRTHPLACE_REGION_STARTS = [
    (1, "OtherUSState"),
    (60, "USTerritory"),
    (100, "Europe"),
    (200, "Asia"),
    (300, "Canada_Bermuda"),
    (303, "Mexico_CentralAmerica"),
    (320, "Caribbean"),
    (360, "SouthAmerica"),
    (375, "Americas_NotSpecified"),
    (400, "Africa"),
    (500, "Oceania_Other"),
]


def find_range_label(code_number, range_starts):
    """
    Return the label of the range a number falls into.
    range_starts is a list of (first code, label) sorted by first code.
    """
    label = "Unknown"
    for first_code, range_label in range_starts:
        if code_number >= first_code:
            label = range_label
        else:
            # The list is sorted, so no later range can match
            break
    return label


def group_industry(code_text):
    """Map one INDP code (text like "7860") to its industry sector name."""
    return find_range_label(int(code_text), INDUSTRY_SECTOR_STARTS)


def group_birthplace(code_text):
    """Map one POBP code (text like "036") to a birthplace region name."""
    code_number = int(code_text)
    if code_number == 36:
        return "NewYork"
    if code_number == 72:
        return "PuertoRico"
    return find_range_label(code_number, BIRTHPLACE_REGION_STARTS)


def map_codes(codes, mapping_function):
    """
    Apply mapping_function to every code in a Series.
    Each distinct code is mapped only once (there are only a few hundred),
    then the results are looked up for all rows. Blanks stay blank.
    """
    mapping = {}
    for code in codes.dropna().unique():
        mapping[code] = mapping_function(code)
    return codes.map(mapping)


# ---------------------------------------------------------------------------
# Stage 1: pandas cleaning
# ---------------------------------------------------------------------------

def filter_adults(people, min_age=18):
    """Keep only people aged min_age or older (voting age by default)."""
    is_adult = people["AGEP"] >= min_age
    adults = people[is_adult].reset_index(drop=True)
    print("Kept", len(adults), "of", len(people), "people aged", min_age, "or older")
    return adults


def signed_log(values):
    """
    sign(x) * log(1 + |x|)
    Like a log, but also works for 0 and negative values (income can be
    negative because of business losses). It shrinks huge values, so a few
    very high incomes do not dominate.
    """
    return np.sign(values) * np.log1p(np.abs(values))


def prepare_fields(people):
    """
    Return a new DataFrame with one cleaned column per field, ready for
    the scikit-learn transformer. The input DataFrame is not modified.
    """
    prepared = pd.DataFrame(index=people.index)

    # --- Continuous fields: blank means "not applicable", which we treat as 0 ---
    for field in CONTINUOUS_FIELDS:
        values = people[field].astype("float64").fillna(0.0)
        if field in LOG_FIELDS:
            values = signed_log(values)
        prepared[field] = values.astype("float32")

    # --- SCHL: the code number already follows the education order ---
    prepared["SCHL"] = people["SCHL"].astype("float32")

    # --- ENG: blank means "speaks only English", the easiest case -> 0 ---
    prepared["ENG"] = people["ENG"].astype("float32").fillna(0.0)

    # --- Binary fields: compare against the code that means "yes" ---
    prepared["SEX"] = (people["SEX"] == "2").astype("float32")        # 2 = female
    prepared["NATIVITY"] = (people["NATIVITY"] == "2").astype("float32")  # 2 = foreign born
    prepared["DIS"] = (people["DIS"] == "1").astype("float32")        # 1 = with a disability

    # --- Category fields: work with plain text codes ---
    for field in CATEGORY_FIELDS:
        prepared[field] = people[field].astype("string")

    # Occupation: the first 2 digits of the SOC code are the major group
    # (e.g. "15" = computer and math, "47" = construction)
    prepared["SOCP"] = prepared["SOCP"].str[:2]

    # Industry: code ranges -> sector names
    prepared["INDP"] = map_codes(prepared["INDP"], group_industry)

    # Field of degree: the first 2 digits are the broad field (e.g. "24" = engineering)
    prepared["FOD1P"] = prepared["FOD1P"].str[:2]

    # Place of birth: code ranges -> regions
    prepared["POBP"] = map_codes(prepared["POBP"], group_birthplace)

    # Blank category values become their own "NA" category
    for field in CATEGORY_FIELDS:
        prepared[field] = prepared[field].fillna("NA").astype(object)

    return prepared


# ---------------------------------------------------------------------------
# Stage 2: scikit-learn transformer
# ---------------------------------------------------------------------------

def build_feature_transformer(field_weights=None):
    """
    Build (but do not fit) the ColumnTransformer.
    There is one step per field, named after the field, so that
    field_weights (a dict field -> weight) can scale each field's columns.
    """
    if field_weights is None:
        field_weights = FIELD_WEIGHTS

    steps = []

    # Continuous and ordered fields: mean 0, variance 1
    for field in CONTINUOUS_FIELDS + ORDINAL_FIELDS:
        steps.append((field, StandardScaler(), [field]))

    # Yes/no fields: already 0/1, keep as they are
    for field in BINARY_FIELDS:
        steps.append((field, "passthrough", [field]))

    # Category fields: one column per (common) category
    for field in CATEGORY_FIELDS:
        encoder = OneHotEncoder(
            min_frequency=MIN_CATEGORY_FREQUENCY,     # merge rare categories
            handle_unknown="infrequent_if_exist",     # unseen codes -> "infrequent" column
            sparse_output=False,                      # plain (dense) numpy array
            dtype=np.float32,                         # half the memory of float64
        )
        steps.append((field, encoder, [field]))

    transformer = ColumnTransformer(
        transformers=steps,
        transformer_weights=field_weights,  # multiply each field's columns by its weight
        verbose_feature_names_out=True,     # output names look like "SOCP__SOCP_15"
    )
    return transformer


def build_feature_matrix(people, field_weights=None):
    """
    Full pipeline: clean the fields, then fit and apply the transformer.

    Returns
    -------
    features : numpy array, shape (number of people, number of feature columns)
    feature_names : list of str, one name per column, e.g. "SOCP__SOCP_15"
    transformer : the fitted ColumnTransformer (reuse it with .transform on new data)
    """
    prepared = prepare_fields(people)

    transformer = build_feature_transformer(field_weights)
    features = transformer.fit_transform(prepared)

    # Make sure everything is float32 to save memory
    features = np.asarray(features, dtype=np.float32)

    feature_names = list(transformer.get_feature_names_out())
    return features, feature_names, transformer


def field_of_feature(feature_name):
    """Return the field a feature column came from: "SOCP__SOCP_15" -> "SOCP"."""
    return feature_name.split("__")[0]


# ---------------------------------------------------------------------------
# Command-line entry point: build the matrix and describe it
# ---------------------------------------------------------------------------

def main():
    """Load NY adults, build the feature matrix, and print a short description."""
    from pums_loader import load_pums_persons

    people = load_pums_persons(state="NY")
    adults = filter_adults(people)

    features, feature_names, transformer = build_feature_matrix(adults)

    print()
    print("Feature matrix shape (people, columns):", features.shape)
    print("Memory: about", round(features.nbytes / (1024 * 1024)), "MB")

    # Count how many columns each field produced
    print()
    print("Columns per field:")
    columns_per_field = {}
    for name in feature_names:
        field = field_of_feature(name)
        columns_per_field[field] = columns_per_field.get(field, 0) + 1
    for field, count in columns_per_field.items():
        print("  " + field.ljust(10) + str(count))


# Run main() only when this file is executed directly (not when imported)
if __name__ == "__main__":
    main()
