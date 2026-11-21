"""
pums_loader.py

Downloads and loads ACS PUMS (American Community Survey - Public Use Microdata
Sample) PERSON records for one U.S. state.

Each row of the person file is one surveyed person. Important notes:
  * PWGTP is the "person weight": how many real people this one row represents.
  * PUMA is the smallest geography in PUMS (areas of ~100,000+ people).
    PUMA codes are only unique when combined with STATE.
  * Many columns are numeric CODES (e.g. OCCP = occupation code), not text.
    We keep those as strings so leading zeros are not lost ("0010" != 10).
  * Blank cells mean "not applicable" (e.g. wages for a 5-year-old) and are
    loaded as missing values (NaN).

Data source:      https://www2.census.gov/programs-surveys/acs/data/pums/
Data dictionary:  https://www.census.gov/programs-surveys/acs/microdata/documentation.html

Usage from the command line (run from the project root folder):
    python -m src.pums_loader --state NY

Usage from other Python code:
    from src.pums_loader import load_pums_persons
    people = load_pums_persons(state="NY")
"""

import argparse
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

# Project root = the folder that contains "src" (one level above this file's folder)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# All raw PUMS files are stored here. The "data" folder is ignored by git.
PUMS_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "pums"

# Base URL of the Census Bureau's PUMS download area
PUMS_BASE_URL = "https://www2.census.gov/programs-surveys/acs/data/pums"

# Default survey settings for this project
DEFAULT_STATE = "NY"
DEFAULT_YEAR = 2024
DEFAULT_PERIOD = "5-Year"

# Columns that hold numbers we may want to do math on (ages, dollars, hours, weights).
# Everything else is treated as a text code.
NUMERIC_COLUMNS = [
    "PWGTP",   # Person weight (how many people this row represents)
    "AGEP",    # Age
    "WKHP",    # Usual hours worked per week
    "JWMNP",   # Travel time to work (minutes)
    "WAGP",    # Wages or salary income, past 12 months
    "PERNP",   # Total person's earnings
    "PINCP",   # Total person's income
    "ADJINC",  # Income adjustment factor (6 implied decimal places)
]

# A starting set of person characteristics for building "community" features.
# Pass columns=None to load_pums_persons() to load ALL ~290 columns instead.
DEFAULT_COLUMNS = [
    # --- Identifiers and geography ---
    "SERIALNO",  # Housing unit / group quarters serial number (household ID)
    "SPORDER",   # Person number within the household
    "STATE",     # State FIPS code (NY = 36)
    "PUMA",      # Public Use Microdata Area code (2020 Census definition)
    "PWGTP",     # Person weight
    # --- Demographics ---
    "AGEP",      # Age
    "SEX",       # Sex
    "MAR",       # Marital status
    "RAC1P",     # Race (recoded)
    "HISP",      # Hispanic origin (detailed)
    "NATIVITY",  # Native or foreign born
    "CIT",       # Citizenship status
    "POBP",      # Place of birth
    "ANC1P",     # Ancestry (first entry)
    "DIS",       # Disability
    "MIL",       # Military service
    "MIG",       # Lived in the same house 1 year ago?
    # --- Language ---
    "LANP",      # Language other than English spoken at home
    "ENG",       # Ability to speak English
    # --- Education ---
    "SCHL",      # Educational attainment
    "FOD1P",     # Field of degree (first entry)
    # --- Work ---
    "ESR",       # Employment status
    "COW",       # Class of worker (private, government, self-employed, ...)
    "OCCP",      # Occupation code
    "SOCP",      # Standard Occupational Classification code
    "INDP",      # Industry code
    "WKHP",      # Usual hours worked per week
    "JWTRNS",    # Means of transportation to work
    "JWMNP",     # Travel time to work
    # --- Income ---
    "WAGP",      # Wages or salary income
    "PERNP",     # Total earnings
    "PINCP",     # Total income
    "ADJINC",    # Income adjustment factor (needed to compare years)
]

# Columns holding dollar amounts that should be adjusted with ADJINC
INCOME_COLUMNS = ["WAGP", "PERNP", "PINCP"]


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def build_pums_person_url(state, year, period):
    """
    Build the download URL of the person-record CSV zip for one state.
    Example result:
        https://www2.census.gov/programs-surveys/acs/data/pums/2024/5-Year/csv_pny.zip
    """
    # The Census file names use the lowercase 2-letter state abbreviation
    state_lower = state.lower()

    # "csv_p" = CSV format, person records ("csv_h" would be housing records)
    file_name = "csv_p" + state_lower + ".zip"

    url = PUMS_BASE_URL + "/" + str(year) + "/" + period + "/" + file_name
    return url


def get_zip_path(state, year, period):
    """
    Return the local path where the downloaded zip for this state/year/period is saved.
    Example: data/raw/pums/2024_5-Year/csv_pny.zip
    """
    folder = PUMS_DATA_DIR / (str(year) + "_" + period)
    file_name = "csv_p" + state.lower() + ".zip"
    return folder / file_name


def download_pums_zip(state=DEFAULT_STATE, year=DEFAULT_YEAR, period=DEFAULT_PERIOD, force=False):
    """
    Download the person-record zip file for one state, unless it is already on disk.
    Set force=True to download it again anyway.
    Returns the path to the zip file.
    """
    url = build_pums_person_url(state, year, period)
    zip_path = get_zip_path(state, year, period)

    # If we already downloaded this file before, skip the download
    if zip_path.exists() and not force:
        print("Zip already downloaded:", zip_path)
        return zip_path

    # Make sure the destination folder exists
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    # Download to a temporary ".part" file first, so an interrupted download
    # never leaves a broken file with the final name
    temp_path = zip_path.with_suffix(".zip.part")

    print("Downloading:", url)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (pums-loader)"})

    with urllib.request.urlopen(request) as response:
        # Total size in bytes (used only to print progress)
        total_bytes = int(response.headers.get("Content-Length", 0))
        downloaded_bytes = 0
        chunk_size = 1024 * 1024  # read 1 MB at a time
        last_printed_percent = -10

        with open(temp_path, "wb") as output_file:
            while True:
                chunk = response.read(chunk_size)

                # An empty chunk means the download is finished
                if not chunk:
                    break

                output_file.write(chunk)
                downloaded_bytes = downloaded_bytes + len(chunk)

                # Print progress every 10%
                if total_bytes > 0:
                    percent = int(downloaded_bytes * 100 / total_bytes)
                    if percent >= last_printed_percent + 10:
                        downloaded_mb = downloaded_bytes / (1024 * 1024)
                        total_mb = total_bytes / (1024 * 1024)
                        print("  " + str(percent) + "%  (" + str(round(downloaded_mb)) + " / " + str(round(total_mb)) + " MB)")
                        last_printed_percent = percent

    # Download finished: give the file its final name
    temp_path.replace(zip_path)
    print("Saved to:", zip_path)
    return zip_path


def extract_person_csvs(zip_path):
    """
    Extract the person CSV file(s) from a PUMS zip.
    The zip contains a file like "psam_p36.csv" (36 = NY's state code) plus a PDF readme.
    Some large states are split into several CSVs, so we return a list.
    Files that were already extracted are not extracted again.
    """
    extracted_paths = []

    with zipfile.ZipFile(zip_path) as zip_file:
        for member_name in zip_file.namelist():
            # Use only the file name (drop any folders inside the zip) for safety
            file_name = Path(member_name).name

            # Keep only the person-record CSVs
            is_person_csv = file_name.startswith("psam_p") and file_name.endswith(".csv")
            if not is_person_csv:
                continue

            output_path = zip_path.parent / file_name

            # Extract only if we have not done it before
            if not output_path.exists():
                print("Extracting:", file_name)
                with zip_file.open(member_name) as source, open(output_path, "wb") as destination:
                    # Copy in 1 MB chunks so we never hold the whole file in memory
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        destination.write(chunk)

            extracted_paths.append(output_path)

    # Fail loudly if the zip did not contain what we expected
    if len(extracted_paths) == 0:
        raise FileNotFoundError("No psam_p*.csv file found inside " + str(zip_path))

    return extracted_paths


# ---------------------------------------------------------------------------
# Loading and cleaning
# ---------------------------------------------------------------------------

def build_column_types(column_names):
    """
    Decide the pandas data type each column is READ as:
      * numeric columns  -> float (blank cells become NaN)
      * all other codes  -> string (keeps leading zeros like "0010")
    """
    column_types = {}
    for column in column_names:
        if column in NUMERIC_COLUMNS:
            column_types[column] = "float64"
        else:
            column_types[column] = "string"
    return column_types


def convert_codes_to_categories(people):
    """
    Convert text code columns (OCCP, SCHL, PUMA, ...) to the pandas "category" type.

    "category" stores each distinct code once and a small integer per row,
    which uses far less memory than a separate string in every row
    (about 170 MB instead of 1.3 GB for New York).

    This is done AFTER reading as strings on purpose: reading directly as
    "category" makes pandas guess the codes are numbers and drop leading zeros.
    SERIALNO stays a string because it is an ID with almost no repeated values.
    """
    for column in people.columns:
        is_text_column = people[column].dtype == "string"
        if is_text_column and column != "SERIALNO":
            people[column] = people[column].astype("category")
    return people


def adjust_income_to_constant_dollars(people):
    """
    A 5-year file mixes answers from 5 different survey years. ADJINC converts
    every income value to dollars of the latest year so they can be compared.
    ADJINC has 6 implied decimal places, e.g. 1019518 means 1.019518.
    This creates new columns like "PINCP_ADJ" and keeps the original columns.
    """
    # Without ADJINC there is nothing to adjust
    if "ADJINC" not in people.columns:
        print("ADJINC column not loaded; skipping income adjustment.")
        return people

    adjustment_factor = people["ADJINC"] / 1_000_000

    for column in INCOME_COLUMNS:
        if column in people.columns:
            new_column = column + "_ADJ"
            people[new_column] = people[column] * adjustment_factor

    return people


def load_pums_persons(state=DEFAULT_STATE,
                      year=DEFAULT_YEAR,
                      period=DEFAULT_PERIOD,
                      columns=DEFAULT_COLUMNS,
                      adjust_income=True):
    """
    Main entry point: download (if needed), extract, and load PUMS person records
    for one state into a pandas DataFrame.

    Parameters
    ----------
    state : str
        Two-letter state abbreviation, e.g. "NY".
    year : int
        Last year of the survey period, e.g. 2024.
    period : str
        "5-Year" (larger sample, ~5% of people) or "1-Year" (~1% of people).
    columns : list of str, or None
        Which columns to load. None loads every column (uses much more memory).
    adjust_income : bool
        If True, add *_ADJ income columns in constant (latest-year) dollars.

    Returns
    -------
    pandas.DataFrame with one row per surveyed person.
    """
    # Step 1: make sure the zip is on disk
    zip_path = download_pums_zip(state, year, period)

    # Step 2: make sure the CSV(s) are extracted next to the zip
    csv_paths = extract_person_csvs(zip_path)

    # Step 3: read each CSV and collect the pieces
    pieces = []
    for csv_path in csv_paths:
        # Read only the header row to see which columns actually exist
        header = pd.read_csv(csv_path, nrows=0)
        available_columns = list(header.columns)

        # Decide which columns to read
        if columns is None:
            columns_to_read = available_columns
        else:
            # Warn about requested columns that are not in this file
            missing_columns = []
            for column in columns:
                if column not in available_columns:
                    missing_columns.append(column)
            if len(missing_columns) > 0:
                print("Warning: these columns are not in", csv_path.name, "->", missing_columns)

            # Keep only the requested columns that do exist
            columns_to_read = []
            for column in columns:
                if column in available_columns:
                    columns_to_read.append(column)

        # Read the selected columns with the right data types
        print("Reading:", csv_path.name, "(" + str(len(columns_to_read)) + " columns)")
        piece = pd.read_csv(
            csv_path,
            usecols=columns_to_read,
            dtype=build_column_types(columns_to_read),
        )
        pieces.append(piece)

    # Step 4: stack the pieces into one table (only matters for split states)
    people = pd.concat(pieces, ignore_index=True)

    # Step 5: put columns back in the order the caller asked for
    if columns is not None:
        ordered_columns = []
        for column in columns:
            if column in people.columns:
                ordered_columns.append(column)
        people = people[ordered_columns]

    # Step 6: store text codes as categories to save memory
    people = convert_codes_to_categories(people)

    # Step 7: optional income adjustment to constant dollars
    if adjust_income:
        people = adjust_income_to_constant_dollars(people)

    return people


def print_summary(people):
    """Print a quick overview of a loaded PUMS person table."""
    print()
    print("Rows (surveyed people):", len(people))
    print("Columns:", len(people.columns))

    # The sum of person weights estimates the state's real population
    if "PWGTP" in people.columns:
        estimated_population = int(people["PWGTP"].sum())
        print("Estimated population (sum of PWGTP):", estimated_population)

    # Number of distinct PUMAs = number of small geographic areas available
    if "PUMA" in people.columns:
        print("Distinct PUMAs:", people["PUMA"].nunique())

    # Approximate memory used by the DataFrame
    memory_mb = people.memory_usage(deep=True).sum() / (1024 * 1024)
    print("Memory used: about", round(memory_mb), "MB")

    print()
    print("First 5 rows:")
    print(people.head())


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------

def main():
    """Parse command-line options, load the data, and print a summary."""
    parser = argparse.ArgumentParser(description="Download and load ACS PUMS person records for one state.")
    parser.add_argument("--state", default=DEFAULT_STATE, help="Two-letter state abbreviation (default: NY)")
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR, help="Survey end year (default: 2024)")
    parser.add_argument("--period", default=DEFAULT_PERIOD, choices=["1-Year", "5-Year"], help="Survey period (default: 5-Year)")
    parser.add_argument("--all-columns", action="store_true", help="Load every column instead of the default subset")
    arguments = parser.parse_args()

    # None means "load every column"
    if arguments.all_columns:
        columns = None
    else:
        columns = DEFAULT_COLUMNS

    people = load_pums_persons(
        state=arguments.state,
        year=arguments.year,
        period=arguments.period,
        columns=columns,
    )
    print_summary(people)


# Run main() only when this file is executed directly (not when imported)
if __name__ == "__main__":
    main()
