"""
puma_profiles.py   (pipeline step 2)

PUMS does not say where a person lives, only their PUMA (an area of
~100,000-200,000 people). So districts are built from whole PUMAs, and each
PUMA needs one "community profile" vector.

This script:
  1. Builds the adult feature vectors (src/features.py) and reduces them with
     PCA to NUMBER_OF_COMPONENTS numbers per person.
  2. Averages those PCA coordinates over the adults of each PUMA, weighting
     each person by PWGTP (how many real people the row represents).
  3. Counts the population of each PUMA using EVERYONE, children included,
     because districts must have equal TOTAL population.

Output: results/puma_profiles.csv with one row per PUMA:
    PUMA, population, adults_sampled, PC1, PC2, ..., PC26

Usage (run from the project root folder, with .venv activated):
    python src\\puma_profiles.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from features import build_feature_matrix, filter_adults
from pums_loader import load_pums_persons


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# Number of PCA components kept per person. 26 components explained ~80% of the
# variance in the PCA experiment (src/run_pca.py). It is a coincidence that NY
# also has 26 congressional districts; the two numbers are unrelated.
NUMBER_OF_COMPONENTS = 26

# Output folder for results (created if missing)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PROFILES_PATH = RESULTS_DIR / "puma_profiles.csv"


# ---------------------------------------------------------------------------
# Step 2a: PCA coordinates for every adult
# ---------------------------------------------------------------------------

def compute_pca_coordinates(adults, number_of_components=NUMBER_OF_COMPONENTS):
    """
    Build the feature matrix for the adults and project it onto the first
    principal components.

    Returns
    -------
    coordinates : numpy array, shape (number of adults, number_of_components)
    pca : the fitted PCA object (pca.explained_variance_ratio_ etc.)
    """
    features, feature_names, transformer = build_feature_matrix(adults)
    print("Feature matrix shape (people, columns):", features.shape)

    pca = PCA(n_components=number_of_components, svd_solver="covariance_eigh")
    coordinates = pca.fit_transform(features)

    explained = 100 * pca.explained_variance_ratio_.sum()
    print("PCA with", number_of_components, "components explains", round(explained, 1), "% of the variance")
    return coordinates, pca


# ---------------------------------------------------------------------------
# Step 2b: average the coordinates inside each PUMA
# ---------------------------------------------------------------------------

def component_column_names(number_of_components):
    """Return ["PC1", "PC2", ...] for the given number of components."""
    names = []
    for index in range(number_of_components):
        names.append("PC" + str(index + 1))
    return names


def build_puma_profiles(adults, coordinates):
    """
    Weighted average of the adults' PCA coordinates per PUMA.

    For each PUMA:  profile = sum(weight * coordinates) / sum(weight)
    where weight = PWGTP (number of real people each surveyed person represents).

    Returns a DataFrame indexed by PUMA with columns PC1..PCn and adults_sampled.
    """
    pc_columns = component_column_names(coordinates.shape[1])

    # Put the coordinates in a table next to each adult's PUMA and weight
    table = pd.DataFrame(coordinates, columns=pc_columns)
    table["PUMA"] = adults["PUMA"].astype(str).to_numpy()
    weights = adults["PWGTP"].to_numpy()

    # Multiply every coordinate by the person's weight
    weighted = table[pc_columns].multiply(weights, axis=0)
    weighted["PUMA"] = table["PUMA"]
    weighted["weight"] = weights

    # Add up per PUMA, then divide by the total weight of that PUMA
    sums = weighted.groupby("PUMA").sum()
    profiles = sums[pc_columns].divide(sums["weight"], axis=0)

    # Also keep how many surveyed adults each profile is based on
    profiles["adults_sampled"] = table.groupby("PUMA").size()
    return profiles


# ---------------------------------------------------------------------------
# Step 2c: total population per PUMA (children included)
# ---------------------------------------------------------------------------

def compute_puma_population(people):
    """Estimated total population of each PUMA = sum of PWGTP over all people (all ages)."""
    population = people.groupby(people["PUMA"].astype(str), observed=True)["PWGTP"].sum()
    population.name = "population"
    return population


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Build the PUMA profiles for NY and save them to results/puma_profiles.csv."""
    # Load everyone (for population) and the adults (for profiles)
    people = load_pums_persons(state="NY")
    adults = filter_adults(people)

    # PCA coordinates for every adult
    coordinates, pca = compute_pca_coordinates(adults)

    # One profile per PUMA, plus its total population
    profiles = build_puma_profiles(adults, coordinates)
    population = compute_puma_population(people)

    # Join them into one table: PUMA, population, adults_sampled, PC1..PCn
    result = profiles.join(population)
    pc_columns = component_column_names(coordinates.shape[1])
    result = result[["population", "adults_sampled"] + pc_columns]
    result.index.name = "PUMA"
    result = result.sort_index()

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(PROFILES_PATH)

    # Short summary
    print()
    print("PUMAs:", len(result))
    print("Total population:", int(result["population"].sum()))
    print("Population per PUMA: min", int(result["population"].min()),
          "| median", int(result["population"].median()),
          "| max", int(result["population"].max()))
    print("Surveyed adults per PUMA: min", int(result["adults_sampled"].min()),
          "| max", int(result["adults_sampled"].max()))
    print("Saved:", PROFILES_PATH)


# Run main() only when this file is executed directly (not when imported)
if __name__ == "__main__":
    main()
