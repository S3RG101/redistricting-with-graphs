"""
run_pca.py

PCA experiment on the person feature vectors, to see how much each field
(age, income, occupation, race, ...) drives the result before we decide on
field weights.

It prints four reports:
  1. Input variance per field: how much "spread" each field puts into the
     matrix. PCA chases variance, so fields with more variance get more say.
  2. Explained variance: how much of the total spread each principal
     component captures, and how many components are needed for 50/80/90/95%.
  3. Top features of the first components: which columns each component
     is mostly made of (its largest loadings).
  4. Field contribution to the top components: the share of the top
     components that comes from each field.

Usage (run from the project root folder, with .venv activated):
    python -m src.run_pca
    python -m src.run_pca --sample 200000     # faster, uses a random subset of people
    python -m src.run_pca --components 20     # analyze the first 20 components
"""

import argparse

import numpy as np
from sklearn.decomposition import PCA

from features import build_feature_matrix, field_of_feature, filter_adults
from pums_loader import load_pums_persons


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def print_title(title):
    """Print a section title surrounded by lines."""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def sum_by_field(values, feature_names):
    """
    Add up one number per feature column into one number per field.
    Example: the 23 SOCP columns are added together into "SOCP".
    Returns a dict field -> total, in the order fields first appear.
    """
    totals = {}
    for value, name in zip(values, feature_names):
        field = field_of_feature(name)
        totals[field] = totals.get(field, 0.0) + float(value)
    return totals


def print_field_table(totals, value_header):
    """Print a dict field -> value as a table sorted from largest to smallest, with percentages."""
    grand_total = sum(totals.values())
    sorted_fields = sorted(totals.items(), key=lambda item: item[1], reverse=True)

    print("  " + "field".ljust(12) + value_header.rjust(12) + "share".rjust(10))
    for field, value in sorted_fields:
        share = 100.0 * value / grand_total
        print("  " + field.ljust(12) + ("%.3f" % value).rjust(12) + ("%.1f%%" % share).rjust(10))


# ---------------------------------------------------------------------------
# The four reports
# ---------------------------------------------------------------------------

def report_input_variance(features, feature_names):
    """Report 1: variance of each input column, added up per field."""
    print_title("1. Input variance per field (what PCA sees before it starts)")

    # Variance of every column (computed in float64 for accuracy)
    column_variances = np.var(features, axis=0, dtype=np.float64)

    variance_per_field = sum_by_field(column_variances, feature_names)
    print_field_table(variance_per_field, "variance")


def report_explained_variance(pca, number_to_show):
    """Report 2: variance captured by each component, and how many components reach key levels."""
    print_title("2. Explained variance by component")

    ratios = pca.explained_variance_ratio_
    cumulative = np.cumsum(ratios)

    print("  " + "component".ljust(12) + "explained".rjust(12) + "cumulative".rjust(12))
    for index in range(number_to_show):
        print("  " + ("PC" + str(index + 1)).ljust(12)
              + ("%.2f%%" % (100 * ratios[index])).rjust(12)
              + ("%.2f%%" % (100 * cumulative[index])).rjust(12))

    print()
    for target in [0.50, 0.80, 0.90, 0.95]:
        # First component index where the cumulative share reaches the target
        components_needed = int(np.searchsorted(cumulative, target) + 1)
        print("  Components needed for " + str(int(target * 100)) + "% of the variance:", components_needed)


def report_top_loadings(pca, feature_names, number_of_components, features_per_component=8):
    """Report 3: the columns with the largest loadings (positive or negative) in each component."""
    print_title("3. Top features of the first " + str(number_of_components) + " components")

    for index in range(number_of_components):
        loadings = pca.components_[index]

        # Column indices sorted by absolute loading, largest first
        order = np.argsort(np.abs(loadings))[::-1]

        print()
        print("  PC" + str(index + 1) + "  (" + ("%.2f%%" % (100 * pca.explained_variance_ratio_[index])) + " of variance)")
        for column_index in order[:features_per_component]:
            print("    " + ("%+.3f" % loadings[column_index]).rjust(8) + "  " + feature_names[column_index])


def report_field_contribution(pca, feature_names, number_of_components):
    """
    Report 4: share of the top components that comes from each field.

    Each component is a unit-length direction, so its squared loadings add up
    to 1 and say how much of that component each column makes up. We add the
    squared loadings of each field's columns, weight every component by the
    variance it explains, and add over the top components.
    """
    print_title("4. Field contribution to the top " + str(number_of_components) + " components (variance-weighted)")

    contribution_per_column = np.zeros(len(feature_names))
    for index in range(number_of_components):
        squared_loadings = pca.components_[index] ** 2
        component_weight = pca.explained_variance_ratio_[index]
        contribution_per_column = contribution_per_column + component_weight * squared_loadings

    contribution_per_field = sum_by_field(contribution_per_column, feature_names)
    print_field_table(contribution_per_field, "contrib.")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    """Load data, build features, fit PCA, and print the reports."""
    parser = argparse.ArgumentParser(description="PCA experiment on PUMS person feature vectors.")
    parser.add_argument("--state", default="NY", help="Two-letter state abbreviation (default: NY)")
    parser.add_argument("--sample", type=int, default=None, help="Use a random subset of this many adults (default: all)")
    parser.add_argument("--components", type=int, default=10, help="How many components to analyze in reports 2-4 (default: 10)")
    arguments = parser.parse_args()

    # Step 1: load people and keep adults only
    people = load_pums_persons(state=arguments.state)
    adults = filter_adults(people)

    # Step 2 (optional): random subset for a faster run; random_state makes it repeatable
    if arguments.sample is not None and arguments.sample < len(adults):
        adults = adults.sample(n=arguments.sample, random_state=0).reset_index(drop=True)
        print("Using a random sample of", len(adults), "adults")

    # Step 3: build the feature matrix
    features, feature_names, transformer = build_feature_matrix(adults)
    print("Feature matrix shape (people, columns):", features.shape)

    # Step 4: fit PCA with all components so we see the full variance picture.
    # "covariance_eigh" is fast when there are many more rows than columns.
    pca = PCA(n_components=None, svd_solver="covariance_eigh")
    pca.fit(features)

    # Step 5: print the reports
    number_of_components = min(arguments.components, len(feature_names))
    report_input_variance(features, feature_names)
    report_explained_variance(pca, number_of_components)
    report_top_loadings(pca, feature_names, min(5, number_of_components))
    report_field_contribution(pca, feature_names, number_of_components)


# Run main() only when this file is executed directly (not when imported)
if __name__ == "__main__":
    main()
