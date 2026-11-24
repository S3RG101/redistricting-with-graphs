"""
cluster_districts.py   (pipeline step 4)

Groups the 144 New York PUMAs into 26 districts so that:
  * each district is ONE connected piece on the map, and
  * PUMAs in the same district have similar community profiles, and
  * district populations are as equal as whole PUMAs allow.

Clustering: scikit-learn AgglomerativeClustering (Ward linkage) with a
connectivity matrix, so only neighboring groups can ever be merged. That makes
every district connected by construction. It does not look at population.

Population repair: balance_population repeatedly moves one border PUMA to a
neighboring district when that makes populations more equal (or passes PUMAs
along a chain of districts when no single move helps), without ever splitting
a district, until every district is within the tolerance or nothing helps.

Inputs:  results/puma_profiles.csv (step 2), results/puma_edges.csv (step 3)
Output:  results/districts_agglomerative.csv with one row per PUMA:
    PUMA, population, district_raw, district

Usage (run from the project root folder, with .venv activated):
    python src\\cluster_districts.py
    python src\\cluster_districts.py --tolerance 0.10
"""

import argparse

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering

from puma_graph import find_connected_pieces, load_edges, load_profiles, neighbors_from_edges, profile_columns
from puma_profiles import RESULTS_DIR


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# New York's number of congressional districts (2020 apportionment)
NUMBER_OF_DISTRICTS = 26

# Balancing stops once every district is within +-5% of the ideal population
POPULATION_TOLERANCE = 0.05

OUTPUT_PATH = RESULTS_DIR / "districts_agglomerative.csv"


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def build_connectivity_matrix(pumas, neighbors):
    """Sparse 0/1 matrix: entry (i, j) = 1 if PUMA i and PUMA j are neighbors."""
    position = {}
    for index, puma in enumerate(pumas):
        position[puma] = index

    rows = []
    columns = []
    for puma in pumas:
        for neighbor in neighbors[puma]:
            rows.append(position[puma])
            columns.append(position[neighbor])

    values = np.ones(len(rows))
    return csr_matrix((values, (rows, columns)), shape=(len(pumas), len(pumas)))


def cluster_agglomerative(pumas, vectors, neighbors, number_of_districts):
    """
    Ward agglomerative clustering, allowed to merge only neighboring groups.
    Returns a Series PUMA -> district number.
    """
    connectivity = build_connectivity_matrix(pumas, neighbors)
    model = AgglomerativeClustering(
        n_clusters=number_of_districts,
        connectivity=connectivity,
        linkage="ward",
    )
    labels = model.fit_predict(vectors)
    return pd.Series(labels, index=pumas)


# ---------------------------------------------------------------------------
# Population balancing
# ---------------------------------------------------------------------------

def members_of(assignment, district):
    """List of PUMAs currently assigned to a district."""
    return list(assignment.index[assignment == district])


def stays_connected_without(assignment, district, removed_puma, neighbors):
    """True if the district is still one piece (and not empty) after removing removed_puma."""
    remaining = []
    for puma in members_of(assignment, district):
        if puma != removed_puma:
            remaining.append(puma)
    if len(remaining) == 0:
        return False
    return len(find_connected_pieces(remaining, neighbors)) == 1


def squared_deviation_total(assignment, population, ideal, districts):
    """Sum of (district population - ideal)^2 over the given districts."""
    district_population = population.groupby(assignment).sum()
    total = 0.0
    for district in districts:
        total = total + (district_population[district] - ideal) ** 2
    return total


def find_best_single_move(assignment, neighbors, population, ideal):
    """
    Look at every border PUMA and every neighboring district it could join.
    Return the (puma, destination) move that most reduces the sum of squared
    population deviations, or None if no move reduces it.
    A move is only allowed if the district it leaves stays in one piece.
    """
    district_population = population.groupby(assignment).sum()
    best_gain = 0.0
    best_move = None

    for puma in assignment.index:
        source = assignment[puma]

        # Districts this PUMA borders (other than its own)
        destinations = set()
        for neighbor in neighbors[puma]:
            if assignment[neighbor] != source:
                destinations.add(assignment[neighbor])

        puma_population = population[puma]
        source_population = district_population[source]

        for destination in destinations:
            destination_population = district_population[destination]

            # Squared deviations of the two districts before and after the move
            before = (source_population - ideal) ** 2 + (destination_population - ideal) ** 2
            after = ((source_population - puma_population - ideal) ** 2
                     + (destination_population + puma_population - ideal) ** 2)
            gain = before - after

            if gain > best_gain:
                # Only check connectivity for moves that would be the new best (it is the slow part)
                if stays_connected_without(assignment, source, puma, neighbors):
                    best_gain = gain
                    best_move = (puma, destination)

    return best_move


def district_neighbors(assignment, neighbors):
    """Dict district -> set of districts it shares a border with."""
    result = {}
    for district in assignment.unique():
        result[district] = set()
    for puma in assignment.index:
        for neighbor in neighbors[puma]:
            if assignment[neighbor] != assignment[puma]:
                result[assignment[puma]].add(assignment[neighbor])
    return result


def shortest_district_paths(start, adjacent_districts):
    """
    Breadth-first search over districts. Returns a dict district -> path
    (list of districts from start to that district), nearest districts first.
    """
    paths = {start: [start]}
    queue = [start]
    while len(queue) > 0:
        current = queue.pop(0)
        for neighbor in sorted(adjacent_districts[current]):
            if neighbor not in paths:
                paths[neighbor] = paths[current] + [neighbor]
                queue.append(neighbor)
    return paths


def find_chain_move(assignment, neighbors, population, ideal):
    """
    Used when no single move helps, typically because an overpopulated
    district only borders other overpopulated districts.

    For an overpopulated district A and an underpopulated district Z, follow
    the shortest chain of neighboring districts A -> B -> ... -> Z and pass
    one border PUMA along each link (A gives one to B, B gives one to C, ...).
    The net effect moves population from A toward Z.

    Returns the new assignment if some chain reduces the sum of squared
    population deviations, otherwise None.
    """
    district_population = population.groupby(assignment).sum()
    deviation = district_population - ideal
    adjacent_districts = district_neighbors(assignment, neighbors)

    # Try the most overpopulated districts first
    overpopulated = list(deviation[deviation > 0].sort_values(ascending=False).index)

    for start in overpopulated:
        paths = shortest_district_paths(start, adjacent_districts)

        # Try underpopulated districts, nearest first
        for target, path in paths.items():
            if deviation[target] >= 0 or len(path) < 3:
                continue  # need an underpopulated target; chains of 2 are single moves

            # Simulate passing one PUMA along every link of the chain
            trial = assignment.copy()
            chain_works = True
            for link in range(len(path) - 1):
                giver = path[link]
                receiver = path[link + 1]

                # Border PUMAs of the giver that touch the receiver and can leave safely
                candidates = []
                for puma in members_of(trial, giver):
                    touches_receiver = False
                    for neighbor in neighbors[puma]:
                        if trial[neighbor] == receiver:
                            touches_receiver = True
                    if touches_receiver and stays_connected_without(trial, giver, puma, neighbors):
                        candidates.append(puma)

                if len(candidates) == 0:
                    chain_works = False
                    break

                # How many people the giver has above the ideal right now
                # (including anything it just received earlier in this chain)
                giver_population = population[members_of(trial, giver)].sum()
                giver_excess = giver_population - ideal

                # Pass the PUMA whose population is closest to that excess
                candidates.sort(key=lambda puma: abs(population[puma] - giver_excess))
                trial[candidates[0]] = receiver

            if not chain_works:
                continue

            before = squared_deviation_total(assignment, population, ideal, path)
            after = squared_deviation_total(trial, population, ideal, path)
            if after < before:
                return trial

    return None


def balance_population(assignment, neighbors, population, tolerance, max_steps=1000):
    """
    Greedy local search that makes district populations more equal while
    keeping every district in one piece. At each step:
      1. make the best single PUMA move, if one reduces
         sum over districts of (population - ideal)^2;
      2. otherwise try a chain move (see find_chain_move).
    Stops when all districts are within the tolerance, or nothing helps.
    Every step strictly lowers the sum of squares, so it cannot loop forever.
    Returns (new assignment Series, single moves made, chain moves made).
    """
    assignment = assignment.copy()
    ideal = population.sum() / assignment.nunique()
    single_moves = 0
    chain_moves = 0

    for step in range(max_steps):
        district_population = population.groupby(assignment).sum()
        worst_deviation = ((district_population - ideal).abs() / ideal).max()
        if worst_deviation <= tolerance:
            break

        # 1. Best single move
        move = find_best_single_move(assignment, neighbors, population, ideal)
        if move is not None:
            puma, destination = move
            assignment[puma] = destination
            single_moves = single_moves + 1
            continue

        # 2. Chain move
        new_assignment = find_chain_move(assignment, neighbors, population, ideal)
        if new_assignment is None:
            break
        assignment = new_assignment
        chain_moves = chain_moves + 1

    return assignment, single_moves, chain_moves


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def profile_variance_explained(assignment, vectors, population):
    """
    How well the districts group similar PUMAs, from 0 to 1 (like R^2):
        1 - (variation of profiles inside districts) / (total variation of profiles)
    Every PUMA counts in proportion to its population. Higher = more similar
    communities inside each district.
    """
    weights = population[vectors.index].to_numpy()
    values = vectors.to_numpy()

    # Total variation around the statewide average profile
    overall_mean = np.average(values, axis=0, weights=weights)
    total = np.sum(weights * np.sum((values - overall_mean) ** 2, axis=1))

    # Variation around each district's own average profile
    within = 0.0
    for district in assignment.unique():
        is_member = (assignment[vectors.index] == district).to_numpy()
        member_values = values[is_member]
        member_weights = weights[is_member]
        district_mean = np.average(member_values, axis=0, weights=member_weights)
        within = within + np.sum(member_weights * np.sum((member_values - district_mean) ** 2, axis=1))

    return 1.0 - within / total


def summarize(stage_name, assignment, neighbors, population, vectors):
    """Print one line with the key quality numbers of an assignment."""
    ideal = population.sum() / assignment.nunique()
    district_population = population.groupby(assignment).sum()
    deviation = (district_population - ideal) / ideal

    # Check that every district is one connected piece
    disconnected = 0
    for district in assignment.unique():
        if len(find_connected_pieces(members_of(assignment, district), neighbors)) > 1:
            disconnected = disconnected + 1

    explained = profile_variance_explained(assignment, vectors, population)

    print("  " + stage_name.ljust(18)
          + "districts " + str(assignment.nunique()).rjust(3)
          + " | split districts " + str(disconnected).rjust(2)
          + " | worst pop. deviation " + ("%+.1f%%" % (100 * deviation.abs().max())).rjust(7)
          + " | profile variance explained " + ("%.1f%%" % (100 * explained)).rjust(6))


def print_district_table(assignment, population):
    """Print each district's number of PUMAs, population, and deviation from the ideal."""
    ideal = population.sum() / assignment.nunique()
    print()
    print("  Ideal population per district:", int(round(ideal)))
    print("  " + "district".ljust(10) + "PUMAs".rjust(6) + "population".rjust(12) + "deviation".rjust(11))
    for district in sorted(assignment.unique()):
        members = members_of(assignment, district)
        district_population = population[members].sum()
        deviation = (district_population - ideal) / ideal
        print("  " + str(district).ljust(10) + str(len(members)).rjust(6)
              + str(int(district_population)).rjust(12) + ("%+.1f%%" % (100 * deviation)).rjust(11))


def renumber_districts(assignment):
    """
    Renumber districts 1, 2, 3, ... in the order they first appear when PUMAs
    are sorted by code (low codes are western NY, high codes are NYC/Long Island).
    """
    new_numbers = {}
    for puma in sorted(assignment.index):
        old = assignment[puma]
        if old not in new_numbers:
            new_numbers[old] = len(new_numbers) + 1
    return assignment.map(new_numbers)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Load the step 2 and 3 results, cluster, balance, evaluate, and save."""
    parser = argparse.ArgumentParser(description="Group PUMAs into contiguous, population-balanced districts.")
    parser.add_argument("--districts", type=int, default=NUMBER_OF_DISTRICTS,
                        help="Number of districts (default: 26)")
    parser.add_argument("--tolerance", type=float, default=POPULATION_TOLERANCE,
                        help="Allowed population deviation, e.g. 0.05 = +-5%% (default: 0.05)")
    arguments = parser.parse_args()

    # Inputs from steps 2 and 3
    profiles = load_profiles()
    edges = load_edges()
    neighbors = neighbors_from_edges(edges)

    pumas = list(profiles.index)
    vectors = profiles[profile_columns(profiles)]
    population = profiles["population"]

    print("PUMAs:", len(pumas), "| districts:", arguments.districts,
          "| population tolerance: +-" + str(round(100 * arguments.tolerance, 1)) + "%")
    print()

    # Step 1: agglomerative clustering (districts are connected by construction)
    raw = cluster_agglomerative(pumas, vectors.to_numpy(), neighbors, arguments.districts)
    summarize("raw clustering", raw, neighbors, population, vectors)

    # Step 2: population balance
    balanced, single_moves, chain_moves = balance_population(raw, neighbors, population, arguments.tolerance)
    summarize("after balancing", balanced, neighbors, population, vectors)
    print("  (balancing made", single_moves, "single moves and", chain_moves, "chain moves)")

    # Consistent district numbers 1..26 for both stages
    final = renumber_districts(balanced)
    print_district_table(final, population)

    # Save one row per PUMA with both stages
    result = pd.DataFrame({
        "population": population[pumas],
        "district_raw": renumber_districts(raw),
        "district": final,
    })
    result.index.name = "PUMA"
    result.to_csv(OUTPUT_PATH)
    print()
    print("  Saved:", OUTPUT_PATH)


# Run main() only when this file is executed directly (not when imported)
if __name__ == "__main__":
    main()
