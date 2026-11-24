"""
puma_graph.py   (pipeline step 3)

Builds the graph that the district clustering works on:
  * nodes = PUMAs
  * edges = PUMAs whose borders touch (so districts can be kept connected)
  * edge weight = how similar the two PUMAs' community profiles are

Steps:
  1. Download the Census TIGER/Line PUMA boundary file for New York.
  2. Find neighboring PUMAs ("queen" contiguity: sharing any border point).
  3. If some PUMAs form a separate piece (an island with no land border),
     connect that piece to its geographically closest PUMA, and print the
     link so we can check it makes sense.
  4. Weight every edge with  similarity = exp(-distance^2 / sigma^2), where
     distance is between the two profile vectors and sigma is the median
     distance over all edges. Identical profiles -> 1, very different -> ~0.

Output: results/puma_edges.csv with one row per edge:
    PUMA_A, PUMA_B, profile_distance, similarity, added_link

Usage (run from the project root folder, with .venv activated):
    python src\\puma_graph.py
"""

import urllib.request
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from libpysal.weights import Queen

from puma_profiles import PROFILES_PATH, RESULTS_DIR


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# TIGER/Line PUMA boundaries (2020 PUMA definitions, published in the 2024 release)
TIGER_YEAR = 2024
STATE_FIPS = "36"  # New York
TIGER_URL = ("https://www2.census.gov/geo/tiger/TIGER" + str(TIGER_YEAR)
             + "/PUMA20/tl_" + str(TIGER_YEAR) + "_" + STATE_FIPS + "_puma20.zip")
TIGER_DIR = PROJECT_ROOT / "data" / "raw" / "tiger"
TIGER_ZIP_PATH = TIGER_DIR / ("tl_" + str(TIGER_YEAR) + "_" + STATE_FIPS + "_puma20.zip")

# Projected coordinate system for New York (UTM zone 18N, units = meters),
# used to measure real distances between shapes
METRIC_CRS = "EPSG:32618"

EDGES_PATH = RESULTS_DIR / "puma_edges.csv"


# ---------------------------------------------------------------------------
# Step 3a: PUMA boundary shapes
# ---------------------------------------------------------------------------

def download_puma_shapes(force=False):
    """Download the PUMA boundary zip for New York, unless it is already on disk."""
    if TIGER_ZIP_PATH.exists() and not force:
        print("Shapes already downloaded:", TIGER_ZIP_PATH)
        return TIGER_ZIP_PATH

    TIGER_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading:", TIGER_URL)
    request = urllib.request.Request(TIGER_URL, headers={"User-Agent": "Mozilla/5.0 (pums-loader)"})
    with urllib.request.urlopen(request) as response:
        content = response.read()
    with open(TIGER_ZIP_PATH, "wb") as output_file:
        output_file.write(content)
    print("Saved to:", TIGER_ZIP_PATH)
    return TIGER_ZIP_PATH


def load_puma_shapes():
    """
    Return a GeoDataFrame with one row per PUMA, indexed by the 5-digit PUMA
    code (same format as the PUMS "PUMA" column, e.g. "01500"), with columns:
        name      - the Census description of the area
        geometry  - the PUMA boundary polygon
    """
    zip_path = download_puma_shapes()

    # geopandas can read the shapefile directly from the zip
    shapes = gpd.read_file(zip_path)

    # Keep only what we need, with simple column names
    shapes = shapes[["PUMACE20", "NAMELSAD20", "geometry"]]
    shapes = shapes.rename(columns={"PUMACE20": "PUMA", "NAMELSAD20": "name"})
    shapes = shapes.set_index("PUMA").sort_index()
    return shapes


# ---------------------------------------------------------------------------
# Step 3b: which PUMAs touch each other
# ---------------------------------------------------------------------------

def find_neighbors(shapes):
    """
    Return a dict PUMA -> set of neighboring PUMAs.
    "Queen" contiguity: two PUMAs are neighbors if their borders share at
    least one point.
    """
    weights = Queen.from_dataframe(shapes, use_index=True, silence_warnings=True)

    neighbors = {}
    for puma in shapes.index:
        neighbors[puma] = set(weights.neighbors[puma])
    return neighbors


def find_connected_pieces(pumas, neighbors):
    """
    Split a list of PUMAs into connected pieces (groups that can reach each
    other by stepping between neighbors). Only steps inside the given list
    are allowed. Returns a list of sets, largest first.
    """
    remaining = set(pumas)
    pieces = []

    while len(remaining) > 0:
        # Start a new piece from any remaining PUMA and explore outward (breadth-first search)
        start = remaining.pop()
        piece = {start}
        to_visit = [start]

        while len(to_visit) > 0:
            current = to_visit.pop()
            for neighbor in neighbors[current]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    piece.add(neighbor)
                    to_visit.append(neighbor)

        pieces.append(piece)

    pieces.sort(key=len, reverse=True)
    return pieces


def connect_separate_pieces(shapes, neighbors):
    """
    If the PUMAs do not form one connected graph (e.g. an island with no land
    border), link each smaller piece to the closest PUMA outside it, measured
    between the boundary shapes. Repeats until everything is connected.

    Modifies neighbors in place and returns the list of added links (PUMA_A, PUMA_B).
    """
    metric_shapes = shapes.to_crs(METRIC_CRS)
    added_links = []

    pieces = find_connected_pieces(list(shapes.index), neighbors)
    while len(pieces) > 1:
        # Take the smallest piece and find its closest PUMA in any other piece
        smallest_piece = pieces[-1]
        best_distance = None
        best_link = None

        for inside_puma in smallest_piece:
            inside_shape = metric_shapes.geometry[inside_puma]
            for outside_puma in metric_shapes.index:
                if outside_puma in smallest_piece:
                    continue
                distance = inside_shape.distance(metric_shapes.geometry[outside_puma])
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_link = (inside_puma, outside_puma)

        # Add the link in both directions
        puma_a, puma_b = best_link
        neighbors[puma_a].add(puma_b)
        neighbors[puma_b].add(puma_a)
        added_links.append(best_link)
        print("Added link between separate pieces:", puma_a, "<->", puma_b,
              "(" + str(round(best_distance / 1000, 1)) + " km apart)")
        print("   ", shapes.loc[puma_a, "name"])
        print("   ", shapes.loc[puma_b, "name"])

        pieces = find_connected_pieces(list(shapes.index), neighbors)

    return added_links


# ---------------------------------------------------------------------------
# Step 3c: edge weights from profile similarity
# ---------------------------------------------------------------------------

def load_profiles():
    """Read results/puma_profiles.csv (from step 2), keeping PUMA codes as text."""
    profiles = pd.read_csv(PROFILES_PATH, dtype={"PUMA": str})
    profiles = profiles.set_index("PUMA").sort_index()
    return profiles


def profile_columns(profiles):
    """Return the names of the PCA columns (PC1, PC2, ...) of the profiles table."""
    columns = []
    for column in profiles.columns:
        if column.startswith("PC"):
            columns.append(column)
    return columns


def build_edge_table(neighbors, profiles, added_links):
    """
    One row per pair of neighboring PUMAs (each pair listed once), with the
    distance between their profile vectors and the similarity weight.
    """
    pc_columns = profile_columns(profiles)
    vectors = profiles[pc_columns]

    # Set of added links, in both orders, to mark them in the table
    added_set = set()
    for puma_a, puma_b in added_links:
        added_set.add((puma_a, puma_b))
        added_set.add((puma_b, puma_a))

    rows = []
    for puma_a in sorted(neighbors):
        for puma_b in sorted(neighbors[puma_a]):
            # List each pair once (A < B)
            if puma_a >= puma_b:
                continue
            difference = vectors.loc[puma_a].to_numpy() - vectors.loc[puma_b].to_numpy()
            distance = float(np.sqrt(np.sum(difference ** 2)))
            rows.append({
                "PUMA_A": puma_a,
                "PUMA_B": puma_b,
                "profile_distance": distance,
                "added_link": (puma_a, puma_b) in added_set,
            })

    edges = pd.DataFrame(rows)

    # sigma = median edge distance, so a "typical" pair of neighbors gets similarity exp(-1) = 0.37
    sigma = edges["profile_distance"].median()
    edges["similarity"] = np.exp(-(edges["profile_distance"] ** 2) / (sigma ** 2))
    print("Similarity scale sigma (median neighbor distance):", round(sigma, 4))
    return edges


def load_edges():
    """Read results/puma_edges.csv (from this step), keeping PUMA codes as text."""
    return pd.read_csv(EDGES_PATH, dtype={"PUMA_A": str, "PUMA_B": str})


def neighbors_from_edges(edges):
    """Rebuild the dict PUMA -> set of neighbors from the edge table."""
    neighbors = {}
    for puma_a, puma_b in zip(edges["PUMA_A"], edges["PUMA_B"]):
        neighbors.setdefault(puma_a, set()).add(puma_b)
        neighbors.setdefault(puma_b, set()).add(puma_a)
    return neighbors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Build the PUMA graph for NY and save the edges to results/puma_edges.csv."""
    shapes = load_puma_shapes()
    profiles = load_profiles()

    # The shapes and the survey data must describe the same set of PUMAs
    pumas_in_shapes = set(shapes.index)
    pumas_in_profiles = set(profiles.index)
    if pumas_in_shapes != pumas_in_profiles:
        print("Only in shapes:", sorted(pumas_in_shapes - pumas_in_profiles))
        print("Only in profiles:", sorted(pumas_in_profiles - pumas_in_shapes))
        raise ValueError("PUMA codes in the shapefile and in the PUMS data do not match")
    print("PUMAs:", len(shapes))

    # Neighbors from the shapes, then make sure the whole graph is connected
    neighbors = find_neighbors(shapes)
    pieces_before = find_connected_pieces(list(shapes.index), neighbors)
    print("Connected pieces from land borders:", len(pieces_before))
    added_links = connect_separate_pieces(shapes, neighbors)

    # Edge table with similarity weights
    edges = build_edge_table(neighbors, profiles, added_links)

    neighbor_counts = []
    for puma in neighbors:
        neighbor_counts.append(len(neighbors[puma]))
    print("Edges:", len(edges), "| neighbors per PUMA: min", min(neighbor_counts),
          "| median", int(np.median(neighbor_counts)), "| max", max(neighbor_counts))
    print("Similarity: min", round(edges["similarity"].min(), 3),
          "| median", round(edges["similarity"].median(), 3),
          "| max", round(edges["similarity"].max(), 3))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    edges.to_csv(EDGES_PATH, index=False)
    print("Saved:", EDGES_PATH)


# Run main() only when this file is executed directly (not when imported)
if __name__ == "__main__":
    main()
