"""
draw_districts.py   (pipeline step 5)

Draws the districts from step 4 (results/districts_agglomerative.csv) on a map
of New York State, with a zoomed panel for New York City (about a third of the
PUMAs are in NYC, so its districts are too small to see on the state view).

It saves three images in results/maps/:
  * districts_agglomerative.png      final districts (after population balancing)
  * districts_agglomerative_raw.png  districts straight from the clustering
  * population_deviation.png         how far each final district is from the
                                     ideal population (blue = too few people,
                                     red = too many)

How the map is built:
  * Each district's shape = its PUMA shapes merged together ("dissolve").
  * The PUMA shapes extend into lakes and the ocean, so the merged shapes are
    trimmed to New York's land outline (Census cartographic state boundary).
  * Colors only separate neighboring districts (no two neighbors share a
    color; four colors are always enough on a map). The number printed on
    each district is what identifies it.

Usage (run from the project root folder, with .venv activated):
    python src\\draw_districts.py
"""

import urllib.request

import geopandas as gpd
import matplotlib
import pandas as pd

matplotlib.use("Agg")  # draw straight to image files, no window
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter

from puma_graph import METRIC_CRS, TIGER_DIR, load_edges, load_puma_shapes, neighbors_from_edges
from puma_profiles import RESULTS_DIR


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

DISTRICTS_PATH = RESULTS_DIR / "districts_agglomerative.csv"
MAPS_DIR = RESULTS_DIR / "maps"

# Census cartographic state outlines (trimmed to the shoreline)
STATE_OUTLINE_URL = "https://www2.census.gov/geo/tiger/GENZ2024/shp/cb_2024_us_state_500k.zip"
STATE_OUTLINE_PATH = TIGER_DIR / "cb_2024_us_state_500k.zip"
NEW_YORK_FIPS = "36"

# Four district colors (blue, yellow, magenta, green). This set was checked
# with a color-vision-deficiency validator: every pair stays distinguishable.
DISTRICT_COLORS = ["#2a78d6", "#eda100", "#e87ba4", "#008300"]

# Number labels are white, except on yellow where dark text reads better
LABEL_INK = {"#2a78d6": "#ffffff", "#eda100": "#0b0b0b", "#e87ba4": "#ffffff", "#008300": "#ffffff"}

# Page and text colors
SURFACE = "#fcfcfb"        # background, and the thin gap drawn between districts
PRIMARY_INK = "#0b0b0b"    # titles
SECONDARY_INK = "#52514e"  # subtitles
MUTED_INK = "#898781"      # zoom-box outline and small notes

# Diverging colors for the population map: blue (too few) - gray (ideal) - red (too many)
UNDER_COLOR = "#2a78d6"
IDEAL_COLOR = "#f0efec"
OVER_COLOR = "#e34948"

plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]


# ---------------------------------------------------------------------------
# Loading shapes and districts
# ---------------------------------------------------------------------------

def load_land_outline():
    """Return New York's land outline (one shape, in meters), downloading it once."""
    if not STATE_OUTLINE_PATH.exists():
        TIGER_DIR.mkdir(parents=True, exist_ok=True)
        print("Downloading:", STATE_OUTLINE_URL)
        request = urllib.request.Request(STATE_OUTLINE_URL, headers={"User-Agent": "Mozilla/5.0 (pums-loader)"})
        with urllib.request.urlopen(request) as response:
            content = response.read()
        with open(STATE_OUTLINE_PATH, "wb") as output_file:
            output_file.write(content)

    states = gpd.read_file(STATE_OUTLINE_PATH)
    new_york = states[states["STATEFP"] == NEW_YORK_FIPS].to_crs(METRIC_CRS)
    return new_york.geometry.union_all()


def load_district_table():
    """Read results/districts_agglomerative.csv (from step 4), keeping PUMA codes as text."""
    table = pd.read_csv(DISTRICTS_PATH, dtype={"PUMA": str})
    return table.set_index("PUMA")


def build_district_shapes(puma_shapes, assignment, land):
    """
    Merge the PUMA shapes of each district into one shape, then trim it to land.
    Returns a GeoDataFrame indexed by district number.
    """
    shapes = puma_shapes.to_crs(METRIC_CRS).copy()
    shapes["district"] = assignment[shapes.index]
    districts = shapes[["district", "geometry"]].dissolve(by="district")
    districts["geometry"] = districts.geometry.intersection(land)
    return districts


def find_nyc_box(puma_shapes, margin_meters=4000):
    """Bounding box (xmin, ymin, xmax, ymax) around the NYC PUMAs, plus a margin."""
    nyc = puma_shapes[puma_shapes["name"].str.startswith("NYC-")].to_crs(METRIC_CRS)
    xmin, ymin, xmax, ymax = nyc.total_bounds
    return (xmin - margin_meters, ymin - margin_meters, xmax + margin_meters, ymax + margin_meters)


# ---------------------------------------------------------------------------
# Coloring
# ---------------------------------------------------------------------------

def district_adjacency(assignment, neighbors):
    """Dict district -> set of districts that share a border with it."""
    adjacency = {}
    for district in assignment.unique():
        adjacency[district] = set()
    for puma in assignment.index:
        for neighbor in neighbors[puma]:
            if assignment[neighbor] != assignment[puma]:
                adjacency[assignment[puma]].add(assignment[neighbor])
    return adjacency


def color_districts(adjacency, number_of_colors):
    """
    Give every district a color number (0 .. number_of_colors - 1) so that
    neighboring districts never get the same one. Backtracking search: color
    districts one by one (most neighbors first); if a district has no free
    color, go back and change an earlier choice.
    """
    order = sorted(adjacency, key=lambda district: len(adjacency[district]), reverse=True)
    colors = {}

    def color_from(position):
        # All districts colored: done
        if position == len(order):
            return True
        district = order[position]
        for color in range(number_of_colors):
            # Is this color already used by a neighbor?
            used_by_neighbor = False
            for neighbor in adjacency[district]:
                if colors.get(neighbor) == color:
                    used_by_neighbor = True
            if used_by_neighbor:
                continue
            colors[district] = color
            if color_from(position + 1):
                return True
            del colors[district]  # undo and try the next color
        return False

    if not color_from(0):
        raise ValueError("Could not color the districts with " + str(number_of_colors) + " colors")
    return colors


def text_ink_for(fill_color):
    """Dark text on light fills, white text on dark fills."""
    red, green, blue = to_rgb(fill_color)
    brightness = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    if brightness > 0.55:
        return PRIMARY_INK
    return "#ffffff"


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def shape_parts(shape):
    """Split a shape into its separate pieces (a district can have several, e.g. islands)."""
    if shape.is_empty:
        return []
    if shape.geom_type in ("MultiPolygon", "GeometryCollection"):
        return list(shape.geoms)
    return [shape]


def draw_labels(axis, districts, labels, label_inks, min_area, keep_point):
    """
    Write each district's label on every piece of it larger than min_area
    (square meters), so pieces that are far apart can still be identified.
    keep_point(x, y) decides whether a label position is used in this panel.
    """
    for district, shape in districts.geometry.items():
        for part in shape_parts(shape):
            if part.geom_type != "Polygon" or part.area < min_area:
                continue
            point = part.representative_point()
            if not keep_point(point.x, point.y):
                continue
            axis.text(point.x, point.y, labels[district], ha="center", va="center",
                      fontsize=9, fontweight="bold", color=label_inks[district])


def draw_map(districts, fill_colors, labels, label_inks, nyc_box, title, subtitle, output_path, colorbar=None):
    """
    Two panels: the whole state (left) and NYC zoomed in (right).
    fill_colors, labels and label_inks are dicts district -> value.
    colorbar (optional) is (colormap, norm, caption) for the population map.
    """
    figure, (state_axis, nyc_axis) = plt.subplots(
        1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.6, 1]}, facecolor=SURFACE)

    colors_in_order = []
    for district in districts.index:
        colors_in_order.append(fill_colors[district])

    # Fill every district; the thin surface-colored edge is the gap between districts
    for axis in [state_axis, nyc_axis]:
        districts.plot(ax=axis, color=colors_in_order, edgecolor=SURFACE, linewidth=1.2)
        axis.set_axis_off()
        axis.set_aspect("equal")

    xmin, ymin, xmax, ymax = nyc_box

    # State panel: outline of the zoomed area, and labels away from it
    # (pieces near NYC are crowded there, and are labeled in the NYC panel)
    state_axis.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, edgecolor=MUTED_INK, linewidth=1))
    crowded_margin = 45000  # meters around the zoom box with no state-panel labels

    def far_from_nyc_box(x, y):
        inside_x = xmin - crowded_margin <= x <= xmax + crowded_margin
        inside_y = ymin - crowded_margin <= y <= ymax + crowded_margin
        return not (inside_x and inside_y)

    draw_labels(state_axis, districts, labels, label_inks, 40e6, far_from_nyc_box)  # pieces > 40 km2

    # NYC panel: zoom in, and label every sizable piece inside the box
    nyc_axis.set_xlim(xmin, xmax)
    nyc_axis.set_ylim(ymin, ymax)
    box_area = (xmax - xmin) * (ymax - ymin)
    clipped = districts.clip((xmin, ymin, xmax, ymax))

    def anywhere(x, y):
        return True

    draw_labels(nyc_axis, clipped, labels, label_inks, 0.004 * box_area, anywhere)
    nyc_axis.set_title("New York City", fontsize=12, color=SECONDARY_INK, loc="left")

    # Optional color scale (population map)
    if colorbar is not None:
        colormap, norm, caption = colorbar
        scale = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
        bar = figure.colorbar(scale, ax=[state_axis, nyc_axis], orientation="horizontal",
                              fraction=0.04, pad=0.02, aspect=50)
        bar.set_label(caption, color=SECONDARY_INK)
        bar.ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))  # 0.1 -> "10%"
        bar.outline.set_visible(False)
        bar.ax.tick_params(colors=SECONDARY_INK)

    # Title and subtitle, left-aligned
    figure.text(0.02, 0.96, title, fontsize=16, fontweight="bold", color=PRIMARY_INK)
    figure.text(0.02, 0.925, subtitle, fontsize=11, color=SECONDARY_INK)

    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    print("Saved:", output_path)


# ---------------------------------------------------------------------------
# The three maps
# ---------------------------------------------------------------------------

def population_deviation(assignment, population):
    """Dict district -> deviation from the ideal population (0.10 = 10% too many people)."""
    district_population = population.groupby(assignment).sum()
    ideal = population.sum() / assignment.nunique()
    deviation = {}
    for district in district_population.index:
        deviation[district] = (district_population[district] - ideal) / ideal
    return deviation


def draw_district_map(puma_shapes, land, neighbors, table, column, nyc_box, title, output_path):
    """Map of the districts in one column of the table, colored so neighbors differ."""
    assignment = table[column]
    districts = build_district_shapes(puma_shapes, assignment, land)

    color_numbers = color_districts(district_adjacency(assignment, neighbors), len(DISTRICT_COLORS))
    fill_colors = {}
    label_inks = {}
    labels = {}
    for district in districts.index:
        fill_colors[district] = DISTRICT_COLORS[color_numbers[district]]
        label_inks[district] = LABEL_INK[fill_colors[district]]
        labels[district] = str(district)

    deviation = population_deviation(assignment, table["population"])
    worst = max(deviation.values(), key=abs)
    subtitle = (str(len(districts)) + " districts built from whole PUMAs  ·  worst population deviation "
                + ("%+.1f%%" % (100 * worst)) + " from the ideal")
    draw_map(districts, fill_colors, labels, label_inks, nyc_box, title, subtitle, output_path)


def draw_deviation_map(puma_shapes, land, table, nyc_box, output_path):
    """Map of the final districts colored by how far their population is from the ideal."""
    assignment = table["district"]
    districts = build_district_shapes(puma_shapes, assignment, land)
    deviation = population_deviation(assignment, table["population"])

    # Symmetric scale around 0, rounded up to the next 5%
    largest = max(abs(value) for value in deviation.values())
    limit = 0.05 * (int(largest / 0.05) + 1)
    colormap = LinearSegmentedColormap.from_list("deviation", [UNDER_COLOR, IDEAL_COLOR, OVER_COLOR])
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    fill_colors = {}
    label_inks = {}
    labels = {}
    for district in districts.index:
        fill_colors[district] = colormap(norm(deviation[district]))
        label_inks[district] = text_ink_for(fill_colors[district])
        labels[district] = str(district) + "\n" + ("%+.0f%%" % (100 * deviation[district]))

    colorbar = (colormap, norm, "Population difference from the ideal district")
    draw_map(districts, fill_colors, labels, label_inks, nyc_box,
             "How equal are the district populations?",
             "Final agglomerative districts  ·  blue = fewer people than the ideal, red = more",
             output_path, colorbar)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Load shapes and step 4 results, then draw and save the three maps."""
    puma_shapes = load_puma_shapes()
    land = load_land_outline()
    neighbors = neighbors_from_edges(load_edges())
    table = load_district_table()
    nyc_box = find_nyc_box(puma_shapes)

    draw_district_map(puma_shapes, land, neighbors, table, "district", nyc_box,
                      "New York districts: agglomerative clustering + population balancing",
                      MAPS_DIR / "districts_agglomerative.png")
    draw_district_map(puma_shapes, land, neighbors, table, "district_raw", nyc_box,
                      "New York districts: agglomerative clustering only (before balancing)",
                      MAPS_DIR / "districts_agglomerative_raw.png")
    draw_deviation_map(puma_shapes, land, table, nyc_box, MAPS_DIR / "population_deviation.png")


# Run main() only when this file is executed directly (not when imported)
if __name__ == "__main__":
    main()
