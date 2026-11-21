"""
check_environment.py

Quick sanity check that the virtual environment has everything it needs.
Run it from the project root with the virtual environment activated:
    python check_environment.py
"""

import sys


def check_python():
    """Print the Python version and where it is running from."""
    print("Python version:", sys.version.split()[0])
    print("Python executable:", sys.executable)


def check_numpy_and_pandas():
    """Import numpy and pandas and print their versions."""
    import numpy
    import pandas

    print("numpy version:", numpy.__version__)
    print("pandas version:", pandas.__version__)


def check_sklearn():
    """Import scikit-learn and run a tiny K-Means to confirm it works."""
    import numpy
    import sklearn
    from sklearn.cluster import KMeans

    print("scikit-learn version:", sklearn.__version__)

    # Two obvious groups of 2D points: one near (0, 0) and one near (10, 10)
    points = numpy.array([
        [0.0, 0.0],
        [0.5, 0.2],
        [10.0, 10.0],
        [10.3, 9.8],
    ])

    # Ask K-Means to find 2 clusters
    model = KMeans(n_clusters=2, n_init=10, random_state=0)
    labels = model.fit_predict(points)

    # The first two points should share a label, and the last two another
    same_first_group = labels[0] == labels[1]
    same_second_group = labels[2] == labels[3]
    different_groups = labels[0] != labels[2]

    if same_first_group and same_second_group and different_groups:
        print("  K-Means test: OK")
    else:
        print("  K-Means test: UNEXPECTED labels", labels)


def check_torch():
    """Import PyTorch, report which device is available, and do a small tensor operation."""
    import torch

    print("torch version:", torch.__version__)

    # Pick the best available device (CUDA on NVIDIA GPUs, MPS on Apple Silicon, else CPU)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("  torch device:", device)

    # Multiply a 2x2 matrix by the identity matrix; the result should be unchanged
    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    identity = torch.eye(2, device=device)
    result = matrix @ identity

    if torch.equal(result, matrix):
        print("  tensor test: OK")
    else:
        print("  tensor test: UNEXPECTED result", result)


def main():
    """Run every check in order."""
    check_python()
    check_numpy_and_pandas()
    check_sklearn()
    check_torch()
    print()
    print("Environment looks good.")


# Run main() only when this file is executed directly (not when imported)
if __name__ == "__main__":
    main()
