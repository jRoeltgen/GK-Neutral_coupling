import sys
target_dir = "../common/"
sys.path.insert(0, target_dir)
import numpy as np
import pytest
from triangle_mesh import triangle_mesh
from hypothesis import given, strategies as st


def test_calc_incenter_equilateral():
    mesh = triangle_mesh()

    # Equilateral triangle with side length 2
    # Coordinates:
    # (0,0), (2,0), (1, sqrt(3))
    mesh.nodes = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [1.0, np.sqrt(3)]
    ])

    # One triangle (1-based indexing in cells)
    mesh.cells = np.array([[1, 2, 3]])

    mesh.calc_incenter()

    # For equilateral triangle, incenter = centroid
    expected = np.array([[1.0, np.sqrt(3)/3]])

    assert np.allclose(mesh.incenter, expected)


def test_calc_incenter_scalene():
    mesh = triangle_mesh()

    mesh.nodes = np.array([
        [0.0, 0.0],
        [4.0, 0.0],
        [0.0, 3.0]
    ])

    mesh.cells = np.array([[1, 2, 3]])

    mesh.calc_incenter()

    # Known incenter for this triangle
    # side lengths: a=5, b=3, c=4
    # weighted average formula
    expected = np.array([[1.0, 1.0]])

    assert np.allclose(mesh.incenter, expected)


def test_calc_incenter_multiple():
    mesh = triangle_mesh()

    mesh.nodes = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [1.0, np.sqrt(3)],
        [4.0, 0.0],
        [6.0, 0.0],
        [5.0, np.sqrt(3)]
    ])

    mesh.cells = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    mesh.calc_incenter()

    expected = np.array([
        [1.0, np.sqrt(3)/3],
        [5.0, np.sqrt(3)/3]
    ])

    assert np.allclose(mesh.incenter, expected)


def test_calc_incenter_recalculate_flag():
    mesh = triangle_mesh()

    mesh.nodes = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [1.0, np.sqrt(3)]
    ])
    mesh.cells = np.array([[1, 2, 3]])

    mesh.calc_incenter()
    first = mesh.incenter.copy()

    # Modify nodes
    mesh.nodes[0] = [10.0, 10.0]

    # Without recalc → should NOT change
    mesh.calc_incenter(recalculate=False)
    assert np.allclose(mesh.incenter, first)

    # With recalc → should change
    mesh.calc_incenter(recalculate=True)
    assert not np.allclose(mesh.incenter, first)


def test_calc_incenter_requires_data():
    mesh = triangle_mesh()

    with pytest.raises(AttributeError):
        mesh.calc_incenter()


@given(st.lists(st.tuples(st.floats(-10, 10), st.floats(-10, 10)), min_size=3, max_size=3))
def test_incenter_properties(points):
    pts = np.array(points)

    # Skip degenerate triangles
    area = np.linalg.det(np.array([
        [pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]],
        [pts[2][0] - pts[0][0], pts[2][1] - pts[0][1]],
    ]))
    if abs(area) < 1e-8:
        return

    mesh = triangle_mesh()
    mesh.nodes = pts
    mesh.cells = np.array([[1, 2, 3]])

    mesh.calc_incenter()

    # Property 1: finite
    assert np.isfinite(mesh.incenter).all()

    # Property 2: translation invariance
    shift = np.array([5.0, -2.0])
    mesh2 = triangle_mesh()
    mesh2.nodes = pts + shift
    mesh2.cells = mesh.cells
    mesh2.calc_incenter()

    assert np.allclose(mesh2.incenter, mesh.incenter + shift)