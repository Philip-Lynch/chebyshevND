"""Tests for ChebyshevGrid: node order, physical points, chunks and fits (Step 5b)."""

import numpy as np
import pytest

from chebyshevnd import (
    Axis,
    AxisMap,
    ChebyshevGrid,
    ChebyshevInterpolantND,
    MappedInterpolant,
    chebyshev_gauss_lobatto_nodes,
    chebyshev_nodes,
)


def plain_grid(nodes="Chebyshev"):
    return ChebyshevGrid([Axis("x", 7, (0.0, 2.0)), Axis("y", 5, (-1.0, 3.0)), Axis("z", 4, (1.0, 4.0))],
                         nodes=nodes)


def f(x, y, z):
    return np.sin(x) * np.cos(0.3 * y) + 0.1 * z**2


def g(x, y, z):
    return np.exp(1j * x) * y + z


def mapped_grid(boundary_scale=1.0):
    rn = chebyshev_nodes(30, 2.0, 10.0)
    boundary = ChebyshevInterpolantND((boundary_scale * np.sqrt(rn))[:, None], [(2.0, 10.0)],
                                      axis_names=["r"], output_names=["ref"])
    inv = AxisMap("power", power=-1)
    return ChebyshevGrid([Axis("r", 30, inv.table_domain(2.0, 10.0), inv),
                          Axis("p", 6, (-0.5, 0.5), AxisMap(lower="ref", width=0.2))],
                         boundary=boundary)


def test_axis_validation():
    with pytest.raises(ValueError, match="positive integer"):
        Axis("x", 0, (0, 1))
    with pytest.raises(ValueError, match="lower bound"):
        Axis("x", 4, (1, 0))
    with pytest.raises(ValueError, match="non-empty"):
        Axis("", 4, (0, 1))
    with pytest.raises(TypeError, match="AxisMap"):
        Axis("x", 4, (0, 1), map="log")
    with pytest.raises(ValueError, match="unique"):
        ChebyshevGrid([Axis("x", 3, (0, 1)), Axis("x", 3, (0, 1))])
    with pytest.raises(ValueError, match="at least 2"):
        ChebyshevGrid([Axis("x", 1, (0, 1))], nodes="GaussLobatto")


@pytest.mark.parametrize("nodes", ["Chebyshev", "GaussLobatto"])
def test_order_matches_meshgrid(nodes):
    grid = plain_grid(nodes)
    assert grid.shape == (7, 5, 4) and grid.size == 140
    node_fn = chebyshev_nodes if nodes == "Chebyshev" else chebyshev_gauss_lobatto_nodes
    for axis_nodes, axis in zip(grid.table_nodes(), grid.axes):
        np.testing.assert_array_equal(axis_nodes, node_fn(axis.n, *axis.domain))
    mesh = np.meshgrid(*grid.table_nodes(), indexing="ij")
    expected = np.stack([m.ravel() for m in mesh], axis=1)
    np.testing.assert_array_equal(grid.table_points(), expected)
    np.testing.assert_array_equal(grid.table_points(17, 40), expected[17:40])
    # Without maps, physical = table coordinates
    np.testing.assert_array_equal(grid.physical_points(), expected)
    assert grid.linear_index(grid.multi_index(37)) == 37
    assert grid.multi_index(37) == (1, 4, 1)


def test_fit_matches_direct_fit():
    grid = plain_grid()
    x = grid.physical_points()
    interp = grid.fit({"f": f(*x.T), "g": g(*x.T)})
    assert isinstance(interp, ChebyshevInterpolantND)
    assert interp.output_names == ("f", "g") and interp.axis_names == ("x", "y", "z")
    mesh = np.meshgrid(*grid.table_nodes(), indexing="ij")
    direct = ChebyshevInterpolantND(np.stack([f(*mesh), g(*mesh)], axis=-1), grid.domains,
                                    output_names=["f", "g"])
    np.testing.assert_array_equal(interp.get_coefficients(), direct.get_coefficients())
    single = grid.fit(f(*x.T).reshape(grid.shape))
    assert single.output_names == ()
    np.testing.assert_array_equal(single.get_coefficients(),
                                  ChebyshevInterpolantND(f(*mesh), grid.domains).get_coefficients())
    with pytest.raises(ValueError, match="shape"):
        grid.fit(np.zeros(5))


def test_mapped_grid_physical_points_and_fit():
    grid = mapped_grid()
    y = grid.table_points()
    x = grid.physical_points()
    np.testing.assert_allclose(x[:, 0], 1 / y[:, 0], rtol=1e-14)
    ref = grid.boundary.evaluate_batch(x[:, :1])[:, 0]
    np.testing.assert_allclose(x[:, 1], ref + 0.2 * y[:, 1], rtol=1e-14)
    np.testing.assert_allclose(ref, np.sqrt(x[:, 0]), rtol=1e-13)
    interp = grid.fit({"h": x[:, 1] ** 2 / x[:, 0]})
    assert isinstance(interp, MappedInterpolant)
    np.testing.assert_allclose(interp.to_physical(y), x, rtol=1e-13)
    r, p = 5.0, np.sqrt(5.0) + 0.03
    assert interp.evaluate(r, p)[0] == pytest.approx(p**2 / r, rel=1e-11)


def test_chunks_save_load_fit(tmp_path):
    grid = plain_grid()
    assert grid.chunks(50) == [(0, 50), (50, 100), (100, 140)]
    x = grid.physical_points()
    paths = []
    for i, (start, stop) in enumerate(grid.chunks(50)):
        pts = grid.physical_points(start, stop)
        path = tmp_path / f"chunk{i}.npz"
        grid.save_chunk(path, start, stop, {"f": f(*pts.T), "g": g(*pts.T)})
        paths.append(path)
    values = grid.load_chunks(reversed(paths))  # order of files does not matter
    assert set(values) == {"f", "g"} and values["g"].dtype == np.complex128
    np.testing.assert_array_equal(values["f"], f(*x.T).reshape(grid.shape))
    merged = grid.fit(values)
    direct = grid.fit({"f": f(*x.T), "g": g(*x.T)})
    np.testing.assert_array_equal(merged.get_coefficients(), direct.get_coefficients())


def test_load_chunks_validation(tmp_path):
    grid = plain_grid()

    def save(name, start, stop, target=grid, funcs=("f",)):
        pts = target.physical_points(start, stop)
        path = tmp_path / name
        target.save_chunk(path, start, stop, {fn: f(*pts.T) for fn in funcs})
        return path

    a, b = save("a.npz", 0, 70), save("b.npz", 70, 140)
    with pytest.raises(ValueError, match=r"Nodes \[70, 140\) are in no chunk"):
        grid.load_chunks([a])
    with pytest.raises(ValueError, match="overlap"):
        grid.load_chunks([a, b, save("c.npz", 60, 80)])
    with pytest.raises(ValueError, match=r"Nodes \[0, 70\)"):
        grid.load_chunks([b])
    other = ChebyshevGrid([Axis("x", 7, (0.0, 2.0)), Axis("y", 5, (-1.0, 3.0)), Axis("z", 4, (1.0, 5.0))])
    with pytest.raises(ValueError, match="different grid"):
        grid.load_chunks([a, save("d.npz", 70, 140, target=other)])
    with pytest.raises(ValueError, match="holds functions"):
        grid.load_chunks([a, save("e.npz", 70, 140, funcs=("f", "h"))])
    with pytest.raises(ValueError, match="shape"):
        grid.save_chunk(tmp_path / "x.npz", 0, 10, {"f": np.zeros(9)})
    with pytest.raises(ValueError, match="start <= stop"):
        grid.save_chunk(tmp_path / "x.npz", 10, 200, {"f": np.zeros(190)})
    with pytest.raises(ValueError, match="No chunk files"):
        grid.load_chunks([])


def test_grid_hash():
    assert plain_grid().grid_hash() == plain_grid().grid_hash()
    assert len(plain_grid().grid_hash()) == 64
    variants = [
        plain_grid("GaussLobatto"),
        ChebyshevGrid([Axis("x", 8, (0.0, 2.0)), Axis("y", 5, (-1.0, 3.0)), Axis("z", 4, (1.0, 4.0))]),
        ChebyshevGrid([Axis("x", 7, (0.0, 2.5)), Axis("y", 5, (-1.0, 3.0)), Axis("z", 4, (1.0, 4.0))]),
        ChebyshevGrid([Axis("x", 7, (0.0, 2.0)), Axis("w", 5, (-1.0, 3.0)), Axis("z", 4, (1.0, 4.0))]),
        ChebyshevGrid([Axis("x", 7, (0.0, 2.0)), Axis("y", 5, (-1.0, 3.0)),
                       Axis("z", 4, (1.0, 4.0), AxisMap("log"))]),
    ]
    hashes = {grid.grid_hash() for grid in variants} | {plain_grid().grid_hash()}
    assert len(hashes) == len(variants) + 1
    assert mapped_grid().grid_hash() == mapped_grid().grid_hash()
    assert mapped_grid().grid_hash() != mapped_grid(boundary_scale=1.0 + 1e-15).grid_hash()


def test_boundary_must_cover_grid():
    rn = chebyshev_nodes(8, 3.0, 10.0)
    boundary = ChebyshevInterpolantND(np.sqrt(rn)[:, None], [(3.0, 10.0)], axis_names=["r"],
                                      output_names=["ref"])
    with pytest.raises(ValueError, match="covers r in"):
        ChebyshevGrid([Axis("r", 5, (2.0, 10.0)), Axis("p", 4, (0.0, 1.0), AxisMap(lower="ref"))],
                      boundary=boundary)
