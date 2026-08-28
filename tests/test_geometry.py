import unittest
from collections import defaultdict, deque

import numpy as np
import trimesh
from trimesh.intersections import mesh_plane

import griddogs as g


def as_mesh(manifold):
    mesh = manifold.to_mesh()
    return trimesh.Trimesh(
        np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3],
        np.asarray(mesh.tri_verts, dtype=np.int64),
        process=True,
    )


def section_component_count(mesh, z):
    graph = defaultdict(set)
    for line in mesh_plane(mesh, [0, 0, 1], [0, 0, z]):
        a = tuple(np.round(line[0], 5))
        b = tuple(np.round(line[1], 5))
        graph[a].add(b)
        graph[b].add(a)

    unseen = set(graph)
    count = 0
    while unseen:
        count += 1
        queue = deque([unseen.pop()])
        while queue:
            for neighbour in graph[queue.popleft()]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    queue.append(neighbour)
    return count


class ParameterTests(unittest.TestCase):
    def test_parameter_couplings(self):
        self.assertAlmostEqual(g.HOLE_PITCH, 14.0)
        self.assertAlmostEqual(g.PEG_L, g.HOLE_DEPTH - g.PEG_CLEAR)
        self.assertGreater(g.PLATE_T, g.HOLE_DEPTH)
        self.assertGreater(g.PEG_SLOT_ROOT, 0)
        self.assertLess(
            g.PEG_ROOT_D,
            g.HOLE_D + 2 * g.HOLE_CHAMF,
        )
        self.assertGreater(g.WALL_T, g.PEG_ROOT_D)
        self.assertGreater(g.CURVE_SIZE, g.PEG_ROOT_D)
        self.assertLess(g.CURVE_SIZE, g.HOLE_PITCH)

    def test_tile_catalogue_contains_every_canonical_h2d_rectangle(self):
        expected = [
            (cols, rows)
            for cols in range(1, 8)
            for rows in range(cols, 8)
        ]
        self.assertEqual(g.TILES, expected)

    def test_h2d_maximum_tile_size(self):
        size_7 = 7 * g.GRID - 2 * g.CLEAR
        size_8 = 8 * g.GRID - 2 * g.CLEAR
        self.assertAlmostEqual(size_7, 293.5)
        self.assertLessEqual(size_7, 320.0)
        self.assertGreater(size_8, 325.0)


class GeometryTests(unittest.TestCase):
    def assert_watertight_size(self, manifold, expected):
        mesh = as_mesh(manifold)
        self.assertTrue(mesh.is_watertight)
        np.testing.assert_allclose(mesh.extents, expected, atol=1e-3)

    def test_rectangular_tile(self):
        self.assert_watertight_size(g.tile(2, 3), [83.5, 125.5, 9.5])

    def test_anchor_variants(self):
        for height in g.ANCHOR_H:
            with self.subTest(part="round", height=height):
                self.assert_watertight_size(
                    g.anchor_round(height),
                    [g.ROUND_D, g.ROUND_D, height + g.PEG_L],
                )
            for pegs in (2, 3):
                with self.subTest(part=f"wall_{pegs}", height=height):
                    length = (pegs - 1) * g.HOLE_PITCH + g.WALL_T + 2.0
                    self.assert_watertight_size(
                        g.anchor_wall(pegs, height),
                        [length, g.WALL_T, height + g.PEG_L],
                    )
            for name, builder in (
                ("standard", g.anchor_curve_standard),
                ("deep", g.anchor_curve_deep),
                ("bowl", g.anchor_curve_bowl),
                ("centre_standard", g.anchor_curve_centre_standard),
                ("centre_deep", g.anchor_curve_centre_deep),
            ):
                with self.subTest(part=name, height=height):
                    self.assert_watertight_size(
                        builder(height),
                        [g.CURVE_SIZE, height + g.PEG_L, g.CURVE_SIZE],
                    )

    def test_peg_slot_terminates_before_anchor_head(self):
        peg = as_mesh(g.peg())
        self.assertEqual(section_component_count(peg, -0.1), 1)
        self.assertEqual(section_component_count(peg, -1.2), 2)

    def test_bone_and_fit_coupon(self):
        self.assert_watertight_size(g.anchor_bone(), [24.0, 17.0, 12.0 + g.PEG_L])
        coupon_length = g.HOLE_PITCH * 5 + 6
        self.assert_watertight_size(g.fit_coupon(), [coupon_length, 15.0, g.PLATE_T])


if __name__ == "__main__":
    unittest.main()
