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


def section_extents(mesh, z):
    lines = mesh_plane(mesh, [0, 0, 1], [0, 0, z])
    points = lines.reshape(-1, 3)
    return points.max(axis=0) - points.min(axis=0)


class ParameterTests(unittest.TestCase):
    def test_parameter_couplings(self):
        self.assertAlmostEqual(g.HOLE_PITCH, 14.0)
        self.assertLessEqual(g.PEG_L, g.HOLE_DEPTH - g.PEG_CLEAR)
        self.assertAlmostEqual(g.WALL_PEG_L, g.HOLE_DEPTH - g.PEG_CLEAR)
        self.assertAlmostEqual(g.SINGLE_PEG_L, g.HOLE_DEPTH - g.PEG_CLEAR)
        self.assertGreater(sum(g.FOOT_H) + g.PLATE_T, g.HOLE_DEPTH)
        self.assertGreater(g.PEG_SLOT_ROOT, 0)
        self.assertLess(
            g.PEG_ROOT_D,
            g.HOLE_D + 2 * g.HOLE_CHAMF,
        )
        self.assertGreater(g.WALL_T, g.PEG_ROOT_D)
        self.assertGreater(g.CURVE_SIZE, g.PEG_ROOT_D)
        self.assertLess(g.CURVE_SIZE, g.HOLE_PITCH)
        self.assertGreater(g.PEG_RIDGE_D, g.PEG_D)
        self.assertAlmostEqual(g.HOLE_DEPTH - g.SINGLE_PEG_L, g.PEG_CLEAR)
        magnet_gap = sum(g.FOOT_H) + g.PLATE_T - g.HOLE_DEPTH - g.MAG_DEPTH
        self.assertAlmostEqual(magnet_gap, 1.6)

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
                    [g.ROUND_D, g.ROUND_D, height + g.SINGLE_PEG_L],
                )
            for name, span in (
                ("wall_short", g.HOLE_PITCH),
                ("wall_long", 2 * g.HOLE_PITCH),
            ):
                with self.subTest(part=name, height=height):
                    length = span + g.WALL_T + 2.0
                    self.assert_watertight_size(
                        g.anchor_wall(span, height),
                        [length, g.WALL_T, height + g.WALL_PEG_L],
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
                        [g.CURVE_SIZE, height + g.SINGLE_PEG_L, g.CURVE_SIZE],
                    )

    def test_peg_slot_terminates_before_anchor_head(self):
        peg = as_mesh(g.peg())
        self.assertEqual(section_component_count(peg, -0.1), 1)
        self.assertEqual(section_component_count(peg, -1.2), 2)

    def test_single_peg_ridge(self):
        peg = as_mesh(g.peg(ridge_d=g.PEG_RIDGE_D, peg_l=g.SINGLE_PEG_L))
        self.assertAlmostEqual(
            max(section_extents(peg, -g.PEG_RIDGE_Z)[:2]),
            g.PEG_RIDGE_D,
            places=3,
        )
        self.assertAlmostEqual(
            max(section_extents(peg, -1.5)[:2]),
            g.PEG_D,
            places=3,
        )

    def test_tile_hole_depth(self):
        tile = as_mesh(g.tile(1, 1))
        np.testing.assert_allclose(tile.extents, [41.5, 41.5, 9.5], atol=1e-3)
        self.assertTrue(tile.is_watertight)
        lines = mesh_plane(tile, [0, 1, 0], [0, 0, 0])
        points = lines.reshape(-1, 3)
        floor_points = points[
            (np.abs(points[:, 0]) < g.HOLE_D / 2 - 0.1)
            & (points[:, 2] > sum(g.FOOT_H) - 1.0)
        ]
        hole_floor_z = floor_points[:, 2].min()
        self.assertAlmostEqual(
            sum(g.FOOT_H) + g.PLATE_T - hole_floor_z,
            g.HOLE_DEPTH,
            places=3,
        )

    def test_wall_pegs_are_solid_and_at_the_ends(self):
        h = g.ANCHOR_H[1]
        for span in (g.HOLE_PITCH, 2 * g.HOLE_PITCH):
            with self.subTest(span=span):
                wall = as_mesh(g.anchor_wall(span, h))
                section_z = h + (
                    g.PEG_ROOT_CH + g.WALL_PEG_L - g.PEG_TIP_CH
                ) / 2
                self.assertEqual(section_component_count(wall, section_z), 2)
                section = mesh_plane(wall, [0, 0, 1], [0, 0, section_z])
                points = section.reshape(-1, 3)
                left = points[points[:, 0] < 0]
                right = points[points[:, 0] > 0]
                self.assertAlmostEqual(
                    right[:, 0].mean() - left[:, 0].mean(),
                    span,
                    places=3,
                )
                self.assertAlmostEqual(
                    max(np.ptp(left[:, :2], axis=0)),
                    g.WALL_PEG_D,
                    places=3,
                )

    def test_bone_and_fit_coupon(self):
        self.assert_watertight_size(g.anchor_bone(), [24.0, 17.0, 12.0 + g.PEG_L])
        coupon_length = g.HOLE_PITCH * 5 + 6
        self.assert_watertight_size(g.fit_coupon(), [coupon_length, 15.0, g.PLATE_T])


if __name__ == "__main__":
    unittest.main()
