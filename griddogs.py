#!/usr/bin/env python3
"""
Parametric tool-anchor system that mounts on Gridfinity baseplates.
Generates printable STLs: pegboard Tiles (with Gridfinity feet) + press-fit Anchors.

Tune the PARAMS block, run:  uv run griddogs.py
Requires:  manifold3d, trimesh, numpy (managed via pyproject.toml / uv)
"""

import os

import numpy as np
import trimesh
from manifold3d import CrossSection, JoinType, Manifold, OpType

# ----------------------------- PARAMS ---------------------------------------
GRID        = 42.0    # Gridfinity pitch
CLEAR       = 0.25    # per-side bin clearance -> tile plan = n*42 - 0.5
FOOT_H      = (0.8, 1.8, 2.15)          # bottom chamfer, straight, top chamfer
FOOT_W      = (35.6, 37.2, 41.5)        # widths at each level
FOOT_R      = (0.8, 1.6, 3.75)          # corner radii at each level

PLATE_T     = 4.75    # pegboard plate thickness (total tile height = 9.5)
HOLE_D      = 6.5     # anchor hole diameter
HOLE_DEPTH  = 5.5     # blind hole depth; extends into the Gridfinity foot
HOLES_CELL  = 3       # holes per Gridfinity cell in each axis
HOLE_PITCH  = GRID / HOLES_CELL
HOLE_CHAMF  = 0.6     # insertion chamfer at hole mouth

PEG_D       = 6.5     # peg diameter; anchors are cheaper to tune than tiles
PEG_CLEAR   = 0.2     # bottom clearance between peg tip and blind hole
PEG_L       = 3.8     # short slotted peg retained by the two-peg bone
SINGLE_PEG_L = HOLE_DEPTH - PEG_CLEAR
WALL_PEG_D  = PEG_D   # physically validated solid wall peg
WALL_PEG_L  = SINGLE_PEG_L
PEG_SLOT    = 1.4     # compression slot width through the peg
PEG_TIP_CH  = 0.7     # peg tip chamfer
PEG_ROOT_CH = HOLE_CHAMF  # root flare matches the hole-mouth chamfer
PEG_ROOT_CLEAR = 0.2  # diametral clearance at the hole mouth
PEG_ROOT_D  = HOLE_D + 2 * HOLE_CHAMF - PEG_ROOT_CLEAR
PEG_SLOT_ROOT = PEG_ROOT_CH / 2  # uncut bridge between slot and anchor head
PEG_RIDGE_D = 6.6     # physically validated ridge on single-peg anchors
PEG_RIDGE_H = 0.8       # rounded ridge length along the peg
PEG_RIDGE_Z = SINGLE_PEG_L - PEG_TIP_CH - PEG_RIDGE_H / 2
COUPON_HOLE_DEPTH = 4.0  # keep the plate-only fit coupon blind

PEG_SUPPORT = 1.1     # solid wall on each side of the peg root
WALL_T      = 10.0    # rounded practical width, exceeding required root support
ROUND_D     = 12.0    # compact bumper; adjacent anchors retain a 2 mm gap
ANCHOR_H    = (20.0, 40.0)  # standard short and tall heights above plate
ANCHOR_CLEAR = 0.25   # per-side clearance between adjacent anchor footprints
CURVE_SIZE  = HOLE_PITCH - 2 * ANCHOR_CLEAR
CURVE_SH_MAX = 7.0    # maximum straight shoulder height
CURVE_NECK_DROP = 6.0  # neck starts this far below the tip
CURVE_TIP   = 3.0     # neck thickness and rounded-tip diameter
CURVE_BASE_R = 1.0    # side-profile base corner radius
CURVE_CTRL  = (0.18, 0.48)  # standard sweep control heights
DEEP_CTRL   = (0.08, 0.32)  # deeper sweep
BOWL_CTRL   = (0.03, 0.12)  # most aggressive scoop
SEGS        = 64

MAGNETS     = False   # enable Gridfinity magnet pockets in the feet
MAG_D       = 6.3     # pocket diameter for 6 x 2 mm magnets (spec 6.5; 6.3 = press fit)
MAG_DEPTH   = 2.4
MAG_OFFSET  = 13.0    # +/- from cell centre (26 mm spacing, per Gridfinity spec)
MAG_CHAMF   = 0.4

TILES       = [(cols, rows) for cols in range(1, 8) for rows in range(cols, 8)]
DIAGONAL_WALLS = (("2x2", 1, 1), ("2x3", 1, 2))
OUT         = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stl")
# -----------------------------------------------------------------------------

os.makedirs(OUT, exist_ok=True)

assert sum(FOOT_H) + PLATE_T > HOLE_DEPTH, "blind holes must retain a solid floor"
assert PEG_L > PEG_TIP_CH, "peg must retain a cylindrical press-fit section"
assert SINGLE_PEG_L > PEG_TIP_CH, "single peg must retain a cylindrical section"
assert WALL_PEG_L > PEG_TIP_CH, "wall peg must retain a cylindrical section"
assert WALL_PEG_D == HOLE_D, "wall peg and hole diameters must match"
assert PEG_SUPPORT > 0, "peg root needs positive surrounding wall thickness"
assert WALL_T >= PEG_ROOT_D + 2 * PEG_SUPPORT, "wall does not support the peg root"
assert PEG_ROOT_CLEAR > 0, "peg root flare must not wedge in the hole mouth"
assert PEG_SLOT_ROOT > 0, "compression slot must stop before the anchor head"
assert PEG_D < PEG_RIDGE_D, "ridge must stand proud of its shaft"
assert PEG_RIDGE_D <= PEG_ROOT_D, "ridge must fit beneath the root flare"
assert PEG_RIDGE_Z - PEG_RIDGE_H / 2 > PEG_ROOT_CH, "test ridge must clear the root"
assert PEG_RIDGE_Z + PEG_RIDGE_H / 2 <= SINGLE_PEG_L - PEG_TIP_CH, \
    "ridge must finish before the tip chamfer"
assert sum(FOOT_H) + PLATE_T - HOLE_DEPTH - MAG_DEPTH > 0, \
    "tile holes must not meet optional magnet pockets"
assert PLATE_T > COUPON_HOLE_DEPTH, "coupon holes must retain a solid floor"
assert ANCHOR_CLEAR > 0, "adjacent curved anchors need lateral clearance"
assert CURVE_SIZE > PEG_ROOT_D, "curved anchor body must support the peg root"


def rounded_square(w, r, segs=16):
    """Centered rounded-square polygon, CCW."""
    h = w / 2.0
    c = h - r
    pts = []
    centers = [(c, c, 0), (-c, c, 90), (-c, -c, 180), (c, -c, 270)]
    for cx, cy, a0 in centers:
        for t in np.linspace(np.radians(a0), np.radians(a0 + 90), segs):
            pts.append((cx + r * np.cos(t), cy + r * np.sin(t)))
    return np.array(pts)


def slab(poly, z, t=0.02):
    return Manifold.extrude(CrossSection([poly]), t).translate([0, 0, z])


def gridfinity_foot():
    """One bin foot, z = 0 .. 4.75, built from exact offset profiles via hulls."""
    h1, h2, h3 = FOOT_H
    w1, w2, w3 = FOOT_W
    r1, r2, r3 = FOOT_R
    p1, p2, p3 = rounded_square(w1, r1), rounded_square(w2, r2), rounded_square(w3, r3)
    bottom = Manifold.batch_hull([slab(p1, 0), slab(p2, h1)])
    mid    = Manifold.extrude(CrossSection([p2]), h2).translate([0, 0, h1])
    top    = Manifold.batch_hull([slab(p2, h1 + h2), slab(p3, h1 + h2 + h3 - 0.02)])
    return bottom + mid + top


def rounded_rect_cs(L, W, r):
    cs = CrossSection.square([L - 2 * r, W - 2 * r], True)
    return cs.offset(r, JoinType.Round, circular_segments=SEGS)


def chamfered_prism(cs, h, ch):
    """Extrude cross-section with a chamfered top edge.
    Convex cross-sections only: the chamfer band is built as a hull, which
    would bridge any concavity (chamfer concave shapes piecewise instead)."""
    assert cs.hull().area() - cs.area() < 1e-6 * cs.area(), \
        "chamfered_prism: cross-section must be convex"
    body = Manifold.extrude(cs, h - ch)
    csm  = cs.offset(-ch, JoinType.Round, circular_segments=SEGS)
    lo   = Manifold.extrude(cs, 0.02).translate([0, 0, h - ch])
    hi   = Manifold.extrude(csm, 0.02).translate([0, 0, h - 0.02])
    return body + Manifold.batch_hull([lo, hi])


def tile(nx, ny, hole_depth=HOLE_DEPTH):
    assert nx >= 1 and ny >= 1, "tile dimensions must be positive"
    assert hole_depth > 0, "hole depth must be positive"
    foot_h = sum(FOOT_H)
    top_z  = foot_h + PLATE_T
    # feet
    feet = []
    for i in range(nx):
        for j in range(ny):
            cx = (i - (nx - 1) / 2) * GRID
            cy = (j - (ny - 1) / 2) * GRID
            feet.append(gridfinity_foot().translate([cx, cy, 0]))
    # plate
    Lx, Ly = nx * GRID - 2 * CLEAR, ny * GRID - 2 * CLEAR
    r = FOOT_R[2]
    cs = CrossSection.square([Lx - 2 * r, Ly - 2 * r], True).offset(
        r, JoinType.Round, circular_segments=SEGS)
    plate = Manifold.extrude(cs, PLATE_T).translate([0, 0, foot_h])
    solid = Manifold.batch_boolean(feet + [plate], OpType.Add)

    # holes + chamfers
    cutters = []
    hx, hy = HOLES_CELL * nx, HOLES_CELL * ny
    for i in range(hx):
        for j in range(hy):
            x = (i - (hx - 1) / 2) * HOLE_PITCH
            y = (j - (hy - 1) / 2) * HOLE_PITCH
            hole = Manifold.cylinder(hole_depth + 0.01, HOLE_D / 2,
                                     circular_segments=SEGS
                                     ).translate([x, y, top_z - hole_depth])
            cham = Manifold.cylinder(HOLE_CHAMF + 0.01, HOLE_D / 2,
                                     HOLE_D / 2 + HOLE_CHAMF,
                                     circular_segments=SEGS
                                     ).translate([x, y, top_z - HOLE_CHAMF])
            cutters += [hole, cham]
    if MAGNETS:
        for i in range(nx):
            for j in range(ny):
                cx = (i - (nx - 1) / 2) * GRID
                cy = (j - (ny - 1) / 2) * GRID
                for sx in (-1, 1):
                    for sy in (-1, 1):
                        mx, my = cx + sx * MAG_OFFSET, cy + sy * MAG_OFFSET
                        cutters.append(Manifold.cylinder(
                            MAG_DEPTH + 0.01, MAG_D / 2, circular_segments=SEGS
                            ).translate([mx, my, -0.01]))
                        # bottom radius grown by the same 0.01 z-fuzz keeps the
                        # chamfer at exactly 45 deg through the extended cutter
                        cutters.append(Manifold.cylinder(
                            MAG_CHAMF + 0.01, MAG_D / 2 + MAG_CHAMF + 0.01, MAG_D / 2,
                            circular_segments=SEGS).translate([mx, my, -0.01]))
    cut = Manifold.batch_boolean(cutters, OpType.Add)
    return solid - cut


def peg(x=0.0, y=0.0, slot_along_x=False, side_flat=0.0,
        body_d=PEG_D, ridge_d=None, peg_l=PEG_L, slotted=True):
    """Press-fit peg pointing DOWN from z=0 (in-use orientation).
    For parts that print lying on their side: slot_along_x turns the slot 90
    deg so the press-fit crowns sit on cleanly-printed vertical side walls,
    and side_flat provides a level interface for slicer support under the peg."""
    root = Manifold.cylinder(PEG_ROOT_CH, PEG_ROOT_D / 2, body_d / 2,
                             circular_segments=SEGS)
    if ridge_d is None:
        body = Manifold.cylinder(
            peg_l - PEG_TIP_CH, body_d / 2, circular_segments=SEGS,
        )
        tip = Manifold.cylinder(
            PEG_TIP_CH, body_d / 2, body_d / 2 - PEG_TIP_CH,
            circular_segments=SEGS,
        ).translate([0, 0, peg_l - PEG_TIP_CH])
        profile = body + tip + root
    else:
        ridge_start = PEG_RIDGE_Z - PEG_RIDGE_H / 2
        ridge_steps = 8
        stem_cs = [
            (0, 0),
            (PEG_ROOT_D / 2, 0),
            (body_d / 2, PEG_ROOT_CH),
            (body_d / 2, ridge_start),
        ]
        for step in range(1, ridge_steps + 1):
            u = step / ridge_steps
            rise = (1 - np.cos(2 * np.pi * u)) / 2
            radius = (body_d + (ridge_d - body_d) * rise) / 2
            stem_cs.append((radius, ridge_start + u * PEG_RIDGE_H))
        stem_cs += [
            (body_d / 2, peg_l - PEG_TIP_CH),
            (body_d / 2 - PEG_TIP_CH, peg_l),
            (0, peg_l),
        ]
        stem = Manifold.revolve(
            CrossSection([np.array(stem_cs)]),
            circular_segments=SEGS,
        )
        profile = stem
    p = profile.rotate([180, 0, 0])   # now spans z = -peg_l .. 0
    max_d = max(body_d, ridge_d or body_d)
    if slotted:
        sw = [PEG_SLOT, max_d + 2] if slot_along_x else [max_d + 2, PEG_SLOT]
        slot_r = PEG_SLOT / 2
        slot_round_z = -PEG_SLOT_ROOT - slot_r
        slot_bottom = -peg_l - 0.2
        slot_h = slot_round_z - slot_bottom
        slot = Manifold.cube([sw[0], sw[1], slot_h], True)\
                       .translate([0, 0, (slot_bottom + slot_round_z) / 2])
        relief_len = max_d + 2
        if slot_along_x:
            relief = Manifold.cylinder(relief_len, slot_r, circular_segments=32)\
                             .rotate([90, 0, 0])\
                             .translate([0, relief_len / 2, slot_round_z])
        else:
            relief = Manifold.cylinder(relief_len, slot_r, circular_segments=32)\
                             .rotate([0, 90, 0])\
                             .translate([-relief_len / 2, 0, slot_round_z])
        p = p - (slot + relief)
    if side_flat:
        trim = Manifold.cube([max_d + 2, side_flat + 1, peg_l + 1], True)\
                       .translate([0, max_d / 2 - side_flat + (side_flat + 1) / 2,
                                   -peg_l / 2])
        p = p - trim
    return p.translate([x, y, 0])


def to_print_orientation(m, head_h):
    """Flip anchor so head-top sits on the bed, pegs point up."""
    return m.rotate([180, 0, 0]).translate([0, 0, head_h])


def anchor_round(h):
    top_ch = 1.5
    head = Manifold.cylinder(h - top_ch, ROUND_D / 2,
                             circular_segments=SEGS) + \
           Manifold.cylinder(top_ch, ROUND_D / 2, ROUND_D / 2 - top_ch,
                             circular_segments=SEGS
                             ).translate([0, 0, h - top_ch])
    anchor_peg = peg(ridge_d=PEG_RIDGE_D, peg_l=SINGLE_PEG_L)
    return to_print_orientation(head + anchor_peg, h)


def anchor_wall(span, h):
    """Wall anchor with one solid peg at each end."""
    length = span + WALL_T + 2.0
    head = chamfered_prism(rounded_rect_cs(length, WALL_T, 2.0), h, 1.2)
    pegs = [
        peg(
            x,
            0,
            body_d=WALL_PEG_D,
            peg_l=WALL_PEG_L,
            slotted=False,
        )
        for x in (-span / 2, span / 2)
    ]
    return to_print_orientation(
        head + Manifold.batch_boolean(pegs, OpType.Add),
        h,
    )
def curve_sweep(x_outer, x_inner, shoulder_h, neck_h, ctrl):
    assert neck_h - shoulder_h > 1.0, "anchor height is too short for the curved profile"
    sweep_h = neck_h - shoulder_h
    z_ctrl1 = shoulder_h + ctrl[0] * sweep_h
    z_ctrl2 = shoulder_h + ctrl[1] * sweep_h
    points = []
    for t in np.linspace(0.0, 1.0, 64):
        u = 1.0 - t
        x = u ** 3 * x_outer + 3 * u * u * t * x_outer + \
            3 * u * t * t * x_inner + t ** 3 * x_inner
        z = u ** 3 * shoulder_h + 3 * u * u * t * z_ctrl1 + \
            3 * u * t * t * z_ctrl2 + t ** 3 * neck_h
        points.append((x, z))
    return points


def curve_base_arcs(x0, x1):
    back = [(x1 - CURVE_BASE_R + CURVE_BASE_R * np.cos(a),
             CURVE_BASE_R + CURVE_BASE_R * np.sin(a))
            for a in np.linspace(-np.pi / 2, 0.0, 8)]
    front = [(x0 + CURVE_BASE_R + CURVE_BASE_R * np.cos(a),
              CURVE_BASE_R + CURVE_BASE_R * np.sin(a))
             for a in np.linspace(np.pi, 3 * np.pi / 2, 8)]
    return back, front



def finish_curve_anchor(pts):
    body = Manifold.extrude(CrossSection([np.array(pts)]), CURVE_SIZE)\
        .rotate([90, 0, 0]).translate([0, CURVE_SIZE / 2, 0])
    anchor_peg = peg(
        0,
        0,
        slot_along_x=True,
        side_flat=0.5,
        ridge_d=PEG_RIDGE_D,
        peg_l=SINGLE_PEG_L,
    )
    m = body + anchor_peg
    return m.rotate([-90, 0, 0]).translate([0, 0, CURVE_SIZE / 2])


def anchor_curve(ctrl, h):
    """Directional curved stop with a rounded tip and one concave face."""
    x0, x1 = -CURVE_SIZE / 2, CURVE_SIZE / 2
    x_neck = x1 - CURVE_TIP
    neck_r = CURVE_TIP / 2
    shoulder_h = min(CURVE_SH_MAX, h / 4)
    neck_h = h - CURVE_NECK_DROP
    z_cap = h - neck_r
    curve = list(reversed(curve_sweep(x0, x_neck, shoulder_h, neck_h, ctrl)))
    back_base, front_base = curve_base_arcs(x0, x1)
    cap = [(x1 - neck_r + neck_r * np.cos(a),
            z_cap + neck_r * np.sin(a))
           for a in np.linspace(0.0, np.pi, 17)]
    pts = (
        [(x0 + CURVE_BASE_R, 0.0), (x1 - CURVE_BASE_R, 0.0)]
        + back_base[1:] + [(x1, z_cap)] + cap[1:]
        + curve + [(x0, CURVE_BASE_R)] + front_base[1:-1]
    )
    return finish_curve_anchor(pts)


def anchor_curve_centre(ctrl, h):
    """Double-sided curved stop with its rounded peak centred over the peg."""
    x0, x1 = -CURVE_SIZE / 2, CURVE_SIZE / 2
    neck_r = CURVE_TIP / 2
    shoulder_h = min(CURVE_SH_MAX, h / 4)
    neck_h = h - CURVE_NECK_DROP
    z_cap = h - neck_r
    right_curve = curve_sweep(x1, neck_r, shoulder_h, neck_h, ctrl)
    left_curve = [(-x, z) for x, z in reversed(right_curve)]
    back_base, front_base = curve_base_arcs(x0, x1)
    cap = [(neck_r * np.cos(a), z_cap + neck_r * np.sin(a))
           for a in np.linspace(0.0, np.pi, 17)]
    pts = (
        [(x0 + CURVE_BASE_R, 0.0), (x1 - CURVE_BASE_R, 0.0)]
        + back_base[1:] + [(x1, shoulder_h)] + right_curve[1:]
        + [(neck_r, z_cap)] + cap[1:] + left_curve
        + [(x0, CURVE_BASE_R)] + front_base[1:-1]
    )
    return finish_curve_anchor(pts)


def anchor_curve_standard(h):
    return anchor_curve(CURVE_CTRL, h)


def anchor_curve_deep(h):
    return anchor_curve(DEEP_CTRL, h)


def anchor_curve_bowl(h):
    return anchor_curve(BOWL_CTRL, h)


def anchor_curve_centre_standard(h):
    return anchor_curve_centre(CURVE_CTRL, h)


def anchor_curve_centre_deep(h):
    return anchor_curve_centre(DEEP_CTRL, h)


def anchor_bone():
    """The mascot: dog-bone anchor, one peg under each end."""
    c, lobe_r, lobe_dy = HOLE_PITCH / 2, 5.0, 3.5
    h = 12.0
    # the bone outline is concave at the waist, so chamfer each convex piece
    # and union them; chamfered_prism on the whole outline would hull it shut
    pieces = [chamfered_prism(rounded_rect_cs(17.0, 7.0, 2.0), h, 1.2)]
    for sx in (-1, 1):
        for sy in (-1, 1):
            pieces.append(chamfered_prism(
                CrossSection.circle(lobe_r, SEGS).translate([sx * c, sy * lobe_dy]),
                h, 1.2))
    head = Manifold.batch_boolean(pieces, OpType.Add)
    return to_print_orientation(head + peg(-c, 0) + peg(c, 0), h)


def fit_coupon():
    """Quick comparison strip with holes spanning HOLE_D +/- 0.2 mm."""
    dias = [HOLE_D + offset for offset in (-0.2, -0.1, 0.0, 0.1, 0.2)]
    pitch, W, T = HOLE_PITCH, 15.0, PLATE_T
    L = pitch * len(dias) + 6
    cs = rounded_rect_cs(L, W, 2.0)
    m = Manifold.extrude(cs, T)
    cutters = []
    for k, d in enumerate(dias):
        x = -((len(dias) - 1) / 2) * pitch + k * pitch
        cutters.append(Manifold.cylinder(COUPON_HOLE_DEPTH + 0.01, d / 2,
                       circular_segments=SEGS
                       ).translate([x, 0, T - COUPON_HOLE_DEPTH]))
        cutters.append(Manifold.cylinder(HOLE_CHAMF + 0.01, d / 2,
                       d / 2 + HOLE_CHAMF, circular_segments=SEGS
                       ).translate([x, 0, T - HOLE_CHAMF]))
        for n in range(k + 1):                    # notches on front edge
            nx = x - (k) * 1.25 + n * 2.5
            cutters.append(Manifold.cylinder(T + 1, 0.8, circular_segments=24
                           ).translate([nx, -W / 2, -0.5]))
    cut = Manifold.batch_boolean(cutters, OpType.Add)
    return m - cut


def export(m, name):
    mesh = m.to_mesh()
    v = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3]
    f = np.asarray(mesh.tri_verts, dtype=np.int64)
    tm = trimesh.Trimesh(v, f, process=True)
    assert tm.is_watertight, f"{name} is not watertight"
    path = os.path.join(OUT, name)
    tm.export(path)
    b = tm.bounds
    print(f"{name:28s} watertight={tm.is_watertight}  "
          f"size={b[1][0]-b[0][0]:.1f} x {b[1][1]-b[0][1]:.1f} x {b[1][2]-b[0][2]:.1f} mm  "
          f"vol={tm.volume/1000:.1f} cm3")
    return tm


if __name__ == "__main__":
    for nx, ny in TILES:
        export(tile(nx, ny), f"tile_{nx}x{ny}_gridfinity.stl")
    for h in ANCHOR_H:
        hs = f"{int(h)}mm"
        export(anchor_round(h), f"anchor_round_bumper_{hs}.stl")
        export(anchor_wall(HOLE_PITCH, h), f"anchor_wall_short_{hs}.stl")
        export(anchor_wall(2 * HOLE_PITCH, h), f"anchor_wall_long_{hs}.stl")
        for name, dx, dy in DIAGONAL_WALLS:
            span = HOLE_PITCH * np.hypot(dx, dy)
            export(anchor_wall(span, h), f"anchor_wall_diagonal_{name}_{hs}.stl")
        export(anchor_curve_standard(h), f"anchor_curve_standard_{hs}.stl")
        export(anchor_curve_deep(h), f"anchor_curve_deep_{hs}.stl")
        export(anchor_curve_bowl(h), f"anchor_curve_bowl_{hs}.stl")
        export(anchor_curve_centre_standard(h), f"anchor_curve_centre_standard_{hs}.stl")
        export(anchor_curve_centre_deep(h), f"anchor_curve_centre_deep_{hs}.stl")
    export(anchor_bone(),    "anchor_bone.stl")
    export(fit_coupon(),     "fit_test_coupon.stl")