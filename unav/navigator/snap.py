from shapely.geometry import Point, LineString, MultiPoint
from shapely.ops import nearest_points
import math
from typing import List, Tuple, Optional


def snap_toward_nearest_waypoint(
    point,
    walkable_union,
    nav_nodes,
    inward_offset=5.0,
):
    p = Point(*point)
    if walkable_union.contains(p):
        return point

    boundary = walkable_union.boundary

    sorted_nodes = sorted(nav_nodes, key=lambda n: math.hypot(n[0] - point[0], n[1] - point[1]))

    for nav_pt in sorted_nodes:
        ray = LineString([point, nav_pt])
        intersection = boundary.intersection(ray)

        if intersection.is_empty:
            continue

        if intersection.geom_type == 'Point':
            candidates = [intersection]
        elif intersection.geom_type == 'MultiPoint':
            candidates = list(intersection.geoms)
        elif intersection.geom_type == 'GeometryCollection':
            candidates = [g for g in intersection.geoms if g.geom_type == 'Point']
        else:
            continue

        if not candidates:
            continue

        entry = min(candidates, key=lambda c: p.distance(c))

        dx = nav_pt[0] - entry.x
        dy = nav_pt[1] - entry.y
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            return (entry.x, entry.y)

        nudged = Point(entry.x + (dx / norm) * inward_offset,
                       entry.y + (dy / norm) * inward_offset)
        if walkable_union.contains(nudged):
            return (nudged.x, nudged.y)
        else:
            return (entry.x, entry.y)

    return snap_inside_walkable(point, walkable_union, inward_offset)


def snap_inside_walkable(
    point,
    walkable_union,
    inward_offset=20.0,
):
    p = Point(*point)
    if walkable_union.contains(p):
        return point

    nearest_geom, _ = nearest_points(walkable_union, p)

    dx = point[0] - nearest_geom.x
    dy = point[1] - nearest_geom.y
    norm = math.hypot(dx, dy)
    if norm == 0:
        dx, dy, norm = 1.0, 0.0, 1.0

    inward_x = nearest_geom.x - (dx / norm) * inward_offset
    inward_y = nearest_geom.y - (dy / norm) * inward_offset

    inward_point = Point(inward_x, inward_y)
    if walkable_union.contains(inward_point):
        return (inward_x, inward_y)
    else:
        return (nearest_geom.x, nearest_geom.y)
