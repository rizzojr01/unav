import json
import math
import networkx as nx
from shapely.geometry import Polygon as ShapelyPolygon, Point, LineString, MultiLineString
from shapely.ops import unary_union, nearest_points
from typing import Dict, List, Tuple, Any

class PathFinder:
    """
    PathFinder constructs a directed visibility graph from annotated floorplan data,
    supporting efficient shortest path queries and room/region lookups.

    Supports:
      - Arbitrary walkable regions (group 0), obstacles (group 1), doors (group 2)
      - Manual waypoints (group 3), inter-waypoints (group 4), and destinations (group 5)
      - Companion lines (group 6) for inter-waypoint orientation
    """

    def __init__(self, json_path: str):
        # Node (waypoint/destination) data
        self.nodes: Dict[int, Tuple[float, float]] = {}
        self.labels: Dict[int, str] = {}
        self.group_ids: Dict[int, int] = {}
        self.descriptions: Dict[int, str] = {}           # Inter-waypoints
        self.dest_orientations: Dict[int, str] = {}      # Destinations
        self.nav_ids: List[int] = []                     # Navigation/inter-waypoints
        self.dest_ids: List[int] = []                    # Destination IDs
        self.partner_lines: Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]] = {}

        # Region geometry
        self.walkable_polygons: List[ShapelyPolygon] = []
        self.obstacle_polygons: List[ShapelyPolygon] = []
        self.door_polygons: List[Tuple[ShapelyPolygon, str]] = []
        self.room_polygons: List[Tuple[ShapelyPolygon, str]] = []

        self.walkable_union: ShapelyPolygon = None        # Final walkable region

        # Visibility graph
        self.G = nx.DiGraph()

        # Route network (MultiLineString of all graph edges) for snap-to-route
        self.route_network = None

        # Build graph from JSON
        self._load_data(json_path)
        self._build_graph()

    def _euclidean(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Euclidean distance between two points."""
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _load_data(self, json_path: str):
        """Load and parse map JSON, populating region and node data."""
        with open(json_path) as f:
            data = json.load(f)

        node_id = 0
        raw_group6: List[Dict[str, Any]] = []

        for shape in data["shapes"]:
            stype = shape.get("shape_type")
            gid = shape.get("group_id")
            pts = shape.get("points", [])
            label = (shape.get("label") or "").strip()
            desc = (shape.get("description") or "").strip()

            # Parse polygon and rectangle (for regions)
            if stype in ("polygon", "rectangle"):
                if stype == "rectangle" and len(pts) == 2:
                    (x0, y0), (x1, y1) = pts
                    pts = [(x0, y0), (x0, y1), (x1, y1), (x1, y0)]
                poly = ShapelyPolygon(pts)
                if gid == 0:
                    self.walkable_polygons.append(poly)
                    self.room_polygons.append((poly, label))
                elif gid == 1:
                    self.obstacle_polygons.append(poly)
                elif gid == 2:
                    self.door_polygons.append((poly, label))
                continue

            # Points: navigation/inter/destination
            if stype == "point" and pts:
                pt = tuple(pts[0])
                self.nodes[node_id] = pt
                self.labels[node_id] = label
                self.group_ids[node_id] = gid
                if gid == 4:
                    self.descriptions[node_id] = desc
                if gid in (3, 4):
                    self.nav_ids.append(node_id)
                if gid == 5:
                    self.dest_ids.append(node_id)
                    self.dest_orientations[node_id] = desc
                node_id += 1

            # Group 6: inter-waypoint companion lines
            if stype == "line" and gid == 6 and len(pts) == 2:
                raw_group6.append({
                    "label": label,
                    "points": pts
                })

        # Attach companion lines to matching inter-waypoints by label
        for entry in raw_group6:
            pts = entry["points"]
            line = (tuple(pts[0]), tuple(pts[1]))
            lbl = entry["label"]
            for nid, node_lbl in self.labels.items():
                if self.group_ids.get(nid) == 4 and node_lbl == lbl:
                    self.partner_lines[nid] = line
                    break

        # Build walkable region: union walkable + doors, minus obstacles
        merged = unary_union(self.walkable_polygons + [poly for poly, _ in self.door_polygons])
        for obs in self.obstacle_polygons:
            merged = merged.difference(obs)
        self.walkable_union = merged

    def _visible(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> bool:
        """
        True if line p1→p2 is inside the walkable region and does not cross obstacles.
        """
        line = LineString([p1, p2])
        if not self.walkable_union.contains(line):
            return False
        for obs in self.obstacle_polygons:
            if line.crosses(obs) or line.within(obs):
                return False
        return True

    def _build_graph(self):
        """Create the directed visibility graph (waypoints/inter-waypoints/destinations)."""
        # Connect all nav/inter-waypoints with visibility
        for i in self.nav_ids:
            for j in self.nav_ids:
                if i < j and self._visible(self.nodes[i], self.nodes[j]):
                    w = self._euclidean(self.nodes[i], self.nodes[j])
                    self.G.add_edge(i, j, weight=w)
                    self.G.add_edge(j, i, weight=w)
        # Connect to destination nodes
        for nid in self.nav_ids:
            for did in self.dest_ids:
                if self._visible(self.nodes[nid], self.nodes[did]):
                    w = self._euclidean(self.nodes[nid], self.nodes[did])
                    self.G.add_edge(nid, did, weight=w)
        # Build route network: deduplicated line segments from all graph edges
        seen = set()
        segments = []
        for u, v in self.G.edges():
            key = (min(u, v), max(u, v))
            if key not in seen:
                seen.add(key)
                segments.append((self.nodes[u], self.nodes[v]))
        self.route_network = MultiLineString(segments) if segments else None


    def snap_to_route(self, point, threshold=None):
        """
        Snap a point to the nearest location on the route network (graph edges).

        Args:
            point: (x, y) query coordinate.
            threshold: Only snap when distance to network is within this many pixels.

        Returns:
            (x, y) snapped coordinate on the nearest graph edge.
        """
        if self.route_network is None or self.route_network.is_empty:
            return point
        p = Point(*point)
        if threshold is not None and p.distance(self.route_network) > threshold:
            return point
        snapped, _ = nearest_points(self.route_network, p)
        return (snapped.x, snapped.y)

    def get_route_segments(self):
        """Return all route-network edges as [{from, to}] dicts for frontend rendering."""
        seen = set()
        segs = []
        for u, v in self.G.edges():
            key = (min(u, v), max(u, v))
            if key not in seen:
                seen.add(key)
                p1, p2 = self.nodes[u], self.nodes[v]
                segs.append({"from": list(p1), "to": list(p2)})
        return segs

    def find_path(self, start_id: int, dest_id: int) -> Dict[str, Any]:
        """
        Find shortest path between two node IDs using Dijkstra.

        Args:
            start_id (int): Start node ID.
            dest_id (int): Destination node ID.

        Returns:
            Dict: path IDs, coordinates, labels, and total cost.
        """
        if dest_id not in self.dest_ids:
            return {"error": "Destination must be terminal"}
        if start_id == dest_id:
            return {
                "path_ids": [start_id],
                "path_coords": [self.nodes[start_id]],
                "path_labels": [self.labels[start_id]],
                "total_cost": 0.0
            }
        try:
            path = nx.dijkstra_path(self.G, source=start_id, target=dest_id)
            coords = [self.nodes[n] for n in path]
            cost = sum(self._euclidean(coords[i], coords[i + 1]) for i in range(len(coords) - 1))
            return {
                "path_ids": path,
                "path_coords": coords,
                "path_labels": [self.labels[n] for n in path],
                "total_cost": cost
            }
        except nx.NetworkXNoPath:
            return {"error": "No path found"}

    def find_path_from_pose(self, pose_xy: Tuple[float, float], dest_id: int) -> Dict[str, Any]:
        """
        Insert a temporary node at a given coordinate, connect it to visible nodes,
        and compute shortest path to a destination.

        Args:
            pose_xy (tuple): (x, y) pose coordinate.
            dest_id (int): Destination node ID.

        Returns:
            Dict: As in find_path().
        """
        if dest_id not in self.dest_ids:
            return {"error": "Destination must be terminal"}

        # Add virtual node for pose
        vid = -1
        self.nodes[vid] = pose_xy
        self.labels[vid] = "pose"
        self.group_ids[vid] = -1
        self.G.add_node(vid)

        for nid in self.nav_ids:
            if self._visible(pose_xy, self.nodes[nid]):
                w = self._euclidean(pose_xy, self.nodes[nid])
                self.G.add_edge(vid, nid, weight=w)
        for did in self.dest_ids:
            if self._visible(pose_xy, self.nodes[did]):
                w = self._euclidean(pose_xy, self.nodes[did])
                self.G.add_edge(vid, did, weight=w)

        # Run shortest path
        raw = self.find_path(vid, dest_id)

        # Remove virtual node
        if self.G.has_node(vid):
            self.G.remove_node(vid)
        self.nodes.pop(vid, None)
        self.labels.pop(vid, None)
        self.group_ids.pop(vid, None)

        # Prepend pose to result path if found
        if "path_coords" in raw and raw["path_coords"]:
            raw["path_coords"].insert(0, pose_xy)
            raw["path_labels"].insert(0, "start_pose")
            raw["path_ids"].insert(0, vid)

        return raw

    def get_current_room(self, pose_xy: Tuple[float, float]) -> str:
        """
        Return the label of the room polygon containing a point, or 'Unknown'.

        Args:
            pose_xy (tuple): (x, y) coordinate.

        Returns:
            str: Room label, or "Unknown" if not found.
        """
        pt = Point(*pose_xy)
        for poly, lbl in self.room_polygons:
            if poly.contains(pt):
                return lbl or "Unnamed Room"
        return "Unknown"

    def list_all_destinations(self) -> Dict[int, Tuple[str, Tuple[float, float]]]:
        """Return all destinations as {id: (label, coordinates)}."""
        return {d: (self.labels[d], self.nodes[d]) for d in self.dest_ids}

    def get_destination_id_by_name(self, name: str) -> int:
        """Return a destination ID by (case-insensitive) label substring match."""
        for d in self.dest_ids:
            if name.lower() in self.labels[d].lower():
                return d
        return None
