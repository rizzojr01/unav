import os
import json
import math
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional

from unav.navigator.pathfinder import PathFinder
from unav.navigator.snap import snap_inside_walkable, snap_toward_nearest_waypoint
from unav.config import UNavNavigationConfig

class FacilityNavigator:
    """
    Unified navigation system supporting multi-place, multi-building, multi-floor pathfinding,
    with robust cross-floor and cross-building routing via named inter-waypoints (e.g., stairs, elevators).
    All keys are tuples for full clarity and safety.
    """

    def __init__(self, config: UNavNavigationConfig):
        """
        Initialize the facility navigator.

        Args:
            config (UNavNavigationConfig): Unified navigation configuration.
        """
        self.config = config

        # pf_map uses (place, building, floor) as key, PathFinder as value
        self.pf_map: Dict[Tuple[str, str, str], PathFinder] = {}
        # inter_waypoints: label → list of (place, building, floor, node_id)
        self.inter_waypoints: Dict[str, List[Tuple[str, str, str, int]]] = {}

        # Load all PathFinder objects for every place/building/floor
        for place, buildings in self.config.building_jsons.items():
            for building, floors in buildings.items():
                for floor, json_path in floors.items():
                    key = (place, building, floor)
                    pf = PathFinder(json_path)
                    self.pf_map[key] = pf

        self.scales = self._load_scales(self.config.scale_file)
        self.G = nx.DiGraph()
        self._build_unified_graph()

    def _load_scales(self, scale_file: Optional[str]) -> Dict[Tuple[str, str, str], float]:
        """
        Load scaling factors for each floor from a JSON file.

        Args:
            scale_file (Optional[str]): Path to scale JSON file.

        Returns:
            Dict[Tuple[str, str, str], float]: Mapping from (place, building, floor) to scale.
        """
        scales = {key: 1.0 for key in self.pf_map}
        if scale_file and os.path.exists(scale_file):
            with open(scale_file, 'r') as f:
                data = json.load(f)
            for place, buildings in data.items():
                for building, floors in buildings.items():
                    for floor, sc in floors.items():
                        key = (place, building, floor)
                        if key in scales:
                            scales[key] = sc
        return scales

    def _build_unified_graph(self):
        """
        Build a unified directed navigation graph over all places/buildings/floors.
        All graph node keys are (place, building, floor, node_id).
        """
        # Add intra-floor walkable edges
        for key, pf in self.pf_map.items():
            scale = self.scales.get(key, 1.0)
            for u, v, d in pf.G.edges(data=True):
                uid = (*key, u)  # (place, building, floor, node_id)
                vid = (*key, v)
                scaled_weight = d['weight'] * scale
                self.G.add_edge(uid, vid, weight=scaled_weight)
            # Collect labeled inter-waypoints (group 4)
            for nid in pf.nav_ids:
                if pf.group_ids.get(nid) == 4:
                    label = pf.labels[nid]
                    if label:
                        node_key = (*key, nid)
                        self.inter_waypoints.setdefault(label, []).append(node_key)

        # Add cross-floor/building edges for inter-waypoints of same label
        for label, nodes in self.inter_waypoints.items():
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    u, v = nodes[i], nodes[j]
                    pf_u = self.pf_map[u[:3]]
                    desc = pf_u.descriptions.get(u[3], "").lower()
                    try:
                        floor_u = u[2]
                        floor_v = v[2]
                        floor_u_num = int(''.join(filter(str.isdigit, floor_u)))
                        floor_v_num = int(''.join(filter(str.isdigit, floor_v)))
                        jump = abs(floor_u_num - floor_v_num)
                    except Exception:
                        jump = 0
                    # Penalty by description type
                    if "staircase" in desc:
                        penalty_per_jump = 50.0
                        total_penalty = penalty_per_jump * jump
                    elif "elevator" in desc:
                        total_penalty = 3.0
                    else:
                        total_penalty = 0.0
                    self.G.add_edge(u, v, weight=total_penalty)
                    self.G.add_edge(v, u, weight=total_penalty)

    def list_destinations(self) -> Dict[Tuple[str, str, str, int], Tuple[str, Tuple[float, float]]]:
        """
        List all available destinations.

        Returns:
            Dict[Tuple[str, str, str, int], Tuple[label, (x, y)]]
        """
        out = {}
        for key, pf in self.pf_map.items():
            place, building, floor = key
            for did in pf.dest_ids:
                out[(place, building, floor, did)] = (pf.labels[did], pf.nodes[did])
        return out

    def find_path(
        self,
        start_place: str,
        start_building: str,
        start_floor: str,
        start_xy: Tuple[float, float],
        dest_place: str,
        dest_building: str,
        dest_floor: str,
        dest_id: int
    ) -> Dict[str, Any]:
        """
        Compute shortest path from start coordinate to a destination node,
        possibly crossing places/buildings/floors.

        Args:
            start_place (str): Name of start place.
            start_building (str): Name of start building.
            start_floor (str): Name of start floor.
            start_xy (Tuple[float, float]): Start (x, y) in floorplan.
            dest_place (str): Name of dest place.
            dest_building (str): Name of dest building.
            dest_floor (str): Name of dest floor.
            dest_id (int): Destination node id.

        Returns:
            Dict with keys:
                path_coords: List[Tuple[float, float]]
                path_labels: List[str]
                path_keys:   List[Tuple[str, str, str, int] or str]
                path_descriptions: List[str]
                total_cost: float
                error: Optional[str]
        """
        start_key = (start_place, start_building, start_floor)
        target_key = (dest_place, dest_building, dest_floor, dest_id)
        pf0 = self.pf_map[start_key]
        # Snap starting point into walkable region if needed
        scale = self.scales.get(start_key, 1.0)
        nav_coords = [pf0.nodes[nid] for nid in pf0.nav_ids]
        start_xy = snap_toward_nearest_waypoint(start_xy, pf0.walkable_union, nav_coords)

        # Add temporary virtual node for the real start point
        virt = "VIRT"
        self.G.add_node(virt)
        for nid in pf0.nav_ids + pf0.dest_ids:
            if pf0._visible(start_xy, pf0.nodes[nid]):
                w = math.hypot(start_xy[0] - pf0.nodes[nid][0], start_xy[1] - pf0.nodes[nid][1]) * scale
                self.G.add_edge(virt, (*start_key, nid), weight=w)

        try:
            path = nx.dijkstra_path(self.G, virt, target_key)
        except nx.NetworkXNoPath:
            self.G.remove_node(virt)
            return {"error": "No path found"}

        coords = []
        labels = []
        keys = []
        descriptions = []
        cost = 0.0
        prev_pt = start_xy

        for node in path:
            keys.append(node)
            if node == virt:
                coords.append(start_xy)
                labels.append("(start)")
                descriptions.append("")
                continue
            if not (isinstance(node, tuple) and len(node) == 4):
                continue
            place, building, floor, nid = node
            pf = self.pf_map[(place, building, floor)]
            pt = pf.nodes[nid]
            coords.append(pt)
            labels.append(pf.labels[nid])
            if pf.group_ids[nid] == 4:
                desc = pf.descriptions.get(nid, "")
            elif pf.group_ids[nid] == 5:
                desc = pf.dest_orientations.get(nid, "")
            else:
                desc = ""
            descriptions.append(desc)
            cost += math.hypot(prev_pt[0] - pt[0], prev_pt[1] - pt[1])
            prev_pt = pt

        self.G.remove_node(virt)

        return {
            "path_coords": coords,
            "path_labels": labels,
            "path_keys": keys,
            "path_descriptions": descriptions,
            "total_cost": cost
        }
    def get_floor_route_segments(self, place: str, building: str, floor: str):
        """Return route-network segments for one floor as [{from, to}] dicts."""
        key = (place, building, floor)
        pf = self.pf_map.get(key)
        if pf is None:
            return []
        return pf.get_route_segments()
