#!/usr/bin/env python3
"""
Floor Point Cloud Visualization Web Application
Usage: python app.py
Then visit http://localhost:5000
"""

import os
import sys
import json
import random
import subprocess
import threading
from pathlib import Path
from flask import Flask, render_template, jsonify, send_file, abort, request
import yaml
import numpy as np

FLOORMAP_SCRIPT = Path("/home/unav/Desktop/floormap/scripts/visualize_voxel_bev.py")
_bev_jobs = {}  # "place/building/floor" -> {'status': 'running'/'done'/'error', 'error': str}

ZIND_ROOT  = Path("/mnt/data/floorplan-reconstruction/outputs/ZInD")
ZIND_PLACE = "ZInD"

def get_floor_base_dir(place, building, floor):
    """Return the directory where BEV/map files are stored for this floor."""
    if place == ZIND_PLACE:
        return ZIND_ROOT / building / floor
    data_root = get_data_root()
    floor_map_dir = get_floor_map_dir_name()
    return data_root / place / building / floor / floor_map_dir

def zind_floor_is_ready(scene, floor):
    """True if a ZInD floor has a BEV already or can generate one."""
    fd = ZIND_ROOT / scene / floor
    if not fd.is_dir():
        return False
    if (fd / "bev_compare_mesh.npy").exists():
        return True
    return (
        (fd / "depth").is_dir()
        and (fd / "masks").is_dir()
        and (fd / "keyframe_floors").is_dir()
        and (fd / "floor_metadata.json").exists()
    )

app = Flask(__name__)
_NORMAL_CACHE = {}

# Load configuration
SCRIPT_DIR = Path(__file__).parent.parent
CONFIG_PATH = SCRIPT_DIR / "config.yaml"

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

def get_data_root():
    config = load_config()
    return Path(config['data_root'])

def get_slam_dir_name():
    config = load_config()
    return config['dir_names']['slam']

def get_keyframes_dir_name():
    config = load_config()
    return config['dir_names'].get('keyframes', 'keyframes')

def get_floor_map_dir_name():
    config = load_config()
    return config['dir_names'].get('floor_map', 'floor_map')

def get_floorplan_root():
    config = load_config()
    root = config.get('floorplan_root')
    if root:
        return Path(root)
    return Path("/mnt/data/UNav-IO/final")

def _compute_floor_normal_from_ply(ply_path, max_points=50000):
    if not ply_path.exists():
        return None

    vertex_count = None
    samples = []
    seen = 0
    with open(ply_path, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if line.startswith('element vertex'):
                try:
                    vertex_count = int(line.split()[-1])
                except Exception:
                    vertex_count = None
            if line == 'end_header':
                break

        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2])
            except Exception:
                continue
            seen += 1
            if len(samples) < max_points:
                samples.append([x, y, z])
            else:
                j = random.randint(0, seen - 1)
                if j < max_points:
                    samples[j] = [x, y, z]

    if len(samples) < 3:
        return None

    pts = np.asarray(samples, dtype=np.float64)
    mean = pts.mean(axis=0)
    cov = np.cov((pts - mean).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, 0]
    normal_norm = np.linalg.norm(normal)
    if normal_norm < 1e-8:
        return None
    normal = normal / normal_norm
    if normal[1] < 0:
        normal = -normal
    return normal

def _rotation_from_vectors(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    c = float(np.dot(a, b))
    if c > 0.9999:
        return np.eye(3, dtype=np.float64)
    if c < -0.9999:
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, np.array([0.0, 0.0, 1.0]))
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        return _rotation_axis_angle(axis, np.pi)

    v = np.cross(a, b)
    s = np.linalg.norm(v)
    vx = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=np.float64)
    r = np.eye(3, dtype=np.float64) + vx + (vx @ vx) * ((1.0 - c) / (s * s + 1e-12))
    return r

def _rotation_axis_angle(axis, angle):
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ], dtype=np.float64)

def has_images(floor_path):
    """Depth pipeline only: require precomputed web assets."""
    floor_map_dir = get_floor_map_dir_name()
    web_dir = floor_path / floor_map_dir / "web"
    return (web_dir / "keyframe_points_world.json").exists()

def floor_has_images(data_root, place, building, floor):
    """Check if a specific floor has images"""
    floor_path = data_root / place / building / floor
    return has_images(floor_path)

def building_has_images(data_root, place, building):
    """Check if any floor in the building has images"""
    building_dir = data_root / place / building
    if not building_dir.exists():
        return False
    for floor_dir in building_dir.iterdir():
        if floor_dir.is_dir() and has_images(floor_dir):
            return True
    return False

def place_has_images(data_root, place):
    """Check if any building/floor in the place has images"""
    place_dir = data_root / place
    if not place_dir.exists():
        return False
    for building_dir in place_dir.iterdir():
        if building_dir.is_dir():
            for floor_dir in building_dir.iterdir():
                if floor_dir.is_dir() and has_images(floor_dir):
                    return True
    return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/places')
def get_places():
    """Get list of places that have at least one floor with images"""
    data_root = get_data_root()
    places = []
    if data_root.exists():
        places = [d.name for d in data_root.iterdir()
                  if d.is_dir() and place_has_images(data_root, d.name)]
    # ZInD virtual place
    if ZIND_ROOT.exists():
        places.append(ZIND_PLACE)
    return jsonify(sorted(places))

@app.route('/api/buildings/<place>')
def get_buildings(place):
    """Get buildings that have at least one floor with images"""
    if place == ZIND_PLACE:
        if not ZIND_ROOT.exists():
            return jsonify([])
        scenes = sorted(
            d.name for d in ZIND_ROOT.iterdir()
            if d.is_dir() and any(zind_floor_is_ready(d.name, f.name)
                                  for f in d.iterdir() if f.is_dir())
        )
        return jsonify(scenes)
    data_root = get_data_root()
    place_dir = data_root / place
    if not place_dir.exists():
        return jsonify([])
    buildings = [d.name for d in place_dir.iterdir()
                 if d.is_dir() and building_has_images(data_root, place, d.name)]
    return jsonify(sorted(buildings))

@app.route('/api/floors/<place>/<building>')
def get_floors(place, building):
    """Get floors that have at least one of the required images"""
    if place == ZIND_PLACE:
        scene_dir = ZIND_ROOT / building
        if not scene_dir.exists():
            return jsonify([])
        floors = sorted(f.name for f in scene_dir.iterdir()
                        if f.is_dir() and zind_floor_is_ready(building, f.name))
        return jsonify(floors)
    data_root = get_data_root()
    building_dir = data_root / place / building
    if not building_dir.exists():
        return jsonify([])
    floors = [d.name for d in building_dir.iterdir()
              if d.is_dir() and floor_has_images(data_root, place, building, d.name)]
    return jsonify(sorted(floors))

@app.route('/api/images/<place>/<building>/<floor>')
def get_available_images(place, building, floor):
    """Get list of available images for the specified location (depth only)"""
    available = []
    if place == ZIND_PLACE:
        fd = ZIND_ROOT / building / floor
        if fd.is_dir():
            available.append({'type': 'views', 'name': 'Floor Map'})
        return jsonify(available)
    data_root = get_data_root()
    floor_map_dir = get_floor_map_dir_name()
    base_dir = data_root / place / building / floor / floor_map_dir
    web_dir = base_dir / "web"
    if (web_dir / "keyframe_points_world.json").exists():
        available.append({'type': 'views', 'name': 'Floor Map'})
    return jsonify(available)

@app.route('/api/keyframe_points/<place>/<building>/<floor>')
def get_keyframe_points(place, building, floor):
    """Return sampled per-keyframe camera-frame points and poses for client rendering."""
    if place == ZIND_PLACE:
        base_dir = ZIND_ROOT / building / floor / "keyframe_floors"
    else:
        data_root = get_data_root()
        floor_map_dir = get_floor_map_dir_name()
        floor_map_root = data_root / place / building / floor / floor_map_dir
        web_points = floor_map_root / "web" / "keyframe_points_world.json"
        if web_points.exists():
            with open(web_points, 'r') as f:
                return jsonify(json.load(f))
        base_dir = floor_map_root / "keyframe_floors"

    base_dir = Path(base_dir)  # ensure Path type

    if not base_dir.exists():
        return jsonify({'keyframes': {}})

    # Sampling parameters
    try:
        max_points = int(request.args.get('max_points', 200))
    except Exception:
        max_points = 200

    keyframes = {}
    for json_path in sorted(base_dir.glob('keyframe_*.json')):
        try:
            kf_id = int(json_path.stem.split('_')[1])
        except Exception:
            continue
        with open(json_path, 'r') as f:
            data = json.load(f)
        pts = data.get('points_camera', [])
        if max_points > 0 and len(pts) > max_points:
            step = max(1, len(pts) // max_points)
            pts = pts[::step][:max_points]
        keyframes[str(kf_id)] = {
            'pose_cw': data.get('pose_cw'),
            'points_c': pts
        }

    return jsonify({'keyframes': keyframes})

@app.route('/api/floor_normal/<place>/<building>/<floor>')
def get_floor_normal(place, building, floor):
    cache_key = f"{place}/{building}/{floor}"
    if cache_key in _NORMAL_CACHE:
        return jsonify(_NORMAL_CACHE[cache_key])

    data_root = get_data_root()
    floor_map_dir = get_floor_map_dir_name()
    ply_path = data_root / place / building / floor / floor_map_dir / 'floor_points_depth.ply'

    normal = _compute_floor_normal_from_ply(ply_path)
    if normal is None:
        result = {'normal': None, 'rotation': None}
        _NORMAL_CACHE[cache_key] = result
        return jsonify(result)

    target = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    rotation = _rotation_from_vectors(normal, target)
    result = {
        'normal': normal.tolist(),
        'rotation': rotation.tolist(),
    }
    _NORMAL_CACHE[cache_key] = result
    return jsonify(result)
@app.route('/image/<place>/<building>/<floor>/<image_type>')
def get_image(place, building, floor, image_type):
    """Get image"""
    base_dir = get_floor_base_dir(place, building, floor)

    if image_type in ('map', 'views', 'view_top'):
        image_path = base_dir / "floor_map_depth.png"
    elif image_type == 'view_front':
        image_path = base_dir / "view_front.png"
    elif image_type == 'view_side':
        image_path = base_dir / "view_side.png"
    elif image_type == 'bev_compare':
        image_path = base_dir / "bev_compare.png"
    else:
        abort(404)

    if not image_path.exists():
        abort(404)

    return send_file(image_path, mimetype='image/png')

@app.route('/api/view_metadata/<place>/<building>/<floor>')
def get_view_metadata(place, building, floor):
    """Get camera metadata for interactive view"""
    data_root = get_data_root()
    floor_map_dir = get_floor_map_dir_name()
    base_dir = data_root / place / building / floor / floor_map_dir
    metadata_path = base_dir / "view_metadata.json"

    if not metadata_path.exists():
        return jsonify({'cameras': {}, 'bounds': {}})

    with open(metadata_path, 'r') as f:
        return jsonify(json.load(f))

@app.route('/api/has_separate_views/<place>/<building>/<floor>')
def has_separate_views(place, building, floor):
    """Depth pipeline ready check"""
    if place == ZIND_PLACE:
        ready = zind_floor_is_ready(building, floor)
        return jsonify({'has_separate_views': ready})
    data_root = get_data_root()
    floor_map_dir = get_floor_map_dir_name()
    base_dir = data_root / place / building / floor / floor_map_dir
    web_dir = base_dir / "web"
    return jsonify({'has_separate_views': (web_dir / "keyframe_points_world.json").exists()})

@app.route('/floor_mask/<place>/<building>/<floor>/<int:kf_id>')
def get_floor_mask_image(place, building, floor, kf_id):
    """Get floor mask image by keyframe ID"""
    data_root = get_data_root()
    slam_dir = get_slam_dir_name()
    floor_map_dir = get_floor_map_dir_name()
    config = load_config()
    mask_pattern = config.get('extraction', {}).get('mask_pattern', 'image{}_floor_mask.png')
    mask_dir_name = config.get('dir_names', {}).get('mask', 'masks')

    # Try multiple possible locations - floor_map/masks first
    possible_dirs = [
        data_root / place / building / floor / floor_map_dir / mask_dir_name,  # floor_map/masks/
        data_root / place / building / floor / slam_dir / "keyframes_mask",
        data_root / place / building / floor / "keyframes_mask",
    ]

    for base_dir in possible_dirs:
        mask_path = base_dir / mask_pattern.format(kf_id)
        if mask_path.exists():
            return send_file(mask_path, mimetype='image/png')

    abort(404)


@app.route('/keyframe_with_mask/<place>/<building>/<floor>/<int:kf_id>')
def get_keyframe_with_mask(place, building, floor, kf_id):
    """Get keyframe image with floor mask overlay"""
    import cv2
    import numpy as np
    import io

    data_root = get_data_root()
    slam_dir = get_slam_dir_name()
    keyframes_dir = get_keyframes_dir_name()
    config = load_config()
    mask_pattern = config.get('extraction', {}).get('mask_pattern', 'image{}_floor_mask.png')

    # Find keyframe image
    keyframe_path = None
    possible_kf_dirs = [
        data_root / place / building / floor / keyframes_dir,
        data_root / place / building / floor / slam_dir / keyframes_dir,
    ]
    patterns = [f"image{kf_id}.png", f"image{kf_id}.jpg"]

    for base_dir in possible_kf_dirs:
        for pattern in patterns:
            p = base_dir / pattern
            if p.exists():
                keyframe_path = p
                break
        if keyframe_path:
            break

    if not keyframe_path:
        abort(404)

    # Find mask - check floor_map/masks directory first (where SAM3 saves masks)
    floor_map_dir = get_floor_map_dir_name()
    mask_dir_name = config.get('dir_names', {}).get('mask', 'masks')
    mask_path = None
    possible_mask_dirs = [
        data_root / place / building / floor / floor_map_dir / mask_dir_name,  # floor_map/masks/
        data_root / place / building / floor / slam_dir / "keyframes_mask",
        data_root / place / building / floor / "keyframes_mask",
    ]

    for base_dir in possible_mask_dirs:
        p = base_dir / mask_pattern.format(kf_id)
        if p.exists():
            mask_path = p
            break

    # Load keyframe image
    img = cv2.imread(str(keyframe_path))
    if img is None:
        abort(404)

    # Apply mask highlight if exists: color overlay on masked floor
    if mask_path and mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            if mask.shape[:2] != img.shape[:2]:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

            mask_bin = (mask > 128)
            overlay_color = np.array([255, 180, 0], dtype=np.uint8)  # BGR - bright blue
            alpha = 0.5
            overlay = img.copy()
            overlay[mask_bin] = (
                overlay[mask_bin].astype(np.float32) * (1 - alpha) + overlay_color.astype(np.float32) * alpha
            ).astype(np.uint8)
            img = overlay

    # Encode as PNG
    _, buffer = cv2.imencode('.png', img)
    return send_file(io.BytesIO(buffer.tobytes()), mimetype='image/png')


@app.route('/keyframe/<place>/<building>/<floor>/<int:kf_id>')
def get_keyframe_image(place, building, floor, kf_id):
    """Get keyframe image by ID"""
    data_root = get_data_root()
    keyframes_dir = get_keyframes_dir_name()
    slam_dir = get_slam_dir_name()

    # Try multiple possible locations
    possible_dirs = [
        data_root / place / building / floor / keyframes_dir,  # floor/keyframes/
        data_root / place / building / floor / slam_dir / keyframes_dir,  # floor/slam/keyframes/
    ]

    # Try different naming patterns
    patterns = [
        f"image{kf_id}.png",
        f"image{kf_id}.jpg",
        f"frame{kf_id}.png",
        f"frame{kf_id}.jpg",
        f"{kf_id}.png",
        f"{kf_id}.jpg"
    ]

    for base_dir in possible_dirs:
        for pattern in patterns:
            image_path = base_dir / pattern
            if image_path.exists():
                mimetype = 'image/png' if pattern.endswith('.png') else 'image/jpeg'
                return send_file(image_path, mimetype=mimetype)

    abort(404)

@app.route('/floorplan_image/<place>/<building>/<floor>')
def get_floorplan_image(place, building, floor):
    """Get floorplan.png for the selected floor if it exists."""
    floorplan_root = get_floorplan_root()
    floorplan_path = floorplan_root / place / building / floor / "floorplan.png"
    if not floorplan_path.exists():
        abort(404)
    return send_file(floorplan_path, mimetype='image/png')

@app.route('/api/floorplan_transform/<place>/<building>/<floor>')
def get_floorplan_transform(place, building, floor):
    """Get floorplan transform matrix if it exists."""
    floorplan_root = get_floorplan_root()
    transform_path = floorplan_root / place / building / floor / "transform_matrix.npy"
    if not transform_path.exists():
        return jsonify({'transform': None})
    try:
        matrix = np.load(str(transform_path))
        return jsonify({'transform': matrix.tolist()})
    except Exception:
        return jsonify({'transform': None})

@app.route('/api/config')
def get_config():
    """Get current configuration"""
    config = load_config()
    return jsonify({
        'data_root': config['data_root'],
        'default_place': config.get('default_place', ''),
        'default_building': config.get('default_building', ''),
        'default_floor': config.get('default_floor', '')
    })

@app.route('/image/<place>/<building>/<floor>/bev_mesh')
def get_bev_mesh_png(place, building, floor):
    """Serve bev_compare_mesh.npy as a clean grayscale PNG (white=occupied, black=free)."""
    import io
    import cv2 as _cv2
    base_dir = get_floor_base_dir(place, building, floor)
    npy_path = base_dir / "bev_compare_mesh.npy"
    if not npy_path.exists():
        abort(404)
    mesh = np.load(str(npy_path))          # (Nz, Nx) float32 in [0,1]
    # numpy row-0 = z_min (matplotlib origin='lower'); flip for PNG top-left origin
    img_u8 = (np.flipud(mesh) * 255).astype(np.uint8)
    _, buf = _cv2.imencode('.png', img_u8)
    return send_file(io.BytesIO(buf.tobytes()), mimetype='image/png')


@app.route('/api/bev_bounds/<place>/<building>/<floor>')
def get_bev_bounds(place, building, floor):
    """Return the BEV coordinate bounds used during mesh generation."""
    base_dir = get_floor_base_dir(place, building, floor)
    meta_path = base_dir / "floor_metadata.json"
    if not meta_path.exists():
        return jsonify({'bounds': None})
    try:
        meta = json.loads(meta_path.read_text())
        bounds = meta.get('bounds', {})
        return jsonify({'bounds': bounds})
    except Exception:
        return jsonify({'bounds': None})


@app.route('/api/bev_contour/<place>/<building>/<floor>')
def get_bev_contour(place, building, floor):
    """Return the outer contour polygon(s) of the BEV mesh as world-coordinate JSON."""
    base_dir = get_floor_base_dir(place, building, floor)
    contour_path = base_dir / "bev_compare_contour.json"
    if not contour_path.exists():
        return jsonify([])
    try:
        return jsonify(json.loads(contour_path.read_text()))
    except Exception:
        return jsonify([])


@app.route('/api/bev_status/<place>/<building>/<floor>')
def get_bev_status(place, building, floor):
    """Check whether bev_compare.png exists and return any running job status."""
    base_dir = get_floor_base_dir(place, building, floor)
    image_path = base_dir / "bev_compare.png"
    key = f"{place}/{building}/{floor}"
    job = _bev_jobs.get(key, {})
    return jsonify({
        'exists': image_path.exists(),
        'status': job.get('status', 'idle'),
        'error': job.get('error', ''),
    })


@app.route('/api/generate_bev/<place>/<building>/<floor>', methods=['POST'])
def generate_bev(place, building, floor):
    """Launch BEV mesh comparison generation in a background thread."""
    key = f"{place}/{building}/{floor}"
    if _bev_jobs.get(key, {}).get('status') == 'running':
        return jsonify({'status': 'running', 'message': 'Already running'})

    if not FLOORMAP_SCRIPT.exists():
        return jsonify({'status': 'error', 'error': f'Script not found: {FLOORMAP_SCRIPT}'}), 500

    base_dir = get_floor_base_dir(place, building, floor)
    output_path = base_dir / "bev_compare.png"

    if place == ZIND_PLACE:
        # Read per-floor depth scale from floor_metadata.json
        depth_scale = 3.0  # fallback
        meta_path = base_dir / "floor_metadata.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                depth_scale = meta.get('depth_scale', {}).get('used_scale', depth_scale)
            except Exception:
                pass
        cmd = [
            sys.executable, str(FLOORMAP_SCRIPT),
            '--data-dir',    str(base_dir),
            '--output',      str(output_path),
            '--depth-scale', str(depth_scale),
            '--subsample',   '8',
            '--every',       '1',
            '--voxel-size',  '0.05',
            '--y-min',       '0.5',
            '--y-max',       '1.5',
            '--jump-ratio',  '1.4',
            '--max-tri-area', '150',
            '--flip-h',
        ]
    else:
        config = load_config()
        depth_scale = config.get('da2', {}).get('depth_scale', 5.0)
        subsample   = config.get('extraction', {}).get('depth_subsample', 12)
        cmd = [
            sys.executable, str(FLOORMAP_SCRIPT),
            '--data-dir',    str(base_dir),
            '--output',      str(output_path),
            '--depth-scale', str(depth_scale),
            '--subsample',   str(subsample),
            '--every',       '5',
            '--voxel-size',  '0.06',
            '--y-min',       '-0.2',
            '--y-max',       '1.8',
        ]

    def _run():
        _bev_jobs[key] = {'status': 'running'}
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                _bev_jobs[key] = {'status': 'done'}
            else:
                _bev_jobs[key] = {'status': 'error', 'error': result.stderr[-800:]}
        except Exception as exc:
            _bev_jobs[key] = {'status': 'error', 'error': str(exc)}

    _bev_jobs[key] = {'status': 'running'}
    threading.Thread(target=_run, daemon=True).start()
    return jsonify({'status': 'running'})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8080, help='Port number')
    args = parser.parse_args()

    print("=" * 60)
    print("Floor Map Visualization Web App")
    print("=" * 60)
    print(f"Config file: {CONFIG_PATH}")
    print(f"Data directory: {get_data_root()}")
    print("")
    print(f"Visit http://localhost:{args.port} to view")
    print("=" * 60)
    debug_env = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug_env, host='0.0.0.0', port=args.port)
