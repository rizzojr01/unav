# 🗺️ UNav: Unified Visual Navigation System

A **modular, scalable visual navigation framework** for large buildings, supporting mapping, localization, and pathfinding across multiple floors and buildings. Designed for real-world deployment in robotics, assistive navigation, and digital twin scenarios.

---

## 📦 Prerequisites

Before using UNav, ensure the following are installed:

- **Python 3.8+** (`requirements.txt`)
- **CUDA-compatible GPU** (recommended for feature extraction/matching)
- **Docker** (required for SLAM mapping with stella_vslam_dense)
- **COLMAP** (for triangulation; should be on your `$PATH`)
- **labelme** (for annotating floorplans)
- **[stella_vslam_dense](https://github.com/RoblabWh/stella_vslam_dense.git)** (SLAM mapping via Docker)
- **[implicit_dist](https://github.com/cvg/implicit_dist.git)** (multi-frame pose refinement)
- **[PoseLib](https://github.com/vlarsson/PoseLib)** (robust pose estimation)

**Quick setup for SLAM:**
```sh
git clone https://github.com/RoblabWh/stella_vslam_dense.git
cd stella_vslam_dense/docker
docker build -t stella_vslam_dense .
```

---

## 🚀 One-command Installation (All-in-One)

Install all UNav functionality (mapping, localization, navigation) with one command:

```sh
pip install git+https://github.com/ai4ce/unav.git
```

---

## 🏁 Getting Started (Full System Workflow)

Below is the **essential workflow to construct a metrically registered, navigation-ready environment**:

1. **Mapping:**  
   Run the mapping pipeline to generate 3D map, slice images, extract features, perform matching, and triangulate points.

   ```sh
   python -m unav.run_mapping <data_temp_root> <data_final_root> <feature_model> <place> <building> <floor>
   ```

   Example:
   ```sh
   python -m unav.run_mapping /mnt/data/UNav-IO/temp /mnt/data/UNav-IO/data DinoV2Salad New_York_City LightHouse 4_floor
   ```

2. **Align 3D Map to Floorplan (Required):**  
   This step **is mandatory** for metric localization and navigation.  
   Launch the alignment GUI to register SLAM map coordinates to the architectural floorplan.

   ```sh
   python -m unav.aligner <data_temp_root> <data_final_root> <place> <building> <floor>
   ```

   You must repeat this for every mapped floor/building.

3. **(Optional) Translation Labels:**  
   This step **is mandatory** for metric localization and navigation.  
   If you need multi-language navigation instructions or place names, launch the web-based label editor.
   This will allow you to translate places, buildings, floors, and destinations into your target languages.
   ```sh
   python -m unav.run_translator <data_final_root> [--port <PORT>]
   ```
   Example:
   ```sh
   python -m unav.run_translator /mnt/data/UNav-IO/data --port 5001
   ```
   Then open http://localhost:5001 in your browser.
   The tool is optional: skip this step if you only need English labels.

4. **Localization & Navigation:**  
   Use the generated outputs for real-time localization and navigation.  
   See example usage and API in the `localizer/` and `navigator/` folders and [project documentation](https://github.com/ai4ce/unav).

---

## 📝 Example Notebooks

- `visualize_mapping.ipynb`  
  Visualize mapping results: point clouds, camera trajectories, feature quality.
- `visualize_localization.ipynb`  
  Inspect localization performance, candidate matches, and pose transformation.
- `visualize_navigation.ipynb`  
  Simulate navigation, visualize multi-floor routes, and review generated commands.

---

## 📖 Full Documentation & Code

For further details, localization, navigation modules, and developer guides, visit:  
[https://github.com/ai4ce/unav](https://github.com/ai4ce/unav)

---

## 👤 Maintainer

- **Developer:** Anbang Yang (`ay1620@nyu.edu`)
- **Last updated:** 2025-05-27

