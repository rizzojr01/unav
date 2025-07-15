import logging
from pathlib import Path
import subprocess
from unav.config import UNavMappingConfig
from unav.core.colmap.database_preparer import create_colmap_database_without_poses

logging.basicConfig(level=logging.INFO)

def run_glomap_segment_pipeline(
    config: UNavMappingConfig,
    glomap_bin: str = "glomap",
    glomap_outdir: str = "sparse",
    database_name: str = "database.db",
    image_parent_dirname: str = "perspectives"
) -> None:
    """
    For each segment, create a COLMAP database and call GloMap mapper.

    Args:
        config (UNavMappingConfig): Mapping configuration object.
        glomap_bin (str): Executable for GloMap (default 'glomap').
        glomap_outdir (str): Output subfolder for glomap results.
        database_name (str): Output database name (default 'database.db').
        image_parent_dirname (str): Name of the folder for segment images (default 'perspectives').
    """
    features_dir = Path(config.feature_extraction_config["output_feature_dir"])
    perspectives_root = Path(config.data_temp_dir) / image_parent_dirname
    colmap_sfm_root = Path(config.data_temp_dir) / "colmap_sfm"
    model_name = config.global_descriptor_model

    for tag_dir in sorted(colmap_sfm_root.iterdir()):
        if not tag_dir.is_dir():
            continue
        tag = tag_dir.name

        global_feat_file = features_dir / f"global_features_{model_name}_{tag}.h5"
        local_feat_file = features_dir / f"local_features_{tag}.h5"
        matches_file = tag_dir / "matches.h5"
        pairs_txt = tag_dir / "pairs.txt"
        database_path = tag_dir / database_name
        image_dir = perspectives_root / tag
        glomap_output_dir = tag_dir / glomap_outdir

        missing_files = []
        for f in [global_feat_file, local_feat_file, matches_file, pairs_txt, image_dir]:
            if not f.exists():
                missing_files.append(str(f))
        if missing_files:
            logging.warning(f"[GloMap] Segment [{tag}]: Missing required files:\n    " + "\n    ".join(missing_files))
            continue

        glomap_output_dir.mkdir(parents=True, exist_ok=True)

        # No cameras_txt now, let the function handle camera from features
        logging.info(f"[GloMap] Preparing COLMAP database for segment [{tag}]...")
        create_colmap_database_without_poses(
            database_path=database_path,
            local_feature_file=local_feat_file,
            matches_file=matches_file,
            pairs_txt=pairs_txt,
            overwrite=True
        )

        cmd = [
            glomap_bin, "mapper",
            "--database_path", str(database_path),
            "--image_path", str(image_dir),
            "--output_path", str(glomap_output_dir)
        ]
        logging.info(f"[GloMap] Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            logging.info(f"[GloMap] Segment [{tag}] GloMap mapping finished.")
        except subprocess.CalledProcessError as e:
            logging.error(f"[GloMap] GloMap mapping failed for segment [{tag}]: {e}")
