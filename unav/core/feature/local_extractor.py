import os
import sys
import numpy as np
import torch
import cv2

from unav.core.third_party.SuperPoint_SuperGlue.base_model import dynamic_load
from unav.core.third_party.SuperPoint_SuperGlue import extractors, matchers
from unav.core.third_party.LightGlue.lightglue import LightGlue


def _ensure_mast3r_importable():
    """
    Make MASt3R importable across installations.

    Resolution order:
      1. Already importable (e.g. ``pip install unav[mast3r]`` or editable install)
      2. ``MAST3R_PATH`` environment variable (absolute path to mast3r repo root)
      3. Common fallback locations (Docker ``/workspace/mast3r``, legacy
         ``~/Desktop/mast3r``) — kept for backward compatibility only.

    Call this before any ``from mast3r.xxx import ...`` statement. It is
    idempotent and cheap to call repeatedly.
    """
    try:
        import mast3r  # noqa: F401
        return
    except ImportError:
        pass

    candidates = []
    env_path = os.environ.get("MAST3R_PATH")
    if env_path:
        candidates.append(env_path)
    # Backward-compat fallbacks (Docker image + legacy UNav host layout)
    candidates.extend([
        "/workspace/mast3r",
        os.path.expanduser("~/Desktop/mast3r"),
    ])

    for p in candidates:
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
            try:
                import mast3r  # noqa: F401
                return
            except ImportError:
                continue

    raise ImportError(
        "Could not import 'mast3r'. Install with `pip install unav[mast3r]`, "
        "or set the MAST3R_PATH environment variable to the mast3r repo root."
    )


class Superpoint:
    """
    SuperPoint local feature extractor wrapper.
    """
    def __init__(self, device, conf):
        """
        Args:
            device (str): Device string ("cuda" or "cpu").
            conf (dict): SuperPoint configuration dictionary.
        """
        Model_sp = dynamic_load(extractors, conf["detector_name"])
        self.local_feature_extractor = (
            Model_sp({
                "name": conf["detector_name"],
                "nms_radius": conf["nms_radius"],
                "max_keypoints": conf["max_keypoints"],
            })
            .eval()
            .to(device)
        )
        self.device = device

    def prepare_data(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert BGR image to normalized torch tensor (1, 1, H, W) for inference.
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        image = image[None]
        data = torch.from_numpy(image / 255.0).unsqueeze(0)
        return data

    def extract_local_features(self, image0: np.ndarray) -> dict:
        """
        Extract local features from the given image.

        Args:
            image0 (np.ndarray): Input BGR image.

        Returns:
            dict: { 'keypoints', 'scores', 'descriptors', 'image_size', ... }
        """
        data0 = self.prepare_data(image0)
        pred0 = self.local_feature_extractor(data0.to(self.device))
        del data0
        torch.cuda.empty_cache()
        pred0 = {k: v[0].cpu().detach().numpy() for k, v in pred0.items()}
        if "keypoints" in pred0:
            pred0["keypoints"] = (pred0["keypoints"] + 0.5) - 0.5
        pred0["image_size"] = np.array([image0.shape[1], image0.shape[0]])
        return pred0

class Local_extractor:
    """
    Factory for local feature extractors and matchers.
    Supports SuperPoint+SuperGlue, SuperPoint+LightGlue, and extension to SIFT/SURF.
    """
    def __init__(self, configs: dict):
        """
        Args:
            configs (dict): Configuration for all extractors/matchers.
        """
        self.configs = configs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def lightglue(self, conf: dict):
        """
        Initialize LightGlue matcher.

        Args:
            conf (dict): LightGlue configuration.
        Returns:
            LightGlue model (callable)
        """
        return LightGlue(pretrained="superpoint", **conf["match_conf"]).eval()

    def superglue(self, conf: dict):
        """
        Initialize SuperGlue matcher.

        Args:
            conf (dict): SuperGlue configuration.
        Returns:
            SuperGlue model (callable)
        """
        Model_sg = dynamic_load(matchers, conf["matcher_name"])
        return Model_sg({
            "name": conf["matcher_name"],
            "weights": conf["weights"],
            "sinkhorn_iterations": conf["sinkhorn_iterations"],
        }).eval()

    def extractor(self):
        """
        Returns the local feature extractor function for the specified configuration.

        Returns:
            Callable: Function that takes an image and returns extracted features.
        """
        for name, content in self.configs.items():
            if name == "mast3r":
                return None  # MASt3R does joint extraction+matching
            if name == "superpoint+superglue":
                superpoint = Superpoint(self.device, self.configs["superpoint+superglue"])
                return superpoint.extract_local_features
            elif name == "superpoint+lightglue":
                superpoint = Superpoint(self.device, self.configs["superpoint+lightglue"])
                return superpoint.extract_local_features
            elif name == "sift":
                # TODO: Implement SIFT extractor if needed
                pass
            elif name == "surf":
                # TODO: Implement SURF extractor if needed
                pass
        raise ValueError("No supported local extractor config found.")

    def matcher(self):
        """
        Returns the local feature matcher for the specified configuration.

        Returns:
            Callable: Matcher model.
        """
        for name, content in self.configs.items():
            if name == "mast3r":
                return MASt3RExtractor(content, device=self.device)
            if name == "superpoint+superglue":
                return self.superglue(self.configs["superpoint+superglue"])
            elif name == "superpoint+lightglue":
                return self.lightglue(self.configs["superpoint+lightglue"])
            elif name == "sift":
                # TODO: Implement SIFT matcher if needed
                pass
            elif name == "surf":
                # TODO: Implement SURF matcher if needed
                pass
        raise ValueError("No supported matcher config found.")


# ═══════════════════════════════════════════════════════════
# MASt3R Dense Matcher
# ═══════════════════════════════════════════════════════════
import sys
import numpy as np
from PIL import Image

class MASt3RExtractor:
    """
    Dense matcher using MASt3R (ECCV 2024).
    Replaces SuperPoint+LightGlue for textureless environments.
    """

    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self):
        _ensure_mast3r_importable()
        from mast3r.model import AsymmetricMASt3R
        self.model = AsymmetricMASt3R.from_pretrained(
            self.config.get("model_name", "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
        ).to(self.device)
        self.model.eval()
        # torch.compile disabled (CUDA graph conflict with MASt3R position cache)
        print(f"[MASt3R] Loaded on {self.device}")

    def match_pair(self, query_img_path, db_img_path):
        """
        Run MASt3R on a (query, DB) image pair.

        Returns:
            query_2d: (N, 2) matched pixel coords in query (at original resolution)
            db_2d: (N, 2) matched pixel coords in DB (at original resolution)
            confidence: (N,) confidence scores
            Or (None, None, None) on failure.
        """
        import torch
        _ensure_mast3r_importable()
        from mast3r.fast_nn import fast_reciprocal_NNs
        from dust3r.inference import inference
        from dust3r.utils.image import load_images

        mast3r_size = self.config.get("mast3r_size", 512)
        max_matches = self.config.get("max_matches", 2000)
        subsample = self.config.get("subsample", 8)

        try:
            images = load_images([db_img_path, query_img_path], size=mast3r_size)
            with torch.no_grad(), torch.cuda.amp.autocast():
                output = inference([tuple(images)], self.model, self.device,
                                   batch_size=1, verbose=False)

            desc1 = output['pred1']['desc'].squeeze(0).detach()
            desc2 = output['pred2']['desc'].squeeze(0).detach()
            conf1 = output['pred1']['conf'].squeeze(0).detach().cpu().numpy()
            conf2 = output['pred2']['conf'].squeeze(0).detach().cpu().numpy()

            matches_db, matches_query = fast_reciprocal_NNs(
                desc1, desc2, subsample_or_initxy1=subsample,
                device=self.device, dist='dot', block_size=2**13
            )

            if len(matches_db) < 6:
                return None, None, None

            # Confidence filter
            c1 = conf1[matches_db[:, 1].astype(int), matches_db[:, 0].astype(int)]
            c2 = conf2[matches_query[:, 1].astype(int), matches_query[:, 0].astype(int)]
            conf_scores = c1 * c2
            top_k = min(max_matches, len(conf_scores))
            top_idx = np.argsort(conf_scores)[-top_k:]

            m_db = matches_db[top_idx].astype(np.float64)
            m_query = matches_query[top_idx].astype(np.float64)
            conf_out = conf_scores[top_idx]

            # Scale to original resolution
            # Use cv2 to get actual pixel dimensions (ignoring EXIF rotation)
            import cv2 as _cv2
            _db_img = _cv2.imread(db_img_path)
            db_h, db_w = _db_img.shape[:2]  # cv2: (H, W, C)
            m_h, m_w = desc1.shape[0], desc1.shape[1]  # desc: (H, W, D)
            m_db[:, 0] *= db_w / m_w
            m_db[:, 1] *= db_h / m_h

            _q_img = _cv2.imread(query_img_path)
            q_h, q_w = _q_img.shape[:2]
            mq_h, mq_w = desc2.shape[0], desc2.shape[1]
            m_query[:, 0] *= q_w / mq_w
            m_query[:, 1] *= q_h / mq_h

            return m_query, m_db, conf_out

        except Exception as e:
            print(f"[MASt3R] match_pair error: {e}")
            return None, None, None


    def match_batch(self, query_img_path, db_img_paths):
        """
        Run MASt3R on multiple (query, DB) pairs in one batch.

        Returns:
            list of (query_2d, db_2d, confidence) tuples, one per db_img_path.
            None entries for failed pairs.
        """
        import torch
        _ensure_mast3r_importable()
        from mast3r.fast_nn import fast_reciprocal_NNs
        from dust3r.inference import inference
        from dust3r.utils.image import load_images

        mast3r_size = self.config.get("mast3r_size", 512)
        max_matches = self.config.get("max_matches", 2000)
        subsample = self.config.get("subsample", 8)

        # Build all pairs: each is (db, query)
        valid_indices = []
        all_pairs = []
        for i, db_path in enumerate(db_img_paths):
            if not os.path.exists(db_path):
                continue
            try:
                imgs = load_images([db_path, query_img_path], size=mast3r_size)
                all_pairs.append(tuple(imgs))
                valid_indices.append(i)
            except Exception:
                continue

        if not all_pairs:
            return [None] * len(db_img_paths)

        # Batch inference with AMP
        with torch.no_grad(), torch.cuda.amp.autocast():
            output = inference(all_pairs, self.model, self.device,
                               batch_size=len(all_pairs), verbose=False)

        # Process each pair result
        import cv2 as _cv2
        results = [None] * len(db_img_paths)

        for batch_idx, orig_idx in enumerate(valid_indices):
            try:
                desc1 = output['pred1']['desc'][batch_idx].detach()
                desc2 = output['pred2']['desc'][batch_idx].detach()
                conf1 = output['pred1']['conf'][batch_idx].detach().cpu().numpy()
                conf2 = output['pred2']['conf'][batch_idx].detach().cpu().numpy()

                matches_db, matches_query = fast_reciprocal_NNs(
                    desc1, desc2, subsample_or_initxy1=subsample,
                    device=self.device, dist='dot', block_size=2**13
                )

                if len(matches_db) < 6:
                    continue

                c1 = conf1[matches_db[:, 1].astype(int), matches_db[:, 0].astype(int)]
                c2 = conf2[matches_query[:, 1].astype(int), matches_query[:, 0].astype(int)]
                conf_scores = c1 * c2
                top_k = min(max_matches, len(conf_scores))
                top_idx = np.argsort(conf_scores)[-top_k:]

                m_db = matches_db[top_idx].astype(np.float64)
                m_query = matches_query[top_idx].astype(np.float64)
                conf_out = conf_scores[top_idx]

                # Scale to original resolution using cv2
                db_path = db_img_paths[orig_idx]
                _db_img = _cv2.imread(db_path)
                db_h, db_w = _db_img.shape[:2]
                m_h, m_w = desc1.shape[0], desc1.shape[1]
                m_db[:, 0] *= db_w / m_w
                m_db[:, 1] *= db_h / m_h

                _q_img = _cv2.imread(query_img_path)
                q_h, q_w = _q_img.shape[:2]
                mq_h, mq_w = desc2.shape[0], desc2.shape[1]
                m_query[:, 0] *= q_w / mq_w
                m_query[:, 1] *= q_h / mq_h

                results[orig_idx] = (m_query, m_db, conf_out)
            except Exception:
                continue

        return results


    def match_pair_with_pts3d(self, query_img_path, db_img_path):
        """
        MASt3R matching + return ref's 3D pointmap at matched locations.
        For RelPose method (no colmap 3D needed).

        Returns: (query_2d, pts3d_matched, n_matches) or None
        """
        import torch
        _ensure_mast3r_importable()
        from mast3r.fast_nn import fast_reciprocal_NNs
        from dust3r.inference import inference
        from dust3r.utils.image import load_images

        mast3r_size = self.config.get("mast3r_size", 512)
        max_matches = self.config.get("max_matches", 500)
        subsample = self.config.get("subsample", 16)

        try:
            images = load_images([db_img_path, query_img_path], size=mast3r_size)
            with torch.no_grad(), torch.cuda.amp.autocast():
                output = inference([tuple(images)], self.model, self.device,
                                   batch_size=1, verbose=False)

            pts3d_ref = output['pred1']['pts3d'].squeeze(0).cpu().numpy()
            desc1 = output['pred1']['desc'].squeeze(0).detach()
            desc2 = output['pred2']['desc'].squeeze(0).detach()
            conf1 = output['pred1']['conf'].squeeze(0).detach().cpu().numpy()
            conf2 = output['pred2']['conf'].squeeze(0).detach().cpu().numpy()

            matches_db, matches_query = fast_reciprocal_NNs(
                desc1, desc2, subsample_or_initxy1=subsample,
                device=self.device, dist='dot', block_size=2**13
            )
            if len(matches_db) < 6:
                return None

            c1 = conf1[matches_db[:, 1].astype(int), matches_db[:, 0].astype(int)]
            c2 = conf2[matches_query[:, 1].astype(int), matches_query[:, 0].astype(int)]
            top_k = min(max_matches, len(c1))
            top_idx = np.argsort(c1 * c2)[-top_k:]

            # 3D from ref pointmap
            pts3d_matched = pts3d_ref[
                matches_db[top_idx, 1].astype(int),
                matches_db[top_idx, 0].astype(int)
            ]

            # Query 2D scaled to original
            import cv2 as _cv2
            m_query = matches_query[top_idx].astype(np.float64)
            _q_img = _cv2.imread(query_img_path)
            q_h, q_w = _q_img.shape[:2]
            mq_h, mq_w = desc2.shape[0], desc2.shape[1]
            m_query[:, 0] *= q_w / mq_w
            m_query[:, 1] *= q_h / mq_h

            return m_query, pts3d_matched.astype(np.float64), len(m_query)

        except Exception as e:
            print(f"[MASt3R] match_pair_with_pts3d error: {e}")
            return None

    def dummy_extract(self, image):
        """No-op local feature extraction (MASt3R does joint matching)."""
        return None
