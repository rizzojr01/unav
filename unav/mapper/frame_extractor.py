import os
import cv2
from tqdm import tqdm
from unav.config import UNavMappingConfig

def extract_frames_from_videos(config: UNavMappingConfig, frame_interval: int = 1, img_ext: str = "jpg") -> None:
    """
    Extract frames from all video files in the input folder and save to dedicated subfolders.

    Args:
        config (UNavMappingConfig): Configuration object containing input and output folders.
        frame_interval (int): Extract one frame every `frame_interval` frames. Default is 1 (extract every frame).
        img_ext (str): Image file extension to save extracted frames (e.g., 'jpg', 'png').

    Raises:
        FileNotFoundError: If the input folder does not exist.
        ValueError: If no video files are found in the input folder.
    """
    input_folder = config.frame_extractor_config['input_folder']
    output_folder = config.frame_extractor_config['output_folder']

    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")
    os.makedirs(output_folder, exist_ok=True)

    # Supported video file extensions
    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm"}
    video_files = [
        f for f in os.listdir(input_folder)
        if os.path.splitext(f)[1].lower() in VIDEO_EXTS
    ]

    if not video_files:
        raise ValueError(f"No video files found in {input_folder}")

    for video_file in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(input_folder, video_file)
        video_name = os.path.splitext(video_file)[0]
        video_output_dir = os.path.join(output_folder, video_name)
        os.makedirs(video_output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Failed to open video {video_path}. Skipping.")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0
        saved_idx = 0

        with tqdm(total=total_frames, desc=f"Extracting {video_name}", leave=False) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_interval == 0:
                    img_name = f"{video_name}_frame{saved_idx:06d}.{img_ext}"
                    img_path = os.path.join(video_output_dir, img_name)
                    cv2.imwrite(img_path, frame)
                    saved_idx += 1
                frame_idx += 1
                pbar.update(1)
        cap.release()
        print(f"Extracted {saved_idx} frames from {video_file} to {video_output_dir}")