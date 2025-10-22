"""
Dataset Preparation Script for Trajectory-Guided LTX Video Training

Processes videos to create paired dataset:
- Input: Original videos
- Output: Videos + trajectory visualizations + captions
"""

import sys
sys.path.append('/home/user/LTX_video_training')

import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import argparse
import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.preprocessing.trajectory_extractor import TrajectoryExtractor
from src.preprocessing.trajectory_visualizer import TrajectoryVisualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetPreparator:
    """
    Prepares training dataset with videos and trajectory fields
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        visualization_type: str = 'multi',
        min_frames: int = 60,
        max_frames: int = 300,
        target_fps: int = 30,
        target_resolution: tuple = (704, 1216),
        num_workers: int = 4
    ):
        """
        Initialize dataset preparator

        Args:
            input_dir: Directory containing input videos
            output_dir: Directory to save processed dataset
            visualization_type: Type of trajectory visualization
            min_frames: Minimum number of frames per clip
            max_frames: Maximum number of frames per clip
            target_fps: Target FPS for output videos
            target_resolution: Target resolution (height, width)
            num_workers: Number of parallel workers
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.visualization_type = visualization_type
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.target_fps = target_fps
        self.target_resolution = target_resolution
        self.num_workers = num_workers

        # Create output directories
        self.videos_dir = self.output_dir / "videos"
        self.trajectories_dir = self.output_dir / "trajectories"
        self.captions_dir = self.output_dir / "captions"
        self.metadata_dir = self.output_dir / "metadata"

        for dir_path in [self.videos_dir, self.trajectories_dir,
                        self.captions_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize processors
        self.trajectory_extractor = TrajectoryExtractor()
        self.trajectory_visualizer = TrajectoryVisualizer(
            visualization_type=visualization_type
        )

    def prepare_dataset(self):
        """
        Main dataset preparation pipeline
        """
        logger.info(f"Preparing dataset from {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")

        # Find all video files
        video_files = self._find_video_files()
        logger.info(f"Found {len(video_files)} video files")

        if len(video_files) == 0:
            logger.error("No video files found!")
            return

        # Process videos
        dataset_metadata = []
        failed_videos = []

        if self.num_workers > 1:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {
                    executor.submit(self._process_video, video_path, idx): (video_path, idx)
                    for idx, video_path in enumerate(video_files)
                }

                for future in tqdm(as_completed(futures), total=len(futures),
                                  desc="Processing videos"):
                    video_path, idx = futures[future]
                    try:
                        metadata = future.result()
                        if metadata:
                            dataset_metadata.append(metadata)
                        else:
                            failed_videos.append(video_path)
                    except Exception as e:
                        logger.error(f"Failed to process {video_path}: {e}")
                        failed_videos.append(video_path)
        else:
            # Sequential processing
            for idx, video_path in enumerate(tqdm(video_files, desc="Processing videos")):
                try:
                    metadata = self._process_video(video_path, idx)
                    if metadata:
                        dataset_metadata.append(metadata)
                    else:
                        failed_videos.append(video_path)
                except Exception as e:
                    logger.error(f"Failed to process {video_path}: {e}")
                    failed_videos.append(video_path)

        # Save dataset metadata
        self._save_dataset_metadata(dataset_metadata, failed_videos)

        logger.info(f"Dataset preparation complete!")
        logger.info(f"Successfully processed: {len(dataset_metadata)} videos")
        logger.info(f"Failed: {len(failed_videos)} videos")

    def _find_video_files(self) -> List[Path]:
        """
        Find all video files in input directory
        """
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        video_files = []

        for ext in video_extensions:
            video_files.extend(self.input_dir.glob(f"**/*{ext}"))

        return sorted(video_files)

    def _process_video(self, video_path: Path, video_idx: int) -> Optional[Dict]:
        """
        Process a single video

        Returns metadata dictionary or None if failed
        """
        logger.info(f"Processing {video_path.name}")

        try:
            # Load and validate video
            video_info = self._load_video_info(video_path)
            if not video_info:
                logger.warning(f"Failed to load video info: {video_path}")
                return None

            # Check if video meets requirements
            if video_info['num_frames'] < self.min_frames:
                logger.warning(f"Video too short ({video_info['num_frames']} frames): {video_path}")
                return None

            # Split into clips if too long
            clips = self._split_into_clips(video_path, video_info)

            metadata_list = []
            for clip_idx, clip_info in enumerate(clips):
                clip_metadata = self._process_clip(
                    video_path, clip_info, video_idx, clip_idx
                )
                if clip_metadata:
                    metadata_list.append(clip_metadata)

            return metadata_list if len(metadata_list) > 0 else None

        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _load_video_info(self, video_path: Path) -> Optional[Dict]:
        """
        Load video information
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        info = {
            'num_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }

        cap.release()
        return info

    def _split_into_clips(
        self,
        video_path: Path,
        video_info: Dict
    ) -> List[Dict]:
        """
        Split video into clips of appropriate length
        """
        num_frames = video_info['num_frames']
        fps = video_info['fps']

        if num_frames <= self.max_frames:
            # Single clip
            return [{
                'start_frame': 0,
                'end_frame': num_frames,
                'num_frames': num_frames
            }]

        # Multiple clips with overlap
        clips = []
        overlap = int(self.max_frames * 0.1)  # 10% overlap
        start = 0

        while start < num_frames:
            end = min(start + self.max_frames, num_frames)

            if end - start >= self.min_frames:
                clips.append({
                    'start_frame': start,
                    'end_frame': end,
                    'num_frames': end - start
                })

            start = end - overlap

            # Stop if remaining frames are too short
            if num_frames - start < self.min_frames:
                break

        return clips

    def _process_clip(
        self,
        video_path: Path,
        clip_info: Dict,
        video_idx: int,
        clip_idx: int
    ) -> Optional[Dict]:
        """
        Process a single clip from video
        """
        clip_name = f"video_{video_idx:06d}_clip_{clip_idx:03d}"
        logger.info(f"Processing clip: {clip_name}")

        try:
            # Extract clip frames
            frames = self._extract_clip_frames(video_path, clip_info)
            if frames is None or len(frames) == 0:
                return None

            # Resize frames
            frames = self._resize_frames(frames)

            # Save processed video
            video_output_path = self.videos_dir / f"{clip_name}.mp4"
            self._save_video(frames, video_output_path)

            # Extract trajectories
            trajectory_data = self.trajectory_extractor.extract_from_video(
                str(video_output_path)
            )

            # Create trajectory visualization
            trajectory_output_path = self.trajectories_dir / f"{clip_name}_traj.mp4"
            vis_frames = self.trajectory_visualizer.visualize(
                trajectory_data,
                first_frame=frames[0],
                output_path=str(trajectory_output_path)
            )

            # Generate caption
            caption = self._generate_caption(
                video_path, clip_info, trajectory_data
            )

            # Save caption
            caption_path = self.captions_dir / f"{clip_name}.txt"
            with open(caption_path, 'w') as f:
                f.write(caption)

            # Create metadata
            metadata = {
                'clip_name': clip_name,
                'original_video': str(video_path),
                'video_path': str(video_output_path),
                'trajectory_path': str(trajectory_output_path),
                'caption_path': str(caption_path),
                'num_frames': len(frames),
                'resolution': self.target_resolution,
                'fps': self.target_fps,
                'clip_info': clip_info
            }

            return metadata

        except Exception as e:
            logger.error(f"Error processing clip {clip_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _extract_clip_frames(
        self,
        video_path: Path,
        clip_info: Dict
    ) -> Optional[np.ndarray]:
        """
        Extract frames for a clip
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        start_frame = clip_info['start_frame']
        end_frame = clip_info['end_frame']

        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        for _ in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()

        if len(frames) == 0:
            return None

        return np.stack(frames)

    def _resize_frames(self, frames: np.ndarray) -> np.ndarray:
        """
        Resize frames to target resolution
        """
        target_h, target_w = self.target_resolution
        resized = []

        for frame in frames:
            resized_frame = cv2.resize(frame, (target_w, target_h))
            resized.append(resized_frame)

        return np.stack(resized)

    def _save_video(self, frames: np.ndarray, output_path: Path):
        """
        Save frames as video
        """
        T, H, W, C = frames.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            float(self.target_fps),
            (W, H)
        )

        for frame in frames:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(bgr)

        out.release()

    def _generate_caption(
        self,
        video_path: Path,
        clip_info: Dict,
        trajectory_data: Dict
    ) -> str:
        """
        Generate caption for the clip

        For now, uses simple template. Can be enhanced with:
        - BLIP/CLIP for visual captioning
        - Motion analysis for trajectory description
        - Scene understanding
        """
        # Analyze motion
        trajectories = trajectory_data['trajectories']
        flow_magnitude = np.sqrt(
            trajectories[:, :, :, 0]**2 +
            trajectories[:, :, :, 1]**2
        )
        avg_motion = flow_magnitude.mean()

        # Classify motion
        if avg_motion < 1.0:
            motion_desc = "static scene with minimal movement"
        elif avg_motion < 5.0:
            motion_desc = "slow motion"
        elif avg_motion < 15.0:
            motion_desc = "medium speed motion"
        else:
            motion_desc = "fast motion"

        # Detect dominant motion direction
        flow_x = trajectories[:, :, :, 0].mean()
        flow_y = trajectories[:, :, :, 1].mean()

        if abs(flow_x) > abs(flow_y):
            if flow_x > 0:
                direction = "moving right"
            else:
                direction = "moving left"
        else:
            if flow_y > 0:
                direction = "moving down"
            else:
                direction = "moving up"

        # Count occlusions
        if 'occlusions' in trajectory_data:
            num_occlusions = trajectory_data['occlusions'].sum()
            occlusion_desc = f", {int(num_occlusions)} occlusions"
        else:
            occlusion_desc = ""

        # Build caption
        caption = (
            f"A video clip with {motion_desc}, {direction}"
            f"{occlusion_desc}. "
            f"Source: {video_path.stem}"
        )

        return caption

    def _save_dataset_metadata(
        self,
        dataset_metadata: List[Dict],
        failed_videos: List[Path]
    ):
        """
        Save overall dataset metadata
        """
        # Flatten metadata if nested
        flat_metadata = []
        for item in dataset_metadata:
            if isinstance(item, list):
                flat_metadata.extend(item)
            else:
                flat_metadata.append(item)

        # Save as JSON
        metadata_path = self.output_dir / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                'num_clips': len(flat_metadata),
                'clips': flat_metadata,
                'failed_videos': [str(p) for p in failed_videos]
            }, f, indent=2)

        logger.info(f"Saved dataset metadata to {metadata_path}")

        # Create training CSV
        csv_path = self.output_dir / "train.csv"
        with open(csv_path, 'w') as f:
            f.write("video_path,trajectory_path,caption_path\n")
            for item in flat_metadata:
                f.write(f"{item['video_path']},{item['trajectory_path']},{item['caption_path']}\n")

        logger.info(f"Saved training CSV to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset for trajectory-guided LTX Video training"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input videos"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save processed dataset"
    )
    parser.add_argument(
        "--visualization_type",
        type=str,
        default="multi",
        choices=['flow', 'depth', 'confidence', 'multi'],
        help="Type of trajectory visualization"
    )
    parser.add_argument(
        "--min_frames",
        type=int,
        default=60,
        help="Minimum frames per clip"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=121,
        help="Maximum frames per clip"
    )
    parser.add_argument(
        "--target_fps",
        type=int,
        default=30,
        help="Target FPS"
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="704x1216",
        help="Target resolution (HxW)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers"
    )

    args = parser.parse_args()

    # Parse resolution
    h, w = map(int, args.resolution.split('x'))
    target_resolution = (h, w)

    # Create preparator
    preparator = DatasetPreparator(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        visualization_type=args.visualization_type,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        target_fps=args.target_fps,
        target_resolution=target_resolution,
        num_workers=args.num_workers
    )

    # Prepare dataset
    preparator.prepare_dataset()


if __name__ == "__main__":
    main()
