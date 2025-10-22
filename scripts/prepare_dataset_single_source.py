"""
Enhanced Dataset Preparation for Trajectory-Guided Training
SINGLE-SOURCE APPROACH: Same high-quality videos for both trajectory extraction and training targets

Workflow:
1. Input: High-quality realistic videos
2. Extract trajectories from SAME videos using Trace Anything (or optical flow)
3. Create trajectory visualizations (conditioning input)
4. Use ORIGINAL videos as training targets
5. Generate motion-aware captions

Result: Training pairs where model learns (first_frame + trajectory) → realistic_video
"""

import sys
sys.path.append('/home/user/LTX_video_training')

import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
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


class MotionAnalyzer:
    """
    Analyzes trajectory fields to generate descriptive captions
    """

    def analyze_motion(self, trajectory_data: Dict[str, np.ndarray]) -> Dict:
        """
        Analyze trajectory field to determine motion characteristics

        Returns motion statistics for caption generation
        """
        trajectories = trajectory_data['trajectories']
        H, W, T, _ = trajectories.shape

        # Compute per-pixel motion variance
        if T > 1:
            motion = trajectories[:, :, 1:, :2] - trajectories[:, :, :-1, :2]
            variance = np.var(motion, axis=2)
        else:
            variance = np.zeros((H, W, 2))

        # Global variance (camera motion indicator)
        global_variance = np.mean(variance)

        # Local variance (object motion indicator)
        local_variance = np.std(variance)

        # Camera vs object motion
        if local_variance > 1e-6:
            camera_motion_score = min(global_variance / (local_variance + 1e-6), 1.0)
        else:
            camera_motion_score = 1.0

        # Compute mean flow direction
        if T > 1:
            mean_flow = np.mean(motion, axis=(0, 1, 2))
            flow_x, flow_y = mean_flow[0], mean_flow[1]
        else:
            flow_x, flow_y = 0.0, 0.0

        # Motion magnitude
        motion_magnitude = np.sqrt(flow_x**2 + flow_y**2)

        # Classify camera motion
        camera_type = self._classify_camera_motion(flow_x, flow_y, trajectories)

        # Describe motion speed
        motion_description = self._describe_motion_speed(motion_magnitude)

        # Count occlusions if available
        num_occlusions = int(trajectory_data.get('occlusions', np.zeros((H, W, T))).sum())

        return {
            'camera_motion_score': float(camera_motion_score),
            'object_motion_score': float(1.0 - camera_motion_score),
            'camera_type': camera_type,
            'motion_description': motion_description,
            'motion_magnitude': float(motion_magnitude),
            'num_occlusions': num_occlusions,
            'flow_x': float(flow_x),
            'flow_y': float(flow_y)
        }

    def _classify_camera_motion(
        self,
        flow_x: float,
        flow_y: float,
        trajectories: np.ndarray
    ) -> str:
        """
        Classify type of camera motion
        """
        abs_x = abs(flow_x)
        abs_y = abs(flow_y)

        # Check for zoom (radial pattern)
        H, W, T, _ = trajectories.shape
        center_y, center_x = H // 2, W // 2

        if T > 1:
            # Check if motion is radial from center
            motion = trajectories[:, :, -1, :2] - trajectories[:, :, 0, :2]
            y_coords, x_coords = np.meshgrid(range(H), range(W), indexing='ij')
            y_from_center = y_coords - center_y
            x_from_center = x_coords - center_x

            # Compute radial component
            radial = (motion[:, :, 0] * x_from_center + motion[:, :, 1] * y_from_center)
            radial_mean = np.mean(radial)

            if abs(radial_mean) > max(abs_x, abs_y) * 0.5:
                return "zoom in" if radial_mean > 0 else "zoom out"

        # Simple directional motion
        if abs_x > abs_y * 1.5:
            return "pan right" if flow_x > 0 else "pan left"
        elif abs_y > abs_x * 1.5:
            return "tilt down" if flow_y > 0 else "tilt up"
        elif abs_x > 0.5 or abs_y > 0.5:
            # Diagonal
            h_dir = "right" if flow_x > 0 else "left"
            v_dir = "down" if flow_y > 0 else "up"
            return f"pan {h_dir} and tilt {v_dir}"
        else:
            return "static"

    def _describe_motion_speed(self, magnitude: float) -> str:
        """
        Describe motion speed
        """
        if magnitude < 1.0:
            return "minimal motion"
        elif magnitude < 3.0:
            return "slow motion"
        elif magnitude < 8.0:
            return "medium speed motion"
        elif magnitude < 15.0:
            return "fast motion"
        else:
            return "very fast motion"

    def generate_caption(
        self,
        motion_stats: Dict,
        video_name: str
    ) -> str:
        """
        Generate descriptive caption from motion statistics
        """
        caption_parts = []

        # Add motion description
        if motion_stats['camera_motion_score'] > 0.6:
            # Primarily camera motion
            caption_parts.append(f"camera {motion_stats['camera_type']}")
        elif motion_stats['object_motion_score'] > 0.6:
            # Primarily object motion
            caption_parts.append(f"{motion_stats['motion_description']}")
        else:
            # Mixed motion
            caption_parts.append(f"{motion_stats['motion_description']}")
            if motion_stats['camera_type'] != "static":
                caption_parts.append(f"with camera {motion_stats['camera_type']}")

        # Add occlusion info if significant
        if motion_stats['num_occlusions'] > 100:
            caption_parts.append(f"with occlusions")

        # Add source info
        caption_parts.append("realistic footage")

        # Join and clean up
        caption = ", ".join(caption_parts)

        # Capitalize first letter
        caption = caption[0].upper() + caption[1:]

        return caption


class SingleSourceDatasetPreparator:
    """
    Prepares training dataset using SINGLE-SOURCE approach:
    - Same high-quality videos for both trajectory extraction and training targets
    - Ensures perfect alignment between trajectory and realistic output
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        visualization_type: str = 'multi',
        overlay_first_frame: bool = True,
        overlay_alpha: float = 0.3,
        min_frames: int = 60,
        max_frames: int = 121,
        target_fps: int = 30,
        target_resolution: tuple = (704, 1216),
        num_workers: int = 4
    ):
        """
        Initialize single-source dataset preparator

        Args:
            input_dir: Directory with high-quality source videos
            output_dir: Output directory for processed dataset
            visualization_type: 'multi', 'flow', 'depth', or 'confidence'
            overlay_first_frame: Overlay trajectory on first frame (helps preserve structure)
            overlay_alpha: Alpha for first frame overlay (0-1)
            min_frames: Minimum frames per clip
            max_frames: Maximum frames per clip
            target_fps: Target FPS
            target_resolution: Target resolution (H, W)
            num_workers: Parallel workers
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.visualization_type = visualization_type
        self.overlay_first_frame = overlay_first_frame
        self.overlay_alpha = overlay_alpha
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.target_fps = target_fps
        self.target_resolution = target_resolution
        self.num_workers = num_workers

        # Create output structure
        self.videos_dir = self.output_dir / "videos"  # Original videos (training targets)
        self.trajectories_dir = self.output_dir / "trajectories"  # Trajectory viz (conditioning)
        self.captions_dir = self.output_dir / "captions"  # Auto-generated captions
        self.metadata_dir = self.output_dir / "metadata"  # Processing metadata

        for dir_path in [self.videos_dir, self.trajectories_dir,
                        self.captions_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize processors
        self.trajectory_extractor = TrajectoryExtractor(enable_physics_smoothing=True)
        self.trajectory_visualizer = TrajectoryVisualizer(
            visualization_type=visualization_type,
            overlay_first_frame=overlay_first_frame
        )
        self.motion_analyzer = MotionAnalyzer()

        logger.info("=" * 70)
        logger.info("SINGLE-SOURCE Dataset Preparation")
        logger.info("=" * 70)
        logger.info(f"Approach: Same videos for trajectory extraction AND training targets")
        logger.info(f"Input: {self.input_dir}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Trajectory visualization: {visualization_type}")
        logger.info(f"Overlay first frame: {overlay_first_frame} (alpha={overlay_alpha})")
        logger.info("=" * 70)

    def prepare_dataset(self):
        """
        Main preparation pipeline: process all videos
        """
        # Find source videos
        video_files = self._find_video_files()
        logger.info(f"Found {len(video_files)} source videos")

        if len(video_files) == 0:
            logger.error("No video files found!")
            return

        # Process all videos
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
                            if isinstance(metadata, list):
                                dataset_metadata.extend(metadata)
                            else:
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
                        if isinstance(metadata, list):
                            dataset_metadata.extend(metadata)
                        else:
                            dataset_metadata.append(metadata)
                    else:
                        failed_videos.append(video_path)
                except Exception as e:
                    logger.error(f"Failed to process {video_path}: {e}")
                    failed_videos.append(video_path)

        # Save dataset metadata
        self._save_dataset_metadata(dataset_metadata, failed_videos)

        logger.info("=" * 70)
        logger.info("Dataset Preparation Complete!")
        logger.info(f"Successfully processed: {len(dataset_metadata)} clips")
        logger.info(f"Failed: {len(failed_videos)} videos")
        logger.info(f"Output saved to: {self.output_dir}")
        logger.info("=" * 70)

    def _process_video(self, video_path: Path, video_idx: int) -> Optional[List[Dict]]:
        """
        Process single source video to create training pair(s)

        From ONE video:
        1. Load frames (high-quality realistic content)
        2. Extract trajectory field
        3. Create trajectory visualization (conditioning input)
        4. Save original as training target
        5. Generate caption from motion analysis

        Returns:
            List of metadata dicts (one per clip if video is split)
        """
        logger.info(f"\nProcessing: {video_path.name}")

        try:
            # Load and validate
            video_info = self._load_video_info(video_path)
            if not video_info or video_info['num_frames'] < self.min_frames:
                logger.warning(f"Video too short or invalid: {video_path}")
                return None

            # Split into clips if needed
            clips = self._split_into_clips(video_path, video_info)

            metadata_list = []
            for clip_idx, clip_info in enumerate(clips):
                clip_metadata = self._process_clip(
                    video_path, clip_info, video_idx, clip_idx
                )
                if clip_metadata:
                    metadata_list.append(clip_metadata)

            return metadata_list

        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _process_clip(
        self,
        video_path: Path,
        clip_info: Dict,
        video_idx: int,
        clip_idx: int
    ) -> Optional[Dict]:
        """
        Process single clip: extract trajectory and create training pair
        """
        clip_name = f"video_{video_idx:06d}_clip_{clip_idx:03d}"

        try:
            # 1. Extract clip frames (high-quality realistic content)
            frames = self._extract_clip_frames(video_path, clip_info)
            if frames is None or len(frames) == 0:
                return None

            # Resize to target resolution
            frames = self._resize_frames(frames)

            # Save processed video (training target)
            video_output_path = self.videos_dir / f"{clip_name}.mp4"
            self._save_video(frames, video_output_path)

            logger.info(f"  Saved training target: {video_output_path.name}")

            # 2. Extract trajectory field from SAME video
            logger.info(f"  Extracting trajectories...")
            trajectory_data = self.trajectory_extractor.extract_from_video(
                str(video_output_path)
            )

            # 3. Analyze motion for caption
            motion_stats = self.motion_analyzer.analyze_motion(trajectory_data)

            # 4. Create trajectory visualization (conditioning input)
            logger.info(f"  Creating trajectory visualization...")
            trajectory_output_path = self.trajectories_dir / f"{clip_name}_traj.mp4"

            # For overlay, pass first frame
            first_frame = frames[0] if self.overlay_first_frame else None

            vis_frames = self.trajectory_visualizer.visualize(
                trajectory_data=trajectory_data,
                first_frame=first_frame,
                output_path=str(trajectory_output_path)
            )

            logger.info(f"  Saved conditioning input: {trajectory_output_path.name}")

            # 5. Generate caption
            caption = self.motion_analyzer.generate_caption(motion_stats, clip_name)

            # Save caption
            caption_path = self.captions_dir / f"{clip_name}.txt"
            with open(caption_path, 'w') as f:
                f.write(caption)

            logger.info(f"  Caption: {caption}")

            # Create metadata
            metadata = {
                'clip_name': clip_name,
                'original_video': str(video_path),
                'video_path': str(video_output_path),  # Training target
                'trajectory_path': str(trajectory_output_path),  # Conditioning input
                'caption_path': str(caption_path),
                'caption': caption,
                'num_frames': len(frames),
                'resolution': self.target_resolution,
                'fps': self.target_fps,
                'motion_stats': motion_stats,
                'clip_info': clip_info
            }

            return metadata

        except Exception as e:
            logger.error(f"Error processing clip {clip_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _find_video_files(self) -> List[Path]:
        """Find all video files in input directory"""
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        videos = []
        for ext in extensions:
            videos.extend(self.input_dir.glob(f"**/*{ext}"))
        return sorted(videos)

    def _load_video_info(self, video_path: Path) -> Optional[Dict]:
        """Load basic video information"""
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

    def _split_into_clips(self, video_path: Path, video_info: Dict) -> List[Dict]:
        """Split long videos into clips"""
        num_frames = video_info['num_frames']

        if num_frames <= self.max_frames:
            return [{
                'start_frame': 0,
                'end_frame': num_frames,
                'num_frames': num_frames
            }]

        # Multiple clips with overlap
        clips = []
        overlap = int(self.max_frames * 0.1)
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

            if num_frames - start < self.min_frames:
                break

        return clips

    def _extract_clip_frames(
        self,
        video_path: Path,
        clip_info: Dict
    ) -> Optional[np.ndarray]:
        """Extract frames for clip"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        start_frame = clip_info['start_frame']
        end_frame = clip_info['end_frame']

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
        """Resize frames to target resolution"""
        target_h, target_w = self.target_resolution
        resized = []

        for frame in frames:
            resized_frame = cv2.resize(frame, (target_w, target_h))
            resized.append(resized_frame)

        return np.stack(resized)

    def _save_video(self, frames: np.ndarray, output_path: Path):
        """Save frames as video"""
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

    def _save_dataset_metadata(
        self,
        dataset_metadata: List[Dict],
        failed_videos: List[Path]
    ):
        """Save dataset metadata and training CSV"""
        # Save full metadata as JSON
        metadata_path = self.output_dir / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                'num_clips': len(dataset_metadata),
                'clips': dataset_metadata,
                'failed_videos': [str(p) for p in failed_videos],
                'config': {
                    'visualization_type': self.visualization_type,
                    'overlay_first_frame': self.overlay_first_frame,
                    'target_resolution': self.target_resolution,
                    'target_fps': self.target_fps
                }
            }, f, indent=2)

        logger.info(f"Saved metadata to {metadata_path}")

        # Create training CSV
        csv_path = self.output_dir / "train.csv"
        with open(csv_path, 'w') as f:
            f.write("video_path,trajectory_path,caption_path\n")
            for item in dataset_metadata:
                f.write(f"{item['video_path']},{item['trajectory_path']},{item['caption_path']}\n")

        logger.info(f"Saved training CSV to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset for trajectory-guided LTX Video training (SINGLE-SOURCE APPROACH)"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory with high-quality source videos"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for processed dataset"
    )
    parser.add_argument(
        "--visualization_type",
        type=str,
        default="multi",
        choices=['flow', 'depth', 'confidence', 'multi'],
        help="Trajectory visualization type"
    )
    parser.add_argument(
        "--overlay_first_frame",
        action="store_true",
        default=True,
        help="Overlay trajectory on first frame (recommended)"
    )
    parser.add_argument(
        "--overlay_alpha",
        type=float,
        default=0.3,
        help="Alpha for first frame overlay (0-1)"
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
    preparator = SingleSourceDatasetPreparator(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        visualization_type=args.visualization_type,
        overlay_first_frame=args.overlay_first_frame,
        overlay_alpha=args.overlay_alpha,
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
