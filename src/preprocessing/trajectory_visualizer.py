"""
Trajectory Visualization Tools
Creates rich visual encodings of trajectory fields for IC-LoRA conditioning
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Literal
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrajectoryVisualizer:
    """
    Creates multi-modal visualizations of trajectory fields
    """

    def __init__(
        self,
        visualization_type: Literal['flow', 'depth', 'confidence', 'multi'] = 'multi',
        overlay_first_frame: bool = True,
        color_scheme: str = 'hsv'
    ):
        """
        Initialize visualizer

        Args:
            visualization_type: Type of visualization to create
                - 'flow': Optical flow style (direction + magnitude)
                - 'depth': Depth map from trajectory Z-component
                - 'confidence': Confidence/variance map
                - 'multi': Multi-channel encoding
            overlay_first_frame: Whether to overlay on first frame
            color_scheme: Color scheme for visualization ('hsv', 'jet', 'magma')
        """
        self.visualization_type = visualization_type
        self.overlay_first_frame = overlay_first_frame
        self.color_scheme = color_scheme

    def visualize(
        self,
        trajectory_data: Dict[str, np.ndarray],
        first_frame: Optional[np.ndarray] = None,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Create visualization of trajectory field

        Args:
            trajectory_data: Dictionary with 'trajectories', 'confidence', 'occlusions'
            first_frame: Optional first frame for overlay
            output_path: Optional path to save video

        Returns:
            Visualization frames (T, H, W, 3)
        """
        trajectories = trajectory_data['trajectories']
        confidence = trajectory_data.get('confidence')
        occlusions = trajectory_data.get('occlusions')

        H, W, T, _ = trajectories.shape

        logger.info(f"Creating {self.visualization_type} visualization")

        if self.visualization_type == 'flow':
            vis_frames = self._create_flow_visualization(trajectories)
        elif self.visualization_type == 'depth':
            vis_frames = self._create_depth_visualization(trajectories)
        elif self.visualization_type == 'confidence':
            vis_frames = self._create_confidence_visualization(trajectories, confidence)
        elif self.visualization_type == 'multi':
            vis_frames = self._create_multi_channel_visualization(
                trajectories, confidence, occlusions
            )
        else:
            raise ValueError(f"Unknown visualization type: {self.visualization_type}")

        # Overlay on first frame if requested
        if self.overlay_first_frame and first_frame is not None:
            vis_frames = self._overlay_on_frame(vis_frames, first_frame)

        # Save if requested
        if output_path:
            self._save_video(vis_frames, output_path)

        return vis_frames

    def _create_flow_visualization(
        self,
        trajectories: np.ndarray
    ) -> np.ndarray:
        """
        Create optical flow style visualization

        Uses HSV color space:
        - Hue: Flow direction
        - Saturation: Full
        - Value: Flow magnitude
        """
        H, W, T, _ = trajectories.shape
        vis_frames = []

        for t in range(T):
            flow_x = trajectories[:, :, t, 0]
            flow_y = trajectories[:, :, t, 1]

            # Convert to polar coordinates
            magnitude, angle = cv2.cartToPolar(flow_x, flow_y)

            # Create HSV image
            hsv = np.zeros((H, W, 3), dtype=np.uint8)
            hsv[..., 0] = (angle * 180 / np.pi / 2).astype(np.uint8)  # Hue
            hsv[..., 1] = 255  # Saturation
            hsv[..., 2] = cv2.normalize(
                magnitude, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)  # Value

            # Convert to RGB
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            vis_frames.append(rgb)

        return np.stack(vis_frames)

    def _create_depth_visualization(
        self,
        trajectories: np.ndarray
    ) -> np.ndarray:
        """
        Create depth map visualization from Z-component
        """
        H, W, T, _ = trajectories.shape
        vis_frames = []

        # Get Z component
        depth = trajectories[:, :, :, 2]

        # Normalize depth across all frames
        depth_min, depth_max = depth.min(), depth.max()
        if depth_max > depth_min:
            depth_normalized = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = np.zeros_like(depth)

        # Get colormap
        if self.color_scheme == 'jet':
            colormap = cv2.COLORMAP_JET
        elif self.color_scheme == 'magma':
            colormap = cv2.COLORMAP_MAGMA
        else:
            colormap = cv2.COLORMAP_VIRIDIS

        for t in range(T):
            depth_frame = (depth_normalized[:, :, t] * 255).astype(np.uint8)
            colored = cv2.applyColorMap(depth_frame, colormap)
            rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
            vis_frames.append(rgb)

        return np.stack(vis_frames)

    def _create_confidence_visualization(
        self,
        trajectories: np.ndarray,
        confidence: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Create confidence map visualization

        Shows motion variance/confidence
        """
        H, W, T, _ = trajectories.shape

        if confidence is None:
            # Compute temporal variance as proxy for confidence
            variance = np.var(trajectories, axis=2)
            confidence = 1.0 - cv2.normalize(
                np.mean(variance, axis=2), None, 0, 1, cv2.NORM_MINMAX
            )
            confidence = np.repeat(confidence[:, :, np.newaxis], T, axis=2)

        vis_frames = []

        for t in range(T):
            conf_frame = (confidence[:, :, t] * 255).astype(np.uint8)
            colored = cv2.applyColorMap(conf_frame, cv2.COLORMAP_HOT)
            rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
            vis_frames.append(rgb)

        return np.stack(vis_frames)

    def _create_multi_channel_visualization(
        self,
        trajectories: np.ndarray,
        confidence: Optional[np.ndarray],
        occlusions: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Create multi-channel visualization combining flow, depth, and confidence

        Encodes:
        - R: Horizontal flow
        - G: Vertical flow
        - B: Depth/magnitude
        """
        H, W, T, _ = trajectories.shape
        vis_frames = []

        # Normalize each component
        flow_x = trajectories[:, :, :, 0]
        flow_y = trajectories[:, :, :, 1]
        depth = trajectories[:, :, :, 2]

        # Normalize to [0, 255]
        flow_x_norm = self._normalize_to_uint8(flow_x)
        flow_y_norm = self._normalize_to_uint8(flow_y)
        depth_norm = self._normalize_to_uint8(depth)

        for t in range(T):
            # Stack as RGB
            rgb = np.stack([
                flow_x_norm[:, :, t],
                flow_y_norm[:, :, t],
                depth_norm[:, :, t]
            ], axis=-1)

            # Highlight occlusions if available
            if occlusions is not None:
                mask = occlusions[:, :, t]
                rgb[mask] = [255, 0, 0]  # Red for occlusions

            vis_frames.append(rgb)

        return np.stack(vis_frames)

    def _normalize_to_uint8(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data to uint8 range [0, 255]
        """
        data_min, data_max = data.min(), data.max()
        if data_max > data_min:
            normalized = (data - data_min) / (data_max - data_min) * 255
        else:
            normalized = np.zeros_like(data)
        return normalized.astype(np.uint8)

    def _overlay_on_frame(
        self,
        vis_frames: np.ndarray,
        first_frame: np.ndarray
    ) -> np.ndarray:
        """
        Overlay visualization on first frame
        """
        T, H, W, C = vis_frames.shape

        # Resize first frame if needed
        if first_frame.shape[:2] != (H, W):
            first_frame = cv2.resize(first_frame, (W, H))

        overlayed = []
        for t in range(T):
            # Blend visualization with first frame
            blended = cv2.addWeighted(
                first_frame, 0.5,
                vis_frames[t], 0.5,
                0
            )
            overlayed.append(blended)

        return np.stack(overlayed)

    def _save_video(self, frames: np.ndarray, output_path: str):
        """
        Save frames as video
        """
        T, H, W, C = frames.shape

        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (W, H))

        for t in range(T):
            # Convert RGB to BGR for OpenCV
            bgr = cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR)
            out.write(bgr)

        out.release()
        logger.info(f"Saved visualization to {output_path}")


class ParticleTrajectoryVisualizer:
    """
    Visualize trajectories as particle traces
    """

    def __init__(
        self,
        num_particles: int = 1000,
        trace_length: int = 10,
        particle_size: int = 2
    ):
        """
        Initialize particle visualizer

        Args:
            num_particles: Number of particles to track
            trace_length: Length of particle traces
            particle_size: Size of particle dots
        """
        self.num_particles = num_particles
        self.trace_length = trace_length
        self.particle_size = particle_size

    def visualize(
        self,
        trajectory_data: Dict[str, np.ndarray],
        first_frame: np.ndarray,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Create particle trajectory visualization

        Args:
            trajectory_data: Trajectory field data
            first_frame: First frame for background
            output_path: Optional output path

        Returns:
            Visualization frames (T, H, W, 3)
        """
        trajectories = trajectory_data['trajectories']
        H, W, T, _ = trajectories.shape

        # Sample particle positions
        particle_positions = self._sample_particles(H, W)

        # Track particles through time
        particle_traces = self._track_particles(
            particle_positions, trajectories
        )

        # Render visualization
        vis_frames = self._render_particles(
            particle_traces, first_frame, H, W, T
        )

        if output_path:
            self._save_video(vis_frames, output_path)

        return vis_frames

    def _sample_particles(self, H: int, W: int) -> np.ndarray:
        """
        Sample random particle positions

        Returns:
            Array of shape (N, 2) with (x, y) coordinates
        """
        x = np.random.randint(0, W, self.num_particles)
        y = np.random.randint(0, H, self.num_particles)
        return np.stack([x, y], axis=1)

    def _track_particles(
        self,
        initial_positions: np.ndarray,
        trajectories: np.ndarray
    ) -> np.ndarray:
        """
        Track particles through trajectory field

        Returns:
            Array of shape (N, T, 2) with particle positions over time
        """
        H, W, T, _ = trajectories.shape
        N = len(initial_positions)

        traces = np.zeros((N, T, 2), dtype=np.float32)
        traces[:, 0] = initial_positions

        for t in range(1, T):
            for i in range(N):
                # Get current position
                x, y = traces[i, t-1].astype(int)

                # Clamp to valid range
                x = np.clip(x, 0, W-1)
                y = np.clip(y, 0, H-1)

                # Get flow at this position
                flow = trajectories[y, x, t-1, :2]

                # Update position
                traces[i, t] = traces[i, t-1] + flow

        return traces

    def _render_particles(
        self,
        particle_traces: np.ndarray,
        first_frame: np.ndarray,
        H: int, W: int, T: int
    ) -> np.ndarray:
        """
        Render particle traces on frames
        """
        N, _, _ = particle_traces.shape

        # Resize first frame if needed
        if first_frame.shape[:2] != (H, W):
            first_frame = cv2.resize(first_frame, (W, H))

        vis_frames = []

        for t in range(T):
            # Start with first frame
            frame = first_frame.copy()

            # Draw particle traces
            for i in range(N):
                # Get trace for this particle
                start_t = max(0, t - self.trace_length)
                trace = particle_traces[i, start_t:t+1]

                # Draw trace as line
                for j in range(len(trace) - 1):
                    pt1 = tuple(trace[j].astype(int))
                    pt2 = tuple(trace[j+1].astype(int))

                    # Color fades over time
                    alpha = (j + 1) / len(trace)
                    color = (
                        int(255 * alpha),
                        int(100 * alpha),
                        int(255 * (1 - alpha))
                    )

                    cv2.line(frame, pt1, pt2, color, 1, cv2.LINE_AA)

                # Draw particle dot
                pos = tuple(trace[-1].astype(int))
                cv2.circle(frame, pos, self.particle_size, (255, 255, 0), -1)

            vis_frames.append(frame)

        return np.stack(vis_frames)

    def _save_video(self, frames: np.ndarray, output_path: str):
        """
        Save frames as video
        """
        T, H, W, C = frames.shape

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (W, H))

        for t in range(T):
            bgr = cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR)
            out.write(bgr)

        out.release()
        logger.info(f"Saved particle visualization to {output_path}")


def main():
    """
    Test trajectory visualization
    """
    import argparse

    parser = argparse.ArgumentParser(description="Visualize trajectories")
    parser.add_argument("--input", type=str, required=True, help="Input video")
    parser.add_argument("--output", type=str, required=True, help="Output visualization")
    parser.add_argument("--type", type=str, default="multi",
                       choices=['flow', 'depth', 'confidence', 'multi', 'particles'])

    args = parser.parse_args()

    # Extract trajectories first
    from trajectory_extractor import TrajectoryExtractor

    extractor = TrajectoryExtractor()
    trajectory_data = extractor.extract_from_video(args.input)

    # Load first frame
    cap = cv2.VideoCapture(args.input)
    ret, first_frame = cap.read()
    cap.release()
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

    # Visualize
    if args.type == 'particles':
        visualizer = ParticleTrajectoryVisualizer()
    else:
        visualizer = TrajectoryVisualizer(visualization_type=args.type)

    vis_frames = visualizer.visualize(
        trajectory_data,
        first_frame=first_frame,
        output_path=args.output
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()
