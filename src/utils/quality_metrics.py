"""
Quality Metrics for Trajectory-Guided Video Generation

Evaluates:
- Trajectory fidelity
- Motion smoothness
- Temporal consistency
- Physical plausibility
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoQualityMetrics:
    """
    Compute quality metrics for generated videos
    """

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize quality metrics computer

        Args:
            device: Device for computation
        """
        self.device = device

    def compute_all_metrics(
        self,
        generated_video: np.ndarray,
        target_video: Optional[np.ndarray] = None,
        trajectory_field: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute all quality metrics

        Args:
            generated_video: Generated video (T, H, W, 3)
            target_video: Optional ground truth video
            trajectory_field: Optional trajectory field for fidelity check

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Temporal consistency
        metrics['temporal_consistency'] = self.compute_temporal_consistency(
            generated_video
        )

        # Motion smoothness
        metrics['motion_smoothness'] = self.compute_motion_smoothness(
            generated_video
        )

        # Physical plausibility
        metrics['physical_plausibility'] = self.compute_physical_plausibility(
            generated_video
        )

        # If target video provided, compute reconstruction metrics
        if target_video is not None:
            metrics.update(self.compute_reconstruction_metrics(
                generated_video, target_video
            ))

        # If trajectory provided, compute fidelity
        if trajectory_field is not None:
            metrics['trajectory_fidelity'] = self.compute_trajectory_fidelity(
                generated_video, trajectory_field
            )

        return metrics

    def compute_temporal_consistency(self, video: np.ndarray) -> float:
        """
        Compute temporal consistency using frame differences

        Lower values = more consistent
        """
        T = video.shape[0]
        differences = []

        for t in range(T - 1):
            diff = np.abs(video[t + 1] - video[t])
            differences.append(diff.mean())

        # Average frame difference
        avg_diff = np.mean(differences)

        # Normalize to [0, 1] range
        consistency_score = 1.0 - min(avg_diff / 255.0, 1.0)

        return float(consistency_score)

    def compute_motion_smoothness(self, video: np.ndarray) -> float:
        """
        Compute motion smoothness using optical flow

        Measures jerk (derivative of acceleration)
        """
        T, H, W, C = video.shape

        # Convert to grayscale
        gray_frames = [
            cv2.cvtColor(video[t], cv2.COLOR_RGB2GRAY)
            for t in range(T)
        ]

        # Compute optical flow
        flows = []
        for t in range(T - 1):
            flow = cv2.calcOpticalFlowFarneback(
                gray_frames[t],
                gray_frames[t + 1],
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
            flows.append(flow)

        flows = np.stack(flows)  # (T-1, H, W, 2)

        # Compute velocity magnitude
        velocity = np.sqrt(flows[..., 0]**2 + flows[..., 1]**2)

        # Compute acceleration (change in velocity)
        acceleration = np.diff(velocity, axis=0)

        # Compute jerk (change in acceleration)
        jerk = np.diff(acceleration, axis=0)

        # Smoothness = low jerk
        jerk_magnitude = np.abs(jerk).mean()

        # Normalize (lower jerk = higher smoothness)
        smoothness_score = 1.0 / (1.0 + jerk_magnitude)

        return float(smoothness_score)

    def compute_physical_plausibility(self, video: np.ndarray) -> float:
        """
        Compute physical plausibility score

        Checks for:
        - Sudden appearance/disappearance (conservation of matter)
        - Unrealistic accelerations
        - Color consistency
        """
        scores = []

        # Check for sudden appearance/disappearance
        brightness_score = self._check_brightness_consistency(video)
        scores.append(brightness_score)

        # Check for reasonable motion magnitudes
        motion_score = self._check_motion_magnitudes(video)
        scores.append(motion_score)

        # Check color consistency
        color_score = self._check_color_consistency(video)
        scores.append(color_score)

        return float(np.mean(scores))

    def _check_brightness_consistency(self, video: np.ndarray) -> float:
        """
        Check if brightness changes are gradual
        """
        brightness = video.mean(axis=(1, 2, 3))  # (T,)
        brightness_changes = np.abs(np.diff(brightness))

        # Penalize large sudden changes
        max_change = brightness_changes.max()
        score = 1.0 - min(max_change / 128.0, 1.0)

        return score

    def _check_motion_magnitudes(self, video: np.ndarray) -> float:
        """
        Check if motion magnitudes are reasonable
        """
        T, H, W, C = video.shape

        # Compute frame differences
        diffs = np.abs(np.diff(video, axis=0))
        motion_magnitude = diffs.mean(axis=(1, 2, 3))

        # Check for unrealistic motion
        # Typical motion should be < 50 in pixel value change
        reasonable_motion = (motion_magnitude < 50).astype(float).mean()

        return float(reasonable_motion)

    def _check_color_consistency(self, video: np.ndarray) -> float:
        """
        Check if colors remain consistent across frames
        """
        # Compute color histograms for each frame
        histograms = []
        for t in range(video.shape[0]):
            hist = []
            for c in range(3):  # RGB
                h = np.histogram(video[t, :, :, c], bins=32, range=(0, 256))[0]
                h = h / h.sum()  # Normalize
                hist.append(h)
            histograms.append(np.concatenate(hist))

        histograms = np.stack(histograms)  # (T, 96)

        # Compute histogram similarity between consecutive frames
        similarities = []
        for t in range(len(histograms) - 1):
            # Cosine similarity
            sim = np.dot(histograms[t], histograms[t + 1]) / (
                np.linalg.norm(histograms[t]) * np.linalg.norm(histograms[t + 1])
            )
            similarities.append(sim)

        return float(np.mean(similarities))

    def compute_reconstruction_metrics(
        self,
        generated: np.ndarray,
        target: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute reconstruction metrics compared to target

        Returns:
            - mse: Mean squared error
            - psnr: Peak signal-to-noise ratio
            - ssim: Structural similarity (frame-wise average)
        """
        # Ensure same shape
        if generated.shape != target.shape:
            logger.warning(f"Shape mismatch: {generated.shape} vs {target.shape}")
            return {'mse': float('inf'), 'psnr': 0.0, 'ssim': 0.0}

        # MSE
        mse = np.mean((generated - target) ** 2)

        # PSNR
        if mse > 0:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        else:
            psnr = float('inf')

        # SSIM (frame-wise)
        ssim_scores = []
        for t in range(generated.shape[0]):
            ssim = self._compute_ssim(generated[t], target[t])
            ssim_scores.append(ssim)

        avg_ssim = np.mean(ssim_scores)

        return {
            'mse': float(mse),
            'psnr': float(psnr),
            'ssim': float(avg_ssim)
        }

    def _compute_ssim(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        K1: float = 0.01,
        K2: float = 0.03,
        win_size: int = 11
    ) -> float:
        """
        Compute SSIM between two images

        Simplified version - for production use scikit-image
        """
        from scipy.ndimage import uniform_filter

        C1 = (K1 * 255) ** 2
        C2 = (K2 * 255) ** 2

        # Convert to float
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        # Compute means
        mu1 = uniform_filter(img1, win_size, mode='reflect')
        mu2 = uniform_filter(img2, win_size, mode='reflect')

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        # Compute variances and covariance
        sigma1_sq = uniform_filter(img1 ** 2, win_size, mode='reflect') - mu1_sq
        sigma2_sq = uniform_filter(img2 ** 2, win_size, mode='reflect') - mu2_sq
        sigma12 = uniform_filter(img1 * img2, win_size, mode='reflect') - mu1_mu2

        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return float(ssim_map.mean())

    def compute_trajectory_fidelity(
        self,
        generated_video: np.ndarray,
        target_trajectory: np.ndarray
    ) -> float:
        """
        Compute how well the generated video follows the target trajectory

        Extracts motion from generated video and compares to target
        """
        T, H, W, C = generated_video.shape

        # Extract motion from generated video
        generated_flow = self._extract_optical_flow(generated_video)

        # Compare with target trajectory
        target_flow = target_trajectory[:, :, :, :2]  # X, Y components

        # Resize if needed
        if generated_flow.shape != target_flow.shape:
            target_flow = self._resize_flow(target_flow, generated_flow.shape)

        # Compute angular error (common metric for optical flow)
        angular_errors = []
        for t in range(min(generated_flow.shape[2], target_flow.shape[2])):
            gen_flow = generated_flow[:, :, t]
            tgt_flow = target_flow[:, :, t]

            # Normalize
            gen_flow_norm = gen_flow / (np.linalg.norm(gen_flow, axis=2, keepdims=True) + 1e-8)
            tgt_flow_norm = tgt_flow / (np.linalg.norm(tgt_flow, axis=2, keepdims=True) + 1e-8)

            # Cosine similarity
            cos_sim = (gen_flow_norm * tgt_flow_norm).sum(axis=2)
            angular_error = np.arccos(np.clip(cos_sim, -1, 1))

            angular_errors.append(angular_error.mean())

        # Convert angular error to fidelity score
        avg_angular_error = np.mean(angular_errors)
        fidelity = 1.0 - (avg_angular_error / np.pi)  # Normalize to [0, 1]

        return float(fidelity)

    def _extract_optical_flow(self, video: np.ndarray) -> np.ndarray:
        """
        Extract optical flow from video

        Returns flow of shape (H, W, T-1, 2)
        """
        T, H, W, C = video.shape

        gray_frames = [
            cv2.cvtColor(video[t], cv2.COLOR_RGB2GRAY)
            for t in range(T)
        ]

        flows = []
        for t in range(T - 1):
            flow = cv2.calcOpticalFlowFarneback(
                gray_frames[t],
                gray_frames[t + 1],
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
            flows.append(flow)

        return np.stack(flows, axis=2)  # (H, W, T-1, 2)

    def _resize_flow(self, flow: np.ndarray, target_shape: Tuple) -> np.ndarray:
        """
        Resize flow field to target shape
        """
        H_tgt, W_tgt, T_tgt, _ = target_shape
        H_src, W_src, T_src, _ = flow.shape

        resized = np.zeros(target_shape, dtype=flow.dtype)

        for t in range(min(T_src, T_tgt)):
            resized[:, :, t, 0] = cv2.resize(flow[:, :, t, 0], (W_tgt, H_tgt))
            resized[:, :, t, 1] = cv2.resize(flow[:, :, t, 1], (W_tgt, H_tgt))

            # Scale flow values
            resized[:, :, t, 0] *= W_tgt / W_src
            resized[:, :, t, 1] *= H_tgt / H_src

        return resized


class TrainingMetricsLogger:
    """
    Logger for training metrics with W&B / TensorBoard integration
    """

    def __init__(
        self,
        log_dir: str,
        use_wandb: bool = True,
        use_tensorboard: bool = True
    ):
        """
        Initialize metrics logger

        Args:
            log_dir: Directory for logs
            use_wandb: Enable Weights & Biases logging
            use_tensorboard: Enable TensorBoard logging
        """
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir)
            except ImportError:
                logger.warning("TensorBoard not available")
                self.tb_writer = None
        else:
            self.tb_writer = None

        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                logger.warning("W&B not available")
                self.wandb = None
        else:
            self.wandb = None

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """
        Log metrics to all enabled backends
        """
        # TensorBoard
        if self.tb_writer:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, step)

        # W&B
        if self.wandb and self.wandb.run is not None:
            self.wandb.log(metrics, step=step)

    def log_video(
        self,
        video: np.ndarray,
        tag: str,
        step: int,
        fps: int = 30
    ):
        """
        Log video to backends
        """
        # TensorBoard
        if self.tb_writer:
            # Convert to (1, T, C, H, W) format for TensorBoard
            video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2).unsqueeze(0)
            self.tb_writer.add_video(tag, video_tensor, step, fps=fps)

        # W&B
        if self.wandb and self.wandb.run is not None:
            self.wandb.log({tag: self.wandb.Video(video, fps=fps)}, step=step)

    def close(self):
        """
        Close loggers
        """
        if self.tb_writer:
            self.tb_writer.close()


def main():
    """
    Test quality metrics
    """
    # Create synthetic test videos
    T, H, W, C = 30, 256, 256, 3

    # Video 1: Smooth motion
    video1 = np.zeros((T, H, W, C), dtype=np.uint8)
    for t in range(T):
        shift = int(t * 5)
        video1[t, :, shift:shift+50] = 255

    # Video 2: Noisy motion
    video2 = np.random.randint(0, 256, (T, H, W, C), dtype=np.uint8)

    # Compute metrics
    metrics_computer = VideoQualityMetrics()

    print("Video 1 (smooth motion):")
    metrics1 = metrics_computer.compute_all_metrics(video1)
    for key, value in metrics1.items():
        print(f"  {key}: {value:.4f}")

    print("\nVideo 2 (random noise):")
    metrics2 = metrics_computer.compute_all_metrics(video2)
    for key, value in metrics2.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
