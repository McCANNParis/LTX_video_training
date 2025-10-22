"""
Quick Start Example for Trajectory-Guided LTX Video

This script demonstrates the complete workflow:
1. Extract trajectories from a video
2. Visualize trajectories
3. (Optional) Train with the data
4. Generate new video with trajectory guidance
"""

import sys
sys.path.append('..')

import numpy as np
from PIL import Image

from src.preprocessing.trajectory_extractor import TrajectoryExtractor
from src.preprocessing.trajectory_visualizer import TrajectoryVisualizer, ParticleTrajectoryVisualizer
from src.inference.trajectory_guided_inference import TrajectoryGuidedVideoGenerator


def example_1_extract_and_visualize():
    """
    Example 1: Extract trajectories from a video and create visualizations
    """
    print("=" * 60)
    print("Example 1: Extract and Visualize Trajectories")
    print("=" * 60)

    # Initialize extractor
    extractor = TrajectoryExtractor()

    # Extract trajectories from video
    video_path = "path/to/your/video.mp4"
    print(f"\nExtracting trajectories from: {video_path}")

    trajectory_data = extractor.extract_from_video(
        video_path=video_path,
        output_path="output/trajectory_extraction.mp4"
    )

    print(f"Trajectory shape: {trajectory_data['trajectories'].shape}")
    print(f"Confidence shape: {trajectory_data['confidence'].shape}")

    # Create different visualizations
    visualizers = {
        'flow': TrajectoryVisualizer(visualization_type='flow'),
        'depth': TrajectoryVisualizer(visualization_type='depth'),
        'multi': TrajectoryVisualizer(visualization_type='multi'),
        'particles': ParticleTrajectoryVisualizer(num_particles=1000)
    }

    import cv2
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    cap.release()
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

    for name, visualizer in visualizers.items():
        print(f"\nCreating {name} visualization...")
        output_path = f"output/trajectory_{name}.mp4"

        vis_frames = visualizer.visualize(
            trajectory_data=trajectory_data,
            first_frame=first_frame,
            output_path=output_path
        )

        print(f"Saved to: {output_path}")

    print("\n✅ Example 1 complete!")


def example_2_prepare_training_data():
    """
    Example 2: Prepare training dataset from a directory of videos
    """
    print("=" * 60)
    print("Example 2: Prepare Training Dataset")
    print("=" * 60)

    from scripts.prepare_dataset import DatasetPreparator

    preparator = DatasetPreparator(
        input_dir="path/to/video/directory",
        output_dir="./dataset_prepared",
        visualization_type='multi',
        min_frames=60,
        max_frames=121,
        target_fps=30,
        target_resolution=(704, 1216),
        num_workers=4
    )

    print("\nProcessing videos...")
    preparator.prepare_dataset()

    print("\n✅ Example 2 complete!")
    print("Dataset ready at: ./dataset_prepared")


def example_3_simple_inference():
    """
    Example 3: Generate video from image + motion description
    """
    print("=" * 60)
    print("Example 3: Simple Inference (Motion Description)")
    print("=" * 60)

    # Initialize generator
    generator = TrajectoryGuidedVideoGenerator(
        model_path="path/to/ltx-video-model",
        lora_path="path/to/trained-lora"
    )

    # Load input image
    input_image = "path/to/input_image.jpg"

    # Generate with motion description
    motion_descriptions = [
        "pan left slowly",
        "pan right",
        "zoom in",
        "zoom out",
        "tilt up"
    ]

    for motion in motion_descriptions:
        print(f"\nGenerating with motion: {motion}")

        output_video = generator.generate_from_image_and_motion_description(
            input_image=input_image,
            motion_description=motion,
            prompt=f"A beautiful landscape, {motion}, cinematic",
            num_frames=121,
            fps=30,
            output_path=f"output/generated_{motion.replace(' ', '_')}.mp4"
        )

        print(f"Generated: output/generated_{motion.replace(' ', '_')}.mp4")

    print("\n✅ Example 3 complete!")


def example_4_motion_transfer():
    """
    Example 4: Transfer motion from one video to a new subject
    """
    print("=" * 60)
    print("Example 4: Motion Transfer")
    print("=" * 60)

    # Initialize generator
    generator = TrajectoryGuidedVideoGenerator(
        model_path="path/to/ltx-video-model",
        lora_path="path/to/trained-lora"
    )

    # Source video (motion to extract)
    source_video = "path/to/source_with_interesting_motion.mp4"

    # Target images (subjects to apply motion to)
    target_images = [
        "path/to/subject1.jpg",
        "path/to/subject2.jpg",
        "path/to/subject3.jpg"
    ]

    for i, target_image in enumerate(target_images):
        print(f"\nTransferring motion to subject {i+1}...")

        output_video = generator.transfer_motion(
            source_video=source_video,
            target_image=target_image,
            prompt=f"Subject {i+1} with transferred motion, cinematic",
            output_path=f"output/motion_transfer_{i+1}.mp4"
        )

        print(f"Generated: output/motion_transfer_{i+1}.mp4")

    print("\n✅ Example 4 complete!")


def example_5_reference_video_generation():
    """
    Example 5: Generate video using reference video trajectory
    """
    print("=" * 60)
    print("Example 5: Reference Video Generation")
    print("=" * 60)

    # Initialize generator
    generator = TrajectoryGuidedVideoGenerator(
        model_path="path/to/ltx-video-model",
        lora_path="path/to/trained-lora"
    )

    # Input image and reference video
    input_image = "path/to/input_image.jpg"
    reference_video = "path/to/reference_motion.mp4"

    print("\nGenerating video with reference trajectory...")

    output_video = generator.generate_from_image_and_reference(
        input_image=input_image,
        reference_video=reference_video,
        prompt="A person walking in a park, smooth camera motion, cinematic",
        num_frames=121,
        fps=30,
        guidance_scale=7.5,
        num_inference_steps=50,
        output_path="output/reference_guided_generation.mp4"
    )

    print("Generated: output/reference_guided_generation.mp4")
    print("\n✅ Example 5 complete!")


def example_6_batch_processing():
    """
    Example 6: Batch process multiple images with the same trajectory
    """
    print("=" * 60)
    print("Example 6: Batch Processing")
    print("=" * 60)

    # Initialize generator
    generator = TrajectoryGuidedVideoGenerator(
        model_path="path/to/ltx-video-model",
        lora_path="path/to/trained-lora"
    )

    # Single reference trajectory
    reference_video = "path/to/reference_motion.mp4"

    # Multiple input images
    input_images = [
        "path/to/image1.jpg",
        "path/to/image2.jpg",
        "path/to/image3.jpg",
    ]

    prompts = [
        "A serene landscape with smooth camera pan",
        "An urban street scene with dynamic motion",
        "A portrait with cinematic camera movement"
    ]

    for i, (image, prompt) in enumerate(zip(input_images, prompts)):
        print(f"\nProcessing image {i+1}/{len(input_images)}")

        generator.generate_from_image_and_reference(
            input_image=image,
            reference_video=reference_video,
            prompt=prompt,
            num_frames=121,
            fps=30,
            output_path=f"output/batch_{i+1}.mp4"
        )

        print(f"Saved: output/batch_{i+1}.mp4")

    print("\n✅ Example 6 complete!")


def main():
    """
    Run all examples (comment out as needed)
    """
    import os
    os.makedirs("output", exist_ok=True)

    print("\n" + "=" * 60)
    print("Trajectory-Guided LTX Video - Quick Start Examples")
    print("=" * 60)

    # Uncomment the examples you want to run:

    # example_1_extract_and_visualize()
    # example_2_prepare_training_data()
    # example_3_simple_inference()
    # example_4_motion_transfer()
    # example_5_reference_video_generation()
    # example_6_batch_processing()

    print("\n" + "=" * 60)
    print("All examples complete!")
    print("=" * 60)

    print("\nNote: Make sure to:")
    print("  1. Update file paths in the examples")
    print("  2. Have trained model weights ready")
    print("  3. Have input videos/images prepared")


if __name__ == "__main__":
    main()
