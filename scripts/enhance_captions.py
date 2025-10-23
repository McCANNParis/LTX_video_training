"""
Enhanced Caption Generation for Trajectory-Guided Training

Implements best practices for video captioning:
- Motion-first structure
- BLIP-2 scene understanding
- Template-based enhancement
- Quality tiers
"""

import sys
sys.path.append('/home/user/LTX_video_training')

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, List
import logging
from tqdm import tqdm
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CaptionEnhancer:
    """
    Enhances auto-generated captions to higher quality tiers
    """

    def __init__(
        self,
        use_blip2: bool = False,
        use_gpt4: bool = False,
        gpt4_api_key: Optional[str] = None
    ):
        """
        Initialize caption enhancer

        Args:
            use_blip2: Use BLIP-2 for scene understanding
            use_gpt4: Use GPT-4 Vision for high-quality captions
            gpt4_api_key: OpenAI API key if using GPT-4
        """
        self.use_blip2 = use_blip2
        self.use_gpt4 = use_gpt4
        self.gpt4_api_key = gpt4_api_key

        # Load models if needed
        if use_blip2:
            try:
                from transformers import Blip2Processor, Blip2ForConditionalGeneration
                logger.info("Loading BLIP-2 model...")
                self.blip2_processor = Blip2Processor.from_pretrained(
                    "Salesforce/blip2-opt-2.7b"
                )
                self.blip2_model = Blip2ForConditionalGeneration.from_pretrained(
                    "Salesforce/blip2-opt-2.7b"
                )
                logger.info("BLIP-2 loaded successfully")
            except ImportError:
                logger.warning("transformers not installed. Install with: pip install transformers")
                self.use_blip2 = False

        if use_gpt4:
            if not gpt4_api_key:
                logger.warning("GPT-4 API key not provided. Disabling GPT-4.")
                self.use_gpt4 = False
            else:
                try:
                    import openai
                    openai.api_key = gpt4_api_key
                    self.openai = openai
                    logger.info("GPT-4 Vision enabled")
                except ImportError:
                    logger.warning("openai not installed. Install with: pip install openai")
                    self.use_gpt4 = False

        # Caption templates by category
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, List[str]]:
        """
        Load caption templates for different video categories
        """
        return {
            'camera_motion': [
                "{motion_type} {scene}, {speed}, {lighting}, {style}",
                "{style} {motion_type} {scene}, {lighting}, {speed}",
                "Cinematic {motion_type} {scene}, {technical}, {mood}"
            ],
            'object_motion': [
                "{subject} {action}, {camera_work}, {lighting}, {style}",
                "{style} shot of {subject} {action}, {lighting}, {mood}",
                "{subject} {action} {direction}, {camera_work}, {atmosphere}"
            ],
            'landscape': [
                "{motion_type} {natural_feature}, {time_of_day}, {weather}, {style}",
                "{style} {motion_type} {natural_feature}, {lighting}, {mood}",
                "{natural_feature} with {motion_type}, {atmosphere}, {technical}"
            ],
            'urban': [
                "{motion_type} {urban_element}, {activity}, {time}, {style}",
                "{style} {motion_type} {urban_scene}, {lighting}, {energy}",
                "{urban_element} {motion_type}, {atmosphere}, {technical}"
            ],
            'people': [
                "{subject_desc} {action}, {camera_work}, {lighting}, {mood}",
                "{style} shot of {subject_desc} {action}, {lighting}, {context}",
                "{subject_desc} {action} {direction}, {technical}, {atmosphere}"
            ]
        }

    def enhance_caption(
        self,
        auto_caption: str,
        video_path: str,
        motion_stats: Dict,
        target_tier: int = 2
    ) -> str:
        """
        Enhance auto-generated caption to target quality tier

        Args:
            auto_caption: Auto-generated caption (Tier 1)
            video_path: Path to video file
            motion_stats: Motion analysis statistics
            target_tier: 1 (auto), 2 (enhanced), 3 (professional)

        Returns:
            Enhanced caption
        """
        if target_tier == 1:
            return auto_caption

        # Extract components from auto caption and motion stats
        components = self._extract_components(auto_caption, motion_stats)

        if target_tier == 2:
            # Template-based enhancement
            return self._template_enhance(components, motion_stats)

        elif target_tier == 3:
            # Professional enhancement with AI models
            return self._professional_enhance(components, video_path, motion_stats)

        return auto_caption

    def _extract_components(
        self,
        auto_caption: str,
        motion_stats: Dict
    ) -> Dict:
        """
        Extract components from auto caption and motion stats
        """
        # Parse auto caption
        parts = [p.strip() for p in auto_caption.split(',')]

        components = {
            'motion_type': motion_stats.get('camera_type', 'camera motion'),
            'speed': motion_stats.get('motion_description', 'medium speed'),
            'camera_motion_score': motion_stats.get('camera_motion_score', 0.5),
            'motion_magnitude': motion_stats.get('motion_magnitude', 1.0)
        }

        # Try to extract subject from caption
        if len(parts) > 1:
            components['subject'] = parts[0]
        else:
            components['subject'] = "scene"

        return components

    def _template_enhance(
        self,
        components: Dict,
        motion_stats: Dict
    ) -> str:
        """
        Enhance using templates (Tier 2)
        """
        # Determine video category
        camera_score = components.get('camera_motion_score', 0.5)

        if camera_score > 0.7:
            category = 'camera_motion'
        else:
            category = 'object_motion'

        # Select template
        template = self.templates[category][0]

        # Fill in components
        enhanced = template.format(
            motion_type=components.get('motion_type', 'camera motion'),
            scene=components.get('subject', 'scene'),
            speed=components.get('speed', 'medium speed'),
            lighting="natural lighting",  # Default
            style="cinematic composition",  # Default
            subject="subject",
            action="moving",
            camera_work="steady shot",
            direction="forward"
        )

        # Clean up
        enhanced = enhanced.replace('  ', ' ').strip()

        return enhanced

    def _professional_enhance(
        self,
        components: Dict,
        video_path: str,
        motion_stats: Dict
    ) -> str:
        """
        Professional enhancement using AI models (Tier 3)
        """
        enhanced = components.get('subject', 'scene')

        # Add BLIP-2 scene understanding
        if self.use_blip2:
            try:
                first_frame = self._load_first_frame(video_path)
                scene_desc = self._blip2_caption(first_frame)
                enhanced = scene_desc
            except Exception as e:
                logger.warning(f"BLIP-2 failed: {e}")

        # Add motion description
        motion_desc = f"{components['motion_type']}, {components['speed']}"

        # Combine
        professional = f"{motion_desc}, {enhanced}, cinematic composition"

        # Optional: Refine with GPT-4
        if self.use_gpt4:
            try:
                professional = self._gpt4_refine(professional, video_path)
            except Exception as e:
                logger.warning(f"GPT-4 failed: {e}")

        return professional

    def _load_first_frame(self, video_path: str) -> np.ndarray:
        """
        Load first frame from video
        """
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Failed to load frame from {video_path}")

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _blip2_caption(self, frame: np.ndarray) -> str:
        """
        Generate caption using BLIP-2
        """
        from PIL import Image

        # Convert numpy to PIL
        image = Image.fromarray(frame)

        # Process
        inputs = self.blip2_processor(images=image, return_tensors="pt")

        # Generate
        generated_ids = self.blip2_model.generate(**inputs)
        caption = self.blip2_processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()

        return caption

    def _gpt4_refine(self, draft_caption: str, video_path: str) -> str:
        """
        Refine caption using GPT-4 Vision
        """
        # Load multiple frames
        frames = self._load_sample_frames(video_path, num_frames=3)

        # Encode frames as base64
        import base64
        from io import BytesIO
        from PIL import Image

        frame_urls = []
        for frame in frames:
            img = Image.fromarray(frame)
            buffer = BytesIO()
            img.save(buffer, format='JPEG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            frame_urls.append(f"data:image/jpeg;base64,{img_str}")

        # Create prompt
        prompt = f"""
        Refine this video caption for training a video generation model.
        Current caption: "{draft_caption}"

        Requirements:
        1. Start with motion type (camera or object motion)
        2. Include scene description
        3. Add lighting and style
        4. Keep it concise (1-2 sentences)
        5. Use professional cinematography terms

        Output only the refined caption, no explanation.
        """

        # Call GPT-4 Vision
        response = self.openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *[{"type": "image_url", "image_url": {"url": url}} for url in frame_urls]
                ]
            }],
            max_tokens=100
        )

        refined = response.choices[0].message.content.strip()

        # Remove quotes if present
        refined = refined.strip('"\'')

        return refined

    def _load_sample_frames(
        self,
        video_path: str,
        num_frames: int = 3
    ) -> List[np.ndarray]:
        """
        Load sample frames from video
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample evenly
        frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()

        return frames

    def validate_caption(self, caption: str, motion_stats: Dict) -> Dict:
        """
        Validate caption quality

        Returns validation metrics
        """
        metrics = {}

        # Check length
        word_count = len(caption.split())
        metrics['length_ok'] = 10 <= word_count <= 50
        metrics['word_count'] = word_count

        # Check motion alignment
        caption_lower = caption.lower()
        motion_type = motion_stats.get('camera_type', '').lower()

        # Check if motion is mentioned
        motion_keywords = ['pan', 'tilt', 'zoom', 'dolly', 'track', 'static', 'motion']
        metrics['has_motion'] = any(kw in caption_lower for kw in motion_keywords)

        # Check motion alignment
        if motion_type and motion_type in caption_lower:
            metrics['motion_aligned'] = True
        else:
            metrics['motion_aligned'] = False

        # Check detail richness
        detail_keywords = [
            'cinematic', 'lighting', 'slow', 'fast', 'professional',
            'natural', 'golden', 'atmospheric', 'smooth', 'dramatic'
        ]
        detail_count = sum(1 for kw in detail_keywords if kw in caption_lower)
        metrics['detail_richness'] = detail_count / len(detail_keywords)

        # Overall score
        metrics['quality_score'] = (
            int(metrics['length_ok']) * 0.3 +
            int(metrics['has_motion']) * 0.3 +
            int(metrics['motion_aligned']) * 0.2 +
            metrics['detail_richness'] * 0.2
        )

        return metrics


def enhance_dataset_captions(
    dataset_dir: str,
    target_tier: int = 2,
    use_blip2: bool = False,
    use_gpt4: bool = False,
    gpt4_api_key: Optional[str] = None,
    priority_only: bool = False
):
    """
    Enhance all captions in dataset

    Args:
        dataset_dir: Dataset directory
        target_tier: 1 (keep auto), 2 (template enhance), 3 (AI enhance)
        use_blip2: Use BLIP-2 for scene understanding
        use_gpt4: Use GPT-4 Vision (Tier 3 only)
        gpt4_api_key: OpenAI API key
        priority_only: Only enhance high-priority videos
    """
    dataset_path = Path(dataset_dir)

    # Load metadata
    metadata_path = dataset_path / "dataset_metadata.json"
    if not metadata_path.exists():
        logger.error(f"Metadata not found: {metadata_path}")
        return

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Initialize enhancer
    enhancer = CaptionEnhancer(
        use_blip2=use_blip2,
        use_gpt4=use_gpt4,
        gpt4_api_key=gpt4_api_key
    )

    # Process each clip
    clips = metadata['clips']
    enhanced_count = 0
    validation_results = []

    for clip in tqdm(clips, desc=f"Enhancing to Tier {target_tier}"):
        # Check if priority only
        if priority_only:
            # TODO: Add priority scoring logic
            pass

        # Load current caption
        caption_path = Path(clip['caption_path'])
        with open(caption_path, 'r') as f:
            auto_caption = f.read().strip()

        # Enhance caption
        enhanced_caption = enhancer.enhance_caption(
            auto_caption=auto_caption,
            video_path=clip['video_path'],
            motion_stats=clip.get('motion_stats', {}),
            target_tier=target_tier
        )

        # Validate
        validation = enhancer.validate_caption(
            enhanced_caption,
            clip.get('motion_stats', {})
        )
        validation['clip_name'] = clip['clip_name']
        validation['original_caption'] = auto_caption
        validation['enhanced_caption'] = enhanced_caption
        validation_results.append(validation)

        # Save enhanced caption
        if enhanced_caption != auto_caption:
            with open(caption_path, 'w') as f:
                f.write(enhanced_caption)
            enhanced_count += 1

    # Save validation report
    report_path = dataset_path / f"caption_enhancement_tier{target_tier}_report.json"
    with open(report_path, 'w') as f:
        json.dump({
            'target_tier': target_tier,
            'total_clips': len(clips),
            'enhanced_count': enhanced_count,
            'validation_results': validation_results,
            'average_quality_score': np.mean([v['quality_score'] for v in validation_results])
        }, f, indent=2)

    logger.info(f"\nEnhancement Complete!")
    logger.info(f"Enhanced: {enhanced_count}/{len(clips)} captions")
    logger.info(f"Average quality score: {np.mean([v['quality_score'] for v in validation_results]):.2f}")
    logger.info(f"Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Enhance video captions for training"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Dataset directory"
    )
    parser.add_argument(
        "--target_tier",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Target quality tier: 1 (auto), 2 (template), 3 (AI)"
    )
    parser.add_argument(
        "--use_blip2",
        action="store_true",
        help="Use BLIP-2 for scene understanding (Tier 3)"
    )
    parser.add_argument(
        "--use_gpt4",
        action="store_true",
        help="Use GPT-4 Vision for refinement (Tier 3)"
    )
    parser.add_argument(
        "--gpt4_api_key",
        type=str,
        help="OpenAI API key for GPT-4"
    )
    parser.add_argument(
        "--priority_only",
        action="store_true",
        help="Only enhance high-priority videos"
    )

    args = parser.parse_args()

    enhance_dataset_captions(
        dataset_dir=args.dataset_dir,
        target_tier=args.target_tier,
        use_blip2=args.use_blip2,
        use_gpt4=args.use_gpt4,
        gpt4_api_key=args.gpt4_api_key,
        priority_only=args.priority_only
    )


if __name__ == "__main__":
    main()
