# Video Captioning Best Practices for Trajectory-Guided Training

## Overview

High-quality captions are **crucial** for trajectory-guided video generation because they help the model:
1. **Associate motion patterns** with semantic concepts
2. **Understand context** (camera motion vs object motion)
3. **Generate appropriate content** for the motion type
4. **Improve generalization** to new prompts at inference

## Caption Structure

### Optimal Caption Formula

```
[MOTION TYPE] + [SUBJECT/SCENE] + [STYLE/QUALITY] + [TECHNICAL DETAILS]
```

### Components Breakdown

#### 1. Motion Type (Required)
Describes the primary motion in the video.

**Camera Motion:**
- `"camera pan left"`
- `"camera tilt up"`
- `"camera dolly forward"`
- `"camera zoom in"`
- `"smooth tracking shot"`
- `"handheld camera movement"`
- `"drone shot ascending"`

**Object Motion:**
- `"person walking towards camera"`
- `"car driving past"`
- `"leaves falling"`
- `"dancer spinning"`
- `"ball bouncing"`

**Speed Modifiers:**
- `"slow motion"`
- `"fast motion"`
- `"medium speed"`
- `"very slow"`

#### 2. Subject/Scene (Highly Recommended)
What is in the video.

**Examples:**
- `"a woman in red dress"`
- `"a busy city street"`
- `"mountain landscape"`
- `"coffee being poured"`
- `"ocean waves"`

#### 3. Style/Quality (Recommended)
Visual characteristics and mood.

**Cinematic Descriptors:**
- `"cinematic composition"`
- `"professional footage"`
- `"documentary style"`
- `"amateur video"`
- `"surveillance camera angle"`

**Lighting:**
- `"golden hour lighting"`
- `"soft natural light"`
- `"dramatic shadows"`
- `"overcast day"`
- `"neon lights at night"`

**Mood:**
- `"serene atmosphere"`
- `"energetic scene"`
- `"melancholic tone"`

#### 4. Technical Details (Optional but Helpful)
Helps model understand quality expectations.

- `"shot on 35mm film"`
- `"4K resolution"`
- `"shallow depth of field"`
- `"wide angle lens"`
- `"film grain texture"`
- `"high contrast"`

## Caption Quality Tiers

### ⭐ Tier 1: Minimal (Auto-Generated)
**What auto-captioning provides:**
```
Camera pan left, slow motion, realistic footage
```

**Pros:** Fast, consistent format
**Cons:** Generic, lacks detail, misses context

**Use case:** Initial testing, quick iteration

### ⭐⭐ Tier 2: Enhanced (Auto + Manual Touch)
**Auto-generated + manual enhancement:**
```
Smooth camera pan left across a mountain landscape, slow motion,
natural lighting, cinematic composition
```

**Pros:** Better than auto, still scalable
**Cons:** Requires manual work

**Use case:** Production training (500-2000 videos)

### ⭐⭐⭐ Tier 3: Professional (Full Manual)
**Detailed, curated captions:**
```
Cinematic slow pan left across misty mountain peaks at sunrise,
revealing layers of ridges in soft golden light, smooth gimbal
movement, atmospheric depth, professional nature cinematography,
film grain aesthetic, 4K quality
```

**Pros:** Maximum training quality, rich context
**Cons:** Very time-consuming

**Use case:** High-value production, <500 videos

## Best Practices

### ✅ DO

1. **Start with Motion**
   ```
   ✅ Good: "Camera pan left across a busy street"
   ❌ Bad: "A busy street with camera panning"
   ```

2. **Be Specific About Motion Type**
   ```
   ✅ Good: "Slow dolly forward towards subject"
   ❌ Bad: "Camera moving forward"
   ```

3. **Include Speed When Relevant**
   ```
   ✅ Good: "Fast pan right following a car"
   ❌ Bad: "Pan right following a car" (ambiguous speed)
   ```

4. **Describe Visual Quality**
   ```
   ✅ Good: "Cinematic shot with shallow depth of field"
   ❌ Bad: "Shot of person" (no quality info)
   ```

5. **Use Consistent Terminology**
   - Don't mix: "camera pan left" and "panning camera left"
   - Pick one format and stick with it

6. **Match Caption to Video Quality**
   ```
   ✅ If video is HD, cinematic: "Professional cinematic footage"
   ❌ If video is HD, cinematic: "Low quality phone video"
   ```

7. **Include Lighting Info**
   ```
   ✅ Good: "Golden hour lighting, warm tones"
   ❌ Bad: (no lighting description)
   ```

### ❌ DON'T

1. **Don't Be Vague**
   ```
   ❌ Bad: "Video of something moving"
   ✅ Good: "Medium speed pan following a cyclist"
   ```

2. **Don't Contradict the Trajectory**
   ```
   ❌ Bad: Caption says "static shot" but trajectory shows movement
   ✅ Good: Ensure caption matches extracted motion
   ```

3. **Don't Use Ambiguous Terms**
   ```
   ❌ Bad: "Nice video"
   ❌ Bad: "Cool shot"
   ✅ Good: "Cinematic composition with dynamic motion"
   ```

4. **Don't Overload with Irrelevant Details**
   ```
   ❌ Bad: "Shot on Tuesday at 3pm with Canon EOS R5 at f/2.8..."
   ✅ Good: "Professional camera work, shallow depth of field"
   ```

5. **Don't Use First/Second Person**
   ```
   ❌ Bad: "I filmed this walking down the street"
   ✅ Good: "Handheld camera walking down a street"
   ```

6. **Don't Include Timestamps or Metadata**
   ```
   ❌ Bad: "video_2024_01_15_final_v3.mp4"
   ✅ Good: Actual description of content
   ```

## Caption Templates

### Template 1: Camera Motion Focused

```
[CAMERA_MOVEMENT] [SUBJECT/SCENE], [SPEED], [LIGHTING], [STYLE]
```

**Examples:**
- `"Camera pan left across sunset beach, slow motion, golden hour lighting, cinematic"`
- `"Smooth dolly forward towards person, medium speed, soft natural light, professional"`
- `"Handheld tilt up revealing skyscraper, fast motion, bright daylight, documentary style"`

### Template 2: Object Motion Focused

```
[SUBJECT] [ACTION] [DIRECTION], [CAMERA_TYPE], [LIGHTING], [STYLE]
```

**Examples:**
- `"Person walking towards camera, static shot, golden hour, cinematic composition"`
- `"Car driving past from left to right, tracking shot, daytime, professional footage"`
- `"Dancer spinning gracefully, slow motion, studio lighting, artistic cinematography"`

### Template 3: Scene Dynamics

```
[SCENE_DESCRIPTION] with [MOTION_DESCRIPTION], [CAMERA_WORK], [ATMOSPHERE]
```

**Examples:**
- `"Busy city intersection with pedestrians crossing, static wide angle, urban atmosphere"`
- `"Waterfall flowing over rocks with mist rising, slow pan right, serene natural scene"`
- `"Crowded market with vendors and shoppers moving, handheld walk-through, vibrant colors"`

### Template 4: Cinematic (Production Quality)

```
[CINEMATIC_DESCRIPTOR] of [SUBJECT] [ACTION], [TECHNICAL_DETAILS], [MOOD]
```

**Examples:**
- `"Cinematic shot of woman in flowing dress walking through field, golden hour backlighting, shallow depth of field, dreamy atmosphere"`
- `"Aerial drone footage ascending over coastal cliffs, smooth gimbal work, dramatic sunset colors, epic scale"`
- `"Intimate close-up of hands crafting pottery, slow motion, soft diffused lighting, meditative mood"`

## Domain-Specific Captioning

### Nature/Landscape
```
[MOTION] [NATURAL_FEATURE], [TIME_OF_DAY], [WEATHER], [STYLE]

"Slow pan across mountain range, sunrise, misty atmosphere, cinematic landscape"
"Tilt up revealing waterfall, midday, clear skies, nature documentary style"
```

### Urban/Cityscape
```
[MOTION] [URBAN_ELEMENT], [ACTIVITY_LEVEL], [TIME], [STYLE]

"Dolly through busy street, high pedestrian traffic, evening, urban documentary"
"Pan across city skyline, calm night scene, neon lights, cinematic establishing shot"
```

### People/Portrait
```
[SUBJECT_DESCRIPTION] [ACTION], [CAMERA_WORK], [LIGHTING], [MOOD]

"Young woman walking confidently towards camera, steady tracking, golden hour light, fashion editorial"
"Elderly man sitting on bench reading, slow push in, soft morning light, contemplative"
```

### Action/Sports
```
[ATHLETE/SUBJECT] [ACTION], [CAMERA_TECHNIQUE], [SPEED], [ENERGY]

"Skateboarder performing trick, slow motion tracking shot, fast action, dynamic"
"Runner sprinting on track, panning follow shot, high speed, intense energy"
```

### Product/Commercial
```
[PRODUCT] [PRESENTATION], [CAMERA_MOVE], [LIGHTING], [STYLE]

"Coffee cup on table with steam rising, slow dolly in, soft studio lighting, commercial quality"
"Car driving on highway, tracking shot, dramatic sunset, automotive commercial"
```

## Enhancement Workflow

### Step 1: Use Auto-Generated Caption as Base

```python
# Auto-generated:
"Camera pan left, slow motion, realistic footage"
```

### Step 2: Add Subject/Scene Context

```python
# Enhanced:
"Camera pan left across mountain landscape, slow motion, realistic footage"
```

### Step 3: Add Visual Quality Descriptors

```python
# More enhanced:
"Camera pan left across mountain landscape, slow motion, natural lighting, cinematic composition"
```

### Step 4: Add Atmosphere/Mood

```python
# Fully enhanced:
"Cinematic pan left across misty mountain landscape at sunrise, slow motion, soft golden light, serene atmosphere"
```

## Tools and Techniques

### Automated Captioning Tools

#### 1. BLIP-2 (Recommended for Initial Pass)
```python
from transformers import Blip2Processor, Blip2ForConditionalGeneration

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

# Generate caption from first frame
caption = model.generate(first_frame)
```

**Pros:** Good scene understanding
**Cons:** Doesn't understand motion well

#### 2. LLaVA (For Detailed Descriptions)
```python
from llava import LlavaModel

# Can process multiple frames for motion understanding
caption = llava.generate_caption(frames, prompt="Describe the camera motion and scene")
```

**Pros:** Better motion understanding
**Cons:** Slower, needs more compute

#### 3. GPT-4 Vision (Best Quality, API Cost)
```python
import openai

response = openai.ChatCompletion.create(
    model="gpt-4-vision-preview",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this video's motion, subject, lighting, and style in one sentence suitable for video generation training."},
            {"type": "image_url", "image_url": first_frame},
            {"type": "image_url", "image_url": middle_frame},
            {"type": "image_url", "image_url": last_frame}
        ]
    }]
)
```

**Pros:** Excellent quality, understands nuance
**Cons:** API costs, rate limits

### Hybrid Approach (Recommended)

```python
def generate_caption(video_path, trajectory_data):
    # 1. Extract motion statistics
    motion_stats = analyze_motion(trajectory_data)
    motion_desc = f"{motion_stats['camera_type']}, {motion_stats['motion_description']}"

    # 2. Use BLIP-2 for scene description
    first_frame = load_first_frame(video_path)
    scene_desc = blip2_caption(first_frame)

    # 3. Combine with template
    caption = f"{motion_desc}, {scene_desc}, cinematic footage"

    # 4. Optional: Refine with GPT-4 (for high-value videos)
    if is_high_priority(video_path):
        caption = gpt4_refine(caption, [first_frame, middle_frame, last_frame])

    return caption
```

## Batch Captioning Strategy

### For Large Datasets (1000+ videos)

**Phase 1: Auto-Generate (Day 1)**
```bash
python scripts/prepare_dataset_single_source.py \
    --auto_caption_only \
    --caption_model blip2
```
Result: Basic captions for all videos

**Phase 2: Priority Enhancement (Day 2-3)**
```python
# Identify high-quality/important videos
priority_videos = select_best_videos(dataset, criteria=['quality', 'diversity'])

# Manually enhance these captions
for video in priority_videos:
    auto_caption = load_caption(video)
    enhanced_caption = manual_enhance(auto_caption)  # Your curation
    save_caption(video, enhanced_caption)
```
Result: 100-200 high-quality captions

**Phase 3: Template-Based Enhancement (Day 4-5)**
```python
# Use templates to enhance remaining captions
for video in remaining_videos:
    auto_caption = load_caption(video)
    template_caption = apply_template(auto_caption, get_video_category(video))
    save_caption(video, template_caption)
```
Result: All captions enhanced to Tier 2 quality

## Caption Quality Metrics

### How to Measure Caption Quality

#### 1. Motion Alignment Score
```python
def motion_alignment_score(caption, trajectory_data):
    """
    Check if caption motion description matches extracted trajectory
    """
    caption_motion = extract_motion_from_caption(caption)
    actual_motion = classify_motion(trajectory_data)

    return similarity(caption_motion, actual_motion)
```

**Target:** > 0.8 alignment

#### 2. Detail Richness Score
```python
def detail_richness(caption):
    """
    Count informative words
    """
    keywords = ['cinematic', 'lighting', 'motion', 'slow', 'fast', 'pan', 'tilt', 'zoom']
    score = sum(1 for kw in keywords if kw in caption.lower())
    return score / len(keywords)
```

**Target:** > 0.4 (at least 4 keywords)

#### 3. Length Appropriateness
```python
def length_check(caption):
    """
    Captions should be 10-50 words
    """
    word_count = len(caption.split())
    return 10 <= word_count <= 50
```

**Target:** 15-30 words (sweet spot)

## Examples: Good vs Bad Captions

### Example 1: Walking Person

**❌ Bad:**
```
"Person walking"
```
**Issues:** Too vague, no motion details, no context

**⭐ Better:**
```
"Person walking forward, camera static, realistic footage"
```
**Issues:** Still generic, lacks visual quality info

**⭐⭐ Good:**
```
"Woman in casual clothes walking towards camera on city sidewalk, steady shot, natural daylight, documentary style"
```
**Better:** Specific subject, motion, setting, lighting

**⭐⭐⭐ Excellent:**
```
"Professional shot of confident woman in business attire walking towards camera on urban sidewalk, smooth steadicam follow, golden hour side lighting, shallow depth of field, cinematic composition"
```
**Best:** Rich detail, clear motion, professional descriptors

### Example 2: Landscape Pan

**❌ Bad:**
```
"Mountains"
```

**⭐ Better:**
```
"Camera pan across mountains, realistic footage"
```

**⭐⭐ Good:**
```
"Slow camera pan left across mountain range at sunrise, misty atmosphere, cinematic landscape"
```

**⭐⭐⭐ Excellent:**
```
"Cinematic slow pan left revealing layered mountain peaks emerging from morning mist, golden hour backlighting, smooth gimbal movement, atmospheric depth, nature documentary quality"
```

### Example 3: Car Driving

**❌ Bad:**
```
"Car video"
```

**⭐ Better:**
```
"Car driving, camera tracking, realistic footage"
```

**⭐⭐ Good:**
```
"Black sports car driving along coastal highway, smooth tracking shot, sunny day, automotive footage"
```

**⭐⭐⭐ Excellent:**
```
"Cinematic tracking shot of sleek black sports car driving along coastal highway at sunset, smooth drone follow, dramatic golden light reflecting off ocean, professional automotive cinematography"
```

## Caption Validation Checklist

Before finalizing captions, verify:

- [ ] **Motion type is explicit** (pan, tilt, dolly, zoom, static, etc.)
- [ ] **Motion speed is mentioned** (slow, medium, fast) if relevant
- [ ] **Subject/scene is described** (what's in the video)
- [ ] **Camera work is specified** (handheld, gimbal, drone, static)
- [ ] **Lighting is mentioned** (golden hour, natural, studio, etc.)
- [ ] **Style/quality descriptor** (cinematic, professional, documentary)
- [ ] **Caption matches trajectory** (motion description aligns with actual motion)
- [ ] **Length is appropriate** (15-30 words ideal)
- [ ] **Grammar is correct** (no typos or fragments)
- [ ] **Terminology is consistent** (same terms across dataset)

## Production Workflow

### For 500-2000 Video Dataset

**Week 1: Auto-Generation**
```bash
# Generate all captions automatically
python scripts/prepare_dataset_single_source.py \
    --input_dir /workspace/raw_videos \
    --output_dir ./dataset \
    --enable_auto_captions \
    --caption_model blip2
```
**Output:** Basic captions (Tier 1)

**Week 2: Quality Control**
```python
# Review and categorize
python scripts/categorize_videos.py --dataset ./dataset
# Creates: high_quality.txt, medium_quality.txt, low_quality.txt
```

**Week 3: Enhancement**
```python
# Enhance high-priority videos manually
# Use Jupyter notebook or custom UI
# Target: 200-500 videos → Tier 3 quality

# Template-enhance medium priority
# Target: 500-1000 videos → Tier 2 quality

# Keep auto-captions for rest
# Target: Remaining videos → Tier 1 quality
```

**Week 4: Validation**
```python
# Validate all captions
python scripts/validate_captions.py --dataset ./dataset
# Fixes common issues, checks alignment
```

**Result:** Production-ready dataset with mixed-quality captions
- 20-30% Tier 3 (excellent)
- 40-50% Tier 2 (good)
- 20-30% Tier 1 (acceptable)

## Advanced: Prompt Engineering for Training

### Caption as Prompt Template

During training, captions can be used as-is or enhanced:

```python
# Original caption
caption = "Camera pan left across beach, slow motion, golden hour"

# Training prompt variations (data augmentation)
variations = [
    caption,  # Original
    f"A cinematic video showing {caption.lower()}",  # Wrapped
    f"{caption}, professional cinematography",  # Enhanced
    f"{caption.split(',')[0]} in a {caption.split(',')[1]} style"  # Restructured
]
```

### Motion-Conditional Prompting

```python
# During inference, combine user prompt with trajectory
user_prompt = "A serene mountain landscape"
trajectory_motion = "slow pan left"

combined_prompt = f"{user_prompt}, {trajectory_motion}, cinematic"
# → "A serene mountain landscape, slow pan left, cinematic"
```

## Summary

### Key Takeaways

1. **Motion First**: Always start caption with motion description
2. **Be Specific**: "Slow dolly forward" > "Camera moving"
3. **Add Context**: Include subject, lighting, style
4. **Quality Tiers**: Aim for Tier 2 minimum (enhanced auto)
5. **Consistency**: Use same terminology throughout dataset
6. **Validation**: Check alignment with trajectory data
7. **Hybrid Approach**: Auto-generate + manual enhancement for key videos

### Recommended Strategy

**For Best Results:**
- Auto-generate all captions (fast baseline)
- Manually enhance 20-30% of high-priority videos
- Use templates to improve remaining 70-80%
- Validate motion alignment for all
- Iterate based on training results

### Quality vs Quantity

**Better to have:**
- 500 videos with Tier 2-3 captions
- Than 2000 videos with Tier 1 captions

**But acceptable to have:**
- Mixed quality: 20% Tier 3, 50% Tier 2, 30% Tier 1
- This balances quality and dataset size

---

**Remember:** Caption quality directly impacts the model's ability to understand and follow motion + content instructions during inference. Invest time in good captions for better training results!
