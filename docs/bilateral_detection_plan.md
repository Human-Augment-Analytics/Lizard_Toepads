# Bilateral Toepad Detection Plan

## Problem Analysis

Currently, our YOLOv11 model only detects toepads on one side (lower half) of lizard specimens. This is due to training data bias - the TPS annotation files primarily label landmarks on the lower portion of specimens.

### Current Detection Limitations
- **Training Data**: TPS files contain coordinates mainly for lower half toepads
- **Model Bias**: Model learned to associate toepads with specific Y-coordinate ranges
- **Missing Coverage**: Upper half toepads remain undetected despite similar morphology

## Technical Approaches

### Approach 1: Data Augmentation Strategy 🔄

**Concept**: Use image transformations to create synthetic training data for upper toepads.

#### Implementation Steps:
1. **Vertical Flip Augmentation**
   ```yaml
   # Add to training config
   augmentation:
     flipud: 0.5  # 50% chance vertical flip
     fliplr: 0.0  # No horizontal flip (preserves anatomy)
   ```

2. **Label Coordinate Transformation**
   - Automatically flip bounding box coordinates for flipped images
   - Ensure anatomical correctness (don't flip left/right sides)

3. **Training Pipeline Integration**
   ```bash
   # Create H2 config with augmentation
   cp configs/H1.yaml configs/H2_bilateral.yaml
   # Modify H2 config to include augmentation
   # Retrain model
   ```

#### Pros:
- ✅ Quick implementation
- ✅ Leverages existing annotations
- ✅ Minimal data preparation

#### Cons:
- ❌ May create unnatural orientations
- ❌ Doesn't capture natural variation in upper toepads

### Approach 2: Enhanced Annotation Strategy 📍

**Concept**: Expand TPS annotations to include both sides of specimens.

#### Implementation Steps:
1. **TPS File Analysis**
   - Audit existing TPS files for coverage gaps
   - Identify specimens with incomplete annotations

2. **Annotation Expansion**
   ```python
   # Modify TPS processing to detect both upper/lower regions
   def detect_bilateral_landmarks(tps_data, image_height):
       upper_landmarks = filter_by_y_range(tps_data, 0, image_height/2)
       lower_landmarks = filter_by_y_range(tps_data, image_height/2, image_height)
       return upper_landmarks, lower_landmarks
   ```

3. **Semi-Automated Annotation**
   - Use current model to propose upper toepad locations
   - Manual verification and correction
   - Generate new TPS files with bilateral coverage

#### Pros:
- ✅ Most accurate approach
- ✅ Captures natural anatomical variation
- ✅ Creates comprehensive dataset

#### Cons:
- ❌ Time-intensive annotation process
- ❌ Requires domain expertise for verification

### Approach 3: Multi-Scale Detection 🔍

**Concept**: Train model to detect toepads at multiple scales and orientations.

#### Implementation Steps:
1. **Sliding Window Approach**
   ```python
   def detect_bilateral_toepads(image, model):
       # Split image into upper/lower regions
       upper_half = image[:image.height//2, :]
       lower_half = image[image.height//2:, :]

       # Run detection on both regions
       upper_detections = model.predict(upper_half)
       lower_detections = model.predict(lower_half)

       # Combine and adjust coordinates
       return combine_detections(upper_detections, lower_detections)
   ```

2. **Ensemble Detection**
   - Train separate models for upper/lower regions
   - Combine predictions with confidence weighting

#### Pros:
- ✅ Works with existing annotations
- ✅ Flexible detection strategy
- ✅ Can be implemented post-training

#### Cons:
- ❌ Increased computational cost
- ❌ May miss toepads spanning the middle region

### Approach 4: Class Expansion Strategy 🏷️

**Concept**: Expand from 3 to 5+ classes to explicitly model bilateral anatomy.

#### New Class Structure:
```yaml
dataset:
  nc: 6
  names: ["upper_finger", "lower_finger", "upper_toe", "lower_toe", "left_ruler", "right_ruler"]
```

#### Implementation Steps:
1. **TPS Processing Enhancement**
   ```python
   def classify_landmarks_by_region(landmarks, image_dimensions):
       height_threshold = image_dimensions[1] / 2

       for landmark in landmarks:
           if landmark.y < height_threshold:
               landmark.class = f"upper_{landmark.type}"
           else:
               landmark.class = f"lower_{landmark.type}"
   ```

2. **Dataset Regeneration**
   - Reprocess all TPS files with new class structure
   - Regenerate YOLO label files
   - Update train/val splits

3. **Model Retraining**
   - Train new model with expanded class set
   - Adjust confidence thresholds per class

#### Pros:
- ✅ Explicit bilateral modeling
- ✅ Clear anatomical distinction
- ✅ Enables specialized analysis

#### Cons:
- ❌ Requires complete dataset regeneration
- ❌ Increased model complexity

## Recommended Implementation Roadmap

### Phase 1: Quick Validation (1-2 days)
1. **Implement Approach 3** - Multi-scale detection
2. Test on current model without retraining
3. Evaluate detection coverage improvement

### Phase 2: Data Augmentation (3-5 days)
1. **Implement Approach 1** - Vertical flip augmentation
2. Create H2_bilateral config with augmentation
3. Retrain model and compare results

### Phase 3: Enhanced Detection (1-2 weeks)
1. **Implement Approach 4** - Class expansion
2. Reprocess entire dataset with bilateral classes
3. Train production model with bilateral detection

### Phase 4: Validation & Optimization (1 week)
1. Comprehensive testing on held-out specimens
2. Performance benchmarking across approaches
3. Error analysis and model refinement

## Success Metrics

### Quantitative Metrics:
- **Detection Coverage**: % of toepads detected (target: >90% both sides)
- **Precision/Recall**: Maintain >95% precision, >90% recall per side
- **Cross-Side Consistency**: Similar detection confidence upper/lower

### Qualitative Metrics:
- **Anatomical Accuracy**: Biologically plausible detections
- **Robustness**: Performance across specimen orientations
- **Scalability**: Processing time remains practical

## Technical Considerations

### Data Pipeline Modifications:
```python
# Enhanced TPS processing for bilateral detection
class BilateralTpsProcessor:
    def __init__(self, upper_lower_threshold=0.5):
        self.threshold = upper_lower_threshold

    def process_bilateral_landmarks(self, tps_file, image_dims):
        # Implementation details...
        pass
```

### Model Architecture:
- Current YOLOv11n sufficient for expanded classes
- Consider YOLOv11s if accuracy degrades with more classes
- H200 GPU can handle larger models if needed

### Inference Pipeline:
```python
# Updated inference for bilateral detection
def bilateral_inference(image_path, model):
    results = model.predict(image_path)

    # Group detections by anatomical region
    upper_detections = filter_by_y_range(results, 0, 0.5)
    lower_detections = filter_by_y_range(results, 0.5, 1.0)

    return {
        'upper': upper_detections,
        'lower': lower_detections,
        'total_coverage': calculate_coverage(upper_detections, lower_detections)
    }
```

## Next Steps

1. **Immediate**: Implement and test Approach 3 (multi-scale detection)
2. **Short-term**: Set up Approach 1 (data augmentation) experiment
3. **Medium-term**: Begin annotation expansion for Approach 2
4. **Long-term**: Full bilateral class expansion (Approach 4)

This plan provides multiple pathways to achieve bilateral toepad detection, with increasing levels of sophistication and annotation requirements.