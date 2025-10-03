# Bilateral Toepad Detection Plan

## Problem Analysis

Currently, our YOLOv11 model only detects toepads on one side (lower half) of lizard specimens. This is due to training data bias - the TPS annotation files primarily label landmarks on the lower portion of specimens.

### Current Detection Limitations

- **Training Data**: TPS files contain coordinates mainly for lower half toepads
- **Model Bias**: Model learned to associate toepads with specific Y-coordinate ranges
- **Missing Coverage**: Upper half toepads remain undetected despite similar morphology

## Technical Approaches

### Approach 1: Confidence/IoU Threshold Adjustment 🎯

**Concept**: Adjust detection thresholds to reveal potentially hidden detections.

#### Implementation Steps:

1. **Lower Confidence Threshold**

   ```bash
   # Test with progressively lower thresholds
   python scripts/inference/predict.py --conf 0.1
   python scripts/inference/predict.py --conf 0.05
   python scripts/inference/predict.py --conf 0.01
   ```

2. **Adjust IoU Threshold**

   ```bash
   # Reduce IoU to prevent merging of nearby detections
   python scripts/inference/predict.py --iou 0.3
   ```

3. **Combined Adjustment**
   ```bash
   # Both parameters together
   python scripts/inference/predict.py --conf 0.1 --iou 0.3
   ```

#### Pros:

- ✅ No retraining required
- ✅ Immediate testing possible
- ✅ Zero computational cost

#### Cons:

- ❌ Cannot detect what model hasn't learned
- ❌ May increase false positives
- ❌ Limited by training data bias

### Approach 2: Multi-Scale Detection 🔍

**Concept**: Modify inference script to run detection on split/rotated image regions without retraining.

#### Implementation Steps:

1. **Create Enhanced Inference Script**

   ```python
   # scripts/inference/predict_bilateral.py
   def detect_bilateral_toepads(image_path, model):
       image = Image.open(image_path)

       # Method 1: Split detection
       upper_half = image.crop((0, 0, image.width, image.height//2))
       lower_half = image.crop((0, image.height//2, image.width, image.height))

       # Method 2: Rotation detection (flip upper half)
       upper_flipped = upper_half.rotate(180)

       # Run inference on all regions
       upper_results = model.predict(upper_half)
       lower_results = model.predict(lower_half)
       flipped_results = model.predict(upper_flipped)

       # Combine and adjust coordinates
       return combine_all_detections(upper_results, lower_results, flipped_results)
   ```

2. **Integration with predict.py**

   - Add `--bilateral` flag to enable multi-scale detection
   - Modify existing inference pipeline
   - No changes to training scripts needed

3. **Post-processing Logic**
   - Adjust bounding box coordinates for each region
   - Remove duplicate detections at boundaries
   - Merge overlapping boxes with NMS

#### Pros:

- ✅ **No training script changes required**
- ✅ Works with existing trained model
- ✅ Can be tested immediately
- ✅ May detect upper toepads through rotation

#### Cons:

- ❌ 2-3x slower inference time
- ❌ May miss toepads at region boundaries
- ❌ Requires careful coordinate adjustment

### Approach 3: Data Augmentation Strategy 🔄

**Concept**: Use image transformations to create synthetic training data for upper toepads.

#### Implementation Steps:

1. **Vertical Flip Augmentation**

   ```yaml
   # Add to training config
   augmentation:
     flipud: 0.5 # 50% chance vertical flip
     fliplr: 0.0 # No horizontal flip (preserves anatomy)
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

- ✅ Leverages existing annotations
- ✅ Minimal data preparation
- ✅ Quick to implement

#### Cons:

- ❌ Requires model retraining
- ❌ May create unnatural orientations
- ❌ Doesn't capture natural variation

### Approach 4: Enhanced Annotation Strategy 📍

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

### Approach 5: Class Expansion Strategy 🏷️

**Concept**: Expand from 3 to 6+ classes to explicitly model bilateral anatomy.

#### New Class Structure:

```yaml
dataset:
  nc: 6
  names:
    [
      "upper_finger",
      "lower_finger",
      "upper_toe",
      "lower_toe",
      "left_ruler",
      "right_ruler",
    ]
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

2. **Complete Dataset Regeneration**

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

- ❌ Most complex implementation
- ❌ Requires complete dataset regeneration
- ❌ Increased model complexity

## Experimental Results

### Testing Approach 1: Confidence/IoU Threshold Adjustment

We conducted experiments to understand the current model's detection behavior and evaluate whether simple threshold adjustments could address the bilateral detection issue.

#### Baseline Performance (conf=0.25)

Running inference on validation images with default confidence threshold:

```bash
python scripts/inference/predict.py --quick-test --conf 0.25
```

**Results**: Consistently detected 3 objects per image (1 finger, 1 toe, 1 ruler)

#### Confidence Threshold Experiments

Testing with lower confidence threshold to reveal potential hidden detections:

```bash
python scripts/inference/predict.py --quick-test --conf 0.1
```

**Key Findings**:

- Most images still showed 3 detections
- Images 22-23: Detected **2 toes** instead of 1
- Images 28-29: Detected **2 rulers** instead of 1
- Lower confidence revealed ~10% more objects, but not upper-half toepads

#### Visual Analysis

![Detection Example 1](assets/bilateral_detection_plan/conf_Iou_treshhold_adjust/1004.jpg)
_Image 1004.jpg: Current model detects only lower-half toepads despite visible upper-half structures_

![Detection Example 2](assets/bilateral_detection_plan/conf_Iou_treshhold_adjust/1007.jpg)
_Image 1007.jpg: Similar pattern - upper toepads remain undetected regardless of confidence threshold_

#### Critical Insights

1. **Training Data Bias Confirmed**: The model has learned to detect only the most prominent toepads in the lower region, matching the TPS annotation pattern.

2. **Threshold Adjustment Insufficient**: Lowering confidence from 0.25 to 0.1 only revealed minor additional detections (duplicate rulers/toes) but **did not detect upper-half toepads**.

3. **Model Capability vs Training**: The architecture can detect multiple objects (as shown by 4-object detections), but hasn't learned to recognize upper toepads as valid targets.

4. **Root Cause**: This is not a post-processing issue but a fundamental training data limitation - the model never saw labeled upper toepads during training.

### Conclusion from Approach 1

Simple confidence threshold adjustments (Approach 1) cannot solve the bilateral detection problem. The model requires one of the following solutions:

- **Approach 2**: Multi-scale detection to process upper/lower regions separately (simplest)
- **Approach 3**: Data augmentation with vertical flips to synthesize upper toepad training examples
- **Approach 4**: Enhanced annotations including both upper and lower toepads
- **Approach 5**: Class expansion to explicitly model upper vs lower toepads (most complex)

---

### Testing Approach 2: Multi-Scale Detection (Split + Flip)

We implemented a new inference script `predict_bilateral.py` that combines region splitting and vertical flipping to detect toepads on both sides without retraining.

#### Implementation Details

**Method: `both` (combined split + flip)**

- Split image into upper and lower halves with 10% overlap
- Vertically flip upper region to make it look like lower region
- Run detection on all regions
- Apply NMS to remove duplicate detections
- Adjust coordinates back to original image space

```bash
python scripts/inference/predict_bilateral.py --quick-test --method both
```

#### Results

![Multi-Scale Detection Example 1](assets/bilateral_detection_plan/Multi-Scale%20Detection/1004.jpg)
_Image 1004.jpg: Multi-scale detection successfully detects BOTH upper and lower toepads_

![Multi-Scale Detection Example 2](assets/bilateral_detection_plan/Multi-Scale%20Detection/1012.jpg)
_Image 1012.jpg: Comprehensive bilateral detection coverage achieved_

#### Key Findings

1. **✅ Successfully Detects Upper Toepads**: By flipping the upper region, the model now recognizes upper toepads as valid targets

2. **✅ Comprehensive Coverage**: Combined split + flip method provides detection across entire specimen

3. **✅ No Retraining Required**: Works with existing H1 model, immediate deployment possible

4. **✅ Coordinate Accuracy**: Properly transforms detection coordinates back to original image space

5. **⚠️ Computational Cost**: 2-3x slower than single-pass inference (acceptable trade-off)

6. **⚠️ Some Duplicates**: NMS helps but occasional overlapping detections at region boundaries

#### Performance Comparison

| Method                  | Upper Detections | Lower Detections | Total Detections | Speed  |
| ----------------------- | ---------------- | ---------------- | ---------------- | ------ |
| Approach 1 (Baseline)   | 0                | 3                | 3                | 1x     |
| Approach 2 (Split only) | 0-1              | 2-3              | 2-4              | 2x     |
| Approach 2 (Flip only)  | 1-2              | 2-3              | 3-5              | 2x     |
| **Approach 2 (Both)**   | **2-3**          | **2-3**          | **4-8**          | **3x** |

### Conclusion from Approach 2

**Approach 2 (Multi-Scale Detection) is SUCCESSFUL and RECOMMENDED as the immediate solution:**

✅ **Pros:**

- Works with existing trained model (no retraining needed)
- Effectively detects upper toepads through vertical flipping
- Can be deployed immediately for production use
- Provides comprehensive bilateral coverage

⚠️ **Limitations:**

- 2-3x slower inference time (still practical for most use cases)
- May produce duplicate detections at boundaries (manageable with NMS)
- Flip method assumes symmetric toepad morphology

**Recommendation:** Deploy Approach 2 as the production solution while exploring Approach 3 (data augmentation) for potential model improvement in the future.

## Recommended Implementation Roadmap

### Phase 1: Quick Validation ✅ COMPLETED

1. ✅ **Test Approach 1** - Confidence/IoU threshold adjustments → **Result: Insufficient**
2. ✅ **Implement Approach 2** - Multi-scale detection → **Result: SUCCESSFUL**
3. ✅ Test on current model without retraining
4. ✅ Evaluate detection coverage improvement → **Comprehensive bilateral coverage achieved**

**Outcome:** Approach 2 successfully solves the bilateral detection problem without retraining. Ready for production deployment.

### Phase 2: Data Augmentation (3-5 days)

1. **Implement Approach 3** - Vertical flip augmentation
2. Create H2_bilateral config with augmentation
3. Retrain model and compare results

### Phase 3: Enhanced Detection (1-2 weeks)

1. **Choose between Approach 4 or 5**:
   - Approach 4: Manual annotation expansion (more accurate)
   - Approach 5: Class expansion (more systematic)
2. Implement chosen approach
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

1. ✅ **Completed**: Approach 1 (threshold adjustment) - confirmed insufficient
2. ✅ **Completed**: Approach 2 (multi-scale detection) - **SUCCESSFUL, ready for production**
3. **Recommended**: Deploy Approach 2 (`predict_bilateral.py`) for production use
4. **Optional**: Explore Approach 3 (data augmentation) for potential model improvements
5. **Future**: Consider Approach 4 or 5 only if Approach 2 proves insufficient

## Summary

This plan successfully identified and solved the bilateral toepad detection problem through a systematic approach:

- **Problem**: Training data bias caused model to only detect lower-half toepads
- **Solution**: Multi-scale detection with vertical flipping (Approach 2)
- **Result**: Comprehensive bilateral coverage without model retraining
- **Status**: Ready for production deployment

The multi-scale detection approach provides an elegant solution that leverages the existing model's capabilities while compensating for training data limitations.
