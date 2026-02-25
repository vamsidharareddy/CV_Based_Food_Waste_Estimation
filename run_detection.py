"""
Food & Shape Detection with GPU Acceleration
Optimized for NVIDIA RTX 4070
"""

import os
import cv2
import json
import torch
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import time

############################################
# GPU Verification
############################################
print("="*60)
print("🚀 Food & Shape Detection System")
print("="*60)

if not torch.cuda.is_available():
    print("❌ CUDA not available! This script requires GPU.")
    exit()

print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"✅ CUDA Version: {torch.version.cuda}")
print(f"✅ PyTorch Version: {torch.__version__}")
print("="*60)

############################################
# Model Paths
############################################
food_model_dir = r"D:\Food\food_type_model"
shape_model_dir = r"D:\Food\shape_model"

# Verify paths
if not os.path.exists(food_model_dir):
    print(f"❌ Food model not found: {food_model_dir}")
    exit()
if not os.path.exists(shape_model_dir):
    print(f"❌ Shape model not found: {shape_model_dir}")
    exit()

print("✅ Model directories found\n")

############################################
# Load Model Function
############################################
def load_model(model_dir, dataset_name):
    """Load Detectron2 model with GPU optimization"""
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(model_dir, "config.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join(model_dir, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda"  # Force GPU
    
    with open(os.path.join(model_dir, "metadata.json"), "r") as f:
        metadata_dict = json.load(f)
    
    metadata = MetadataCatalog.get(dataset_name)
    metadata.set(thing_classes=metadata_dict["thing_classes"])
    
    predictor = DefaultPredictor(cfg)
    return predictor, metadata

############################################
# Load Models
############################################
print("="*60)
print("Loading Models...")
print("="*60 + "\n")

# Load Food Model
start = time.time()
print("🍔 Loading food detection model...")
food_predictor, food_metadata = load_model(food_model_dir, "food_dataset")
food_load_time = time.time() - start
print(f"   ✅ Loaded in {food_load_time:.2f}s")
print(f"   📋 Classes: {food_metadata.thing_classes}\n")

# Load Shape Model
start = time.time()
print("🔷 Loading shape detection model...")
shape_predictor, shape_metadata = load_model(shape_model_dir, "shape_dataset")
shape_load_time = time.time() - start
print(f"   ✅ Loaded in {shape_load_time:.2f}s")
print(f"   📋 Classes: {shape_metadata.thing_classes}\n")

print(f"⏱️ Total loading time: {food_load_time + shape_load_time:.2f}s")
print("="*60 + "\n")

############################################
# Load Image
############################################
print("📂 Select an image...")
Tk().withdraw()
image_path = filedialog.askopenfilename(
    title="Select an Image",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
)

if not image_path:
    print("❌ No image selected.")
    exit()

image = cv2.imread(image_path)

if image is None:
    print(f"❌ Failed to load: {image_path}")
    exit()

print(f"✅ Image loaded: {os.path.basename(image_path)}")
print(f"📐 Dimensions: {image.shape[1]}x{image.shape[0]} (WxH)")
print(f"💾 Size: {os.path.getsize(image_path) / 1024:.1f} KB\n")

############################################
# Run Inference
############################################
print("="*60)
print("Running Inference on GPU...")
print("="*60 + "\n")

# Food Detection
start = time.time()
print("🍔 Detecting food items...")
food_outputs = food_predictor(image)
food_time = time.time() - start
food_count = len(food_outputs["instances"])
print(f"   ✅ Found {food_count} food items in {food_time:.3f}s\n")

# Shape Detection
start = time.time()
print("🔷 Detecting shapes...")
shape_outputs = shape_predictor(image)
shape_time = time.time() - start
shape_count = len(shape_outputs["instances"])
print(f"   ✅ Found {shape_count} shapes in {shape_time:.3f}s\n")

print(f"⚡ Total inference time: {food_time + shape_time:.3f}s")
print(f"🎯 GPU Speed: {1000*(food_time + shape_time):.1f}ms")
print("="*60 + "\n")

############################################
# Extract Bounding Boxes
############################################
food_instances = food_outputs["instances"].to("cpu")
shape_instances = shape_outputs["instances"].to("cpu")

food_boxes = food_instances.pred_boxes.tensor.numpy()
food_classes = food_instances.pred_classes.numpy()
food_scores = food_instances.scores.numpy()

shape_boxes = shape_instances.pred_boxes.tensor.numpy()
shape_classes = shape_instances.pred_classes.numpy()
shape_scores = shape_instances.scores.numpy()

############################################
# IoU Function
############################################
def compute_iou(boxA, boxB):
    """Compute Intersection over Union (IoU)"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

############################################
# Match Food to Shapes
############################################
print("="*60)
print("Mapping Food Items to Shapes")
print("="*60 + "\n")

iou_threshold = 0.3
matches_found = 0

if len(food_boxes) == 0:
    print("⚠️ No food items detected")
elif len(shape_boxes) == 0:
    print("⚠️ No shapes detected")
else:
    for i, food_box in enumerate(food_boxes):
        food_label = food_metadata.thing_classes[food_classes[i]]
        food_score = food_scores[i]
        
        best_match = None
        best_iou = iou_threshold
        
        for j, shape_box in enumerate(shape_boxes):
            shape_label = shape_metadata.thing_classes[shape_classes[j]]
            iou = compute_iou(food_box, shape_box)
            
            if iou > best_iou:
                best_iou = iou
                best_match = (shape_label, shape_scores[j])
        
        if best_match:
            print(f"✅ {food_label} (conf: {food_score:.2f}) → {best_match[0]} (IoU: {best_iou:.2f})")
            matches_found += 1
        else:
            print(f"⚠️ {food_label} (conf: {food_score:.2f}) → No matching shape")
    
    print(f"\n📊 Summary: {matches_found}/{len(food_boxes)} food items matched to shapes")

print("="*60 + "\n")

############################################
# Visualize Results
############################################
print("🎨 Creating visualizations...\n")

# Food visualization
v1 = Visualizer(image[:, :, ::-1], metadata=food_metadata, scale=1.0)
food_vis = v1.draw_instance_predictions(food_outputs["instances"].to("cpu"))

# Shape visualization
v2 = Visualizer(image[:, :, ::-1], metadata=shape_metadata, scale=1.0)
shape_vis = v2.draw_instance_predictions(shape_outputs["instances"].to("cpu"))

# Display
plt.figure(figsize=(16, 7))

plt.subplot(1, 2, 1)
plt.title(f"Food Detection ({food_count} items)", fontsize=14, fontweight='bold')
plt.imshow(food_vis.get_image())
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f"Shape Detection ({shape_count} shapes)", fontsize=14, fontweight='bold')
plt.imshow(shape_vis.get_image())
plt.axis('off')

plt.tight_layout()
plt.show()

print("✅ Visualization complete\n")

############################################
# Save Results
############################################
save_results = input("Save results? (y/n): ").lower() == 'y'

if save_results:
    output_dir = r"D:\Food\results"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save visualizations
    food_output = os.path.join(output_dir, f"food_{timestamp}.jpg")
    shape_output = os.path.join(output_dir, f"shape_{timestamp}.jpg")
    
    cv2.imwrite(food_output, cv2.cvtColor(food_vis.get_image(), cv2.COLOR_RGB2BGR))
    cv2.imwrite(shape_output, cv2.cvtColor(shape_vis.get_image(), cv2.COLOR_RGB2BGR))
    
    print(f"\n✅ Results saved to: {output_dir}")
    print(f"   📄 {os.path.basename(food_output)}")
    print(f"   📄 {os.path.basename(shape_output)}")
else:
    print("\n⏭️ Results not saved")

print("\n" + "="*60)
print("✅ Detection Complete!")
print("="*60)