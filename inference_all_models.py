import os
import time
import torch
import numpy as np
from mmseg.apis import init_model, inference_model
import cv2
from thop import profile
import torchmetrics
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from collections import defaultdict
import subprocess
import warnings
import glob
import pandas as pd

# Suppress TorchScript deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning)

CITYSCAPES_DIR = "cityscapes"
TEST_IMAGES_DIR = os.path.join(CITYSCAPES_DIR, "leftImg8bit/test")
VAL_ANN_DIR = os.path.join(CITYSCAPES_DIR, "gtFine/test")

MODEL_CONFIGS = {
    "FCN": {"mim_config": "fcn_r50-d8_4xb2-40k_cityscapes-512x1024"},
    "U-Net": {"mim_config": "unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024"},
    "DeepLabV3+": {"mim_config": "deeplabv3plus_r101-d8_4xb2-40k_cityscapes-512x1024"},
    "PSPNet": {"mim_config": "pspnet_r50-d8_4xb2-40k_cityscapes-512x1024"},
    "HRNet+OCR": {"mim_config": "ocrnet_hr48_4xb2-80k_cityscapes-512x1024"},
    "SETR": {"mim_config": "setr_vit-l_naive_8xb1-80k_cityscapes-768x768"},
    "SegFormer-B5": {"mim_config": "segformer_mit-b5_8xb1-160k_cityscapes-1024x1024"},
    "BiSeNet V2": {"mim_config": "bisenetv2_fcn_4xb4-160k_cityscapes-1024x1024"},
    "Fast-SCNN": {"mim_config": "fast_scnn_8xb4-160k_cityscapes-512x1024"},
    "DDRNet-23": {"mim_config": "ddrnet_23-slim_in1k-pre_2xb6-120k_cityscapes-1024x1024"}
}

CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
    "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]

results = defaultdict(dict)

def download_weights(model_name, mim_config):
    weight_dir = "weights"
    os.makedirs(weight_dir, exist_ok=True)
    config_path = os.path.join(weight_dir, f"{mim_config}.py")
    if not os.path.exists(config_path) or not any(f.endswith(".pth") for f in os.listdir(weight_dir) if mim_config.split('_')[0] in f):
        print(f"\U0001F4E6 Downloading weights and config for {model_name} via MIM...")
        subprocess.run([
            "mim", "download", "mmsegmentation",
            "--config", mim_config,
            "--dest", weight_dir
        ], check=True)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found after download.")
    checkpoint_pattern = os.path.join(weight_dir, "*.pth")
    if model_name == "U-Net":
        checkpoint_files = [f for f in glob.glob(checkpoint_pattern) if "unet_s5-d16" in f]
    else:
        checkpoint_files = [f for f in glob.glob(checkpoint_pattern) if mim_config.split('_')[0] in f]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint file found for {model_name} using pattern {mim_config.split('_')[0]}*.pth")
    checkpoint_path = checkpoint_files[0]
    print(f"✅ Using checkpoint: {checkpoint_path}")
    return config_path, checkpoint_path

def compute_flops_params(model, input_shape=(3, 1024, 2048)):
    model.eval()
    input_tensor = torch.randn(1, *input_shape).cuda()
    flops, params = profile(model, inputs=(input_tensor,))
    return flops / 1e9, params / 1e6

def measure_fps(model, input_shape=(3, 1024, 2048), num_runs=50):
    model.eval()
    input_tensor = torch.randn(1, *input_shape).cuda()
    with torch.no_grad():
        for _ in range(10): model(input_tensor)
        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)
        start.record()
        for _ in range(num_runs): model(input_tensor)
        end.record()
        torch.cuda.synchronize()
        latency = start.elapsed_time(end) / num_runs
        return 1000 / latency, latency

def measure_memory(model, input_shape=(3, 1024, 2048)):
    model.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        model(torch.randn(1, *input_shape).cuda())
    return torch.cuda.memory_allocated() / 1e9

def evaluate_model(model, val_images, val_labels):
    predictions, ground_truths = [], []
    for img_path, label_path in zip(val_images, val_labels):
        result = inference_model(model, img_path)
        pred_mask = result.pred_sem_seg.data.cpu().numpy()
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if label is None: continue
        label[label > 18] = 255
        predictions.append(pred_mask)
        ground_truths.append(label)
    predictions = torch.tensor(np.array(predictions)).flatten().to(device)
    ground_truths = torch.tensor(np.array(ground_truths)).flatten().to(device)
    mask = ground_truths != 255
    predictions = predictions[mask]
    ground_truths = ground_truths[mask]
    iou_metric = torchmetrics.JaccardIndex(num_classes=len(CITYSCAPES_CLASSES), task="multiclass").to(device)
    mIoU = iou_metric(predictions, ground_truths).item() * 100
    per_class_iou = {}
    for cls_idx, cls_name in enumerate(CITYSCAPES_CLASSES):
        mask_pred = (predictions == cls_idx).float()
        mask_gt = (ground_truths == cls_idx).float()
        intersection = (mask_pred * mask_gt).sum()
        union = mask_pred.sum() + mask_gt.sum() - intersection
        per_class_iou[cls_name] = (intersection / (union + 1e-6)).item() * 100
    return mIoU, per_class_iou

val_images, val_labels = [], []
for city in os.listdir(TEST_IMAGES_DIR):
    for file in os.listdir(os.path.join(TEST_IMAGES_DIR, city)):
        if file.endswith("_leftImg8bit.png"):
            img_path = os.path.join(TEST_IMAGES_DIR, city, file)
            label_path = os.path.join(VAL_ANN_DIR, city, file.replace("_leftImg8bit.png", "_gtFine_labelIds.png"))
            if os.path.exists(label_path):
                val_images.append(img_path)
                val_labels.append(label_path)

val_images, val_labels = val_images[:500], val_labels[:500]  # Uncomment for testing with a smaller dataset
if len(val_images) != len(val_labels): raise ValueError("Mismatch between images and labels.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for model_name, cfg in MODEL_CONFIGS.items():
    print(f"\n🚀 Running {model_name}...")
    try:
        config_path, checkpoint_path = download_weights(model_name, cfg["mim_config"])
        model = init_model(config_path, checkpoint_path, device=device)
        flops, params = compute_flops_params(model)
        fps, latency = measure_fps(model)
        memory = measure_memory(model)
        mIoU, per_class_iou = evaluate_model(model, val_images, val_labels)
        results[model_name].update({
            "FLOPs (G)": flops, "Params (M)": params,
            "FPS": fps, "Latency (ms)": latency,
            "Memory (GB)": memory, "mIoU (%)": mIoU
        })
        results[model_name].update({f"{cls} IoU (%)": iou for cls, iou in per_class_iou.items()})
    except Exception as e:
        print(f"❌ Error running {model_name}: {e}")
        results[model_name]["Error"] = str(e)

# === Plotting ===
df = pd.DataFrame(results).T

all_columns = df.columns.tolist()
class_iou_cols = [col for col in all_columns if "IoU (%)" in col and not col.startswith("mIoU")]
miou_col = [col for col in all_columns if col == "mIoU (%)"]
other_cols = [col for col in all_columns if col not in class_iou_cols + miou_col]
df = df[class_iou_cols + miou_col + other_cols]

df.to_csv("all_model_results.csv")

metrics_to_plot = ["mIoU (%)", "FPS", "Latency (ms)", "FLOPs (G)", "Params (M)", "Memory (GB)"]

# Radar Chart
from math import pi

def plot_radar_chart(df, metrics):
    labels = metrics
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    cmap = plt.get_cmap("tab10")
    for idx, (model, row) in enumerate(df.iterrows()):
        values = row[labels].tolist()
        values += values[:1]
        ax.plot(angles, values, label=model, color=cmap(idx))
        ax.fill(angles, values, alpha=0.1, color=cmap(idx))
    ax.set_title("Radar Chart of Model Performance", size=15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels([])
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig("Radar_Chart_Model_Performance.png")
    plt.show()

# Heatmap

def plot_heatmap(df, metrics):
    fig, ax = plt.subplots(figsize=(10, 6))
    data = df[metrics].copy()
    norm = Normalize(vmin=np.min(data.values), vmax=np.max(data.values))
    im = ax.imshow(data.values, cmap="coolwarm", aspect="auto")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_yticklabels(df.index)
    plt.colorbar(im, ax=ax)
    ax.set_title("Heatmap of Model Metrics", fontsize=14)
    plt.tight_layout()
    plt.savefig("Heatmap_Model_Performance.png")
    plt.show()

# Bar Charts

def plot_bar_charts(df, metrics):
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        df_sorted = df.sort_values(by=metric, ascending=("Latency" in metric))
        bars = plt.bar(df_sorted.index, df_sorted[metric], color=plt.cm.viridis(np.linspace(0, 1, len(df_sorted))))
        plt.title(f"{metric} Comparison")
        plt.xlabel("Model")
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"BarChart_{metric.replace(' (%)', '').replace(' ', '_')}.png")
        plt.show()

plot_radar_chart(df, metrics_to_plot)
plot_heatmap(df, metrics_to_plot)
plot_bar_charts(df, metrics_to_plot)

# Save CSV
df.to_csv("all_model_results.csv")