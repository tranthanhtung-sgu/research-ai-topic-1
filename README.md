# 🧠 Research AI Topic 1 – Real-time Semantic Segmentation for ADAS

This project focuses on real-time semantic segmentation using MMSegmentation for freight trucks on rural Australian roads. It includes benchmarking multiple models on the Cityscapes dataset with a clean environment setup.

---

## 📦 Features

- ⚡ Benchmarking multiple segmentation models (FCN, DeepLabV3+, SegFormer, etc.)
- 🚛 Targeted for real-time performance in Advanced Driver-Assistance Systems (ADAS)
- 🗂️ Supports Cityscapes dataset and custom evaluation
- 🔍 Includes FPS, mIoU, memory usage, and FLOPs analysis

---

## 🛠️ Environment Setup

This project uses **Conda** to manage dependencies. To get started:

### 📁 1. Clone the repository

```bash
git clone https://github.com/tranthanhtung-sgu/research-ai-topic-1.git
cd research-ai-topic-1
```

---

### 🐍 2. Create the Conda environment

```bash
conda env create -f environment.yml
```

💡 This will install all required packages, including PyTorch, MMSegmentation, OpenCV, and more.

---

### 🚀 3. Activate the environment

```bash
conda activate research-ai-topic-1
```

🔁 You can rename the environment by changing the first line of `environment.yml`:

```yaml
name: research-ai-topic-1
```

---

### ⚠️ 4. Troubleshooting

If you face issues creating the environment (especially on Windows), try:

```bash
conda env create -f environment.yml --no-builds
```

Or install pip packages manually:

```bash
pip install -r requirements.txt
```

---

## 📁 Dataset Setup (Optional)

The project uses the [Cityscapes dataset](https://www.cityscapes-dataset.com/). Due to size, it's not included in this repository.

1. Download Cityscapes manually (registration required)  
2. Place it inside the project like this:

```
research-ai-topic-1/
├── cityscapes/
│   ├── leftImg8bit/
│   └── gtFine/
```

3. Update dataset paths in your config files or scripts if needed.

---

## 🚦 How to Run

To perform inference using all selected models, run:

```bash
python inference_all_models.py
```

Ensure the `weights/` folder contains both:
- `.pth` checkpoint files
- Corresponding `.py` MMSegmentation config files

---

## 📂 Project Structure

```
research-ai-topic-1/
├── inference_all_models.py
├── weights/
│   ├── *.pth
│   └── *.py
├── cityscapes/ 
├── environment.yml
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📜 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## 🙏 Acknowledgements

- Built with [OpenMMLab MMSegmentation](https://github.com/open-mmlab/mmsegmentation)  
- Cityscapes dataset: © Cityscapes Consortium
