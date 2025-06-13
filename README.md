


# Document_analysis 📝

A Python-based toolkit for **document layout analysis and classification**, leveraging deep learning (CNNs/Transformers) to detect and segment document elements such as text, figures, tables, headers, and footnotes.

## 🚀 Features
- Detects and segments layout components: **titles, text blocks, images, tables**, etc.
- Trains on large-scale datasets like **PubLayNet**.
- End-to-end pipeline: Data preparation ➜ Model training ➜ Inference ➜ Evaluation.
- Supports **state-of-the-art architectures**: Faster R-CNN, Cascade R-CNN, or Transformer-based detectors.
- Metrics: mAP, IoU for layout components.

## 🗂️ Table of Contents
1. [Installation](#installation)  
2. [Dataset](#dataset)  
3. [Usage](#usage)  
4. [Examples](#examples)  
5. [Project Structure](#project-structure)  
6. [Configuration](#configuration)  
7. [Troubleshooting](#troubleshooting)  
8. [License](#license)  
9. [Contact](#contact)

---

## Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/suyashsachdeva/Document_analysis.git
   cd Document_analysis
```

2. (Recommended) Create a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### 1. Data Preparation

Prepare and preprocess data:

```bash
python scripts/prepare_data.py \
  --input_dir data/publaynet/images \
  --anno_dir data/publaynet/annotations \
  --output_dir processed_data
```

### 2. Train the Model

Train a layout detection model:

```bash
python train.py \
  --data_dir processed_data \
  --model_dir models/layout_detector \
  --epochs 30 \
  --batch_size 8 \
  --lr 1e-4
```

### 3. Run Inference

Detect layouts on new PDF or image files:

```bash
python inference.py \
  --model_dir models/layout_detector \
  --input_file samples/sample_page.jpg \
  --output_file results/prediction.json
```

---

## 📸 Examples

Visuals of layout detection overlaid on sample documents can be found in `/results/`.

---

## 🗂️ Project Structure

```text
Document_analysis/
├── data/
│   └── publaynet/
├── processed_data/
├── models/
│   └── layout_detector/
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   └── inference.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Configuration

Customize parameters via CLI flags or config files:

* `--epochs`, `--batch_size`, `--lr`
* Paths: `--data_dir`, `--model_dir`, `--input_file`, `--output_file`
* Backbone model params (ResNet, Transformer)

---

## 🛠️ Troubleshooting

* **CUDA errors**: Ensure CUDA toolkit and GPU drivers are installed correctly.
* **Slow performance**: Reduce batch size or lower backbone resolution.
* **Low accuracy**: Check dataset labels, augmentation pipeline, or model depth.

---


## 📬 Contact

For queries or issues, open an [issue](https://github.com/suyashsachdeva/Document_analysis/issues) or contact **Suyash Sachdeva**.

---
