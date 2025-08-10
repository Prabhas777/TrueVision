Facial Emotion Recognition for Visually Impaired Assistance
A deep learning system that recognizes facial emotions in images or video frames and is designed to provide assistive feedback for visually impaired users. The model is trained on FER-style datasets and implements a VGG-13–inspired CNN with L2 regularization, dropout, learning-rate scheduling, and early stopping. The repo includes reproducible training code, evaluation artifacts (confusion matrix, metrics), and Jupyter notebooks documenting experiments.

Features
VGG-13–style CNN (Keras/TensorFlow) with BatchNorm, L2, and Dropout

Robust data pipeline with augmentation and class-imbalance weighting

Reproducible training: saves best model, learning curves, metrics, and confusion matrix

Clear separation of concerns: preprocessing + training scripts in src/

Notebooks for exploratory work and reports for documentation

Ready to extend with real-time inference (OpenCV) and Grad-CAM explainability

Tech Stack
Model/Training: TensorFlow / Keras, NumPy, scikit-learn, Matplotlib

Data Loading & Augmentation: Keras ImageDataGenerator

Optional (real-time & explainability): OpenCV, Grad-CAM

Repository Structure
```
facial_emotion_recognition/
│
├── src/
│   ├── data_preprocessing.py     # Data loading, transformations, train/test split
│   └── model_training.py         # Training loop, evaluation, saving models
│
├── notebooks/
│   ├── checkpoint.ipynb         # Initial experiments & EDA
│   └── Final.ipynb              # optimized model
│
├── reports/
│   ├── checkpoint_report.pdf
│   └── Final_report.pdf
│
├── data/
│   └── README.md                 # Instructions to obtain dataset
│
├── scripts/
│   └── download_data.py          # Dataset download helper (e.g., from Kaggle)
│
├── .gitignore
├── requirements.txt
└── README.md
```
Dataset Layout
This project expects a FER-style folder layout. If you use FER-2013 or a similar dataset, organize it like:

'''
DATASET_ROOT/
├── train/
│   ├── angry/      *.png|*.jpg
│   ├── disgust/    ...
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
├── val/            # optional (if absent, validation split is taken from train/)
│   └── ...
└── test/           # optional but recommended for final evaluation
    └── ...
If val/ is missing, the script will create a validation split from train/ (default 20%).
'''
Setup

# Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
On Apple Silicon, you may prefer tensorflow-macos instead of tensorflow.

Train & Evaluate

# From repo root
python src/model_training.py \
  --data_root /path/to/DATASET_ROOT \
  --out_dir runs/vgg13_fer \
  --epochs 20 \
  --img_size 48,48 \
  --batch_size 64 \
  --lr 1e-3 \
  --l2 1e-4 \
  --dropout 0.5
Outputs (saved in --out_dir)
best_model.h5 — best checkpoint by validation accuracy

accuracy.png, loss.png — training curves

classification_report.json — per-class precision/recall/F1

confusion_matrix.png — confusion matrix on test set (if provided)

metrics.json — summary (val best acc, test acc/loss, hyperparams)

label_map.json (written to DATASET_ROOT/) — class index → class name map

How It Works (Under the Hood)
Data pipeline: src/data_preprocessing.py builds Keras generators for train/val/test, applies standard augmentations (rotation/shift/shear/zoom/flip), rescales pixels, computes class weights, and saves a consistent label map.

Model: src/model_training.py defines a VGG-13–style network with four conv blocks (64→128→256→512), two dense layers (1024 units), and softmax output.

Training: Adam optimizer, categorical cross-entropy, early stopping, ReduceLROnPlateau, and model checkpointing on val_accuracy.

Evaluation: If test/ exists, evaluates and saves confusion matrix + classification report.

Reproducing Notebook Results
The original exploratory work lives in:

notebooks/checkpoint1.ipynb (data prep & baselines)

notebooks/checkpoint2.ipynb (training/evaluation refinements)

These mirror the final code in src/ and can be used for experimentation or visual analysis.

Next Steps (Optional Enhancements)
Real-time demo: Add a small OpenCV script to run live webcam inference with the saved model.

Explainability: Add Grad-CAM heatmaps to visualize salient facial regions for each prediction.

Streamlit app: Build a quick UI to upload an image or use the webcam, then deploy on Hugging Face Spaces.

Commands Cheat Sheet

# Quick smoke test of data loaders
python src/data_preprocessing.py --data_root /path/to/DATASET_ROOT --img_size 48,48 --batch_size 64

# Full training run
python src/model_training.py --data_root /path/to/DATASET_ROOT --out_dir runs/vgg13_fer --epochs 20
Requirements (pip)
See requirements.txt. Typical baseline:

tensorflow
numpy
matplotlib
scikit-learn
opencv-python
Pillow
tqdm
License
Add your preferred license (e.g., MIT) if you plan to make this public/open-source.

Acknowledgments
This project draws on standard FER practices and CNN architectures (e.g., VGG). Datasets like FER-2013 or RAF-DB are commonly used for research and benchmarking.
