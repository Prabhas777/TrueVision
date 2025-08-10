# src/data_preprocessing.py
"""
Data loaders + utilities for FER-style folders:
<dataset_root>/
  train/
    angry/, disgust/, fear/, happy/, neutral/, sad/, surprise/
  val/      (optional)
  test/

Outputs:
- Keras ImageDataGenerators (train/val/test)
- class weights dict (for imbalanced classes)
- label_map.json (index -> class name)
"""

import json
import os
from typing import Tuple, Optional, Dict

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight


DEFAULT_CLASSES = ['angry','disgust','fear','happy','neutral','sad','surprise']


def build_datagens(
    data_root: str,
    img_size: Tuple[int, int] = (48, 48),
    batch_size: int = 64,
    use_val_split_if_missing: float = 0.2,
    seed: int = 42
):
    """
    Creates train/val/test iterators. If 'val/' doesn't exist, split from 'train/'.
    """
    train_dir = os.path.join(data_root, "train")
    val_dir   = os.path.join(data_root, "val")
    test_dir  = os.path.join(data_root, "test")

    # Augmentation for train, light rescale for val/test
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.08,
        height_shift_range=0.08,
        shear_range=0.08,
        zoom_range=0.08,
        horizontal_flip=True,
        validation_split=use_val_split_if_missing if not os.path.isdir(val_dir) else 0.0
    )
    plain_datagen = ImageDataGenerator(rescale=1./255)

    if os.path.isdir(val_dir):
        train_gen = train_datagen.flow_from_directory(
            directory=train_dir,
            target_size=img_size,
            color_mode="rgb",   # convert 48x48 gray to 3-channel for VGG-style models
            class_mode="categorical",
            batch_size=batch_size,
            shuffle=True,
            seed=seed
        )
        val_gen = plain_datagen.flow_from_directory(
            directory=val_dir,
            target_size=img_size,
            color_mode="rgb",
            class_mode="categorical",
            batch_size=batch_size,
            shuffle=False
        )
    else:
        # Split from train dir
        train_gen = train_datagen.flow_from_directory(
            directory=train_dir,
            target_size=img_size,
            color_mode="rgb",
            class_mode="categorical",
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
            subset="training"
        )
        val_gen = train_datagen.flow_from_directory(
            directory=train_dir,
            target_size=img_size,
            color_mode="rgb",
            class_mode="categorical",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            subset="validation"
        )

    test_gen = None
    if os.path.isdir(test_dir):
        test_gen = plain_datagen.flow_from_directory(
            directory=test_dir,
            target_size=img_size,
            color_mode="rgb",
            class_mode="categorical",
            batch_size=batch_size,
            shuffle=False
        )

    # Save label map for consistency
    label_map = {v: k for k, v in train_gen.class_indices.items()}  # idx -> class
    with open(os.path.join(data_root, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    # Compute class weights using training samples
    y_indices = train_gen.classes  # numeric labels for each file in generator
    classes = np.unique(y_indices)
    class_weights_vec = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_indices
    )
    class_weights = {int(cls): float(w) for cls, w in zip(classes, class_weights_vec)}

    return train_gen, val_gen, test_gen, class_weights, label_map


if __name__ == "__main__":
    # Quick smoke test
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Path to dataset root containing train/ (and optionally val/, test/)")
    ap.add_argument("--img_size", default="48,48")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    h, w = map(int, args.img_size.split(","))
    tg, vg, eg, cw, lm = build_datagens(args.data_root, (h, w), args.batch_size)
    print("Train batches:", len(tg), "Val batches:", len(vg), "Test batches:", len(eg) if eg else 0)
    print("Class weights:", cw)
    print("Label map:", lm)
