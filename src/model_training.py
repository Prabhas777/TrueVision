# src/model_training.py
"""
Train & evaluate a VGG-13-style CNN on FER-like data.
Saves:
- best model .h5
- training curves (png)
- confusion matrix (png)
- metrics.json
"""

import json
import os
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks

from data_preprocessing import build_datagens


def vgg13(input_shape=(48,48,3), num_classes=7, l2=1e-4, dropout=0.5):
    L2 = regularizers.l2(l2)

    def conv_block(x, filters):
        x = layers.Conv2D(filters, (3,3), padding="same", activation="relu", kernel_regularizer=L2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, (3,3), padding="same", activation="relu", kernel_regularizer=L2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2,2))(x)
        return x

    inputs = layers.Input(shape=input_shape)
    x = conv_block(inputs, 64)
    x = conv_block(x, 128)
    x = conv_block(x, 256)
    x = conv_block(x, 512)

    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation="relu", kernel_regularizer=L2)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(1024, activation="relu", kernel_regularizer=L2)(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model


def plot_curves(history, out_path):
    plt.figure()
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_path, "accuracy.png"))
    plt.close()

    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_path, "loss.png"))
    plt.close()


def plot_confusion(y_true, y_pred, class_names, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label', xlabel='Predicted label',
           title='Confusion Matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(os.path.join(out_path, "confusion_matrix.png"))
    plt.close(fig)


def main(
    data_root: str,
    out_dir: str,
    img_size: Tuple[int,int] = (48,48),
    batch_size: int = 64,
    epochs: int = 20,
    lr: float = 1e-3,
    l2: float = 1e-4,
    dropout: float = 0.5
):
    os.makedirs(out_dir, exist_ok=True)

    train_gen, val_gen, test_gen, class_weights, label_map = build_datagens(
        data_root=data_root,
        img_size=img_size,
        batch_size=batch_size
    )

    num_classes = len(train_gen.class_indices)
    model = vgg13(input_shape=(img_size[0], img_size[1], 3),
                  num_classes=num_classes, l2=l2, dropout=dropout)

    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

    cbs = [
        callbacks.ModelCheckpoint(
            filepath=os.path.join(out_dir, "best_model.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max"
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=cbs
    )

    # Curves
    plot_curves(history, out_dir)

    # Evaluation on test if available
    metrics = {}
    if test_gen is not None:
        test_loss, test_acc = model.evaluate(test_gen, verbose=0)
        metrics["test_loss"] = float(test_loss)
        metrics["test_acc"]  = float(test_acc)

        # Predictions for confusion matrix
        preds = model.predict(test_gen, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        y_true = test_gen.classes
        class_names = [label_map[str(i)] for i in range(len(label_map))]
        plot_confusion(y_true, y_pred, class_names, out_dir)

        # Classification report
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        with open(os.path.join(out_dir, "classification_report.json"), "w") as f:
            json.dump(report, f, indent=2)

    # Save metrics + label map + training params
    metrics.update({
        "epochs_ran": len(history.history["loss"]),
        "val_best_acc": float(max(history.history["val_accuracy"])),
        "train_best_acc": float(max(history.history["accuracy"])),
        "img_size": img_size,
        "batch_size": batch_size,
        "lr": lr,
        "l2": l2,
        "dropout": dropout
    })
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete. Artifacts saved to:", out_dir)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Dataset root containing train/(val/)(test/)")
    ap.add_argument("--out_dir",   required=True, help="Where to save model + plots + metrics")
    ap.add_argument("--img_size",  default="48,48")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs",     type=int, default=20)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--l2",         type=float, default=1e-4)
    ap.add_argument("--dropout",    type=float, default=0.5)
    args = ap.parse_args()

    h, w = map(int, args.img_size.split(","))
    main(
        data_root=args.data_root,
        out_dir=args.out_dir,
        img_size=(h, w),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
        dropout=args.dropout
    )
