#!/usr/bin/env python3
"""
predict_my_digits.py

Predict digit images using a CNN trained on MNIST (improved_digit_cnn.pth).
Automatically inverts and normalizes images to match MNIST style.

Usage:
  python predict_my_digits.py --model improved_digit_cnn.pth --images digit2.jpg digit4.jpg digit8.jpg
"""

import argparse
import os
from typing import List
from PIL import Image, ImageOps
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from improved_digit_cnn import CNN


def image_to_mnist_tensor(path: str, device: torch.device, show=False):
    """
    Convert a photo of a handwritten digit into an MNIST-style tensor.
    NOTE: normalization matches training: transforms.Normalize((0.1307,), (0.3081,))
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    # Load as grayscale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")

    # Invert if background is light (MNIST has white digit on black background)
    if np.mean(img) > 127:
        img = cv2.bitwise_not(img)

    # Denoise and normalize contrast
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    # Otsu thresholding / binarization
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find digit bounding box
    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y+h, x:x+w]
    else:
        # Fallback: center crop
        h, w = img.shape
        size = min(h, w)
        top  = (h - size) // 2
        left = (w - size) // 2
        img  = img[top:top+size, left:left+size]

    # Pad to square
    h, w = img.shape
    diff = abs(h - w)
    if h > w:
        img = cv2.copyMakeBorder(img, 0, 0, diff // 2, diff - diff // 2, cv2.BORDER_CONSTANT, value=0)
    elif w > h:
        img = cv2.copyMakeBorder(img, diff // 2, diff - diff // 2, 0, 0, cv2.BORDER_CONSTANT, value=0)

    # Add padding around digit (like MNIST margins) then resize to 28x28
    img = cv2.copyMakeBorder(img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    if show:
        cv2.imshow(f"Preprocessed: {os.path.basename(path)}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Convert to float32 in [0,1] and apply MNIST normalization
    arr = img.astype(np.float32) / 255.0
    arr = (arr - 0.1307) / 0.3081

    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,28,28)
    return tensor.to(device)


def load_trained_model(model_path: str, device: torch.device):
    """
    Load state_dict into CNN. Uses non-strict load and prints a warning if any keys differ.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = CNN()
    model.to(device)

    sd = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        print(f"Warning: strict load failed ({e}). Falling back to non-strict load.")
        model.load_state_dict(sd, strict=False)

    model.eval()
    return model


def predict_images(model_path: str, image_paths: List[str], device_str: str = "cpu", show=False):
    device = torch.device(device_str)
    model  = load_trained_model(model_path, device)

    results = []
    for path in image_paths:
        basename = os.path.basename(path)
        try:
            tensor = image_to_mnist_tensor(path, device, show=show)
            with torch.no_grad():
                logits = model(tensor)
                probs  = F.softmax(logits, dim=1)
                conf, pred = probs.max(dim=1)
            digit = pred.item()
            confidence = conf.item()
            results.append(f"Prediction for {basename}: {digit} (conf={confidence:.2%})")
        except Exception as e:
            results.append(f"Prediction for {basename}: ERROR - {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Predict digit images using improved_digit_cnn.pth.")
    parser.add_argument("--model", type=str, default="improved_digit_cnn.pth", help="Path to model file")
    parser.add_argument(
        "--images",
        nargs="*",
        default=["datasets/digits/digit2.jpg", "datasets/digits/digit4.jpg", "datasets/digits/digit6.jpg", "datasets/digits/digit8.jpg"],
        help="Image files to predict",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu or cuda)")
    parser.add_argument("--show", action="store_true", help="Show preprocessed images for debugging")
    args = parser.parse_args()

    lines = predict_images(args.model, args.images, args.device, show=args.show)
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
