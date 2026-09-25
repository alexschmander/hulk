"""Image conversion and texture-sensitive quality metrics."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from .common import Record


def luma(image: Record) -> NDArray[np.uint8]:
    raw = np.frombuffer(
        image["data"], np.uint8, count=image["height"] * image["step"]
    )
    y = raw.reshape(image["height"], image["step"])[:, : image["width"]]
    return np.rint(
        np.clip((y.astype(np.float32) - 16) * (255 / 219), 0, 255)
    ).astype(np.uint8)


def rgb709(image: Record) -> NDArray[np.uint8]:
    """Limited-range BT.709, matching the source's declared matrix/range.

    Nearest chroma expansion; floating point rounding may differ from Rust's
    fixed-point balanced converter by a small number of channel levels.
    """
    w, h, stride = image["width"], image["height"], image["step"]
    planes = np.frombuffer(image["data"], np.uint8).reshape(h * 3 // 2, stride)
    y = (planes[:h, :w].astype(np.float32) - 16) * (255 / 219)
    uv = (planes[h:, :w].astype(np.float32) - 128) * (255 / 224)
    u = uv[:, 0::2].repeat(2, 0).repeat(2, 1)
    v = uv[:, 1::2].repeat(2, 0).repeat(2, 1)
    rgb = np.stack(
        (y + 1.5748 * v, y - 0.187324273 * u - 0.468124273 * v, y + 1.8556 * u),
        axis=-1,
    )
    return np.rint(np.clip(rgb, 0, 255)).astype(np.uint8)


def sharpness(gray: NDArray[np.uint8]) -> float:
    h, w = gray.shape
    gray = cv2.resize(
        gray, (320, round(h * 320 / w)), interpolation=cv2.INTER_AREA
    )
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return float(lap.var())


def quality(rgb: NDArray[np.uint8]) -> Record:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    small = cv2.resize(
        gray, (320, round(h * 320 / w)), interpolation=cv2.INTER_AREA
    )
    lap = cv2.Laplacian(small, cv2.CV_32F)
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1)
    tiles = [
        float(t.var())
        for row in np.array_split(lap, 3, axis=0)
        for t in np.array_split(row, 3, axis=1)
    ]
    return {
        "laplacian": float(lap.var()),
        "lower_half_laplacian": float(lap[len(lap) // 2 :].var()),
        "tenengrad": float(np.mean(gx * gx + gy * gy)),
        "tile_laplacian": tiles,
        "dark_fraction": float(np.mean(gray < 10)),
        "bright_fraction": float(np.mean(gray > 245)),
    }
