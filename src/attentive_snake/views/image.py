"""The ``"image"`` view payload: an RGB screen picture as a base64-encoded PNG.

::

    {"width": W, "height": H,                  # picture size in pixels
     "png": "iVBORw0KGgo..."}                  # base64 PNG (no "data:" prefix)

A browser shows it with ``<img src="data:image/png;base64,{png}">``. Build it with
``image_payload(rgb)`` from an ``(H, W, 3)`` uint8 array (Gymnasium's
``render_mode="rgb_array"``). An attention heatmap in the watch frame keeps the
observation's own grid shape; how to place it over the picture is up to the display.
``views.image_window`` shows these payloads in a pygame window (``scripts/watch.py``);
``png_bytes`` turns a payload back into the PNG file's bytes.
"""

import base64
import struct
import zlib
from typing import Any

import numpy as np


def image_payload(rgb: np.ndarray) -> dict[str, Any]:
    """The image payload of an ``(H, W, 3)`` uint8 RGB array."""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"expected an (H, W, 3) uint8 RGB array, got {rgb.shape} {rgb.dtype}.")
    height, width, _ = rgb.shape
    return {"width": width, "height": height, "png": base64.b64encode(encode_png(rgb)).decode("ascii")}


def png_bytes(payload: dict[str, Any]) -> bytes:
    """The PNG file of an image payload."""
    return base64.b64decode(payload["png"])


def encode_png(rgb: np.ndarray) -> bytes:
    """A minimal PNG (8-bit RGB, no filtering) of an ``(H, W, 3)`` uint8 array."""
    height, width, _ = rgb.shape
    rows = np.concatenate([np.zeros((height, 1), dtype=np.uint8), rgb.reshape(height, width * 3)], axis=1)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"".join(
        [b"\x89PNG\r\n\x1a\n", _chunk(b"IHDR", header), _chunk(b"IDAT", zlib.compress(rows.tobytes())), _chunk(b"IEND", b"")]
    )


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
