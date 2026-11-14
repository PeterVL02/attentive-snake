"""The watch-frame payload contracts of the two view kinds."""

import base64
import struct
import zlib

import numpy as np
import pytest

from attentive_snake.views.board import board_payload
from attentive_snake.views.image import image_payload
from attentive_snake.watch.frame import WatchFrame


def test_board_payload() -> None:
    assert board_payload(5, 4, [(2, 1), (1, 1)], (4, 3)) == {"width": 5, "height": 4, "snake": [[2, 1], [1, 1]], "food": [4, 3]}
    assert board_payload(5, 4, [], None)["food"] is None


def test_image_payload_is_a_decodable_png() -> None:
    rgb = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    payload = image_payload(rgb)
    assert (payload["width"], payload["height"]) == (3, 2)
    png = base64.b64decode(payload["png"])
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = struct.unpack(">II", png[16:24])
    assert (width, height) == (3, 2)
    idat_length = struct.unpack(">I", png[33:37])[0]
    rows = np.frombuffer(zlib.decompress(png[41 : 41 + idat_length]), dtype=np.uint8).reshape(2, 1 + 9)
    assert (rows[:, 0] == 0).all() and (rows[:, 1:].reshape(2, 3, 3) == rgb).all()
    with pytest.raises(ValueError):
        image_payload(np.zeros((2, 3), dtype=np.uint8))


def test_frame_merges_the_payload_and_rejects_clashes() -> None:
    def frame(view):
        return WatchFrame("board", view, 1.0, 2, None, False, None, None, "", 0)

    merged = frame({"width": 1, "height": 1}).to_dict()
    assert merged["width"] == 1 and merged["score"] == 1.0 and merged["view_kind"] == "board"
    with pytest.raises(ValueError, match="score"):
        frame({"width": 1, "score": 3}).to_dict()


def test_image_window_shows_a_payload_scaled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import pygame

    from attentive_snake.views.image_window import ImageWindow

    window = ImageWindow(3, 2, scale=10)
    window.render(image_payload(np.full((2, 3, 3), 200, dtype=np.uint8)), caption="hi")
    assert pygame.display.get_surface().get_size() == (30, 20)
    assert pygame.display.get_surface().get_at((15, 10))[:3] == (200, 200, 200)
    window.close()


def test_element_weights_are_a_generic_frame_key() -> None:
    frame = WatchFrame("image", {"width": 1, "height": 1, "png": ""}, 0.0, 0, None, False, None, None, "", 0, [0.5, 1.0])
    assert frame.to_dict()["element_weights"] == [0.5, 1.0]
    with pytest.raises(ValueError, match="clash"):
        WatchFrame("image", {"width": 1, "height": 1, "element_weights": 1}, 0.0, 0, None, False, None, None, "", 0).to_dict()
