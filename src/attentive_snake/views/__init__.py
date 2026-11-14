"""What a watched game looks like on screen: the JSON payloads of the two view kinds.

A game declares one ``view_kind`` in its ``GameSpec`` and its ``GameEnv.view()``
returns the matching payload for the current state. Both payloads have ``width`` and
``height``; their other keys must not collide with the generic watch-frame keys
(``score steps event done heatmap frame_weights element_weights caption episode view_kind``), because
``WatchFrame.to_dict`` merges the payload into the top level of the frame.

* ``"board"`` (``views.board``): a small grid with a player body and one target cell,
  drawn cell by cell (Snake). The attention heatmap is drawn over the cells.
* ``"image"`` (``views.image``): an RGB picture of the screen as a base64 PNG (e.g.
  ``env.render()`` of a Gymnasium game).

Displays: the browser draws payloads itself; ``views.board_window`` draws a board in a
pygame window (``scripts/watch.py`` and Snake's human play), ``views.image_window`` shows
an image payload in one (``scripts/watch.py``). Importing this package does
not import pygame.
"""

from typing import Literal

type ViewKind = Literal["board", "image"]
VIEW_KINDS: tuple[str, ...] = ("board", "image")
