"""Where the tokens of an attention policy sit, so visualisation never hard-codes it.

Order of the token axis (attention maps are indexed the same way):

    [ class token | frame 0 tokens | frame 1 tokens | ... | frame K-1 tokens | action 0 .. action K-1 ]

Frames are oldest first (frame K-1 is the newest); each frame contributes
``tokens_per_frame`` tokens (grid cells in row-major order, or a single token for a
whole frame); the action tokens exist only when ``num_action_tokens`` is K.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenLayout:
    num_frames: int
    tokens_per_frame: int
    num_action_tokens: int = 0

    cls_index = 0
    frames_offset = 1

    @property
    def action_offset(self) -> int:
        return self.frames_offset + self.num_frames * self.tokens_per_frame

    @property
    def num_tokens(self) -> int:
        return self.action_offset + self.num_action_tokens

    def frame_slice(self, frame: int) -> slice:
        start = self.frames_offset + frame * self.tokens_per_frame
        return slice(start, start + self.tokens_per_frame)

    def action_index(self, frame: int) -> int:
        if not self.num_action_tokens:
            raise ValueError("this layout has no action tokens")
        return self.action_offset + frame
