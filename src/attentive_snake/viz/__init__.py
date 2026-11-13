"""Model architecture diagrams from the nn.Module tree."""

from attentive_snake.viz.attention import (
    attention_rollout,
    cls_attention_frames,
    cls_attention_grid,
    cls_attention_over_frames,
)
from attentive_snake.viz.diagram import diagram, save_diagram
from attentive_snake.viz.summaries import register_summary

__all__ = ["diagram", "save_diagram", "register_summary", "cls_attention_grid", "attention_rollout", "cls_attention_frames", "cls_attention_over_frames"]
