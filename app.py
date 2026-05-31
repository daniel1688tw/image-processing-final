"""Gradio web UI for palette-based color transfer.

Run:
    python3 app.py
Then open the URL shown in the terminal.
"""

from __future__ import annotations

import numpy as np
import gradio as gr

from color_transfer import transfer_color
from color_transfer.transfer import TransferConfig


def run_transfer(
    source_img: np.ndarray,
    reference_img: np.ndarray,
    use_segmentation: bool,
    alpha: float,
    max_palette: int,
    bins: int,
    ot_variant: str,
    ot_epsilon: float,
    ot_tau: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Called by Gradio on every click of Run Transfer.

    Gradio passes images as RGB uint8 numpy arrays; OpenCV needs BGR.
    Returns (source, reference, result) all in RGB for display.
    """
    if source_img is None or reference_img is None:
        raise gr.Error("Please upload both a source and a reference image.")

    src_bgr = source_img[..., ::-1].copy()
    ref_bgr = reference_img[..., ::-1].copy()

    cfg = TransferConfig(
        bins=bins,
        max_palette_size=max_palette,
        lighting_alpha=alpha,
        use_segmentation=use_segmentation,
        ot_variant=ot_variant,
        ot_epsilon=ot_epsilon,
        ot_tau=ot_tau,
    )

    result_bgr = transfer_color(src_bgr, ref_bgr, cfg)
    result_rgb = result_bgr[..., ::-1]

    return source_img, reference_img, result_rgb


with gr.Blocks(title="Color Transfer") as demo:
    gr.Markdown("## Color Transfer Demo")
    gr.Markdown(
        "Upload a **source** image and a **reference** image, "
        "then click **Run Transfer**."
    )

    with gr.Row():
        source_input = gr.Image(label="Source", type="numpy")
        reference_input = gr.Image(label="Reference", type="numpy")

    with gr.Accordion("Advanced Configuration", open=False):
        use_segmentation = gr.Checkbox(label="Use Segmentation", value=True)
        alpha = gr.Slider(minimum=0.0, maximum=1.0, value=0.3, step=0.05, label="Lighting Alpha")
        max_palette = gr.Slider(minimum=1, maximum=128, value=32, step=1, label="Max Palette Size")
        bins = gr.Slider(minimum=10, maximum=256, value=100, step=5, label="Histogram Bins")
        
        with gr.Row():
            ot_variant = gr.Dropdown(choices=["emd", "sinkhorn", "unbalanced"], value="emd", label="OT Variant")
            ot_epsilon = gr.Number(value=0.05, label="Sinkhorn/Unbalanced Epsilon")
            ot_tau = gr.Number(value=0.1, label="Unbalanced OT Tau")

    run_btn = gr.Button("▶ Run Transfer", variant="primary")

    with gr.Row():
        out_source = gr.Image(label="Source", interactive=False)
        out_reference = gr.Image(label="Reference", interactive=False)
        out_result = gr.Image(label="Result", interactive=False)

    run_btn.click(
        fn=run_transfer,
        inputs=[
            source_input, reference_input,
            use_segmentation, alpha, max_palette, bins,
            ot_variant, ot_epsilon, ot_tau
        ],
        outputs=[out_source, out_reference, out_result],
    )


if __name__ == "__main__":
    demo.launch()
