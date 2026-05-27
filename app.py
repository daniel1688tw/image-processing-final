"""Gradio web UI for palette-based color transfer.

Run:
    python3 app.py
Then open the URL shown in the terminal.
"""

from __future__ import annotations

import numpy as np
import gradio as gr

from color_transfer import transfer_color


def run_transfer(
    source_img: np.ndarray,
    reference_img: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Called by Gradio on every click of Run Transfer.

    Gradio passes images as RGB uint8 numpy arrays; OpenCV needs BGR.
    Returns (source, reference, result) all in RGB for display.
    """
    if source_img is None or reference_img is None:
        raise gr.Error("Please upload both a source and a reference image.")

    src_bgr = source_img[..., ::-1].copy()
    ref_bgr = reference_img[..., ::-1].copy()

    result_bgr = transfer_color(src_bgr, ref_bgr)
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

    run_btn = gr.Button("▶ Run Transfer", variant="primary")

    with gr.Row():
        out_source = gr.Image(label="Source", interactive=False)
        out_reference = gr.Image(label="Reference", interactive=False)
        out_result = gr.Image(label="Result", interactive=False)

    run_btn.click(
        fn=run_transfer,
        inputs=[source_input, reference_input],
        outputs=[out_source, out_reference, out_result],
    )


if __name__ == "__main__":
    demo.launch()
