# Standard libraries
from pathlib import Path
from typing import Tuple

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def get_valid_clines(
    rgb: np.ndarray,
    seg: np.ndarray,
    max_line_size_half: int = 60,
    min_line_size_half: int = 8,
    lines_padding: str = "zero",  # "repeat" or "zero"
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract valid correspondence lines from stacked RGB lines and the corresponding
    segmentation masks. The length of the lines is randomly chosen. Lines are padded
    in order to keep the same length after the clipping (with zeros or repeating the
    first and last values).

    Args:
        rgb (np.ndarray): Original RGB correspondence lines. Shape (N, L, 3) and dtype
            np.uint8.
        seg (np.ndarray): Original lines segmentation masks. Shape (N, L) and dtype
            np.uint8.
        max_line_size_half (int, optional): Half of the maximum line size to clip the
            lines. Defaults to 60.
        min_line_size_half (int, optional): Half of the minimum line size to clip the
            lines. Defaults to 8.
        lines_padding (str, optional): Padding to apply to the clipped lines. Can be
            "repeat" or "zero". Defaults to "zero".

    Raises:
        ValueError: If an invalid lines_padding is provided.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing the clipped RGB lines (shape
            (N', L, 3), where N' <= N, and dtype np.uint8) and the
            corresponding segmentation masks (shape (N', L) and dtype np.uint8). For the
            segmentation masks, a value of 127 is used to keep track of the clipped
            points.
    """
    # Get a random clip width
    clip_width = np.random.randint(0, max_line_size_half - min_line_size_half)
    
    if clip_width == 0:
        # Keep only the lines where the change is 255
        lines_to_keep = np.isclose(
                np.sum(np.diff(seg, axis=-1), axis=-1), 255  # 0 - 1
        )
    else:
        # Keep only the lines where the change is 255
        lines_to_keep = np.isclose(
                np.sum(
                    np.diff(seg[:, clip_width:-clip_width], axis=-1), axis=-1
                ), 255  # 0 - 1
        )
        
        if lines_padding == "zero":
            rgb[:, :clip_width] = 0
            rgb[:, -clip_width:] = 0
        elif lines_padding == "repeat":
            # Broadcast the first and last columns to the clipped columns
            rgb[:, :clip_width] = rgb[:, clip_width, None]
            rgb[:, -clip_width:] = rgb[:, -clip_width - 1, None]
        else:
            raise ValueError(f"Invalid lines_padding: {lines_padding}")
        
        # Mask the points which are outside the image or in the border
        seg[:, :clip_width] = 127
        seg[:, -clip_width:] = 127
    
    # Keep only the lines of interest
    rgb = rgb[lines_to_keep]
    
    seg *= 255
    # Keep only the lines of interest
    seg = seg[lines_to_keep]
    
    return rgb, seg


if __name__ == "__main__":
    
    #------------------------------------------#
    # Parameters                               #
    #------------------------------------------#
    SAVE_FIGURES = True
    DATA_PATH = Path("data/webdatasets")
    DATASET_NAME = "gso_1M"
    CHUNK_ID = "00000000"
    MAX_LINE_SIZE_HALF = 60
    MIN_LINE_SIZE_HALF = 8
    #------------------------------------------#

    output_path = DATA_PATH / f"{DATASET_NAME}_clines" / CHUNK_ID
    sample_names = [o.name.split(".")[0] for o in output_path.glob("*.clines.rgb.npy")]

    for sample_name in tqdm(sample_names[:3]):

        rgb = np.load(output_path / f"{sample_name}.clines.rgb.npy")
        seg = np.load(output_path / f"{sample_name}.clines.seg.npy")

        rgb, seg = get_valid_clines(
            rgb,
            seg,
            MAX_LINE_SIZE_HALF,
            MIN_LINE_SIZE_HALF,
            lines_padding="repeat",
        )

        if SAVE_FIGURES:

            fig: plt.Figure
            fig, axes = plt.subplots(1, 2, squeeze=False, sharex=True, sharey=True)
            ax: plt.Axes = axes[0, 0]
            ax.imshow(rgb)
            ax.axis("off")
            ax.set_title("RGB")
            ax: plt.Axes = axes[0, 1]
            ax.imshow(seg, cmap="bwr")
            ax.axis("off")
            ax.set_title("Segmentation")

            fig.savefig(output_path / f"00_{sample_name}.masked.png")
