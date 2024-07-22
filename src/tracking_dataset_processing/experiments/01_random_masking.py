# Standard libraries
from pathlib import Path

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


#------------#
# Parameters #
#------------#
SAVE_FIGURES = True

DATA_PATH = Path("data/webdatasets")
DATASET_NAME = "gso_1M"
CHUNK_ID = "00000000"
MIN_LINE_SIZE = 60
MIN_LINE_SIZE_HALF = 8
#------------#


output_path = DATA_PATH / f"{DATASET_NAME}_clines" / CHUNK_ID
sample_names = [o.name.split(".")[0] for o in output_path.glob("*.clines.rgb.npy")]


for sample_name in tqdm(sample_names[:3]):
    
    rgb = np.load(output_path / f"{sample_name}.clines.rgb.npy")
    seg = np.load(output_path / f"{sample_name}.clines.seg.npy")
    
    mask = np.isnan(seg)
    
    clip_width = np.random.randint(0, MIN_LINE_SIZE - MIN_LINE_SIZE_HALF)
    
    if clip_width == 0:
        m = np.bitwise_and(
            np.isclose(
                np.sum(np.diff(seg, axis=-1), axis=-1), -1
            ),
            np.isclose(
                np.sum(np.abs(np.diff(seg, axis=-1)), axis=-1), 1
            ),
        )
        mask = ~np.repeat(
            m[:, np.newaxis], seg.shape[1] - clip_width * 2, axis=1
        )
    
    else:
        mask[:, :clip_width] = True
        mask[:, -clip_width:] = True

        # mask lines where there is more than one change or the change is not to -1
        m = np.bitwise_and(
            np.isclose(
                np.sum(np.diff(seg[:, clip_width:-clip_width], axis=-1), axis=-1), -1
            ),
            np.isclose(
                np.sum(np.abs(np.diff(seg[:, clip_width:-clip_width], axis=-1)), axis=-1), 1
            ),
        )
        mask[:, clip_width:-clip_width] = ~np.repeat(
            m[:, np.newaxis], seg.shape[1] - clip_width * 2, axis=1
        )

    
    masked_rgb = rgb.copy()
    masked_rgb[mask] = 0
    masked_rgb = masked_rgb.astype(np.uint8)

    masked_seg = 255 * seg.copy()
    masked_seg[mask] = 127
    masked_seg = masked_seg.astype(np.uint8)

    if SAVE_FIGURES:
        
        fig: plt.Figure
        fig, axes = plt.subplots(1, 2, squeeze=False, sharex=True, sharey=True)
        ax: plt.Axes = axes[0, 0]
        ax.imshow(masked_rgb)
        ax.axis("off")
        ax.set_title("RGB")
        ax: plt.Axes = axes[0, 1]
        ax.imshow(masked_seg, cmap="bwr")
        ax.axis("off")
        ax.set_title("Segmentation")
        
        fig.savefig(output_path / f"{sample_name}.masked.png")
