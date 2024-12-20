import argparse
import glob
import os
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm.auto import tqdm

from utils import pad_to_multiple

parser = argparse.ArgumentParser()
parser.add_argument('--mask-dir', type=str, required=True)
parser.add_argument('--image-dir', type=str, required=True)
# parser.add_argument('--label-dir', type=str, required=True)
parser.add_argument('--output-dir', type=str, required=True)


def load_image(image_path):
    image = np.load(image_path)
    image = pad_to_multiple(image, 16)
    return image


def main(args):
    # Prepare output for images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Get data
    mask_paths = sorted(glob.glob(os.path.join(args.mask_dir, "*.npy")))
    print(f"Masks Found: {len(mask_paths)}")

    cmap = ListedColormap(["black", "red"])

    for mask_path in tqdm(mask_paths):
        mask_path = pathlib.Path(mask_path)
        mask = np.load(mask_path)

        image_path = os.path.join(args.image_dir, mask_path.name)
        image = load_image(image_path)

        slice_index = mask.shape[0] // 2

        mask_slice = mask[slice_index, :, :]
        image_slice = image[slice_index, :, :]

        for target_class in range(1, 3):
            alpha_slice = (mask_slice == target_class) * 0.5

            plt.figure()
            plt.axis("off")
            plt.imshow(image_slice, cmap="gray", vmin=0, vmax=1)

            plt.imshow(mask_slice, cmap=cmap, vmin=0, vmax=1, alpha=alpha_slice)
            plt.colorbar(ticks=[])
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, mask_path.stem + f"_{target_class}"))
            plt.close()


if __name__ == '__main__':
    main(parser.parse_args())
