import argparse
import glob
import os
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from utils import pad_to_multiple

parser = argparse.ArgumentParser()
parser.add_argument('--map-dir', type=str, required=True)
parser.add_argument('--image-dir', type=str, required=True)
# parser.add_argument('--label-dir', type=str, required=True)
parser.add_argument('--output-dir', type=str, required=True)


def load_image(image_path):
    image = np.load(image_path)
    image = pad_to_multiple(image, 16)
    image = image.clip(-150, 250)
    image -= image.min()
    image /= image.max()
    return image


def main(args):
    # Prepare output for images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Get data
    map_paths = sorted(glob.glob(os.path.join(args.map_dir, "*.npy")))

    for map_path in tqdm(map_paths):
        map_path = pathlib.Path(map_path)
        grad_map = np.load(map_path)

        image_path = os.path.join(args.image_dir, map_path.name)
        image = load_image(image_path)

        slice_index = grad_map.shape[-1] // 2

        for target_class in range(grad_map.shape[0]):
            grad_map_slice = grad_map[target_class, :, :, slice_index]
            image_slice = image[:, :, slice_index]
            alpha_slice = grad_map_slice / grad_map_slice.max() if grad_map_slice.max() > 0 else 0

            plt.figure()
            plt.axis("off")
            plt.imshow(image_slice, cmap="gray", vmin=0, vmax=1)

            plt.imshow(grad_map_slice, cmap='inferno', vmin=0, alpha=alpha_slice)
            plt.colorbar(ticks=[])
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, map_path.stem + f"_{target_class}"))
            plt.close()


if __name__ == '__main__':
    main(parser.parse_args())
