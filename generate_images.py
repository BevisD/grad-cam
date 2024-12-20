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
parser.add_argument('--map-dir', type=str, required=True)
parser.add_argument('--mask-dir', type=str, required=True)
parser.add_argument('--prediction-dir', type=str, required=True)
parser.add_argument('--image-dir', type=str, required=True)
parser.add_argument('--output-dir', type=str, required=True)


def load_file(file_path):
    image = np.load(file_path)
    image = pad_to_multiple(image, 16)
    return image


def main(args):
    # Prepare output for images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mask_dir = os.path.join(script_dir, args.output_dir, "masks")
    map_dir = os.path.join(script_dir, args.output_dir, "maps")
    prediction_dir = os.path.join(script_dir, args.output_dir, "predictions")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(map_dir, exist_ok=True)
    os.makedirs(prediction_dir, exist_ok=True)

    # Get data
    map_paths = sorted(glob.glob(os.path.join(args.map_dir, "*.npy")))
    print(f"Maps Found: {len(map_paths)}")

    cmap = ListedColormap(["black", "red"])

    for map_path in tqdm(map_paths):
        map_path = pathlib.Path(map_path)
        grad_map = np.load(map_path)

        image_path = os.path.join(args.image_dir, map_path.name)
        image = load_file(image_path)

        mask_path = os.path.join(args.mask_dir, map_path.name)
        mask = load_file(mask_path)

        prediction_path = os.path.join(args.prediction_dir, map_path.name)
        prediction = load_file(prediction_path)

        for target_class in range(1, 3):
            if np.count_nonzero(mask == target_class) == 0:
                slice_index = mask.shape[0] // 2
            else:
                slice_index = np.argmax((mask == target_class).sum(axis=(1, 2)))

            grad_map_slice = grad_map[target_class, slice_index, :, :]
            mask_slice = mask[slice_index, :, :]
            image_slice = image[slice_index, :, :]
            prediction_slice = prediction[slice_index, :, :]
            
            grad_map_slice = pad_to_multiple(grad_map_slice, 16).astype(np.float64)

            map_alpha_slice = grad_map_slice / grad_map_slice.max() if grad_map_slice.max() > 0 else 0
            mask_alpha_slice = (mask_slice == target_class) * 0.5
            prediction_alpha_slice = (prediction_slice == target_class) * 0.5

            # Plot Ground Truth
            plt.figure()
            plt.axis("off")
            plt.imshow(image_slice, cmap="gray", vmin=0, vmax=1)
            plt.imshow(mask_slice, cmap=cmap, vmin=0, vmax=1, alpha=mask_alpha_slice)
            plt.tight_layout()
            plt.savefig(os.path.join(mask_dir, pathlib.Path(mask_path).stem + f"_{target_class}"))
            plt.close()

            # Plot Prediction
            plt.figure()
            plt.axis("off")
            plt.imshow(image_slice, cmap="gray", vmin=0, vmax=1)
            plt.imshow(prediction_slice, cmap=cmap, vmin=0, vmax=1, alpha=prediction_alpha_slice)
            plt.tight_layout()
            plt.savefig(os.path.join(prediction_dir, pathlib.Path(prediction_path).stem + f"_{target_class}"))
            plt.close()

            # Plot Grad Map
            plt.figure()
            plt.axis("off")
            plt.imshow(image_slice, cmap="gray", vmin=0, vmax=1)
            plt.imshow(grad_map_slice, cmap='inferno', vmin=0, alpha=map_alpha_slice)
            plt.colorbar(ticks=[])
            plt.tight_layout()
            plt.savefig(os.path.join(map_dir, map_path.stem + f"_{target_class}"))
            plt.close()


if __name__ == '__main__':
    main(parser.parse_args())
