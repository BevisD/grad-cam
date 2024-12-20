import argparse
import glob
import os
import pathlib

from monai.networks.nets import BasicUNet
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from utils import pad_to_multiple

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained-path', type=str, required=True)
parser.add_argument('--data-dir', type=str, required=True)
parser.add_argument('--output-dir', type=str, required=True)
parser.add_argument("--high-precision", action="store_true")


def load_model(args):
    # Load trained weights
    weights = torch.load(args.pretrained_path, map_location=torch.device('cpu'))["state_dict"]

    # Rename weights
    keys = list(weights.keys())
    for key in keys:
        if key.startswith("unet."):
            new_key = key.removeprefix("unet.")
            weights[new_key] = weights.pop(key)

    model = BasicUNet(
        in_channels=1,
        out_channels=3,
    )
    missing, unexpected = model.load_state_dict(weights, strict=False)
    print(f"Missing Weights: {len(missing)}")
    print(f"Unexpected Weights: {len(unexpected)}")

    return model


def load_image(image_path):
    image = np.load(image_path)
    image = pad_to_multiple(image, 16)
    return torch.from_numpy(np.expand_dims(image, (0, 1)))


def main(args):
    # Hardware setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16 if args.high_precision else torch.bfloat16
    print(f"Device: {device}")

    # Prepare output for predictions
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    model = load_model(args)
    model.eval()
    model = model.to(device)

    # Get data
    image_paths = sorted(glob.glob(os.path.join(args.data_dir, "*.npy")))

    # Generate maps
    for image_path in image_paths:
        filename = pathlib.Path(image_path).name
        print(filename, flush=True)

        image = load_image(image_path)
        image = image.to(device)

        with autocast(dtype=dtype, enabled=torch.cuda.is_available()):
            output = model(image)

        output = output.detach().cpu().numpy()

        np.save(os.path.join(output_dir, filename), output)


if __name__ == '__main__':
    main(parser.parse_args())
