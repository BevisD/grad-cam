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


class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer = dict(self.model.named_modules())[target_layer_name]
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self):
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)

        return heatmap.cpu().detach().numpy()


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
    image = image.clip(-150, 250)
    image -= image.min()
    image /= image.max()
    return torch.from_numpy(np.expand_dims(image, (0, 1)))


def main(args):
    # Hardware setup
    device = torch.device('cuda' if torciuda.is_available() else 'cpu')
    dtype = torch.float16 if args.high_precision else torch.bfloat16
    print(f"Device: {device}")

    # Prepare output for maps
    script_dir = os.path.dirname(os.path.abspath(__file__))
    maps_dir = os.path.join(script_dir, args.output_dir)
    os.makedirs(maps_dir, exist_ok=True)

    model = load_model(args)
    model.to(device)

    # Config Grad-CAM
    target_layer_name = "final_conv"
    target_classes = [0, 1, 2]
    grad_cam = GradCAM(model, target_layer_name)

    # Get data
    image_paths = sorted(glob.glob(os.path.join(args.data_dir, "*.npy")))

    # Generate maps
    for image_path in image_paths:
        filename = pathlib.Path(image_path).name
        print(filename)

        image = load_image(image_path)
        image.to(device)

        cam_classes = []
        for target_class in target_classes:
            with autocast(dtype=dtype, enabled=torch.cuda.is_available()):
                output = model(image)

            loss = output[0, target_class, :, :].mean()

            model.zero_grad()
            loss.backward()

            cam = grad_cam.generate_cam()
            cam_classes.append(cam)

        cam_array = np.stack(cam_classes, axis=0)
        np.save(os.path.join(maps_dir, filename), cam_array)


if __name__ == '__main__':
    main(parser.parse_args())
