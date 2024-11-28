import numpy as np

__all__ = ["calc_response_acc",
           "pad_to_multiple"]


def calc_response_acc(y_pred, y, threshold=0.0):
    assert y.ndim == 1 and y.size() == y_pred.size()
    y_pred = y_pred > threshold
    return (y == y_pred).sum().item() / y.size(0)


def pad_to_multiple(tensor: np.ndarray, n: int):
    new_shape = n * np.ceil(np.divide(tensor.shape, n))
    padding = np.subtract(new_shape, tensor.shape)
    l_padding = (padding // 2).astype(int)
    r_padding = (padding - l_padding).astype(int)

    pad_vector = list(zip(l_padding, r_padding))
    return np.pad(tensor, pad_vector)
