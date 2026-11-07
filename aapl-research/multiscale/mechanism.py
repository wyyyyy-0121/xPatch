"""Fixed shared-horizon objective for the planned, not-yet-run mechanism test."""
import torch


def shared_path_loss(prediction: torch.Tensor, actual: torch.Tensor, horizons=(5, 15, 30, 60)):
    """Equal weight for each prefix path after training-only target scaling."""
    if prediction.ndim != 2 or prediction.shape != actual.shape or not len(horizons):
        raise ValueError("Prediction and target paths must have matching dimensions")
    if tuple(sorted(set(horizons))) != tuple(horizons) or horizons[0] < 1 or horizons[-1] != prediction.shape[1]:
        raise ValueError("Horizons must be increasing and end at the full path length")
    return torch.stack([(prediction[:, :h] - actual[:, :h]).square().mean() for h in horizons]).mean()
