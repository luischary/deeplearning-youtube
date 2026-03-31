import os
import pathlib
from contextlib import contextmanager

import torch


@contextmanager
def _portable_pathlib_load():
    original_posix_path = pathlib.PosixPath
    original_windows_path = pathlib.WindowsPath

    if os.name != "nt":
        yield
        return

    # Checkpoints salvos em Linux/macOS podem conter objetos PosixPath.
    pathlib.PosixPath = pathlib.WindowsPath
    try:
        yield
    finally:
        pathlib.PosixPath = original_posix_path
        pathlib.WindowsPath = original_windows_path


def load_torch_checkpoint(*args, **kwargs):
    with _portable_pathlib_load():
        return torch.load(*args, **kwargs)


class CheckpointHandler:
    def __init__(self, max_saves, mode: str = "min"):
        self.best_metrics = {}
        self.mode = mode
        self.max_saves = max_saves

    def save_last(self, state_dict, ckp_path):
        torch.save(state_dict, ckp_path)

    def save_checkpoint(self, state_dict, ckp_path, metric):
        if len(self.best_metrics) < self.max_saves:
            self.best_metrics[ckp_path] = metric

            torch.save(state_dict, ckp_path)
        else:
            if self.mode == "min":
                odered = sorted(
                    self.best_metrics.items(), key=lambda x: x[1], reverse=True
                )
                self.best_metrics = {x[0]: x[1] for x in odered}

                substitui = ""
                for p, m in self.best_metrics.items():
                    if m > metric:
                        substitui = p
                        break

                if substitui != "":
                    del self.best_metrics[substitui]
                    os.remove(substitui)
                    self.best_metrics[ckp_path] = metric

                    torch.save(state_dict, ckp_path)
            else:
                odered = sorted(
                    self.best_metrics.items(), key=lambda x: x[1], reverse=False
                )
                self.best_metrics = {x[0]: x[1] for x in odered}

                substitui = ""
                for p, m in self.best_metrics.items():
                    if m < metric:
                        substitui = p
                        break

                if substitui != "":
                    del self.best_metrics[substitui]
                    os.remove(substitui)
                    self.best_metrics[ckp_path] = metric

                    torch.save(state_dict, ckp_path)


class EarlyStopper:
    def __init__(self, mode: "max", patience: int = 3):
        self.mode = mode
        self.best = None
        self.patience = patience
        self.patience_count = 0
        self.stop = False

    def update(self, metric):
        if self.best is None:
            self.best = metric
            return True
        elif self.mode == "max":
            if metric > self.best:
                self.best = metric
                self.patience_count = 0
                return True
            else:
                self.patience_count += 1
        else:
            if metric < self.best:
                self.best = metric
                self.patience_count = 0
                return True
            else:
                self.patience_count += 1

        if self.patience_count > self.patience:
            self.stop = True

        return False
