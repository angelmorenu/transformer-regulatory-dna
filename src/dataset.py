from pathlib import Path
import numpy as np


class SequenceDataset:
    """Simple dataset wrapper for preprocessed .npz files.

    The loader is robust to a few common key names used in .npz files (X/y, inputs/labels, arr_0/arr_1).
    This class is defined in a module so it is picklable for multi-worker DataLoader use.
    """
    def __init__(self, path: str):
        npz = np.load(path, allow_pickle=True)
        # common conventions
        if 'X' in npz and 'y' in npz:
            self.inputs = npz['X']
            self.labels = npz['y']
        elif 'inputs' in npz and 'labels' in npz:
            self.inputs = npz['inputs']
            self.labels = npz['labels']
        elif 'x' in npz and 'y' in npz:
            self.inputs = npz['x']
            self.labels = npz['y']
        elif 'arr_0' in npz and 'arr_1' in npz:
            self.inputs = npz['arr_0']
            self.labels = npz['arr_1']
        else:
            # fallback: use first two arrays if possible
            keys = list(npz.keys())
            if len(keys) >= 2:
                self.inputs = npz[keys[0]]
                self.labels = npz[keys[1]]
            else:
                raise ValueError(f'Unrecognized .npz structure: {keys} in {path}')
        assert len(self.inputs) == len(self.labels), 'Inputs and labels length mismatch'

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.labels[idx]
        return {'input': x, 'label': int(y)}
