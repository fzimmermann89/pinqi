import torch

class filterDS(torch.utils.data.Dataset):
    def __init__(self, ds, returnIndices=None):
        """
        Return only some data from each Point of the Dataset
        Args:
            ds: Dataset to filter
            returnIndices: Indices of the Properties to pass through. default=None means return all data (Identity)
        """
        self.ds = ds
        self.returnIndices = None if returnIndices is None else [rid if isinstance(rid,slice) else slice(rid,rid+1) for rid in returnIndices ]

    def __getitem__(self, idx):
        data = self.ds[idx]
        if self.returnIndices is None:
            return data
        return sum((data[rid] for rid in self.returnIndices) ,tuple())

    def __len__(self):
        return len(self.ds)