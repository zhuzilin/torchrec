import os
from typing import List

import torch

try:
    torch.ops.load_library(os.path.join(os.path.dirname(__file__), "tde_cpp.so"))
except Exception as ex:
    print(f"File tde_cpp.so not found {ex}")


__all__ = ["DensePS"]


class DensePS:
    def __init__(self, table_name: str, url: str):
        """
        DensePS table of an embedding table.

        Args:
            table_name: name of the table.
            url: url of the PS.
        """
        self._ps = torch.classes.tde.DensePS(table_name, url)

    def save(self, keys: List[str], tensors: List[torch.Tensor]):
        assert len(keys) == len(tensors)
        wrapped_keys = torch.classes.tde.StringList()
        wrapped_tensors = torch.classes.tde.TensorList()
        for i in range(len(keys)):
            wrapped_keys.append(keys[i])
            wrapped_tensors.append(tensors[i].data)
        self._ps.save(wrapped_keys, wrapped_tensors)

    def load(self, keys: List[str], tensors: List[torch.Tensor]):
        assert len(keys) == len(tensors)
        wrapped_keys = torch.classes.tde.StringList()
        wrapped_tensors = torch.classes.tde.TensorList()
        for i in range(len(keys)):
            wrapped_keys.append(keys[i])
            wrapped_tensors.append(tensors[i].data)
        self._ps.load(wrapped_keys, wrapped_tensors)
