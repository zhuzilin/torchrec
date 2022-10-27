import unittest

import torch
from torchrec_dynamic_embedding.dense_ps import DensePS
from utils import register_memory_io


register_memory_io()


class TestDensePS(unittest.TestCase):
    def testEvictFetch(self):
        x = torch.rand((5, 5)).cuda()
        y = torch.rand((4, 4)).cuda()
        ps = DensePS("table", "memory://")
        ps.save(["x", "y"], [x, y])
        new_x = torch.zeros_like(x)
        new_y = torch.zeros_like(y)
        ps.load(["y", "x"], [new_y, new_x])
        self.assertTrue(torch.allclose(x, new_x))
        self.assertTrue(torch.allclose(y, new_y))


if __name__ == "__main__":
    unittest.main()
