import torch
from torch.utils.data import DataLoader
from ai_library.DataGenerator import DataGeneration
import unittest


class TestDataGeneration(unittest.TestCase):
    def setUp(self):
        self.dir_name = "./ai_library/data/shapes/"
        self.image_size = 64
        self.train_dataset = DataGeneration(self.dir_name, self.image_size, train=True)
        self.test_dataset = DataGeneration(self.dir_name, self.image_size, train=False)


    def test_dataset_size(self):
        self.assertEqual(len(self.train_dataset), 10000)
        self.assertEqual(len(self.test_dataset), 2000)

    def test_getitem(self):
        train_loader = DataLoader(self.train_dataset, batch_size=1, shuffle=True)
        for i, (data, target) in enumerate(train_loader):
            self.assertEqual(data.shape, torch.Size([1, 1, self.image_size, self.image_size]))
            self.assertEqual(target.shape, torch.Size([1]))
            self.assertGreaterEqual(target.item(), 0)
            self.assertLessEqual(target.item(), 3)

if __name__ == '__main__':
    unittest.main()