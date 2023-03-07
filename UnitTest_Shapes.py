import unittest as unittest
from ai_library.RandomShapeDataGeneration import ShapeGenerator


class TestShapeGenerator(unittest.TestCase):
    def test_generate_image(self):
        shape = ShapeGenerator(1)
        num_images = 1000
        for i in range(num_images):
            img, label = shape.generate_image()
            if label[0] == 0:
                self.assertEqual('circle', shape.shapes[label[0]])
            elif label[0] == 1:
                self.assertEqual('square', shape.shapes[label[0]])
            elif label[0] == 2:
                self.assertEqual('rectangle', shape.shapes[label[0]])
            else:
                self.fail(f"Invalid label: {label}")
        
        print('Random Generated Data Test Successful')
        


if __name__ == '__main__':
    unittest.main()

