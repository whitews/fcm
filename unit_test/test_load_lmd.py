import unittest
from fcm import FCSreader


class FCSReaderLMDTestCase(unittest.TestCase):
    def setUp(self):
        self.fcm = FCSreader('sample_data/coulter.lmd').get_FCMdata()
        
    def test_get_points(self):
        self.assertEqual(self.fcm.shape[0], int(self.fcm.notes.text['tot']))

    def test_get_notes(self):
        self.assertEqual(self.fcm.notes.text['cyt'], 'Cytomics FC 500')
        
    def test_get_second_file(self):
        x = FCSreader('sample_data/coulter.lmd')
        z = x.get_FCMdata()
        y = x.get_FCMdata()
        self.assertEqual(z.shape, y.shape, 'Failed to load second dataset')
        self.assertNotEqual(z[0, 0], y[0, 0], 'Failed to load second dataset')

if __name__ == '__main__':
    suite1 = unittest.makeSuite(FCSReaderLMDTestCase, 'test')
    unittest.main()