import unittest
from fcm import FCSreader
from fcm import loadFCS


class FCSReaderTestCase(unittest.TestCase):
    def setUp(self):
        self.fcm = FCSreader('sample_data/3FITC_4PE_004.fcs').get_FCMdata()
        
    def test_get_points(self):
        self.assertEqual(self.fcm.shape[0], int(self.fcm.notes.text['tot']))

    def test_get_notes(self):
        self.assertEqual(self.fcm.notes.text['cyt'], 'FACScan')
        
    @staticmethod
    def test_load_fcs():
        for unused in range(100):
            loadFCS('sample_data/3FITC_4PE_004.fcs', transform=None)

    @staticmethod
    def test_load_fcs_from_memory():
        import io
        with open('sample_data/3FITC_4PE_004.fcs') as f:
            mem_file = io.BytesIO(f.read())
            loadFCS(mem_file)

if __name__ == '__main__':
    suite1 = unittest.makeSuite(FCSReaderTestCase, 'test')
    unittest.main()