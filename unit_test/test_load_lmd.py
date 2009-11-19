import unittest
from io import FCSreader

class FCSreaderLMDTestCase(unittest.TestCase):
    def setUp(self):
        self.fcm = FCSreader('../sample_data/coulter.lmd').get_FCMdata()
        
    def testGetPnts(self):
        self.assertEqual(self.fcm.shape, (18110, 8))

    def testGetNotes(self):
        self.assertEqual(self.fcm.notes.text['cyt'], 'Cytomics FC 500')

if __name__ == '__main__':
    suite1 = unittest.makeSuite(FCSreaderLMDTestCase,'test')

    unittest.main()
