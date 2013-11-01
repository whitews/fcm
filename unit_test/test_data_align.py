import unittest
from fcm.statistics import DPMixture, DPCluster
from fcm.alignment import DiagonalAlignData, CompAlignData, FullAlignData
import numpy as np
import numpy.testing as npt


class DiagAlignTestCase(unittest.TestCase):
    def setUp(self):
        self.mu1 = np.array([0, 0, 0])
        self.sig = 2 * np.eye(3)
        self.mu2 = np.array([5, 5, 5])

        self.clust1 = DPCluster(.5, self.mu1, self.sig)
        self.clust2 = DPCluster(.5, self.mu2, self.sig)
        self.clusters = [self.clust1, self.clust2]
        self.x = DPMixture(self.clusters, niter=1, identified=True)

        self.diag = DiagonalAlignData(self.x, size=100000)
        self.comp = CompAlignData(self.x, size=100000)
        self.full = FullAlignData(self.x, size=200000)

    def test_diag_align(self):
        y = self.x + np.array([1, -1, 1])
        a, b, f, s, m = self.diag.align(y, solver='ralg')
        assert s, 'failed to converge'
        npt.assert_array_almost_equal(a, np.eye(3), decimal=1)
        npt.assert_array_almost_equal(b, np.array([-1, 1, -1]), decimal=1)

    def test_comp_align(self):
        m = np.array([[1, 0, .2], [0, 1, 0], [0, 0, 1]])
        y = self.x * m
        a, b, f, s, _ = self.comp.align(y)
        assert s, 'failed to converge'
        npt.assert_array_almost_equal(a, np.linalg.inv(m), decimal=1)
        npt.assert_array_almost_equal(b, np.array([0, 0, 0]), decimal=1)
        npt.assert_array_almost_equal((y * a).mus, self.x.mus, decimal=1)

    def test_full_align(self):
        m = np.array([[1, 0, .2], [0, 1, 0], [0, 0, 1]])
        y = self.x * m
        a, b, f, s, _ = self.full.align(y)
        assert s, 'failed to converge'
        npt.assert_array_almost_equal(a, np.linalg.inv(m), decimal=1)
        npt.assert_array_almost_equal(b, np.array([0, 0, 0]), decimal=1)
        npt.assert_array_almost_equal(((y * a) + b).mus, self.x.mus, decimal=1)

    def test_full_align_exclude(self):
        m = np.array([[1, 0, .2], [0, 1, 0], [0, 0, 1]])
        y = self.x * m
        x0 = np.hstack((np.eye(3).flatten(), np.zeros(3)))
        full = FullAlignData(self.x, size=200000, exclude=[0])
        a, b, f, s, _ = full.align(y, x0=x0)
        npt.assert_array_almost_equal(a[0], np.array([1, 0, 0]), decimal=1)
        npt.assert_array_almost_equal(a[:, 0], np.array([1, 0, 0]), decimal=1)

if __name__ == '__main__':
    suite1 = unittest.makeSuite(DiagAlignTestCase, 'test')

    unittest.main()
