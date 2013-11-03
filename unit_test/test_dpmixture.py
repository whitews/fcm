"""
Created on Oct 30, 2009

@author: Jacob Frelinger
"""
import unittest
from fcm.statistics import DPCluster, DPMixture
from numpy import array, eye, all, dot
from numpy.testing import assert_array_equal
from numpy.testing.utils import assert_equal


class DPMixtureTestCase(unittest.TestCase):
    def setUp(self):
        self.mu1 = array([0, 0, 0])
        self.sig = eye(3)
        self.mu2 = array([5, 5, 5])

        self.cluster1 = DPCluster(.5/3, self.mu1, self.sig)
        self.cluster2 = DPCluster(.5/3, self.mu2, self.sig)
        self.clusters = [
            self.cluster1,
            self.cluster2,
            self.cluster1,
            self.cluster2,
            self.cluster1,
            self.cluster2
        ]
        self.mix = DPMixture(self.clusters, niter=3, identified=True)

    def tearDown(self):
        pass

    def test_prob(self):
        pnt = array([[1, 1, 1]])
        for i in [self.cluster1, self.cluster2]:
            assert i.prob(pnt) <= 1, 'prob of cluster %s is > 1' % i
            assert i.prob(pnt) >= 0, 'prob of cluster %s is < 0' % i

    def test_mix_prob(self):
        pnt = array([1, 1, 1])
        assert self.mix.prob(pnt)[0] == \
            self.cluster1.prob(pnt), \
            'mixture generates different prob then component 1'
        assert self.mix.prob(pnt)[1] == \
            self.cluster2.prob(pnt), \
            'mixture generates different prob then component 2'

    def test_classify(self):
        pnt = array([self.mu1, self.mu2])
        assert self.mix.classify(pnt)[0] == \
            0, \
            'classify classifies mu1 as belonging to something else'
        assert self.mix.classify(pnt)[1] == \
            1, \
            'classify classifies m21 as belonging to something else'

    def test_make_modal(self):

        modal = self.mix.make_modal()
        assert modal.classify(
            array(
                [
                    self.mu1,
                    self.mu2,
                    self.mu1,
                    self.mu2,
                    self.mu1,
                    self.mu2
                ])).tolist() == [1, 0, 1, 0, 1, 0], 'classify not working'
        #TODO do actual lookup.
        
        modal = self.mix.make_modal(delta=9)
        assert modal.classify(
            array(
                [
                    self.mu1,
                    self.mu2,
                    self.mu1,
                    self.mu2,
                    self.mu1,
                    self.mu2
                ])).tolist() == [0, 0, 0, 0, 0, 0], 'classify not working'

    def test_average(self):
        cluster1 = DPCluster(0.5, self.mu1, self.sig)
        cluster3 = DPCluster(0.5, self.mu1, self.sig)
        cluster5 = DPCluster(0.5, self.mu1, self.sig)
        cluster7 = DPCluster(0.5, self.mu1, self.sig)
        cluster2 = DPCluster(0.5, self.mu2, self.sig)
        cluster4 = DPCluster(0.5, self.mu2, self.sig)
        cluster6 = DPCluster(0.5, self.mu2, self.sig)
        cluster8 = DPCluster(0.5, self.mu2, self.sig)

        mix = DPMixture(
            [
                cluster1,
                cluster2,
                cluster3,
                cluster4,
                cluster5,
                cluster6,
                cluster7,
                cluster8
            ], niter=4)
        avg = mix.average()

        assert len(avg.clusters) == 2
        assert all(avg.mus[0] == self.mu1)
        assert all(avg.mus[1] == self.mu2)
        assert all(avg.sigmas[0] == self.sig)
        assert all(avg.sigmas[1] == self.sig)
        assert avg.pis[0] == 0.5, 'pis should be 0.5 but is %f' % avg.pis()[0]
        assert avg.pis[1] == 0.5, 'pis should be 0.5 but is %f' % avg.pis()[0]

    def test_last(self):
        cluster1 = DPCluster(0.5, self.mu1, self.sig)
        cluster3 = DPCluster(0.5, self.mu1 + 3, self.sig)
        cluster5 = DPCluster(0.5, self.mu1 + 5, self.sig)
        cluster7 = DPCluster(0.5, self.mu1 + 7, self.sig)
        cluster2 = DPCluster(0.5, self.mu2 + 2, self.sig)
        cluster4 = DPCluster(0.5, self.mu2 + 4, self.sig)
        cluster6 = DPCluster(0.5, self.mu2 + 6, self.sig)
        cluster8 = DPCluster(0.5, self.mu2 + 8, self.sig)

        mix = DPMixture(
            [
                cluster1,
                cluster2,
                cluster3,
                cluster4,
                cluster5,
                cluster6,
                cluster7,
                cluster8
            ], niter=4)

        new_r = mix.last()
        assert len(new_r.clusters) == 2
        assert all(new_r.clusters[0].mu == cluster7.mu)
        assert all(new_r.clusters[1].mu == cluster8.mu)

        new_r = mix.last(2)
        assert len(new_r.clusters) == 4
        assert all(new_r.clusters[0].mu == cluster5.mu)
        assert all(new_r.clusters[1].mu == cluster6.mu)
        assert all(new_r.clusters[2].mu == cluster7.mu)
        assert all(new_r.clusters[3].mu == cluster8.mu)

    def test_draw(self):
        x = self.mix.draw(10)
        assert x.shape[0] == 10, "Number of drawed rows is wrong"
        assert x.shape[1] == 3, "number of drawed columns is wrong"

    def test_arithmetic(self):
        adder = 3
        array_adder = array([1, 2, 3])
        mat_adder = 2*eye(3)

        # add
        b = self.mix + adder
        self.assertIsInstance(
            b, DPMixture, 'integer addition return wrong type')
        assert_equal(b.mus[0], self.mix.mus[0] + adder,
                     'integer addition returned wrong value')

        c = self.mix + array_adder
        self.assertIsInstance(
            c, DPMixture, 'array addition return wrong type')
        assert_array_equal(
            c.mus[0],
            self.mix.mus[0] + array_adder,
            'array addition returned wrong value')

        # radd
        b = adder + self.mix
        self.assertIsInstance(
            b, DPMixture, 'integer addition return wrong type')
        assert_array_equal(
            b.mus[0],
            adder + self.mix.mus[0],
            'integer addition returned wrong value')

        c = array_adder + self.mix
        self.assertIsInstance(
            c, DPMixture, 'array addition return wrong type')
        assert_array_equal(
            c.mus[0],
            array_adder + self.mix.mus[0],
            'array addition returned wrong value')

        # sub
        b = self.mix - adder
        self.assertIsInstance(
            b, DPMixture, 'integer subtraction return wrong type')
        assert_array_equal(
            b.mus[0],
            self.mix.mus[0] - adder,
            'integer subtraction returned wrong value')

        c = self.mix - array_adder
        self.assertIsInstance(
            c, DPMixture, 'array subtraction return wrong type')
        assert_array_equal(
            c.mus[0],
            self.mix.mus[0] - array_adder,
            'array subtraction returned wrong value')

        # rsub
        b = adder - self.mix
        self.assertIsInstance(
            b, DPMixture, 'integer subtraction return wrong type')
        assert_array_equal(
            b.mus[0],
            adder - self.mix.mus[0],
            'integer subtraction returned wrong value')

        c = array_adder - self.mix
        self.assertIsInstance(
            c, DPMixture, 'array subtraction return wrong type')
        assert_array_equal(
            c.mus[0],
            array_adder - self.mix.mus[0],
            'array subtraction returned wrong value')

        # mul
        b = self.mix * adder
        self.assertIsInstance(
            b, DPMixture, 'integer multiplication return wrong type')
        assert_array_equal(
            b.mus[0],
            self.mix.mus[0] * adder,
            'integer multiplication returned wrong value')

        c = self.mix * array_adder
        self.assertIsInstance(
            c, DPMixture, 'array multiplication return wrong type')
        assert_array_equal(
            c.mus[0],
            dot(self.mix.mus[0], array_adder),
            'array multiplication returned wrong value')

        d = self.mix * mat_adder
        self.assertIsInstance(
            d, DPMixture, 'array multiplication return wrong type')
        assert_array_equal(
            d.mus[0],
            dot(self.mix.mus[0], mat_adder),
            'array multiplication returned wrong value')

        # rmul
        b = adder * self.mix
        self.assertIsInstance(
            b, DPMixture, 'integer multiplication return wrong type')
        assert_array_equal(
            b.mus[0],
            adder * self.mix.mus[0],
            'integer multiplication returned wrong value')

        c = array_adder * self.mix
        self.assertIsInstance(
            c, DPMixture, 'array multiplication return wrong type')
        assert_array_equal(
            c.mus[0],
            dot(array_adder, self.mix.mus[0]),
            'array multiplication returned wrong value')
        
        d = mat_adder * self.mix
        self.assertIsInstance(
            d, DPMixture, 'array multiplication return wrong type')
        assert_array_equal(
            d.mus[0],
            dot(mat_adder, self.mix.mus[0]),
            'array multiplication returned wrong value')
        
        assert_array_equal(
            d.sigmas[0],
            dot(mat_adder, dot(self.mix.sigmas[0], mat_adder)),
            'array multiplication failed')

    def test_get_item(self):
        assert_equal(self.mu1, self.mix[0].mu, 'getitem failed')
        self.mix[0] = self.cluster2
        assert_equal(self.mu2, self.mix[0].mu, 'getitem failed')
        self.mix[0] = self.cluster1
        
    def test_get_iteration(self):
        self.assertIsInstance(self.mix.get_iteration(2), DPMixture, 
                              'get_iteration failed')
        self.assertEqual(len(self.mix.get_iteration(2).clusters), 2, 
                         'get_iteration return wrong number of clusters')
        self.assertIsInstance(self.mix.get_iteration([0, 2]), DPMixture,
                              'get_iteration failed')
        self.assertEqual(len(self.mix.get_iteration([0, 2]).clusters), 4,
                         'get_iteration return wrong number of clusters')
        
    def test_enumerate_clusters(self):
        for i, j in self.mix.enumerate_clusters():
            self.assertIsInstance(i, int)
            self.assertIs(
                j,
                self.mix[i],
                'failed to return the right cluster when enumerating')
    
    def test_enumerate_pis(self):
        for i, j in self.mix.enumerate_pis():
            self.assertIsInstance(i, int)
            self.assertIs(
                j,
                self.mix[i].pi,
                'failed to return the right pi when enumerating')
    
    def test_enumerate_mus(self):
        for i, j in self.mix.enumerate_mus():
            self.assertIsInstance(i, int)
            self.assertIs(
                j,
                self.mix[i].mu,
                'failed to return the right mean when enumerating')
            
    def test_enumerate_sigmas(self):
        for i, j in self.mix.enumerate_sigmas():
            self.assertIsInstance(i, int)
            self.assertIs(
                j,
                self.mix[i].sigma,
                'failed to return the right covariance when enumerating')

if __name__ == "__main__":
    unittest.main()