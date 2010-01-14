import unittest
from fcm.statistics import DPCluster, ModalDPMixture
from numpy import array, eye


class ModalDp_clusterTestCase(unittest.TestCase):


    def setUp(self):
        self.mu1 = array([0,0,0])
        self.sig = eye(3)
        self.mu2 = array([5,5,5])
        
        self.clst1 = DPCluster(.25, self.mu1, self.sig)
        self.clst2 = DPCluster(.25, self.mu1, self.sig)
        self.clst3 = DPCluster(.5, self.mu2, self.sig)
        self.mix = ModalDPMixture([self.clst1, self.clst2, self.clst3], {0 : [0,1], 1 : [2]})


    def tearDown(self):
        pass


    def testprob(self):
        pnt = array([1,1,1])

        for i in [self.clst1, self.clst2]:
            assert i.prob(pnt)[0] <= 1, 'prob of clst %s is > 1' % i
            assert i.prob(pnt)[0] >= 0, 'prob of clst %s is < 0' % i

        
    def testmixprob(self):
        pnt = array([1,1,1])
        assert self.mix.prob(pnt)[0] == (self.clst1.prob(pnt)[0]+self.clst2.prob(pnt)[0]), 'mixture generates different prob then compoent 1'
        assert self.mix.prob(pnt)[1] == self.clst3.prob(pnt)[0], 'mixture generates different prob then compoent 2'
        
    def testclassify(self):
        pnt = array([self.mu1, self.mu2])
        assert self.mix.classify(pnt)[0] == 0, 'classify classifys mu1 as belonging to something else'
        assert self.mix.classify(pnt)[1] == 1, 'classify classifys m21 as belonging to something else'


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()