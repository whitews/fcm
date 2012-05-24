'''
Created on Oct 30, 2009

@author: Jacob Frelinger
'''
import unittest
from fcm.statistics import DPCluster, DPMixture
from numpy import array, eye, all


class Dp_clusterTestCase(unittest.TestCase):


    def setUp(self):
        self.mu1 = array([0,0,0])
        self.sig = eye(3)
        self.mu2 = array([5,5,5])
        
        self.clst1 = DPCluster(.5, self.mu1, self.sig)
        self.clst2 = DPCluster(.5, self.mu2, self.sig)
        self.mix = DPMixture([self.clst1, self.clst2])


    def tearDown(self):
        pass


    def testprob(self):
        pnt = array([1,1,1])

        for i in [self.clst1, self.clst2]:
            assert i.prob(pnt) <= 1, 'prob of clst %s is > 1' % i
            assert i.prob(pnt) >= 0, 'prob of clst %s is < 0' % i

        
    def testmixprob(self):
        pnt = array([1,1,1])
        assert self.mix.prob(pnt)[0] == self.clst1.prob(pnt), 'mixture generates different prob then compoent 1'
        assert self.mix.prob(pnt)[1] == self.clst2.prob(pnt), 'mixture generates different prob then compoent 2'
        
    def testclassify(self):
        pnt = array([self.mu1, self.mu2])
        assert self.mix.classify(pnt)[0] == 0, 'classify classifys mu1 as belonging to something else'
        assert self.mix.classify(pnt)[1] == 1, 'classify classifys m21 as belonging to something else'

    def testMakeModal(self):

        modal = self.mix.make_modal()
#        modal = ModalDPMixture([self.clst1, self.clst2],
#                                { 0: [0], 1: [1]},
#                                [self.mu1, self.mu2])
        pnt = array([self.mu1, self.mu2])
        assert modal.classify(array([self.mu1, self.mu2, self.mu1, self.mu2, self.mu1, self.mu2])).tolist() == [0,1,0,1,0,1], 'classify not working'
        assert self.mix.classify(self.mu1) == modal.classify(self.mu1), 'derived modal mixture is wrong'
        assert self.mix.classify(pnt)[0] == modal.classify(pnt)[0], 'derived modal mixture is wrong'
        assert self.mix.classify(pnt)[1] == modal.classify(pnt)[1], 'derived modal mixture is wrong'

    def testAverage(self):
        clst1 = DPCluster(0.5, self.mu1, self.sig)
        clst3 = DPCluster(0.5, self.mu1, self.sig)
        clst5 = DPCluster(0.5, self.mu1, self.sig)
        clst7 = DPCluster(0.5, self.mu1, self.sig)
        clst2 = DPCluster(0.5, self.mu2, self.sig)
        clst4 = DPCluster(0.5, self.mu2, self.sig)
        clst6 = DPCluster(0.5, self.mu2, self.sig)
        clst8 = DPCluster(0.5, self.mu2, self.sig)
        
        mix = DPMixture([clst1, clst2, clst3, clst4, clst5, clst6, clst7, clst8], niter=4)
        avg = mix.average()
        
        assert len(avg.clusters) == 2
        assert all(avg.mus()[0] == self.mu1)
        assert all(avg.mus()[1] == self.mu2)
        assert all(avg.sigmas()[0] == self.sig)
        assert all(avg.sigmas()[1] == self.sig)
        assert avg.pis()[0] == 0.5, 'pis should be 0.5 but is %f'% avg.pis()[0]
        assert avg.pis()[1] == 0.5, 'pis should be 0.5 but is %f'% avg.pis()[0]
        
    def testLast(self):
        clst1 = DPCluster(0.5, self.mu1, self.sig)
        clst3 = DPCluster(0.5, self.mu1+3, self.sig)
        clst5 = DPCluster(0.5, self.mu1+5, self.sig)
        clst7 = DPCluster(0.5, self.mu1+7, self.sig)
        clst2 = DPCluster(0.5, self.mu2+2, self.sig)
        clst4 = DPCluster(0.5, self.mu2+4, self.sig)
        clst6 = DPCluster(0.5, self.mu2+6, self.sig)
        clst8 = DPCluster(0.5, self.mu2+8, self.sig)
        
        mix = DPMixture([clst1, clst2, clst3, clst4, clst5, clst6, clst7, clst8], niter=4)
        
        new_r = mix.last()
        assert len(new_r.clusters)==2
        assert all(new_r.clusters[0].mu == clst7.mu)
        assert all(new_r.clusters[1].mu == clst8.mu)
        
        new_r = mix.last(2)
        assert len(new_r.clusters)==4
        assert all(new_r.clusters[0].mu == clst5.mu)
        assert all(new_r.clusters[1].mu == clst6.mu)
        assert all(new_r.clusters[2].mu == clst7.mu)
        assert all(new_r.clusters[3].mu == clst8.mu)
        
        try:
            new_r = mix.last(10)
        except ValueError:
                pass     
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    
