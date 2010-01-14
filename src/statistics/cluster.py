'''
Created on Oct 30, 2009

@author: jolly
'''

from enthought.traits.api import HasTraits
from numpy import zeros, outer, sum

from cdp import cdpcluster
from dp_cluster import DPCluster, DPMixture

class DPMixtureModel(HasTraits):
    '''
    Fits a DP Mixture model to a fcm dataset.
    
    '''


    def __init__(self,fcmdata, nclusts, itter=1000, burnin= 100, last= 5):
        '''
        DPMixtureModel(fcmdata, nclusts, itter=1000, burnin= 100, last= 5)
        fcmdata = a fcm data object
        nclusts = number of clusters to fit
        itter = number of mcmc itterations
        burning = number of mcmc burnin itterations
        last = number of mcmc itterations to draw samples from
        
        '''
        pnts = fcmdata.view()
        self.m = pnts.mean(0)
        self.s = pnts.std(0)
        self.data = (pnts-self.m)/self.s
        
        self.nclusts = nclusts
        self.itter = itter
        self.burnin = burnin
        self.last = last
        
        self.d = self.data.shape[1]
        self.pi = zeros((nclusts*last))
        self.mus = zeros((nclusts*last, self.d))
        self.sigmas = zeros((nclusts*last, self.d, self.d))
        
        self._run = False
        
    def fit(self, verbose=False):
        self.cdp = cdpcluster(self.data)
        self.cdp.setT(self.nclusts)
        self.cdp.setJ(1)
        self.cdp.setBurnin(self.burnin)
        self.cdp.setIter(self.itter-self.last)
        if verbose:
            self.cdp.setVerbose(True)
        self.cdp.run()
        
        self._run = True #we've fit the mixture model
        
        idx = 0
        for i in range(self.last):
            for j in range(self.nclusts):
                self.pi[idx] = self._getpi(j)
                self.mus[idx,:] = self._getmu(j)
                self.sigmas[idx,:,:] = self._getsigma(j)
                idx+=1
        
        
                
    def _getpi(self, idx):
        return self.cdp.getp(idx)
    
    def _getmu(self,idx):
        tmp = zeros(self.d)
        for i in range(self.d):
            tmp[i] = self.cdp.getMu(idx,i)
            
        return tmp*self.s + self.m
    
    def _getsigma(self, idx):
        tmp = zeros((self.d,self.d))
        for i in range(self.d):
            for j in range(self.d):
                tmp[i,j] = self.cdp.getSigma(idx,i,j)    
        return tmp*outer(self.s, self.s)
        
        
    def get_results(self):
        
        self.pi = self.pi/sum(self.pi)
        if self._run:
            rslts = []
            for i in range(self.last * self.nclusts):
                rslts.append(DPCluster(self.pi[i],self.mus[i], self.sigmas[i]))
        
        return DPMixture(rslts)
            