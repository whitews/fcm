from fcm.statistics import DPMixtureModel, DPMixture
import unittest
from numpy import array, eye
import numpy as np
from numpy.random import multivariate_normal
from fcm import FCMdata, FCMCollection

gen_mean = {
    0: [0, 5],
    1: [-5, 0],
    2: [5, 0]
}

gen_sd = {
    0: [0.5, 0.5],
    1: [.5, 1],
    2: [1, .25]
}

gen_corr = {
    0: 0.5,
    1: -0.5,
    2: 0
}

group_weights = [0.4, 0.3, 0.3]


def fit_one(args):
    x, name = args
    m = DPMixtureModel(nclusts=8, niter=100,  burnin=0, last=1)
    r = m.fit(x, verbose=10)
    return r


class DPMixtureModelTestCase(unittest.TestCase):
    def generate_data(self, n=1e4, k=2, ncomps=3, seed=1):
        
        np.random.seed(seed)
        data_concat = []
        labels_concat = []
    
        for j in xrange(ncomps):
            mean = gen_mean[j]
            sd = gen_sd[j]
            corr = gen_corr[j]
    
            cov = np.empty((k, k))
            cov.fill(corr)
            cov[np.diag_indices(k)] = 1
            cov *= np.outer(sd, sd)
    
            num = int(n * group_weights[j])
            rvs = multivariate_normal(mean, cov, size=(num,))
    
            data_concat.append(rvs)
            labels_concat.append(np.repeat(j, num))
    
        return (np.concatenate(labels_concat),
                np.concatenate(data_concat, axis=0))
        
    def test_list_fitting(self):
        true1, data1 = self.generate_data()
        true2, data2 = self.generate_data()

        model = DPMixtureModel(3, 2000, 100, 1, type='BEM')
        rs = model.fit([data1, data2])
        assert(len(rs) == 2)
        for r in rs:
            diffs = {}
            for i in gen_mean:
                diffs[i] = np.min(np.abs(r.mus-gen_mean[i]), 0)
                assert(np.vdot(diffs[i], diffs[i]) < 1)

        fcm1 = FCMdata('test_fcm1', data1, ['fsc', 'ssc'], [0, 1])
        fcm2 = FCMdata('test_fcm2', data2, ['fsc', 'ssc'], [0, 1])
        
        c = FCMCollection('fcms', [fcm1, fcm2])
        
        rs = model.fit(c)
        assert(len(rs) == 2)
        for r in rs:
            diffs = {}
            for i in gen_mean:
                diffs[i] = np.min(np.abs(r.mus-gen_mean[i]), 0)
                assert(np.vdot(diffs[i], diffs[i]) < 1)
    
    def test_bem_fitting(self):
        true, data = self.generate_data()
        
        model = DPMixtureModel(3, 2000, 100, 1, type='BEM')
        model.seed = 1
        r = model.fit(data, verbose=False)

        diffs = {}
        for i in gen_mean:
            diffs[i] = np.min(np.abs(r.mus-gen_mean[i]), 0)
            assert(np.vdot(diffs[i], diffs[i]) < 1)

    def test_mcmc_fitting(self):
        true, data = self.generate_data()
        
        model = DPMixtureModel(3, 100, 100, 1)
        model.seed = 1
        r = model.fit(data, verbose=10)

        diffs = {}
        for i in gen_mean:
            diffs[i] = np.min(np.abs(r.mus-gen_mean[i]), 0)
            assert(np.vdot(diffs[i], diffs[i]) < 1)

    def test_reference(self):
        true, data = self.generate_data()
        
        model = DPMixtureModel(3, 100, 100, 1)
        model.seed = 1
        model.load_ref(array(true))
        r = model.fit(data, verbose=True)

        diffs = {}
        for i in gen_mean:
            diffs[i] = np.abs(r.mus[i]-gen_mean[i])
            assert(np.vdot(diffs[i], diffs[i]) < 1)

        model.load_ref(r)
        r = model.fit(data, verbose=True)

        diffs = {}
        for i in gen_mean:
            diffs[i] = np.abs(r.mus[i]-gen_mean[i])
            assert(np.vdot(diffs[i], diffs[i]) < 1)

    def setUp(self):
        self.mu = array([0, 0])
        self.sig = eye(2)
        self.pnts = multivariate_normal(self.mu, self.sig, size=(1000,))
        self.k = 16
        self.niter = 10
        self.model = DPMixtureModel(self.k, self.niter, 0, 1)

    def test_model(self):
        r = self.model.fit(self.pnts, verbose=False)
        assert(isinstance(r, DPMixture))
        mus = r.mus
        assert(mus.shape == (16, 2))
        
    def test_model_prior(self):
        self.model.load_mu(self.mu.reshape(1, 2))
        self.model.load_sigma(self.sig.reshape(1, 2, 2))
        r = self.model.fit(self.pnts, verbose=False)
        assert(isinstance(r, DPMixture))
        mus = r.mus
        assert(mus.shape == (16, 2))
        
    def test_model_datatypes(self):
        r = self.model.fit(self.pnts.astype('int'))
        self.assertIsInstance(r, DPMixture, 'failed to fit integer data')

        r = self.model.fit(self.pnts.astype('float'))
        self.assertIsInstance(r, DPMixture, 'failed to fit float data')
        
        r = self.model.fit(self.pnts.astype('double'))
        self.assertIsInstance(r, DPMixture, 'failed to fit double data')
        
if __name__ == '__main__':
    unittest.main(verbosity=2)