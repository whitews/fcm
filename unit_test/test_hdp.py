from fcm.statistics import HDPMixtureModel
import unittest
import numpy as np
import numpy.random as npr
from numpy.random import multivariate_normal

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

group_weights1 = [0.4, 0.3, 0.3]
group_weights2 = [.1, .1, .8]


class HDPMixtureModelTestCase(unittest.TestCase):
    def generate_data(self,n=1e4, k=2, ncomps=3, seed=1):
        
        npr.seed(seed)
        data1_concat = []
        data2_concat = []
        labels1_concat = []
        labels2_concat = []
    
        for j in xrange(ncomps):
            mean = gen_mean[j]
            sd = gen_sd[j]
            corr = gen_corr[j]
    
            cov = np.empty((k, k))
            cov.fill(corr)
            cov[np.diag_indices(k)] = 1
            cov *= np.outer(sd, sd)
    
            num1 = int(n * group_weights1[j])
            num2 = int(n * group_weights2[j])
            rvs1 = multivariate_normal(mean, cov, size=(num1,))
            rvs2 = multivariate_normal(mean, cov, size=(num2,))
            data1_concat.append(rvs1)
            data2_concat.append(rvs2)
            labels1_concat.append(np.repeat(j, num1))
            labels2_concat.append(np.repeat(j, num2))
    
        return (
            [
                np.concatenate(labels1_concat),
                np.concatenate(labels2_concat)
            ],
            [
                np.concatenate(data1_concat, axis=0),
                np.concatenate(data2_concat, axis=0)
            ]
        )
        
    def test_mcmc_fitting(self):
        print "starting mcmc"
        true, data = self.generate_data()
        model = HDPMixtureModel(3, 100, 100, 1)
        model.seed = 1
        r = model.fit(data, verbose=10)
        print r.mus
        self.assertEqual(len(r), 2, 'results are the wrong length: %d' % len(r))
        diffs = {}
        for i in gen_mean:
            diffs[i] = np.min(np.abs(r[0].mus-gen_mean[i]), 0)
            self.assertLessEqual(
                np.vdot(diffs[i], diffs[i]),
                1,
                'diff to large: %f' % np.vdot(diffs[i], diffs[i]))
        
        for i in gen_mean:
            diffs[i] = np.min(np.abs(r[1].mus-gen_mean[i]), 0)
            self.assertLessEqual(
                np.vdot(diffs[i], diffs[i]),
                1,
                'diff to large: %f' % np.vdot(diffs[i], diffs[i]))

if __name__ == '__main__':
    unittest.main()