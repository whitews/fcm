"""
Unit test framework
"""

import unittest

# include all the TestCase imports here
from test_Annotation import FCMAnnotationTestCase
from test_FCM_collection import FCMCollectionTestCase
from test_FCM_data import FCMDataTestCase
from test_load_lmd import FCSreaderLMDTestCase
from test_transforms import FCMtransformTestCase
from test_load_fcs import FCSreaderTestCase
from test_tree import TreeTestCase
from test_subsample import SubsampleTestCase
from test_dpmixture import Dp_mixtureTestCase
from test_dpcluster import Dp_clusterTestCase
from test_mdpmixture import ModalDp_clusterTestCase
from test_mvnpdf import mvnpdfTestCase
from test_cluster import DPMixtureModelTestCase
from test_hdp import HDPMixtureModel_TestCase
from test_hdpmixture import HDPMixtureTestCase
from test_mhdpmixture import ModalHDp_clusterTestCase
from test_data_align import DiagAlignTestCase
from test_ordereddpmixture import OrderedDp_mixtureTestCase
from test_cluster_align import ClusterAlignTestCase

if __name__ == "__main__":
    unittest.main()