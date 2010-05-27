from distutils.core import setup, Extension
from numpy import get_include

cdp_extension = Extension('fcm.statistics._cdp',
                          sources = [ 'src/statistics/cdp_ext/%s' %i for i in ['Model.cpp',
                                    'SpecialFunctions.cpp',
                                    'bandmat.cpp',
                                    'cdp.cpp',
                                    'cdp2.cpp',
                                    'cdpbase.cpp',
                                    'cdpcluster.cpp',
                                    'cdpcluster_wrap.cpp',
                                    'cdpp.cpp',
                                    'cdpprior.cpp',
                                    'cdpresult.cpp',
                                    'cholesky.cpp',
                                    'evalue.cpp',
                                    'extreal.cpp',
                                    'fft.cpp',
                                    'hholder.cpp',
                                    'jacobi.cpp',
                                    'ltqnorm.cpp',
                                    'myexcept.cpp',
                                    'mvnpdf.cpp',
                                    'newfft.cpp',
                                    'newmat1.cpp',
                                    'newmat2.cpp',
                                    'newmat3.cpp',
                                    'newmat4.cpp',
                                    'newmat5.cpp',
                                    'newmat6.cpp',
                                    'newmat7.cpp',
                                    'newmat8.cpp',
                                    'newmat9.cpp',
                                    'newmatex.cpp',
                                    'newmatnl.cpp',
                                    'newmatrm.cpp',
                                    'solution.cpp',
                                    'sort.cpp',
                                    'specialfunctions2.cpp',
                                    'submat.cpp',
                                    'svd.cpp']],
                                    include_dirs = [get_include()],
                                    libraries = ['m', 'stdc++'])

setup(name='fcm',
      version='0.01',
      packages=['fcm', 'fcm.core', 'fcm.graphics', 'fcm.gui', 'fcm.io', 'fcm.statistics' ],
      package_dir = {'fcm': 'src'},
      package_data= {'': ['data/*']},
      description='Python Flow Cytometry (FCM) Tools',
      author='Jacob Frelinger',
      author_email='jacob.frelinger@duke.edu',
      ext_modules = [cdp_extension],
      requires=['numpy (>=1.3.0)'], # figure out the rest of what's a required package.
      )
