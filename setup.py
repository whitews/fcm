#from setuptools import setup, Extension
from distutils.core import setup
from numpy import get_include
from Cython.Distutils import Extension
from Cython.Distutils import build_ext

logicle_extension = Extension('fcm.core._logicle',
                              sources=[ 'fcm/core/logicle_ext/%s' % i for i in [
                                'Logicle.cpp',
                                'my_logicle.cpp',
                                'my_logicle_wrapper.cpp']],
                              include_dirs=[get_include()]
                              )

munkres_extension = Extension("fcm.alignment.munkres", ["fcm/alignment/munkres.pyx",
                                         "fcm/alignment/munkres_ext/Munkres.cpp"],
                             include_dirs = [get_include(), 'fcm/alignment/munkres_ext/'],
                             language='c++')
setup(name='fcm',
      version='0.9.5',
      url='http://code.google.com/p/py-fcm/',
      packages=['fcm', 'fcm.core', 'fcm.graphics', 'fcm.gui', 'fcm.io', 'fcm.statistics' , 'fcm.alignment'],
      package_data={'': ['data/*']},
      description='Python Flow Cytometry (FCM) Tools',
      author='Jacob Frelinger',
      author_email='jacob.frelinger@duke.edu',
      cmdclass = {'build_ext': build_ext},
      ext_modules=[logicle_extension, munkres_extension],
      requires=['numpy (==1.7.1)',
                'scipy (==0.12)',
                'dpmix (>=0.2)',
                'matplotlib (>=1.1, <1.3)'],
      )
