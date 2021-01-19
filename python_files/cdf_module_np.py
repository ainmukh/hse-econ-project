from distutils.core import setup, Extension
import numpy

cdf_module_np = Extension('cdf_module_np',
                          sources=['/Users/ainurmukh/work/coursework/project/test.cpp'],
                          include_dirs=[numpy.get_include(),
                                        '/usr/local/include/eigen3',
                                        '/usr/local/include/gsl',
                                        '/usr/local/include'],
                          libraries=['/usr/local/lib/gsl', '/usr/local/lib/gslcblas'],
                          extra_compile_args=['-std=c++17'])

setup(name='cdf_module_np', ext_modules=[cdf_module_np])
