import os


def get_library_dirs():
    result = os.getenv('LIBRARY_DIRS')
    if result is None or result == '':
        return []
    return result.split('%')


def get_include_dirs():
    result = os.getenv('INCLUDE_DIRS')
    if result is None or result == '':
        return []
    return result.split('%')


def get_extra_compile_args():
    result = os.getenv('EXTRA_COMPILE_ARGS')
    if result is None or result == '':
        return []
    return result.split('%')


def get_extra_link_libs():
    result = os.getenv('EXTRA_LINK_LIBS')
    if result is None or result == '':
        return []
    return result.split('%')


def configuration(parent_package='', top_path=None):
    import numpy
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.misc_util import get_info

    # Necessary for the half-float d-type.
    info = get_info('npymath')

    config = Configuration('distribution',
                           parent_package,
                           top_path)
    config.add_extension('cdf_ufunc',
                         sources=['cdf_ufunc.cpp',
                                  'polylogarithm.cpp'],
                         include_dirs=[numpy.get_include(), '/usr/include/eigen3'] + get_include_dirs(),
                         extra_compile_args=get_extra_compile_args(),
                         library_dirs=get_library_dirs(),
                         libraries=['gsl', 'gslcblas'] + get_extra_link_libs(),
                         extra_info=info)

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup
    from cmake_setuptools import *
    setup(configuration=configuration)
