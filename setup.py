#!/usr/bin/env python
"""TryNychka: trying to figure out
And here goes some shitty documentation
Just to try it
"""

DOCLINES = (__doc__ or '').split("\n")

from distutils import version
import os
import sys
import subprocess
import textwrap
import warnings
import sysconfig
from distutils.version import LooseVersion

if sys.version_info[:2] < (3, 7):
    raise RuntimeError("Python version >= 3.7 required.")

import builtins


CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: MIT License
Programming Language :: C
Programming Language :: Python :: 3.7
Topic :: Software Development :: Libraries
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: POSIX :: Linux
Operating System :: MacOS

"""


IS_RELEASE_BRANCH = False

def parse_setuppy_commands():
    """Check the commands and respond appropriately.  Disable broken commands.
    Return a boolean value for whether or not to run the build or not (avoid
    parsing Cython and template files if False).
    """
    args = sys.argv[1:]

    if not args:
        # User forgot to give an argument probably, let setuptools handle that.
        return True

    info_commands = ['--help-commands', '--name', '--version', '-V',
                     '--fullname', '--author', '--author-email',
                     '--maintainer', '--maintainer-email', '--contact',
                     '--contact-email', '--url', '--license', '--description',
                     '--long-description', '--platforms', '--classifiers',
                     '--keywords', '--provides', '--requires', '--obsoletes']

    for command in info_commands:
        if command in args:
            return False

    # Note that 'alias', 'saveopts' and 'setopt' commands also seem to work
    # fine as they are, but are usually used together with one of the commands
    # below and not standalone.  Hence they're not added to good_commands.
    good_commands = ('develop', 'sdist', 'build', 'build_ext', 'build_py',
                     'build_clib', 'build_scripts', 'bdist_wheel', 'bdist_rpm',
                     'bdist_wininst', 'bdist_msi', 'bdist_mpkg')

    for command in good_commands:
        if command in args:
            return True

    # The following commands are supported, but we need to show more
    # useful messages to the user
    if 'install' in args:
        print(textwrap.dedent("""
            Note: for reliable uninstall behaviour and dependency installation
            and uninstallation, please use pip instead of using
            `setup.py install`:
              - `pip install .`       (from a git repo or downloaded source
                                       release)
              - `pip install scipy`   (last SciPy release on PyPI)
            """))
        return True

    if '--help' in args or '-h' in sys.argv[1]:
        print(textwrap.dedent("""
            SciPy-specific help
            -------------------
            To install SciPy from here with reliable uninstall, we recommend
            that you use `pip install .`. To install the latest SciPy release
            from PyPI, use `pip install scipy`.
            For help with build/installation issues, please ask on the
            scipy-user mailing list.  If you are sure that you have run
            into a bug, please report it at https://github.com/scipy/scipy/issues.
            Setuptools commands help
            ------------------------
            """))
        return False


    # The following commands aren't supported.  They can only be executed when
    # the user explicitly adds a --force command-line argument.
    bad_commands = dict(
        test="""
            `setup.py test` is not supported.  Use one of the following
            instead:
              - `python runtests.py`              (to build and test)
              - `python runtests.py --no-build`   (to test installed scipy)
              - `>>> scipy.test()`           (run tests for installed scipy
                                              from within an interpreter)
            """,
        upload="""
            `setup.py upload` is not supported, because it's insecure.
            Instead, build what you want to upload and upload those files
            with `twine upload -s <filenames>` instead.
            """,
        upload_docs="`setup.py upload_docs` is not supported",
        easy_install="`setup.py easy_install` is not supported",
        clean="""
            `setup.py clean` is not supported, use one of the following instead:
              - `git clean -xdf` (cleans all files)
              - `git clean -Xdf` (cleans all versioned files, doesn't touch
                                  files that aren't checked into the git repo)
            """,
        check="`setup.py check` is not supported",
        register="`setup.py register` is not supported",
        bdist_dumb="`setup.py bdist_dumb` is not supported",
        bdist="`setup.py bdist` is not supported",
        flake8="`setup.py flake8` is not supported, use flake8 standalone",
        build_sphinx="`setup.py build_sphinx` is not supported, see doc/README.md",
        )
    bad_commands['nosetests'] = bad_commands['test']
    for command in ('upload_docs', 'easy_install', 'bdist', 'bdist_dumb',
                     'register', 'check', 'install_data', 'install_headers',
                     'install_lib', 'install_scripts', ):
        bad_commands[command] = "`setup.py %s` is not supported" % command

    for command in bad_commands.keys():
        if command in args:
            print(textwrap.dedent(bad_commands[command]) +
                  "\nAdd `--force` to your command to use it anyway if you "
                  "must (unsupported).\n")
            sys.exit(1)

    # Commands that do more than print info, but also don't need Cython and
    # template parsing.
    other_commands = ['egg_info', 'install_egg_info', 'rotate']
    for command in other_commands:
        if command in args:
            return False

    # If we got here, we didn't detect what setup.py command was given
    warnings.warn("Unrecognized setuptools command ('{}'), proceeding with "
                  "generating Cython sources and expanding templates".format(
                  ' '.join(sys.argv[1:])))
    return True


def check_setuppy_command():
    run_build = parse_setuppy_commands()
    return run_build


def configuration(parent_package='', top_path=None):
    from scipy._build_utils.system_info import get_info, NotFoundError
    from numpy.distutils.misc_util import Configuration


    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('nychka')

    return config


def setup_package():
    # In maintenance branch, change np_maxversion to N+3 if numpy is at N
    # Update here, in pyproject.toml, and in scipy/__init__.py
    # Rationale: SciPy builds without deprecation warnings with N; deprecations
    #            in N+1 will turn into errors in N+3
    # For Python versions, if releases is (e.g.) <=3.9.x, set bound to 3.10
    np_minversion = '1.16.5'
    np_maxversion = '9.9.99'
    python_minversion = '3.7'
    python_maxversion = '3.10'
    if IS_RELEASE_BRANCH:
        req_np = 'numpy>={},<{}'.format(np_minversion, np_maxversion)
        req_py = '>={},<{}'.format(python_minversion, python_maxversion)
    else:
        req_np = 'numpy>={}'.format(np_minversion)
        req_py = '>={}'.format(python_minversion)


    req_openmp = 'intel-openmp>=2021.2.0'


    metadata = dict(
        name='nychka',
        version="0.0.1",
        maintainer="Nychka Developers",
        maintainer_email="aynurmukh@gmail.com",
        description=DOCLINES[0],
        long_description="\n".join(DOCLINES[2:]),
        url="https://github.com/ainmukh/nychka",
        download_url="https://github.com/ainmukh/nychka/releases",
        project_urls={
            "Source Code": "https://github.com/ainmukh/nychka",
        },
        license='MIT',
        classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
        platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
        install_requires=[req_np, req_openmp],
        python_requires=req_py,
    )

    if "--force" in sys.argv:
        run_build = True
        sys.argv.remove('--force')
    else:
        # Raise errors for unsupported commands, improve help output, etc.
        run_build = check_setuppy_command()

    # This import is here because it needs to be done before importing setup()
    # from numpy.distutils, but after the MANIFEST removing and sdist import
    # higher up in this file.
    from setuptools import setup

    if run_build:
        from numpy.distutils.core import setup

        metadata['configuration'] = configuration

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
