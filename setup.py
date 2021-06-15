#    GPLv3 License

#    DiscOTimeS: Automated estimation of trends, discontinuities and nonlinearities
#    in geophysical time series
#    Copyright (C) 2021  Julius Oelsmann

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

import re

from codecs import open
from os.path import dirname, join, realpath

from setuptools import setup, find_packages
import os
import sys



DISTNAME = "discotimes"
DESCRIPTION = "Automated estimation of trends, discontinuities and nonlinearities time series using Bayesian Inference"
AUTHOR = "Julius Oelsmann"
AUTHOR_EMAIL = "julius.oelsmann@tum.com"
URL = "https://github.com/oelsmann/discotimes"
LICENSE = "GPLv3 License"

classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "License :: OSI Approved :: GPLv3 License",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Geodesy",    
    "Topic :: Scientific/Engineering :: Climate Science",    
    "Topic :: Scientific/Engineering :: Sea level Science",    
    "Operating System :: OS Independent",
]

PROJECT_ROOT = dirname(realpath(__file__))

# Get the long description from the README file
with open(join(PROJECT_ROOT, "README.md"), encoding="utf-8") as buff:
    LONG_DESCRIPTION = buff.read()

REQUIREMENTS_FILE = join(PROJECT_ROOT, "requirements.txt")

with open(REQUIREMENTS_FILE) as f:
    install_reqs = f.read().splitlines()

test_reqs = ["pytest", "pytest-cov"]


def get_version():
    VERSIONFILE = join("discotimes", "__init__.py")
    lines = open(VERSIONFILE).readlines()
    version_regex = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in lines:
        mo = re.search(version_regex, line, re.M)
        if mo:
            return mo.group(1)
    raise RuntimeError(f"Unable to find version in {VERSIONFILE}.")


if __name__ == "__main__":
    setup(
        name=DISTNAME,
        version=get_version(),
        maintainer=AUTHOR,
        maintainer_email=AUTHOR_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/x-rst",
        packages=find_packages(),
        # because of an upload-size limit by PyPI, we're temporarily removing docs from the tarball.
        # Also see MANIFEST.in
        # package_data={'docs': ['*']},
        include_package_data=True,
        classifiers=classifiers,
        python_requires=">=3.7",
        install_requires=install_reqs,
        tests_require=test_reqs,
        scripts=['discotimes/discofit.py']
        )
    
    


