import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = '1.0.0'
PACKAGE_NAME = 'tsmoothie'
AUTHOR = 'Marco Cerliani'
AUTHOR_EMAIL = 'cerlymarco@gmail.com'
URL = 'https://github.com/cerlymarco/tsmoothie'

LICENSE = 'MIT'
DESCRIPTION = 'A python library for timeseries smoothing and outlier detection in a vectorized way.'
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
      'numpy',
      'scipy',
      'simdkalman'
]

setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESC_TYPE,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      url=URL,
      install_requires=INSTALL_REQUIRES,
      packages=find_packages()
      )