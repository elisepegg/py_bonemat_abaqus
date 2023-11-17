from setuptools import setup, find_packages
from codecs import open
from os import path

# get current path
here = path.abspath(path.dirname(__file__))

# function to open the readme file
def readme():
    with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
        return f.read()

# find the version
exec(open(path.join('py_bonemat_abaqus','version.py')).read())
    
# define setup
setup(name='py_bonemat_abaqus',
      version=__version__,
      description='Assign material properties of bone to a finite element mesh',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Natural Language :: English',
      ],
      keywords=['bone', 'material', 'finite element', 'medical', 'science', 'engineering'],
      url='http://github.com/elisepegg/py_bonemat_abaqus',
      author='Elise Pegg',
      author_email='e.c.pegg@bath.ac.uk',
      license='GPLv3',
      install_requires=['numpy','pydicom'],
      packages=find_packages(exclude=['build', '_docs', 'templates']),
      include_package_data=True,
      package_data={'readme': ['README.rst'],
                    'license': ['LICENSE.txt','gpl.txt'],
                    'example': [path.join('example','example_abaqus_mesh.inp'),
                                path.join('example','example_ct_data.vtk'),
                                path.join('example','example_parameters.txt')],
                    'tests': [path.join('tests','__init__.py'),
                              path.join('tests','tests.py'),
                              path.join('tests','tests_functional.py'),
                              path.join('tests','tests_validation.py')],},
      entry_points={'console_scripts': ['py_bonemat_abaqus = py_bonemat_abaqus.command_line:main'],},
      zip_safe=False)
