=================
py_bonemat_abaqus
=================
:Version: Version 1.0.1
:Author: Dr Elise Pegg, University of Oxford
:Email: elise.pegg@ndorms.ox.ac.uk

------------
Introduction
------------
This python package provides tools to add material properties of bone to an ABAQUS finite element model input file, where the modulus of each element is defined based upon its corresponding CT data using the Hounsfield Unit (HU) and input parameters.

The package aims to be equivalent to BoneMat software developed by researchers in Bologna, Italy, but tailored for ABAQUS finite element users (as BoneMat cannot currently import ABAQUS input files).  The original BoneMat can be found at www.bonemat.org_, and further details can be found in published papers [1]_, [2]_, [3]_, [4]_.

.. _www.bonemat.org: https://www.bonemat.org

Notes:

- When this package is used to add materials to an ABAQUS input file, any model parameters (such as loading, sets, or step definitions) are retained.
- The present version of this package will only work with: 
	+ linear and quadratic tetrahdral elements
	+ linear wedge elements
	+ linear hexahedral elements
- To run the package, you need to have three files: a parameters file (.txt), an ABAQUS input file (.inp), and a CT scan (either as a series of dicom images in a folder, or as a .vtk file).  Example files are provided in the example folder.
- More information on this package can be found at the start of the py_bonemat_abaqus.py code

------------
Installation
------------
The simplest and recommended way to install py_bonemat_abaqus is with pip. You may install the latest stable release from PyPI with pip using the following command::

>>> pip install py_bonemat_abaqus

If you do not have pip, you may use easy_install::

>>> easy_install py_bonemat_abaqus

Alternatively, you may download the source package from the PyPI page, extract it and install using::

>>> python setup.py install

------------
Dependencies
------------
- Numpy - version 1.6 or higher
- PyDicom - version 0.9.7 or higher

-----
Usage
-----
The package can be run either from the terminal, or imported into a python script.

To run from a terminal use the following syntax:

>>> py_bonemat_abaqus -p <parameters file> -ct <ct file/dir> -m <abaqus input file>

To run from within a python script, at the top of the file import the 'run' script from the package, and then execute with::

	run(<parameters file>, < ct scan dir or vtk>, <abaqus input file>)

For example, if a python script containing the following two lines were saved in the examples folder of the source code, it would calculate the material properties of 'example_abaqus_mesh.inp'::

	from py_bonemat_abaqus.run import run

	run('example_parameters.txt','example_ct_data.vtk','example_abaqus_mesh.inp')


----------
References
----------
.. [1] Helgason B, Taddei F, Palsson H, Schileo E, Cristofolini L, Viceconti M, Brynjolfsson S. (2008) Med Eng Phys 30 [4] p444-453: http://dx.doi.org/10.1016/j.medengphy.2007.05.006
.. [2] Taddei F, Schileo E, Helgason B, Cristofolini L, Viceconti M. (2007) Med Eng Phys 29 [9] p973-979: http://dx.doi.org/10.1016/j.medengphy.2006.10.014
.. [3] Taddei F, Pancanti A, Viceconti M. (2004) Med Eng Phys 26 [1] p61-69: http://dx.doi.org/10.1016/S1350-4533(03)00138-3
.. [4] Zannoni C, Mantovani R, Viceconti M. (1998) Med Eng Phys 20 [1] p735-740: http://dx.doi.org/10.1016/S1350-4533(98)00081-2
