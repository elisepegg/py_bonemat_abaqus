#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
py_bonemat_abaqus
=================

Based upon Bonemat software, which was originally developed at the Istituto
Orthopedico Rizzoli in Bologna, Italy.

Input data:
----------- 
          1. A parameters file called *.txt detailing which parameters to use
              - an example parameters file is included in the test/data folder
          2. Name of the CT dicom image file directory or .vtk file
          3. Name of the ABAQUS input file (*.inp) to be modified

Output data:
------------
          Modified ABAQUS input file (*MAT.inp)

Command line usage:
-------------------
          >>> py_bonemat_abaqus -p <parameters file> -ct <ct file/dir> -m <abaqus input file>

Notes:
------
          For the current version, the input file must have *Part and *Assembly
              definitions for it to be read correctly. This is the same as the
              default output format created by Abaqus
          Any already defined materials will not be removed by the software, so
              this may cause some elements to have duplicate materials assigned.
              Therefore to ensure no conflicts, before running program check you
              have not assigned materials to any parts in the input file
"""
#-------------------------------------------------------------------------------
# Import modulus
#-------------------------------------------------------------------------------
from py_bonemat_abaqus.run import run
import sys, os

#-------------------------------------------------------------------------------
# Start program if run from command line
#-------------------------------------------------------------------------------
def main(argv):
    run(argv[0], argv[1], argv[2])
    
if __name__ == '__main__':
    main(sys.argv[1:])
