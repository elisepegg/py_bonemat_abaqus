#!/usr/bin/python
#
# py_bonemat_abaqus - command line
# ================================
#
# Created by Elise Pegg, University of Bath

__all__ = ['main']

#------------------------------------------------------------------------------
# Import modules
#------------------------------------------------------------------------------
from py_bonemat_abaqus.run import run
import sys, os
import argparse

#------------------------------------------------------------------------------
# Run program
#------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(prog="py_bonemat_abaqus")
    parser.add_argument('-p', nargs='?', help='parameters file (*.txt)')
    parser.add_argument('-ct', nargs='?', help='ct dataset (dir or *.vtk)')
    parser.add_argument('-m', nargs='?', help='mesh file (*.inp or *.ntr')
 
    args = parser.parse_args()
 
    run(args.p, args.ct, args.m)
