#!/usr/bin/python
#
# py_bonemat_abaqus - run
# ==========================
#
# Created by Elise Pegg, University of Bath

__all__ = ['run']

#-------------------------------------------------------------------------------
# Import modules
#-------------------------------------------------------------------------------
import sys
import os
from py_bonemat_abaqus import general, data_import, calc, data_output
from py_bonemat_abaqus.version import __version__
import time
from datetime import date

#-------------------------------------------------------------------------------
# Define run program
#-------------------------------------------------------------------------------
def run(argv0, argv1, argv2):
    #---------------------------------------------------------------------------
    # create welcome screen
    #---------------------------------------------------------------------------    
    t = time.time()
    d = date.today()
    month = d.strftime("%B")[:3]
    year = d.isocalendar()[0]
    print("""
    ************** PY_BONEMAT ABAQUS """ + __version__ + """ ************
    *** Elise Pegg,  University of Bath,   """ + month + """ """ + repr(year) + """ ***
    ***************************************************
    """)
    
    #---------------------------------------------------------------------------
    # check input arguments
    #---------------------------------------------------------------------------
    argv = [argv0, argv1, argv2]
    if general.check_argv(argv) == False:
        sys.exit(1)
    param_file = argv[0]
    ct_data = argv[1]
    mesh_file = argv[2]
        
    #---------------------------------------------------------------------------
    # Import parameters
    #---------------------------------------------------------------------------
    print("    Importing parameters file: " + param_file)
    param = data_import.import_parameters(param_file)

    #---------------------------------------------------------------------------
    # Import Abaqus input file data
    #---------------------------------------------------------------------------
    print("    Importing mesh file: " + mesh_file)
    parts = data_import.import_mesh(mesh_file, param)

    #---------------------------------------------------------------------------
    # Import CT data
    #---------------------------------------------------------------------------
    print("    Importing CT data: " + ct_data)
    vtk_data = data_import.import_ct_data(ct_data)
    
    #---------------------------------------------------------------------------
    # Determine material properties for elements within each part
    #---------------------------------------------------------------------------
    print("    Calculating material properties")
    parts = calc.calc_mat_props(parts, param, vtk_data)

    #---------------------------------------------------------------------------
    # Write data to new abaqus input file
    #---------------------------------------------------------------------------
    print("    Writing material data to new abaqus input file:")
    print("\t" + mesh_file[:-4] + "MAT.inp")
    data_output.output_abq_inp(parts, mesh_file, param['poisson'])

    #---------------------------------------------------------------------------
    # End
    #---------------------------------------------------------------------------
    print("""
    **   !!! Bone material assignment complete !!!   **
    ***************************************************
    """)
    tt = time.time() - t
    print(" Elapsed time: " + repr(tt))
    os._exit(0)
