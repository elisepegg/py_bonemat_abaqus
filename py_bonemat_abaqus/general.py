#!/usr/bin/python
#
# py_bonemat_abaqus - general
# ==========================
#
# Created by Elise Pegg, University of Bath

__all__ = ['check_argv']

#-------------------------------------------------------------------------------
# Import modules
#-------------------------------------------------------------------------------
from math import sin, cos, pi
from numpy import diag, outer, array, identity, float64, dot, sqrt, atleast_1d
from numpy import expand_dims
#-------------------------------------------------------------------------------
# Checks
#-------------------------------------------------------------------------------
def check_argv(argv):
    usage = '\nUsage: run(<parameters.txt>, <ct_data>, <job.inp>)'

    # check number of input arguments is correct
    if len(argv) != 3:
        raise IOError(usage + "\n\tIncorrect number of input arguments\n")
        
    # check parameters file is a .txt file
    elif ".txt" not in argv[0]:
        raise IOError(usage + "\n\t" + argv[1] + " is not a .txt file\n")
        
    # check CTdir is a directory or a 
    elif ("." in argv[1]) and (".vtk" not in argv[1]):
        raise IOError(usage + "\n\t" + argv[2] + " is not a directory or .vtk file\n")
        
    # check job.inp is a text file
    elif (".inp" not in argv[2]) and (".ntr" not in argv[2]):
        raise IOError(usage + "\n\t" + argv[2] + " is not an abaqus input file\n")
        
    # return result
    else:
        return True

#-------------------------------------------------------------------------------
# General text functions
#-------------------------------------------------------------------------------
def _read_text(fle):
    """ Reads lines from text file """

    with open(fle, 'r') as inpf:
        lines = inpf.read()

    return lines

def _remove_spaces(lines):
    """ Removes spaces from lines """           
        
    lines = lines.replace(" ", "")
    lines = lines.replace("\r","\n")
    lines = lines.replace("\n\n", "\n")
    
    return lines

def _remove_eol_r(lines):
    """ Replace any windows eol character (\r) with \n """

    lines = lines.replace("\r", "\n")
    lines = lines.replace("\n\n", "\n")

    return lines


def _refine_spaces(lines):
    """ Replaces any multiple spaces with one space """

    for n in range(10):
        lines = lines.replace("  "," ")

    return lines

#-------------------------------------------------------------------------------
# General maths functions
#-------------------------------------------------------------------------------
def _rot_matrix(point, angle, direction):
    """ Calculates a rotation matrix to transform 3D points.

    Code heavily based upon 'transformations.py' by Christoph Gohlke"""

    sina = sin(angle * pi/180)
    cosa = cos(angle * pi/180)
    direction = _unit_vector(direction[:3])
    R = diag([cosa, cosa, cosa])
    R += outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += array([[ 0.0,         -direction[2],  direction[1]],
                [ direction[2], 0.0,          -direction[0]],
                [-direction[1], direction[0],  0.0]])
    M = identity(4)
    M[:3, :3] = R
    if point is not None:
        point = array(point[:3], dtype = float64, copy=False)
        M[:3, 3] = point - dot(R, point)

    return M[:3,:3]

def _unit_vector(data, axis=None, out=None):
    """ Function to calculate unit vector.

    Code heavily based upon 'transformations.py' by Christoph Gohlke"""

    if out is None:
        data = array(data, dtype=float64, copy=True)
        if data.ndim == 1:
            data /= sqrt(dot(data,data))
            return data
    else:
        if out is not data:
            out[:] = array(data, copy=False)
        data = out
    length = atleast_1d(sum(data*data, axis))
    sqrt(length, length)
    if axis is not None:
        length = expand_dims(length, axis)
    data /= length
    if out is None:
        return data
