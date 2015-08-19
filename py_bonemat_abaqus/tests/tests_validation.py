#!/usr/bin/python
#
# py_bonemat_abaqus - validation tests
# ====================================
#
# Equivalence tests to compare the output of py_bonemat_abaqus with the original
# software
#
# This script will create random element shapes within a CT scan then processed
# with this python package, but for the final comparison the files will need to
# be run through the original BoneMat separately
#
# The CT scan which was used for the testing was that of a foot, downloaded from
#     www.osirix-viewer.com/datasets
#
# Created by Elise Pegg, University of Oxford

#------------------------------------------------------------------------------
# Import modules
#------------------------------------------------------------------------------
from random import uniform
from numpy.linalg import det
from itertools import permutations, combinations
from numpy import matrix, mean, arange
from math import atan2, pi, floor, log10
from operator import itemgetter
import os, sys
from unittest import TestCase, main
import py_bonemat_abaqus
from py_bonemat_abaqus import general, data_import, calc, data_output

#----------------------------- Create the files -------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Define variables
#------------------------------------------------------------------------------
# dimensions
vtk_xrange = [-118.0, 117.368]
vtk_yrange = [-97.1, 138.268]
vtk_zrange = [442.62, 440.72]
# file name of the ct scan
ct_file = "iliac1.vtk"
# number of elements to create and test
num_ele = 10
    
#------------------------------------------------------------------------------
# Define sub-functions
#------------------------------------------------------------------------------
def _rand_xyz(x, y, z):
    """ Generates a random co-ordinate within the ranges of x, y, and z """

    return [uniform(x[0],x[1]), uniform(y[0],y[1]), uniform(z[0],z[1])]

def _rand_magnitude():
    """ Generates a volume for the element """

    return uniform(1, 6)

def _jacobian_linear_tet(pts):
    """ Calculates the jacobian matrix of a linear tetrahedron """
    
    return [[1, 1, 1, 1],
            [pts[0][0], pts[1][0], pts[2][0], pts[3][0]],
            [pts[0][1], pts[1][1], pts[2][1], pts[3][1]],
            [pts[0][2], pts[1][2], pts[2][2], pts[3][2]]]

def _jacobian_quad_tet(pts):
    """ Calculates the jacobian matrix of a quadratic tetrahedron """

    # split the quadratic tetrahedron into linear tetrahedrons
    tets = [[pts[0], pts[1], pts[2], pts[3]],
            [pts[0], pts[1], pts[2], pts[7]],
            [pts[0], pts[1], pts[2], pts[8]],
            [pts[0], pts[1], pts[2], pts[9]],
            [pts[1], pts[3], pts[2], pts[5]],
            [pts[1], pts[3], pts[2], pts[6]],
            [pts[1], pts[3], pts[2], pts[7]],
            [pts[0], pts[3], pts[1], pts[4]],
            [pts[0], pts[3], pts[1], pts[8]],
            [pts[0], pts[3], pts[1], pts[6]],
            [pts[2], pts[3], pts[0], pts[5]],
            [pts[2], pts[3], pts[0], pts[4]],
            [pts[2], pts[3], pts[0], pts[9]]]
    
    # calculate the jacobians of each tet
    count = 0
    Jdet=[0.8]
    while (min(Jdet) > 0.2) & (count < len(tets)):
        Jdet.append(det(_jacobian_linear_tet(tets[count])))
        count += 1
        
    return min(Jdet)

def _jacobian_linear_wedge(pts):
    """ Calculates the jacobian matrix of a linear wedge """

    # split the wedge into linear tetrahedrons            
    tets = [[pts[0], pts[1], pts[2], pts[3]],
            [pts[0], pts[1], pts[2], pts[4]],
            [pts[0], pts[1], pts[2], pts[5]],
            [pts[3], pts[5], pts[4], pts[0]],
            [pts[3], pts[5], pts[4], pts[1]],
            [pts[3], pts[5], pts[4], pts[2]],
            [pts[3], pts[1], pts[2], pts[4]],
            [pts[0], pts[2], pts[3], pts[4]],
            [pts[4], pts[1], pts[0], pts[2]],
            [pts[2], pts[1], pts[4], pts[0]]]

    # calculate the jacobians of each tet
    Jdet = [det(_jacobian_linear_tet(t)) for t in tets]

    return min(Jdet)

def _jacobian_linear_hex(pts):
    """ Calculates the jacobian matrix of a linear hexahedral element """

    # split the hex into wedges
    wedges = [[pts[0], pts[1], pts[3], pts[4], pts[5], pts[7]],
              [pts[1], pts[2], pts[3], pts[5], pts[6], pts[7]],
              [pts[4], pts[0], pts[3], pts[5], pts[1], pts[2]],
              [pts[7], pts[4], pts[3], pts[6], pts[5], pts[2]],
              [pts[3], pts[2], pts[7], pts[0], pts[1], pts[4]],
              [pts[2], pts[6], pts[7], pts[1], pts[5], pts[4]]]
    
    # calculate the jacobians of each wedge
    count = 0
    Jdet=[0.8]
    while (min(Jdet) > 0.7) & (count < len(wedges)):
        Jdet.append(_jacobian_linear_wedge(wedges[count]))
        count += 1

    return min(Jdet)

def _calc_dist(pt1, pt2):
    """ Calculates the distance between two 3D points """
    
    vector = (matrix(pt1) - matrix(pt2)).tolist()[0]
    dist = (sum([v**2 for v in vector])) ** 0.5

    return dist
    

def _write_abq_input(pts, elecode, fname, num_pts):
    """ Writes the abaqus input file """
    
    with open(fname,'w') as oupf:
        # write the header
        oupf.write("""*Heading
** Job name: Job-1 Model name: Model-1
** Generated by: Abaqus/CAE 6.12-1
*Preprint, echo=NO, model=NO, history=NO, contact=NO
**
** PARTS
**
*Part, name=Part-1
*Node
""")
        # write the nodes
        count = 1
        for p in pts:
            oupf.write(repr(count))
            for i in range(len(p)):
                oupf.write(',')
                oupf.write(repr(p[i]))
            oupf.write('\r\n')
            count += 1
        # write the elements
        oupf.write('*Element, type =' + elecode + '\r\n')
        oupf.write('1')
        for n in range(num_pts):
            oupf.write(',')
            oupf.write(repr(n+1))
        oupf.write('\r\n*End Part')

def _write_ntr_input(pts, ele_code, fname, num_pts):
    with open(fname, 'w') as oupf:
        # write the title
        title_data = ['25', '0', '0', '1', '0', '0', '0', '0', '0']
        _write_chunks(title_data, oupf)
        oupf.write("PATRAN Neutral, HyperMesh Template PATRAN/GENERAL\r\n")
        # write the summary
        summary_data = ['26', '0', '0', '1', repr(num_pts), '1', '1', '1', '0']
        _write_chunks(summary_data, oupf)
        oupf.write("  07-27-201510:28:00\r\n")
        # write the node data
        for p in pts:
            node_data = ['1', repr(pts.index(p)+1), '0', '2', '0', '0', '0', '0', '0']
            _write_chunks(node_data, oupf)
            _write_nodes(p, oupf)
            oupf.write("1G       6       0       0  000000\r\n")
        # write the element data
        ele_data = ['2', '1', ele_code, '2', '0', '0', '0', '0', '0']
        _write_chunks(ele_data, oupf)
        oupf.write("      ")
        ele_data2 = [repr(num_pts), '1', '4', '0']
        _write_chunks(ele_data2, oupf, False)
        oupf.write(" 0.000000000E+00 0.000000000E+00 0.000000000E+00\r\n")
        oupf.write("      ")
        ele_data3 = [repr(i+1) for i in range(num_pts)]
        _write_chunks(ele_data3, oupf)
        # write the end of file
        oupf.write("99       0       0       1       0       0       0       0       0")

def _write_chunks(data, oupf, write_n = True):
    for i in range(2 - len(data[0])):
        oupf.write(" ")
    oupf.write(data[0])
    for d in data[1:]:
        for i in range(8 - len(d)):
            oupf.write(" ")
        oupf.write(d)
    if write_n:
        oupf.write("\r\n")

def _write_nodes(nodes, oupf):
    for n in nodes:
        # determine E base
        base = floor(log10(abs(n)))
        if n > 0:
            string = '%.10f' % (n/(10**base))
            oupf.write(string)
        else:
            string = '%.9f' % (n/(10**base))
            oupf.write(string)
        oupf.write('E')
        if base<0:
            oupf.write('-')
        else:
            oupf.write('+')
        for i in range(2 - len(repr(int(abs(base))))):
            oupf.write("0")
        oupf.write(repr(int(abs(base))))
    oupf.write('\r\n')

def _calculate_jacobian(pts, ele_type):
    """ Performs the jacobian calculation for the element type """
    
    if ele_type == 'linear_tet':
        J = _jacobian_linear_tet(pts)
    elif ele_type == 'quad_tet':
        J = _jacobian_quad_tet(pts)
    elif ele_type == 'linear_hex':
        J = _jacobian_linear_hex(pts)
    elif ele_type == 'linear_wedge':
        J = _jacobian_linear_wedge(pts)
    else:
        raise IOError("Error: element type not recognised")

    return J

def _reorder_quad(pts):
    """ Determines the best point order for a quadratic tetrahedron """
    
    centroid = mean(matrix(pts),0).tolist()[0]
    # make the centroid = (0,0,0)
    pts = (matrix(pts) - matrix(centroid)).tolist()
    # find the distances of all the points from the centroid
    dists = [_calc_dist(p, centroid) for p in pts]
    corners = [dists.index(d) for d in sorted(dists)[-4:]]
    corner_pts = [pts[c] for c in corners]
    midnodes = [dists.index(d) for d in sorted(dists)[:-4]]
    midnode_pts = [pts[m] for m in midnodes]
    # order the corner nodes using _best_order_linear_tet function
    corner_pts_final = _best_order_linear_tet(corner_pts)
    # for each edge identify the closest midnode
    midnodes_final = []
    for edge in [[0,1],[1,2],[2,0],[0,3],[1,3],[2,3]]:
        midnode_pt, midnode_index = _get_midnode(corner_pts_final[edge[0]], corner_pts_final[edge[1]], midnode_pts)
        midnodes_final.append(midnode_pt)
        midnode_pts.pop(midnode_index)
        midnodes.pop(midnode_index)

    return corner_pts_final + midnodes_final

def _best_order_linear_tet(pts):
    """ Determines the best point order for a linear tetrahedron """
    
    centroid = mean(matrix(pts),0).tolist()[0]
    # make the centroid (0,0,0)
    pts = (matrix(pts) - matrix(centroid)).tolist()
    # find p4 (max z)
    z = [p[2] for p in pts]
    p4indx = z.index(max(z))
    left = [0,1,2,3]
    left.pop(left.index(p4indx))
    # order points 1-3 anticlockwise in the xy-plane
    pts_remain = [pts[n] for n in left]
    new_pts = _anticlockwise(pts_remain)

    return new_pts + [pts[p4indx]]

def _anticlockwise(pts):
    """ Orders any number of points anti-clockwise in the xy-plane """
    
    # for each point find the angle between 0-360 relative to the first point
    p1 = pts[0]
    angles = [(atan2(p1[1],p1[0]) - atan2(p[1], p[0])) * 180 / pi for p in pts]

    # determine anti-clockwise order
    order = sorted(enumerate(angles), key=itemgetter(1))

    # return ordered points
    return [pts[order[len(pts)-1 -n][0]] for n in range(len(pts))]

def _get_midnode(corner1, corner2, midnodes):
    """ Idenfity which mid-node is closest to the midpoint of an edge """
    
    allowable_dist = 0.1
    midpt = mean(matrix([corner1,corner2]),0).tolist()[0]
    dists = [_calc_dist(p, midpt) for p in midnodes]
    # ensure chosen midpoint does not distort element too much
    if min(dists) > allowable_dist:
        chosen_pt = midnodes[dists.index(min(dists))]
        vector = (matrix(chosen_pt) - matrix(midpt)).tolist()[0]
        factor = allowable_dist / min(dists)
        new_vector = [v * factor for v in vector]
        new_pt = (matrix(new_vector) + matrix(midpt)).tolist()[0]

        return new_pt, dists.index(min(dists))
    else:
        return midnodes[dists.index(min(dists))], dists.index(min(dists))

def _create_element(ele_type, num_pts, vtk_xrange, vtk_yrange, vtk_zrange, vtk_data):
    """ Create elements """
    
    ok, ok2, ok3 = False, False, True
    while (ok != True ) | (ok2 != True) | (ok3 != True):
        # randomly create nodes
        pts = [_rand_xyz([-1,1],[-1,1],[-1,1]) for i in range(num_pts)]
        # make element centroid = (0,0,0)
        pts = (matrix(pts) - mean(matrix(pts),0)).tolist()
        # check all edges have a magnitude
        if [0.,0.,0.] in pts:
            continue
        # if quad tet, re-order points so four outermost are the corners
        if ele_type == 'quad_tet':
            pts = _reorder_quad(pts)
        if ele_type == 'linear_tet':
            pts = _best_order_linear_tet(pts)            
        # scale up element to random magnitude
        nodes = (matrix(pts) * _rand_magnitude()).tolist()
        # move element to random location
        centroid = _rand_xyz(vtk_xrange, vtk_yrange, vtk_zrange)
        pts = (matrix(pts) + matrix(centroid)).tolist()
        # calculate the jacobian to check element quality
        J = _calculate_jacobian(pts, ele_type)
        if J > 0.2: # (Burkhart et al.)
            ok = True
        else:
            continue
        # check element is within bony region (HU > 700)
        HU = [vtk_data.interpolateScalar(p) for p in pts]
        if mean(HU) > 700:
            ok2 = True

    return pts

def _create_parameters_file():
    # check if parameters file already exists or not
    fullfile = os.path.join('py_bonemat_abaqus','example','example_parameters.txt')
    if os.path.isfile(fullfile):
        return
    else:        
        parameters = """## parameters.txt
## 	- an input file for the python program 'bonemat_abaqus'
##      - note: any text beginning with a # will be ignored

## BoneMat options
## ===============
    ## define modulus calculation
    ## --------------------------
    # integration = None # equivalent to BoneMat V1
    # integration = HU   # equivalent to BoneMat V2
    integration = E      # equivalent to BoneMat V3

    ## define how to group modulus values into bins
    ## -------------------------------------------
    gapValue = 1
    # groupingDensity = mean
    groupingDensity = max 
    
    ## integration order
    ## -----------------
    intSteps = 4 
    
## CT calibration coefficients [rhoQCT = rhoQCTa + (rhoQCTb * HU)]
## ===============================================================
    rhoQCTa = 1.3152
    rhoQCTb = 0.6944

## CT calibration correction [rhoAsh = rhoAsha + (rhoAshb * rhoQCT)]
## =================================================================
    # calibrationCorrect = True # if true will perform correction
    calibrationCorrect = False # if false then will skip this step

    ## define whether to use a single, or triple, set of correction parameters
    ##------------------------------------------------------------------------
    # numCTparam = single
    numCTparam = triple

    ## define the thresholds
    ## --------------------
    rhoThresh1 = 0 # note: must be lowest threshold
    rhoThresh2 = 5 # note: must be highest threshold

    ## define the first set
    ## --------------------
    rhoAsha1 = 0
    rhoAshb1 = 1
    
    ## define the second set
    ## ---------------------
    rhoAsha2 = 0
    rhoAshb2 = 1

    ## define the third set
    ## --------------------
    rhoAsha3 = 0
    rhoAshb3 = 1

## Modulus calculation [E = a + (b * RhoAsh) ^ c]
## =============================================
    ## define whether to use a single, or triple, set of parameters
    ## ------------------------------------------------------------
    numEparam = single
    #numEparam = triple

    ## define the thresholds
    ## ---------------------
    # (not necessary for single definition)
    # Ethresh1 = 0 # note: must be lowest threshold
    # Ethresh2 = 0 # note: must be highest threshold
    
    ## define the first set
    ## --------------------
    Ea1 = 0
    Eb1 = 0.0105
    Ec1 = 2.29

    ## define the second set
    ## ---------------------
    #Ea2 = 0
    #Eb2 = 1
    #Ec2 = 1

    ## define the third set
    ## --------------------
    #Ea3 = 0
    #Eb3 = 1
    #Ec3 = 1

    ## other
    ## -----
    minVal = 0.000001 # HU below this are assumed to be equal to minVal
    poisson = 0.35 # poisson's ratio to use for bone in the model"""
        with open(fullfile, 'w') as oupf:
            oupf.write(parameters)
        return
    
###------------------------------------------------------------------------------
### Import vtk data
###------------------------------------------------------------------------------
##vtk_data = data_import._import_vtk_ct_data(ct_file)
##param_file = 'param.txt'
###------------------------------------------------------------------------------
### Create linear tetrahedrons
###------------------------------------------------------------------------------
##for ele in range(num_ele):
##    # create element points
##    pts = _create_element('linear_tet', 4, vtk_xrange, vtk_yrange, vtk_zrange, vtk_data)
##    # write mesh
##    filename = 'linear_tet' + repr(ele) + '.ntr'
##    _write_ntr_input(pts, '5', filename, 4)
###------------------------------------------------------------------------------
### Create quadratic tetrahedrons
###------------------------------------------------------------------------------
##for ele in range(num_ele):
##    # create element points
##    pts = _create_element('quad_tet', 10, vtk_xrange, vtk_yrange, vtk_zrange, vtk_data)
##    # write mesh
##    filename = 'quad_tet' + repr(ele) + '.ntr'
##    _write_ntr_input(pts, '5', filename, 10)
###------------------------------------------------------------------------------
### Create linear wedges
###------------------------------------------------------------------------------
##for ele in range(num_ele):
##    # create element points
##    pts = _create_element('linear_wedge', 6, vtk_xrange, vtk_yrange, vtk_zrange, vtk_data)
##    # write mesh
##    filename = 'linear_wedge' + repr(ele) + '.ntr'
##    _write_ntr_input(pts, '7', filename, 6)
###------------------------------------------------------------------------------
### Create linear hexahedrons
###------------------------------------------------------------------------------
##for ele in range(num_ele):
##    # create element points
##    pts = _create_element('linear_hex', 8, vtk_xrange, vtk_yrange, vtk_zrange, vtk_data)
##    # write mesh
##    filename = 'linear_hex' + repr(ele) + '.ntr'
##    _write_ntr_input(pts, '8', filename, 8)
###------------------------------- Run the files --------------------------------
###------------------------------------------------------------------------------
### Run the linear tetrahedron elements through py_bonemat_abaqus
###------------------------------------------------------------------------------
##for ele in range(num_ele):
##    mesh_file = 'linear_tet' + repr(ele) + '.ntr'
##    param = data_import.import_parameters(param_file)
##    parts = data_import.import_mesh(mesh_file, param)
##    parts = calc.calc_mat_props(parts, param, vtk_data)
##    data_output.output_abq_inp(parts, mesh_file, param['poisson'])
###------------------------------------------------------------------------------
### Run the quadratic tetrahedron elements through py_bonemat_abaqus
###------------------------------------------------------------------------------
##for ele in range(num_ele):
##    mesh_file = 'quad_tet' + repr(ele) + '.ntr'
##    param = data_import.import_parameters(param_file)
##    parts = data_import.import_mesh(mesh_file, param)
##    parts = calc.calc_mat_props(parts, param, vtk_data)
##    data_output.output_abq_inp(parts, mesh_file, param['poisson'])
###------------------------------------------------------------------------------
### Run the linear wedge elements through py_bonemat_abaqus
###------------------------------------------------------------------------------
##for ele in range(num_ele):
##    mesh_file = 'linear_wedge' + repr(ele) + '.ntr'
##    param = data_import.import_parameters(param_file)
##    parts = data_import.import_mesh(mesh_file, param)
##    parts = calc.calc_mat_props(parts, param, vtk_data)
##    data_output.output_abq_inp(parts, mesh_file, param['poisson'])
###------------------------------------------------------------------------------
### Run the linear hexahedron elements through py_bonemat_abaqus
###------------------------------------------------------------------------------
##for ele in range(num_ele):
##    mesh_file = 'linear_hex' + repr(ele) + '.ntr'
##    param = data_import.import_parameters(param_file)
##    parts = data_import.import_mesh(mesh_file, param)
##    parts = calc.calc_mat_props(parts, param, vtk_data)
##    data_output.output_abq_inp(parts, mesh_file, param['poisson'])
#------------------------------------------------------------------------------
# Collate the modulus values into a results file
#------------------------------------------------------------------------------
##matfiles = os.listdir('.')
##with open('results.csv','w') as oupf:
##    for f in matfiles:
##        if 'MAT' in f:
##            with open(f,'r') as inpf:
##                lines = inpf.read()
##            oupf.write(f)
##            oupf.write(',')
##            oupf.write(lines.split('*Elastic\r\n')[1].split(',')[0])
##            oupf.write('\r\n')
