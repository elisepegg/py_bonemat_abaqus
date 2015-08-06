#!/usr/bin/python
#
# py_bonemat_abaqus - data import
# ===============================
#
# Version: 1.0.1
# Created by Elise Pegg, University of Oxford, Aug 2015

__all__ = ['import_parameters','import_mesh','import_ct_data']
           
#-------------------------------------------------------------------------------
# Import modules
#-------------------------------------------------------------------------------
import sys, os, dicom
from numpy import mean, linspace, arange, size, zeros, floor, diff, prod, matrix
import re
import string
from py_bonemat_abaqus.general import _read_text, _remove_spaces, _refine_spaces
from py_bonemat_abaqus.general import _remove_eol_r
from py_bonemat_abaqus.general import _rot_matrix
from py_bonemat_abaqus.classes import linear_tet, quad_tet, linear_wedge, linear_hex
from py_bonemat_abaqus.classes import vtk_data, part

#-------------------------------------------------------------------------------
# Functions for importing parameters
#-------------------------------------------------------------------------------
def import_parameters(fle):
    """ Imports parameter data """
    
    # read parameters file
    lines = _read_text(fle) + '\n'
    lines = _remove_spaces(lines)
    lines = _remove_eol_r(lines)
    
    # identify and record data
    param = _get_param(lines)
    
    return param

def _get_param(lines):
    """ Identifies and records data from file """   
    # search lines for ? = ?
    data = re.findall(r'^(\w+)=([-\d\w.;]+).*\n', lines, re.MULTILINE)
    # store data
    param = {}
    for d in data:
        try:
            param[d[0]] = float(d[1])
        except:
            if d[1].lower() == 'true':
                param[d[0]] = True
            elif d[1].lower() == 'false':
                param[d[0]] = False
            elif d[1].lower() == 'none':
                param[d[0]] = None
            else:
                param[d[0]] = d[1]
                
    # if ignore parameter present, separate out names
    if 'ignore' in param.keys():
        param['ignore'] = param['ignore'].split(';')
        
    return param

#-------------------------------------------------------------------------------
# Functions for importing abaqus mesh data
#-------------------------------------------------------------------------------
def import_mesh(fle, param):
    """ Reads in mesh data """

    filetype = _what_mesh_filetype(fle)
    if filetype == '.inp':
        mesh = _import_abq_mesh(fle, param)
    elif filetype == '.ntr':
        mesh = _import_ntr_mesh(fle)
    else:
        raise IOError('Mesh Import Error: Unknown filetype')
        
    return mesh

def _what_mesh_filetype(fle):
    """ Determines mesh file format """
    lines = _read_text(fle)
    if '*Heading' in lines:
        return '.inp'
    elif 'Neutral' in lines:
        return '.ntr'
    elif '.ntr' in fle:
        return '.ntr'
    else:
        return None
    
def _import_abq_mesh(fle, param):
    """ Records mesh data from abaqus input file """
    
    # read abaqus file
    lines = _read_text(fle)
    lines = _remove_spaces(lines)
    lines = _remove_eol_r(lines)

    # check for pre-existing materials
    if '*Materials' in lines:
        print('Warning: Input file: ' + fle + ' already has materials defined')
        print('         Beware duplicate material definition')
       
    # determine how many parts
    parts = _find_part_names(lines)
    
    # store data for each part
    part_data = [0] * len(parts)
    for p in parts:
        if 'ignore' in param.keys():
            if p in param['ignore']:
                part_data[parts.index(p)] = _get_part_data(lines, p, True)
            else:
                part_data[parts.index(p)] = _get_part_data(lines, p)
        else:
            part_data[parts.index(p)] = _get_part_data(lines, p)
        
    return part_data

def _find_part_names(lines):
    """ Identifies and records part names within abaqus input file """
    
    parts = re.findall("\*Part,name=(\S+)\n", lines, re.IGNORECASE)
    
    return parts

def _get_part_data(lines, partname, ignore=False):
    """ Read elements and node data for defined part """
    
    # get lines relevant to part
    partlines = _get_lines(r'\*Part,name='+partname+'\n', r'EndPart\n', lines)
    test_output(partlines)

    # check for any co-ordinate transformations
    transform = _get_transform_data(lines)
    
    # read nodes
    nodes = _get_nodes(partlines)

    # apply any transformations to nodes
    nodes = _apply_transform(nodes, transform)
    
    # read elements
    elename = _get_elename(partlines)
    test_output(elename)
    elements = _get_elements(partlines, elename)
    eletype = _confirm_eletype(elements)
    
    # create part class for part
    part = _create_part(partname, elements, elename, eletype, nodes, transform, ignore)

    return part

def _get_transform_data(lines):
    instancelines = _get_lines(r'\*Instance', r'EndInstance\n', lines)
    test_output(instancelines)
    if len(instancelines.split('\n')) <3:
        transform = [[0,0,0]]
    elif len(instancelines.split('\n')) == 3:
        # just translation
        transform = [[float(n) for n in instancelines.split('\n')[1].split(',')]]
    else:
        # rotation and/or transformation
        transform = [[float(n) for n in instancelines.split('\n')[1].split(',')]]
        transform.append([float(n) for n in instancelines.split('\n')[2].split(',')])

    return transform

def _apply_transform(nodes, transform):
    # apply translation
    for n in nodes.keys():
        nodes[n] = [nodes[n][i]+transform[0][i] for i in [0,1,2]]        
    if len(transform) == 2:
        # apply rotation
        point = transform[1][:3]
        direction = [transform[1][3]-transform[1][0],
                     transform[1][4]-transform[1][1],
                     transform[1][5]-transform[1][2]]
        angle = transform[1][6]
        R = _rot_matrix(point, angle, direction)
        for n in nodes.keys():
            nodes[n] = [i[0] for i in (R*matrix(nodes[n]).T).tolist()]
    return nodes

    

def _confirm_eletype(elements):
    if len(elements[0]) == 5:
        return 'linear_tet'
    elif len(elements[0]) == 11:
        return 'quad_tet'
    elif len(elements[0]) == 9:
        return 'linear_hex'
    elif len(elements[0]) == 7:
        return 'linear_wedge'
    else:
        raise IOError('Can only analyse tetrahedral (linear or quad) and linear hex or wedge elements')

def _get_nodes(lines):
    """ Identifies node data from lines """
    
    # get lines relevant to nodes
    nodelines = _get_lines(r'\*Node\n', r'\*', lines)
    test_output(nodelines)
    
    # identify node data from lines
    nodes_str = [n.split(',') for n in nodelines.split('\n')]
    
    # convert to float
    nodes = _conv_float(nodes_str)
    
    return nodes

def _get_lines(startstring, endstring, lines):
    """ Identifies smallest matching string which meets start/end criteria """
    
    # create pattern
    start = re.compile(startstring, re.IGNORECASE)
    end = re.compile(endstring, re.IGNORECASE)

    # locate all incidences of start and end strings (case insensitive)
    start_i, end_i = [],[]
    i, j = 0, 0
    if startstring == '':
        start_i = [0]
    else:
        while start.search(lines, i) != None:
            start_i.append(start.search(lines, i).end())
            i = start_i[-1]+1
    if endstring == '$':
        end_i = [len(lines)-1]
    else:
        while end.search(lines, j) != None:
            end_i.append(end.search(lines, j).start())
            j = end_i[-1]+1

    # check start and end strings have been found
    if (start_i == []) | (end_i == []):
        return None
    
    # determine minimum string
    comb = []
    [[comb.append((i,j)) for i in start_i] for j in end_i]
    d = [x[1] - x[0] for x in comb]
    pos = [i for i in d if i>0]

    # check string has length
    if pos == []:
        return ''
    
    # return the lines
    startstop = comb[d.index(min(pos))]
    return lines[startstop[0]:startstop[1]]

def _get_elename(lines):
    """ Identifies type of element """
    
    return _get_lines(r'\*Element,type=',r'\n', lines)
    
def _conv_float(data_str):
    """ Converts string data to float, assumes 1st entry is label """
    
    res = {}
    for d in data_str:
        if len(d) > 1:
            res[d[0]] = [float(n) for n in d[1:]]
        
    return res

def _get_elements(lines, elename):
    """ Identifies element data from lines """
    
    # get lines relevant to elements
    elelines = _get_lines(r'\*Element,type='+elename,r'\*', lines)
    test_output(elelines)

    # identify element data from lines
    elements = [e.split(',') for e in elelines.split('\n')]

    # remove blank entries
    elements = [e for e in elements if len(e)>1]
    
    return elements

def _create_part(name, elements, elename, eletype, nodes, transform=[[0.,0.,0]], ignore=False):
    """ Creates part class from input data """
    
    # create the part
    new_part = part(name, elename, eletype, transform, ignore)
    
    # add elements to part
    for e in elements:
        pts = [nodes[n] for n in e[1:]]
        exec('ele = ' + eletype + '(int(e[0]), pts, e[1:])')
        new_part.add_element(ele)
        
    return new_part


#-------------------------------------------------------------------------------
# Functions for importing Patran neutral file mesh data
#-------------------------------------------------------------------------------
def _import_ntr_mesh(fle):
    """ Records mesh data from neutral file format """

    # store data for each part
    part_data = _get_ntr_part_data(fle)

    return [part_data]

def _get_ntr_part_data(fle):
    """ Read elements and node data from .ntr file """

    # read in .ntr file
    lines = _read_text(fle)
    lines = lines.replace(' ', ',')
    for n in range(15):
        lines = lines.replace(',,', ',')
    lines = _remove_eol_r(lines)

    # read nodes
    nodes = {}
    nodelines = re.findall(r'\n(,1,[\d,]+\n[,\d.EG\+-]+\n1[G,\d]+)', lines)
    for n in nodelines:
        node_num = n.split('\n')[0].split(',')[2]
        nodes[node_num] = _get_node_ntr(n.split('\n')[1])

    # read elements
    elements = []
    elelines = re.findall(r'\n,2,([\d,E]+\n[\d\.\+,E]+\n[\d,E]+)', lines)
    for e in elelines:
        # get element label
        ele_num = int(e.split(',')[0])
        # get node connectivity
        elements.append(_get_element_ntr(e.split('\n')[2], ele_num))
        # determine element type
        ele_type_num = int(e.split(',')[1])
        num_ele_nodes = int(e.split('\n')[1].split(',')[1])
    
    # determine element name
    if (ele_type_num == 5) & (num_ele_nodes == 4):
        ele_type = 'linear_tet'
        elename = 'C3D4'
    elif (ele_type_num == 5) & (num_ele_nodes == 10):
        ele_type = 'quad_tet'
        elename = 'C3D10'
    elif (ele_type_num == 7) & (num_ele_nodes == 6): 
        ele_type = 'linear_wedge'
        elename = 'W3D6'
    elif (ele_type_num == 8) & (num_ele_nodes == 8):
        ele_type = 'linear_hex'
        elename = 'C3D8'
    else:
        raise IOError("Element type not recognised or not compatible with py_bonemat_abaqus.py");
    
    # create part
    part = _create_part('Part1', elements, elename, ele_type, nodes)

    return part

def _get_node_ntr(line):
    """ Identifies data for one node from .ntr line """
    
    line = line.strip()
    nodes = re.findall(r'[-\d\.]+E...', line)
    node = [float(n) for n in nodes]
    
    return node

def _get_element_ntr(line, indx):
    """ Identifies data for one element from .ntr line """
    
    line = line.strip()
    element = re.findall(r'[\d]+', line)
    element = [indx] + element
    
    return element

#-------------------------------------------------------------------------------
# Functions for importing CT data
#-------------------------------------------------------------------------------
def import_ct_data(ct_data):
    """ Reads CT dataset from specified file or directory """
    
    if ".vtk" in ct_data:
        data = _import_vtk_ct_data(ct_data)
    else:
        if os.path.isfile(ct_data + ".vtk") == False:
            print("    Converting dicoms to vtk file: " + ct_data + ".vtk")
            _convert_dicom_ct_data(ct_data)
        data = _import_vtk_ct_data(ct_data + ".vtk")
        
    return data

def _convert_dicom_ct_data(ct_data):
    data = _import_dicom_ct_data(ct_data)
    _write_vtk(data, ct_data + ".vtk")

def _write_vtk(vtk, fle):
    with open(fle,'w') as oupf:
        dimen = [len(vtk.x), len(vtk.y), len(vtk.z)]
        oupf.write('# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET RECTILINEAR_GRID\r\n')
        oupf.write('DIMENSIONS ' + repr(dimen[0]) + ' ' + repr(dimen[1]) + ' ' + repr(dimen[2]) + '\r\n')
        oupf.write('X_COORDINATES ' + repr(dimen[0]) + ' double\r\n')
        oupf.write(_create_vtk_coord_str(vtk.x, 9))
        oupf.write('Y_COORDINATES ' + repr(dimen[1]) + ' double\r\n')
        oupf.write(_create_vtk_coord_str(vtk.y, 9))
        oupf.write('Z_COORDINATES ' + repr(dimen[2]) + ' float\r\n')
        oupf.write(_create_vtk_coord_str(vtk.z, 9))
        oupf.write('CELL_DATA ' + repr(prod([d-1 for d in dimen])) + '\r\n')
        oupf.write('POINT_DATA ' + repr(prod(dimen)) + '\r\n')
        oupf.write('SCALARS scalars short\r\n')
        oupf.write('LOOKUP_TABLE default\r\n')
        oupf.write(_create_vtk_coord_str(vtk.lookup, 9, False))
        
def _create_vtk_coord_str(coords, max_num, perform_round = True):
    """ Create string for numbers (rounded if specified) in chunks of max_num """
    
    # perform rounding to 4 decimal places (default)
    if perform_round:
        coords = ["%0.4f" % (c,) for c in coords]
    else:
        coords = [repr(c) for c in coords]
        
    # create string
    coords_str = ''
    for c in range(len(coords)/max_num):
        coords_str += " ".join(coords[c * max_num : (c + 1) * max_num]) + '\r\n'
    if (len(coords) % max_num) > 0:
        coords_str += " ".join(coords[-(len(coords)%max_num):])
    coords_str += '\r\n'
    
    return coords_str

def _import_dicom_ct_data(dicom_dir):
    """ Returns voxel data from dicom CT images which is within elements """
    
    # find files in folder
    dicom_files = os.listdir(dicom_dir)
    
    # load the dicom images
    dicoms = _read_dicoms(dicom_files, dicom_dir)

    # sort by z position
    dicoms = _sort_dicoms(dicoms)
    
    # check validity of image metadata
    if _check_dicom_data(dicoms) is not None:
        raise IOError("Dicom Import Error: " + _check_dicom_data(dicoms))

    # return CT voxel data
    return _convert_data_vtk(dicoms)

def _check_dicom_data(dicoms):
    """ Check that the dicom images are valid """
    
    if not len(set([d.Rows for d in dicoms])) == 1:
        return "Dicom rows different sizes"
    elif not len(set([d.Columns for d in dicoms])) == 1:
        return "Dicom columns different sizes"
    elif not len(set([d.PixelSpacing[0] for d in dicoms])) == 1:
        return "Dicom images have different pixel spacings"
    elif not len(set([d.PixelSpacing[1] for d in dicoms])) == 1:
        return "Dicom images have different pixel spacings"
    elif not len(set([d.SliceThickness for d in dicoms])) == 1:
        return "Dicom images have different thicknesses"
    elif not len(set([d.ImagePositionPatient[0] for d in dicoms])) == 1:
        return "Dicom images have different origins"
    elif not len(set([d.ImagePositionPatient[1] for d in dicoms])) == 1:
        return "Dicom images have different origins"
    elif not len(set(diff([float(d.ImagePositionPatient[2]) for d in dicoms]))) == 1:
        return "Dicom slices are not sequential"

def _read_dicoms(fles, path):
    """ Reads dicom data for all files within directory """

    dicoms = [dicom.read_file(os.path.join(path, ds), force = True) for ds in fles]
    dicoms = [d for d in dicoms if len(d) > 1]

    return dicoms

def _sort_dicoms(dicoms):
    """ Iterates through dicom data and returns sorted data """

    z = [d.ImagePositionPatient[2] for d in dicoms]

    return zip(*sorted(zip(z, dicoms)))[1]

def _convert_data_vtk(dicoms):
    """ Changes dicom data, into vtk-like format """

    rows = dicoms[0].Rows
    columns = dicoms[0].Columns
    slices = [float(d.ImagePositionPatient[2]) for d in dicoms]
    xspacing, yspacing = [float(s) for s in dicoms[0].PixelSpacing]
    slice_thickness = float(dicoms[0].SliceThickness)

    # calculate X-coordinates
    xstart = dicoms[0].ImagePositionPatient[0]
    xstop = xstart + (rows * xspacing)
    X = arange(xstart, xstop, xspacing)
    
    # calculate Y-coordinates
    ystart = dicoms[0].ImagePositionPatient[1]
    ystop = ystart + (columns * yspacing)
    Y = arange(ystart, ystop, yspacing)
    
    # calculate Z-coordinates
    zstart = min(slices)
    zstop = zstart + (slice_thickness * len(dicoms))
    Z = arange(zstart, zstop, slice_thickness)
 
    # calculate lookup table
    lookup = _create_lookup(dicoms)
    
    return vtk_data(X, Y, Z, lookup)

def _create_lookup(dicoms):
    """ Creates a vtk lookup table from dicom dataset """

    data = [d.pixel_array for d in dicoms]
    lookup = []
    for k in range(len(data)):
        for j in range(len(data[0][0])):
            for i in range(len(data[0])):
                lookup.append(data[k][i][j])
                
    return lookup

def _import_vtk_ct_data(ct_data):
    """ Creates python array data from vtk file """
    
    # read in text
    lines = _read_text(ct_data)
    lines = _remove_eol_r(lines)
        
    # read in X data
    xlines = _refine_vtk_lines(lines, 'X_COORDINATES', 'double', 'Y_COORDINATES')
    X = [float(x) for x in xlines.split(' ') if len(x)>0]
    
    # read in Y data
    ylines = _refine_vtk_lines(lines, 'Y_COORDINATES', 'double', 'Z_COORDINATES')
    Y = [float(y) for y in ylines.split(' ') if len(y)>0]
    
    # read in Z data
    zlines = _refine_vtk_lines(lines, 'Z_COORDINATES', 'float', 'CELL_DATA')
    Z = [float(z) for z in zlines.split(' ') if len(z)>0]
    
    # read in lookup data
    lookup_lines = _refine_vtk_lines(lines, 'LOOKUP_TABLE','default','NaN')
    lookup = [int(n) for n in lookup_lines.split(' ') if len(n)>0]

    # return data
    return vtk_data(X, Y, Z, lookup)
    
    
def _refine_vtk_lines(lines, key1, key2, key3):
    """ Find lines bewteen key words and remove end-of-line characters """
    
    # find lines
    lines = lines.split(key1)[1].split(key2)[1].split(key3)[0]
    
    # refine lines
    lines = lines.replace('\n', ' ')
    for n in range(3): lines = lines.replace('  ', ' ')
    
    return lines

def test_output(lines):
    if lines is None:
        raise ValueError("Error reading input file, check format")
