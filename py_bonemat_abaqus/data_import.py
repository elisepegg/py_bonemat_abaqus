#!/usr/bin/python
#
# py_bonemat_abaqus - data import
# ===============================
#
# Created by Elise Pegg, University of Bath

__all__ = ['import_parameters','import_mesh','import_ct_data']
           
#-------------------------------------------------------------------------------
# Import modules
#-------------------------------------------------------------------------------
import sys
import os
import dicom
from numpy import mean, array, concatenate, linspace, arange, size
from numpy import zeros, floor, diff, prod, matrix
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
    
    # check all necessary parameters have been defined
    _checkParamInformation(param)
    
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
    
def _checkParamInformation(param):
    """ Iterates through parameters file to check contains all required information """
    
    # assign default parameters if not defined
    if 'integration' not in param.keys():
        param['integration'] = 'E'
        print("        Note: 'integration' parameter not defined. Assigning to default, E (Equivalent to Bonemat V3)")
        
    if 'groupingDensity' not in param.keys():
        param['groupingDensity'] = 'mean'
        print("        Note: 'groupingDensity' parameter not defined. Assigning to default, 'mean'")
    
    # check for essential information
    _checkNecessaryParam(param, ['integration', 'gapValue','groupingDensity','intSteps','rhoQCTa','rhoQCTb',
                                'calibrationCorrect','minVal','poisson','numEparam'],)
    
    # if appropriate, check for calibration information
    if param['calibrationCorrect'] == True:
        _checkNecessaryParam(param, ['numCTparam'])
        if param['numCTparam'] == 'single':
            _checkNecessaryParam(param, ['rhoAsha1','rhoAshb1'])
        elif param['numCTparam'] == 'triple':
            _checkNecessaryParam(param, ['rhoThresh1','rhoThresh2','rhoAsha1','rhoAshb1', 'rhoAsha2','rhoAshb2',
                                        'rhoAsha3','rhoAshb3'])
        else:
            raise IOError("Error: " + param['numCTparam'] + " is not a valid input for numCTparam.  Must be 'single' or 'triple'")  
    
    # if appropriate check for modulus calculation information
    if param['numEparam'] == 'single':
        _checkNecessaryParam(param, ['Ea1','Eb1','Ec1'])
    elif param['numEparam'] == 'triple':
        _checkNecessaryParam(param, ['Ea1','Eb1','Ec1', 'Ea2','Eb2','Ec2','Ea3','Eb3','Ec3'])
    else:
        raise IOError("Error: " + param['numEparam'] + " is not a valid input for numCTparam. Must be 'single' or 'triple'")  
        
    # check all values which need to be are numerical
    _checkNumericalParam(param)       
    
    
def _checkNumericalParam(param):    
    """ Checks parameters fields which need to be numerical are """
    
    # check ints
    ints = ['intSteps']
    if param['intSteps'] != int(param['intSteps']):
        raise IOError("Error: intSteps parameter must be an integer")
    
    # check floats
    floats = ['gapValue','rhoQCTa','rhoQCTb','rhoThresh1','rhoThresh2','rhoAsha1','rhoAshb1','rhoAsha2','rhoAshb2','rhoAsha3',
              'rhoAshb3', 'Ethresh1','Ethresh2','Ea1','Eb1','Ec1','Ea2','Eb2','Ec2','Ea3','Eb3','Ec3','minVal','poisson']
    for f in floats:
        if f in param.keys():
            if type(param[f]) != float:
                raise IOError("Error: " + param[f] + " must be a numerical value")
    
def _checkNecessaryParam(param, fields):
    for f in fields:
        if f not in param.keys():
            raise IOError("Error: " + f + " is not defined in parameters file")    

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
    elif filetype == '.neutral':
        mesh = _import_ntr_mesh(fle)
    else:
        raise IOError('Mesh Import Error: Unknown filetype')
        
    return mesh

def _what_mesh_filetype(fle):
    """ Determines mesh file format """
    lines = _read_text(fle)
    if '*Heading' in lines:
        return '.inp'
    elif '1G' in lines:
        return '.ntr'
    elif '.ntr' in fle:
        return '.ntr'
    elif '.neutral' in fle:
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
    if '\n*Material' in lines:
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
    assemblylines = _get_lines(r'\*Assembly', r'\*EndAssembly\n', lines)
    transform = _get_transform_data(assemblylines, partname)
    
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

def _get_transform_data(lines, partname):
    """ Finds transformation matrix for nodes, if defined """

    instancelines = _get_lines(r'\*Instance,name=[-\w]+,part=' + partname, r'\*EndInstance\n', lines)
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
    """ Applies transformation matrix for nodes """  

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
    """ Checks element type can be analysed """

    if len(elements[0]) == 5:
        return 'linear_tet'
    elif len(elements[0]) == 11:
        return 'quad_tet'
    elif len(elements[0]) == 9:
        return 'linear_hex'
    elif len(elements[0]) == 7:
        return 'linear_wedge'
    else:
        raise IOError('Can only analyse tetrahedral (linear or quad), linear hex or wedge elements')

def _get_nodes(lines):
    """ Identifies node data from lines """
    
    # get lines relevant to nodes
    lines = lines.replace('*Node','**Node')
    nodetext = re.findall(r'\*Node\n([\d,eE.+\-\n]+)\*', lines)
    nodelines = ''.join(nodetext)
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
        return ''
    
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
    final_lines = lines[startstop[0]:startstop[1]]
    return final_lines

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
        elename = 'C3D6'
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
from numpy import int16, arange

def import_ct_data(ct_data):
    """ Reads CT dataset from specified file or directory """
    
    if ".vtk" in ct_data:
        data = _import_vtk_ct_data(ct_data)
    else:
        data = _import_dicoms(ct_data)
        
    return data

def _import_dicoms(ct_data):
    """ Imports dicom data and rearranges voxels to vtk lookup """
    
    dicom_order, dicom_data = _import_dicom_ct_data(ct_data)
    lookup = _import_dicom_lookup(dicom_order, dicom_data, ct_data)
    X,Y,Z = _calc_dicom_xyz(dicom_data)
    vtk = vtk_data(X,Y,Z, lookup)
    
    return vtk
    
def _import_dicom_lookup(dicom_order, dicom_data, ct_data):
    """ Iterates through sorted dicom slices and stores lookup data """

    lookup = array([])
    for s in dicom_order:
        dic = _read_dicom(s, ct_data)
        pixel_array = dic.pixel_array
        # rescale intensity if required
        if (dicom_data[14][0]!=1.0) or (dicom_data[15][0]!=0.0):
            pixel_array = (pixel_array * dicom_data[14][0]) + dicom_data[15][0]
        # add data to lookup array
        lookup = concatenate((lookup, pixel_array.flatten()))

    return lookup
    
def _calc_dicom_xyz(dicom_data):
    """ Calculates range of X, Y, and Z voxel co-ordinates from dicom information """
    
    X = arange(dicom_data[5][0], dicom_data[5][0] + (dicom_data[2][0] * dicom_data[0][0]), dicom_data[2][0])
    Y = arange(dicom_data[6][0], dicom_data[6][0] + (dicom_data[3][0] * dicom_data[1][0]), dicom_data[3][0])
    Z = arange(dicom_data[7][0], dicom_data[7][0] + (dicom_data[4][0] * len(dicom_data[0])), dicom_data[4][0])
    
    return [X,Y,Z]

def _convert_dicom_ct_data(ct_data):
    """ Converts Dicom scans to a VTK ascii file format """

    dicom_order, dicom_data = _import_dicom_ct_data(ct_data)
    lookup = _import_dicom_lookup(dicom_order, dicom_data, ct_data)
    _write_vtk_header(dicom_data, ct_data + ".vtk")
    _write_vtk_lookup(lookup, ct_data)

def _write_vtk_header(dicom_data, fle):
    """ Writes VTK data header """

    x, y, z = _dicom_xyz_data(dicom_data)
    with open(fle,'w') as oupf:
        dimen = [len(x), len(y), len(z)]
        oupf.write('# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET RECTILINEAR_GRID\n')
        oupf.write('DIMENSIONS ' + repr(dimen[0]) + ' ' + repr(dimen[1]) + ' ' + repr(dimen[2]) + '\n')
        oupf.write('X_COORDINATES ' + repr(dimen[0]) + ' double\n')
        _create_vtk_coord_str(x, 9, oupf)
        oupf.write('Y_COORDINATES ' + repr(dimen[1]) + ' double\n')
        _create_vtk_coord_str(y, 9, oupf)
        oupf.write('Z_COORDINATES ' + repr(dimen[2]) + ' float\n')
        _create_vtk_coord_str(z, 9, oupf)
        oupf.write('CELL_DATA ' + repr(prod([d-1 for d in dimen])) + '\n')
        oupf.write('POINT_DATA ' + repr(prod(dimen)) + '\n')
        oupf.write('SCALARS scalars short\n')
        oupf.write('LOOKUP_TABLE default\n')

def _write_vtk_lookup(lookup, path):
    """ Writes VTK lookup data """

    fle = path + ".vtk"
    count = 0
    for d in lookup:
        with open(fle, 'a') as oupf:
            if count < 8:
                oupf.write(repr(int(d)) + ' ')
                count += 1
            else:
                oupf.write(repr(int(d)) + '\n')
                count = 0
    with open(fle, 'a') as oupf:
        oupf.write('\n')

def _create_vtk_coord_str(coords, max_num, oupf, perform_round=True):
    """ Create string for numbers (rounded if specified in chunks of max_num and write to file """

    # perform rounding to 4 decimal places (default)
    if perform_round:
        coords = ["%0.4f" % (c,) for c in coords]
    else:
        coords = [repr(c) for c in coords]
        
    # create string
    for c in range(len(coords)/max_num):
        coords_str = " ".join(coords[c * max_num : (c + 1) * max_num]) + '\n'
        oupf.write(coords_str)
    if (len(coords) % max_num) > 0:
        coords_str = " ".join(coords[-(len(coords)%max_num):])
        oupf.write(coords_str)
        oupf.write('\n')

    return

def _import_dicom_ct_data(dicom_dir):
    """ Returns dicom CT image data and z order of files """
    
    # find files in folder
    dicom_files = os.listdir(dicom_dir)

    # sort by z position
    dicom_order = _sort_dicoms(dicom_files, dicom_dir)

    # gather data
    dicom_data = _gather_dicom_data(dicom_order, dicom_dir)
    
    # check validity of image metadata
    if _check_dicom_data(dicom_data) is not None:
        raise IOError("Dicom Import Error: " + _check_dicom_data(dicom_data))

    # return CT voxel data
    return dicom_order, dicom_data

def _gather_dicom_data(dicom_order, path):
    """ Collect image data for the dicoms in the folder """

    data = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    for d in dicom_order:
        dic = _read_dicom(d, path)
        data[0].append(dic.Rows)
        data[1].append(dic.Columns)
        data[2].append(float(dic.PixelSpacing[0]))
        data[3].append(float(dic.PixelSpacing[1]))
        data[4].append(float(dic.SpacingBetweenSlices))
        data[5].append(float(dic.ImagePositionPatient[0]))
        data[6].append(float(dic.ImagePositionPatient[1]))
        data[7].append(float(dic.ImagePositionPatient[2]))
        data[8].append(float(dic.ImageOrientationPatient[0]))
        data[9].append(float(dic.ImageOrientationPatient[1]))
        data[10].append(float(dic.ImageOrientationPatient[2]))
        data[11].append(float(dic.ImageOrientationPatient[3]))
        data[12].append(float(dic.ImageOrientationPatient[4]))
        data[13].append(float(dic.ImageOrientationPatient[5]))
        data[14].append(float(dic.RescaleSlope))
        data[15].append(float(dic.RescaleIntercept))
    return data

def _check_dicom_data(dicoms):
    """ Check that the dicom images are valid """
    if len(dicoms[0])==1:
        return None
    else:    
        if not len(set(dicoms[0])) == 1:
            return "Dicom rows different sizes"
        elif not len(set(dicoms[1])) == 1:
            return "Dicom columns different sizes"
        elif not len(set(dicoms[2])) == 1:
            return "Dicom images have different pixel spacings"
        elif not len(set(dicoms[3])) == 1:
            return "Dicom images have different pixel spacings"
        elif not len(set(dicoms[4])) == 1:
            return "Dicom images have different thicknesses"
        elif not len(set(dicoms[5])) == 1:
            return "Dicom images have different origins"
        elif not len(set(dicoms[6])) == 1:
            return "Dicom images have different origins"
        elif not len(set(diff(dicoms[7]))) == 1:
            return "Dicom slices are not sequential"
        elif not len(set(dicoms[8])) == 1:
            return "Dicom images have different row cosine orientation"
        elif not len(set(dicoms[9])) == 1:
            return "Dicom images have different row cosine orientation"
        elif not len(set(dicoms[10])) == 1:
            return "Dicom images have different row cosine orientation"
        elif not len(set(dicoms[11])) == 1:
            return "Dicom images have different col cosine orientation"
        elif not len(set(dicoms[12])) == 1:
            return "Dicom images have different col cosine orientation"
        elif not len(set(dicoms[13])) == 1:
            return "Dicom images have different col cosine orientation"
        elif [dicoms[8][0],dicoms[9][0],dicoms[10][0],dicoms[11][0],dicoms[12][0],dicoms[13][0]] != [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]:
            print([dicoms[8][0],dicoms[9][0],dicoms[10][0],dicoms[11][0],dicoms[12][0],dicoms[13][0]])
            return "ImageOrientation parameter must be [1,0,0,0,1,0]"

def _read_dicom(fle, path):
    """ Reads dicom file information """

    d = dicom.read_file(os.path.join(path, fle), force = True)

    return d

def _sort_dicoms(dicom_files, path):
    """ Iterates through dicom files and returns sorted z order """

    z = [dicom.read_file(os.path.join(path, d), force = True).ImagePositionPatient[2] for d in dicom_files]
    dicom_order = zip(*sorted(zip(z, dicom_files)))[1]

    return dicom_order

def _dicom_xyz_data(dicom_data):
    """ Find X,Y,Z voxel grid increments from dicom data """

    rows = dicom_data[0][0]
    columns = dicom_data[1][0]
    slices = dicom_data[7]
    xspacing, yspacing = [dicom_data[2][0], dicom_data[3][0]]
    slice_thickness = dicom_data[4][0]

    # calculate X-coordinates
    xstart = dicom_data[5][0]
    xstop = xstart + (columns * yspacing)
    X = arange(xstart, xstop, yspacing)
    
    # calculate Y-coordinates
    ystart = dicom_data[6][0]
    ystop = ystart + (rows * xspacing)
    Y = arange(ystart, ystop, xspacing)
    
    # calculate Z-coordinates
    zstart = min(slices)
    zstop = zstart + (slice_thickness * len(dicom_data[0]))
    Z = arange(zstart, zstop, slice_thickness)
    
    return X,Y,Z

def _import_vtk_ct_data(ct_data):
    """ Creates python array data from vtk file """
    
    # read in text
    lines = _read_text(ct_data)
    lines = _remove_eol_r(lines)
    header = lines.split('LOOKUP')[0]
        
    # read in X data
    xlines = _refine_vtk_lines(header, 'X_COORDINATES \d+', 'double', 'Y_COORDINATES')
    X = [float(x) for x in xlines]
    
    # read in Y data
    ylines = _refine_vtk_lines(header, 'Y_COORDINATES \d+', 'double', 'Z_COORDINATES')
    Y = [float(y) for y in ylines]
    
    # read in Z data
    zlines = _refine_vtk_lines(header, 'Z_COORDINATES \d+', 'double', 'CELL_DATA')
    Z = [float(z) for z in zlines]

    # read in lookup data
    lookup_lines = _refine_vtk_lines(lines, 'LOOKUP_TABLE','default','')
    lookup = [int(l) for l in lookup_lines]

    # create vtk class
    vtk = vtk_data(X,Y,Z, lookup)

    # return data
    return vtk
    
    
def _refine_vtk_lines(lines, key1, key2, key3):
    """ Find lines bewteen key words and remove end-of-line characters """
    
    # find lines
    p = re.compile(key1+' '+key2+'\n([-.\d\n ]+)'+key3)
    if len(p.findall(lines))>0:
        lines = p.findall(lines)[0]
    else:
        p = re.compile(key1+' '+'float'+'\n([-.\d\n ]+)'+key3)
        if len(p.findall(lines))>0:
               lines = p.findall(lines)[0]
        else:
               raise ValueError("Error reading VTK header, unrecognised format")
    # find numbers
    p2 = re.compile(r'[-.\d]+')

    return p2.findall(lines)

def test_output(lines):
    """ Check lines information has been imported """

    if lines is None:
        raise ValueError("Error reading input file, check format")
    if lines == []:
        raise ValueError("Error reading input file, check format")
