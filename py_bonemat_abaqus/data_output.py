#!/usr/bin/python
#
# py_bonemat_abaqus - data output
# ===============================
#
# Version: 1.0.1
# Created by Elise Pegg, University of Oxford, Aug 2015

__all__ = ['output_abq_inp']
           

#-------------------------------------------------------------------------------
# Import modules
#-------------------------------------------------------------------------------
from py_bonemat_abaqus.general import _read_text, _remove_eol_r
from py_bonemat_abaqus.data_import import _find_part_names, _get_lines
from py_bonemat_abaqus.calc import _get_all_modulus_values
import os

#-------------------------------------------------------------------------------
# Functions to output new abaqus file
#-------------------------------------------------------------------------------
def output_abq_inp(parts, fle, poisson):
    """ Creates a new abaqus input file containing material definitions """
    
    if '.inp' in fle:
        _update_abq_inp(parts, fle, poisson)
    else:
        _create_abq_inp(parts, fle, poisson)

def _update_abq_inp(parts, fle, poisson):
    """ Adds material data to existing abaqus file and writes to 'jobMAT.inp' """
    
    # create output line string
    lines = ''

    # create materials lines
    materials, materials_data = _create_materials(parts, poisson)

    # add header lines
    orig_input = _read_text(fle)
    orig_input = _remove_eol_r(orig_input)
    header = _get_lines('', '\*Part,', orig_input)
    lines = lines + header

    # modify and add part lines for each part
    for p in parts:
        partlines = _get_lines('\*Part,\s?name=' + p.name, '\*End', orig_input)
        if p.ignore == False:
            sets, set_data, set_name = _create_material_elesets(p)
            sections = _create_sections(p, set_data, materials_data, set_name)
            lines = lines + '*Part, name=' + p.name + partlines + sets + sections + '*End Part\n'
        else:
            lines = lines + '*Part, name=' + p.name + partlines + '*End Part\n'

    # add lines until end of assembly
    lines = lines + _get_lines('\*End Part', '\*End Assembly\n', orig_input) + '*End Assembly\n'

    # add material lines
    lines = lines + materials

    # add any remaining lines
    lines = lines + _get_lines('\*End Assembly\n', '$', orig_input)

    # remove any blank lines and make EOL character for all operating systems
    lines = lines.replace('\r','\n')
    lines = lines.replace('\n\n','\n')
    lines = lines.replace('\n\n','\n')
    lines = lines.replace('\n','\r\n')

    # write to file
    with open(fle[:-4]+'MAT.inp','w') as oupf:
        oupf.write(lines)

def _create_abq_inp(parts, fle, poisson):
    """ Creates brand new abaqus input file with material data 'nameMAT.inp' """

    # create and add header lines
    lines = ''
    header = _create_header(fle)
    lines = lines + header
    
    # create materials lines
    materials, materials_data = _create_materials(parts, poisson)
    
    # create and add lines for each part
    for p in parts:
        nodes = _create_node_lines(p)
        eles = _create_element_lines(p)
        sets, set_data, set_name = _create_material_elesets(p)
        sections = _create_sections(p, set_data, materials_data, set_name)
        lines = lines + nodes + eles + sets + sections + '*End Part\n'
        
    # add assembly and material lines
    assembly = _create_assembly(parts)
    lines = lines + assembly + materials

    # remove any blank lines and make EOL character for all operating systems
    lines = lines.replace('\r','\n')
    lines = lines.replace('\n\n','\n')
    lines = lines.replace('\n\n','\n')
    lines = lines.replace('\n','\r\n')

    # write to file
    with open(fle[:-4]+'MAT.inp','w') as oupf:
        oupf.write(lines)


#-------------------------------------------------------------------------------
# Functions to create sections of output file
#-------------------------------------------------------------------------------
def _create_header(fle):
    """ Text header for abaqus input file """

    header1 = "*Heading\n** Job name: " + os.path.split(fle)[-1][:-4] + " Model name: " + os.path.split(fle)[-1][:-4] + "\n"
    header2 = "** Generated by: BoneMat Abaqus\n"
    header3 = "*Preprint, echo=NO, model=NO, history=NO, contact=NO\n"
    header4 = "**\n** PARTS\n**\n"

    return header1 + header2 + header3 + header4

def _create_node_lines(part):
    """ Node text for abaqus input file """
    
    node_data = _get_node_data(part)
    node1 = "*Part, name=" + part.name +"\n*Node\n"
    node2 = _get_node_text(node_data)
    
    return node1 + node2

def _create_element_lines(part):
    """ Element text for abaqus input file """

    ele_data = _get_ele_data(part)
    ele1 = "*Element, type=" + part.ele_name + "\n"
    ele2 = _get_ele_text(ele_data)

    return ele1 + ele2

def _create_materials(parts, poisson):
    """ Text to define materials """

    materials = '**\n** MATERIALS\n**\n'
    moduli = _get_all_modulus_values(parts)
    materials_data = sorted(list(set(moduli)))
    for m in range(len(materials_data)-1):
        materials = materials + '*Material, name=BoneMat_' + repr(m + 1) + '\n'
        materials = materials + '*Elastic\n' + repr(materials_data[m]) + ', ' + repr(poisson)+ '\n'
    materials = materials + '*Material, name=BoneMat_' + repr(len(materials_data)) + '\n'
    materials = materials + '*Elastic\n' + repr(materials_data[len(materials_data)-1]) + ', ' + repr(poisson)+ '\n'                                                       

    return materials, materials_data

def _create_material_elesets(part):
    """ Creates text to define element sets for each material """

    # list of materials in part
    mat_values = sorted(list(set(part.moduli)))

    # dictionary assigning material to set
    mat_sets = dict([(repr(m + 1), mat_values[m]) for m in range(len(mat_values))])
    
    # dictionary assigning element to each set
    ele_sets = dict([(repr(m + 1),[]) for m in range(len(mat_values))])

    # iterate through elements and add indexes
    for n in range(len(part.elements)):
        mat_index = mat_values.index(part.moduli[n])
        ele_index = part.elements[n].indx
        ele_sets[repr(mat_index + 1)].append(ele_index)

    # create text
    set_name = 'BoneMatSet_' + part.name + '_'
    eleset_text = _get_set_text(ele_sets, set_name)

    return eleset_text, mat_sets, set_name

def _create_sections(part, sets, materials_data, set_name):
    """ Text to define sections """
    
    sections = ''
    set_nums = sorted([int(s) for s in sets.keys()])
    for s in set_nums:
        mat = materials_data.index(sets[repr(s)]) + 1
        sections = sections + '*Solid Section, elset=' + set_name + repr(s)
        sections = sections + ', material=BoneMat_' + repr(mat) + '\n'

    return sections

def _create_assembly(parts):
    """ Assembly text for input file """

    assembly1 = "**\n** ASSEMBLY\n**\n*Assembly, name=Assembly\n"
    assembly2 = ""
    for p in parts:
        assembly2 = assembly2 + "*Instance, name=" + p.name
        assembly2 = assembly2 + ", part=" + p.name + "\n"
    assembly3 = "*End Instance\n**\n*End Assembly\n"

    return assembly1 + assembly2 + assembly3

#-------------------------------------------------------------------------------
# Functions to output in .ntr format
#-------------------------------------------------------------------------------
def _get_nodes(ele_type):
    """ Quantifies number of nodes based on element type """
    
    if (ele_type == 'linear_tet'):
        return 4
    elif (ele_type == 'linear_hex'):
        return 8
    elif (ele_type == 'linear_wedge'):
        return 6
    elif (ele_type == 'quad_tet'):
        return 10
    else:
        raise IOError("Error: element type not recognised")
    
def _get_elecode(ele_type):
    """ Returns Ansys element type code based on element type """
    
    if ((ele_type == 'linear_tet') | (ele_type == 'quad_tet')):
        return '5'
    elif (ele_type == 'linear_hex'):
        return '8'
    elif (ele_type == 'linear_wedge'):
        return '7'
    else:
        raise IOError("Error: element type not recognised")

def _write_chunks(data, oupf, write_n = True):
    """ Writes the element data in chunks """
    
    for i in range(2 - len(data[0])):
        oupf.write(" ")
    oupf.write(data[0])
    for d in data[1:]:
        for i in range(8 - len(d)):
            oupf.write(" ")
        oupf.write(d)
    if write_n:
        oupf.write("\n")

def _write_nodes(nodes, oupf):
    """ Writes the nodes in ntr format """
    
    for n in nodes:
        # determine E base
        if (n != 0):
            base = floor(log10(abs(n)))
        else:
            base = 0;
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
        for i in range(2 - len(repr(int(base)))):
            oupf.write("0")
        oupf.write(repr(int(base)))
    oupf.write('\n')
        
def _write_ntr_input(elements, ele_code, fname, num_pts):
    """ Writes a part as an .ntr mesh file """
    
    with open(fname, 'w') as oupf:
        # determine all the node data
        nodes = {}
        for e in elements:
            for n in range(len(e.nodes)):
                nodes[e.nodes[n]] = e.pts[n]
        node_data = []
        for n in nodes.keys():
            node_data.append([int(n), nodes[n][0], nodes[n][1], nodes[n][2]])
        node_data = sorted(node_data)
        # determine all the element data
        ele_data = [[repr(e.indx)] + e.nodes for e in elements]
        ele_data = [[int(n) for n in d] for d in ele_data]
        # determine the number of elements and nodes
        num_eles = len(elements)
        num_nodes = len(node_data)      
        # write the title
        title_data = ['25', '0', '0', '1', '0', '0', '0', '0', '0']
        _write_chunks(title_data, oupf)
        oupf.write("PATRAN Neutral, HyperMesh Template PATRAN/GENERAL\n")
        # write the summary
        summary_data = ['26', '0', '0', '1', repr(num_nodes), repr(num_eles), '1', '1', '0']
        _write_chunks(summary_data, oupf)
        oupf.write("  07-27-201510:28:00\n")
        # write the node data
        for n in node_data:
            line1 = ['1', repr(n[0]), '0', '2', '0', '0', '0', '0', '0']
            _write_chunks(line1, oupf)
            _write_nodes(n[1:], oupf)
            oupf.write("1G       6       0       0  000000\n")
        # write the element data
        for e in ele_data:
            ele_data = ['2', repr(e[0]), ele_code, '2', '0', '0', '0', '0', '0']
            _write_chunks(ele_data, oupf)
            oupf.write("      ")
            ele_data2 = [repr(num_pts), '1', '4', '0']
            _write_chunks(ele_data2, oupf, False)
            oupf.write(" 0.000000000E+00 0.000000000E+00 0.000000000E+00\n")
            oupf.write("      ")
            ele_data3 = [repr(i) for i in e[1:]]
            _write_chunks(ele_data3, oupf)
        # write the end of file
        oupf.write("99       0       0       1       0       0       0       0       0")
#-------------------------------------------------------------------------------
# Miscellaneous functions
#-------------------------------------------------------------------------------
def _get_node_data(part):
    """ Identifies node data using part class """

    # join node index with co-ordinate data
    nodes = {}
    for e in part.elements:
        for n in range(len(e.nodes)):
            nodes[e.nodes[n]] = e.pts[n]

    # create array of data
    node_data = []
    for n in nodes.keys():
        node_data.append([int(n), nodes[n][0], nodes[n][1], nodes[n][2]])

    return sorted(node_data)

def _get_node_text(node_data):
    """ Modifies node array to create input file text """

    char_length = [7, 13, 13, 13]
    node_text = _set_str_columns(node_data, char_length)
        
    return node_text

def _get_ele_data(part):
    """ Identifies element data from part class """
    
    data = [[repr(e.indx)] + e.nodes for e in part.elements]
    data = [[int(n) for n in d] for d in data]
    
    return data

def _get_ele_text(ele_data):
    """ Modifies ele array to create input file text """

    char_length = [6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
    ele_text = _set_str_columns(ele_data, char_length)

    return ele_text
    
def _set_str_columns(data, char_length):
    """ Iterates through array creates columns of text data """
    
    strings = [_set_str_length(d, char_length) for d in data]

    return '\n'.join(strings) + '\n'

def _set_str_length(d, length):
    """ Converts number to string of set length """

    strings = [' '*(length[n] - len(repr(d[n]))) + repr(d[n]) for n in range(len(d))]
    
    return ','.join(strings)

def _get_set_text(sets, set_name):
    """ Create text for defined sets """

    set_text = ''
    set_nums = sorted([int(s) for s in sets.keys()])
    for m in set_nums:
        set_text = set_text + '*Elset, elset=' + set_name + repr(m) + '\n'
        elechunks = _chunks(sets[repr(m)], 16)
        for chunk in elechunks:
            set_text = set_text + ",".join(["%d" % item for item in chunk])
            set_text = set_text + '\n'

    return set_text

def _chunks(l,n):
    """ Splits up array in arrays of set length (n) """
    
    return [l[i:i+n] for i in range(0, len(l), n)]
