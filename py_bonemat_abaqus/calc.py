#!/usr/bin/python
#
# py_bonemat_abaqus - calc
# ========================
#
# Created by Elise Pegg, University of Oxford

__all__ = ['calc_mat_props']

#-------------------------------------------------------------------------------
# Import modules
#-------------------------------------------------------------------------------
from numpy import mean, arange, digitize, array
from copy import deepcopy
from py_bonemat_abaqus.classes import vtk_data
import sys

#-------------------------------------------------------------------------------
# Functions for calculating material data
#-------------------------------------------------------------------------------
def calc_mat_props(parts, param, vtk):
    """ Find material properties of each part """

    for p in range(len(parts)):
        if parts[p].ignore != True:
            parts[p] = _assign_mat_props(parts[p], param, vtk)
    parts = _refine_materials(parts, param)

    return parts
    
def _assign_mat_props(part, param, vtk):
    """ Find material properties of each element """

    # check integration type has been defined
    if 'integration' not in param.keys():
        # if not, make modulus integration (V3) the default
        param['integration'] = 'E'

    # if V1
    if param['integration'] == None:
        # check that the elements are linear tets
        if part.ele_type != 'linear_tet':
            raise IOError('BoneMat version 1 will only work with linear tets, modify parameters file')
        # find voxels
        voxels = _identify_voxels_in_tets(part, vtk)
        # for each tet find HU for each voxel within
        HU_data = [_get_hu(v, vtk) for v in voxels]
    
        # from HU values calculate apparent density
        app_density = [_calc_app_density(hu, param) for hu in HU_data]

        # if calibration correction requested, perform correction
        if param['calibrationCorrect'] in param.keys():
            if param['calibrationCorrect'] == True:
                app_density = [_correct_calibration(rhoQCT, param) for rhoQCT in app_density]
        
        # from density values calculate modulus
        modulus = [_calc_modulus(d, param) for d in app_density]

        # for each tet average all the modulus values
        mean_modulus = [mean(m) for m in modulus]

        # ensure all modulus values above the minimum value
        mean_modulus = [m if m > param['minVal'] else param['minVal'] for m in mean_modulus]
        
        # save
        part = _save_modulus(part, mean_modulus)

    # if V2
    if param['integration'] == 'HU':
        # find scalar for each element        
        HU_data = [e.integral(param['intSteps'], vtk) for e in part.elements]
        
        # calculate apparent density
        app_density = _calc_app_density(HU_data, param)
        
        # if calibration correction requested, perform correction
        if param['calibrationCorrect'] in param.keys():
            if param['calibrationCorrect'] == True:
                app_density = _correct_calibration(app_density, param)
                
        # calculate the modulus values
        moduli = _calc_modulus(app_density, param)

        # ensure all modulus values above the minimum value
        moduli = [m if m > param['minVal'] else param['minVal'] for m in moduli]
        
        # save
        part = _save_modulus(part, moduli)

    # if V3
    if param['integration'] == 'E':
        # convert vtk data to rhoQCT density values
        app_density_lookup = _calc_app_density(vtk.lookup, param)
        
        # if calibration correction requested, perform correction
        if param['calibrationCorrect'] in param.keys():
            if param['calibrationCorrect'] == True:
                app_density_lookup = _correct_calibration(app_density_lookup, param)
                
        # convert vtk data to modulus values
        moduli_lookup = _calc_modulus(app_density_lookup, param)
        #moduli_lookup = [m if m > param['minVal'] else param['minVal'] for m in moduli_lookup]
        vtk_mod = vtk_data(vtk.x, vtk.y, vtk.z, moduli_lookup)

        # find modulus for each element
        moduli = [e.integral(param['intSteps'], vtk_mod) for e in part.elements]


        # save
        part = _save_modulus(part, moduli)

    return part

def _refine_materials(parts, param):
    """ Group the materials into bins separated by the gapValue parameter """

    moduli = _get_all_modulus_values(parts)
    Ehigh = max(moduli)

    # if maxMaterials defined, calculate gapValue
    if 'gapValue' not in param.keys():
        if 'maxMaterials' in param.keys():
            param['gapValue'] = _get_mod_intervals(moduli, param['maxMaterials'])
        else:
            raise IOError("Error: Neither gapValue or maxMaterials defined in parameters file")
    # if groupingDensity is not defined, use max
    if 'groupingDensity' not in param.keys():
        param['groupingDensity'] = 'max'

    # limit the moduli values for each part
    for p in range(len(parts)):
        if parts[p].ignore != True:
            parts[p].moduli = _limit_num_materials(parts[p].moduli, param['gapValue'], param['minVal'], param['groupingDensity'], Ehigh)

    return parts

#-------------------------------------------------------------------------------
# Functions for processing ct data
#-------------------------------------------------------------------------------
def _identify_voxels_in_tets(part, vtk):
    """ Iterate through each element and identifies voxels within """
    
    voxels = []
    for e in part.elements:
        voxels.append(vtk.get_voxels(e))
        
    return voxels

def _get_hu(voxels, vtk):
    """ Lookup value of voxels within tet """

    return [vtk.lookup[v] for v in voxels]

def _calc_app_density(HU, param):
    """ Calculate the apparenty density of each voxel """
    """      [rhoQCT = rhoQCTa + (rhoQCTb * HU)]      """
    
    if 'rhoQCTa' not in param.keys():
        raise IOError("Error: rhoQCTa not defined in parameters file")
    if 'rhoQCTb' not in param.keys():
        raise IOError("Error: rhoQCTb not defined in parameters file")

    return [param['rhoQCTa'] + (param['rhoQCTb'] * hu) for hu in HU]
    
def _correct_calibration(density, param):
    """ Correct the CT calibration to convert rhoQCT to rhoAsh """

    # check parameters file contains the necessary information
    necessaryParam = ['numCTparam','rhoAsha1','rhoAshb1']
    checkNecessaryParam(param, necessaryParam)
    if param['numCTparam'] == 'triple':
        necessaryParamIfTriple = ['rhoThresh1','rhoThresh2','rhoAsha2','rhoAshb2','rhoAsha3','rhoAshb3']
        checkNecessaryParam(param, necessaryParamIfTriple)

    # calculate rhoAsh
    if param['numCTparam'] == 'single':        
        rhoAsh = [param['rhoAsha1'] + (param['rhoAshb1'] * d) for d in density]
    elif param['numCTparam'] == 'triple':
        rhoAsh = [0] * len(density)
        for d in range(len(density)):
            if density[d] < param['rhoThresh1']:
                rhoAsh[d] = param['rhoAsha1'] + (param['rhoAshb1'] * density[d])
            elif ((density[d] >= param['rhoThresh1']) & (density[d] <= param['rhoThresh2'])):
                rhoAsh[d] = param['rhoAsha2'] + (param['rhoAshb2'] * density[d])
            elif density[d] > param['rhoThresh2']:
                rhoAsh[d] = param['rhoAsha3'] + (param['rhoAshb3'] * density[d])
            else:
                raise ValueError("Error: calibration correction for density value: " +repr(density[d]) + " cannot be calculated")
    else:
        raise IOError("Error: numCTparam must be defined as either single or triple")

    return rhoAsh
    
def _calc_modulus(density, param):
    """ Calculate modulus based upon radiographic density """

    # check parameters file contains the necessary information
    necessaryParam = ['numEparam','Ea1','Eb1','Ec1']
    checkNecessaryParam(param, necessaryParam)
    if param['numEparam'] == 'triple':
        necessaryParamIfTriple = ['Ethresh1','Ethresh2','Ea2','Eb2','Ec2','Ea3','Eb3','Ec3']
        checkNecessaryParam(param, necessaryParamIfTriple)

    # make sure apparent density values are all positive
    density = [i if i >= 0 else 0 for i in density]
        
    # calculate modulus
    if param['numEparam'] == 'single':
        modulus = [param['Ea1'] + (param['Eb1'] * (d ** param['Ec1'])) for d in density]
    elif param['numEparam'] == 'triple':
        modulus = [0] * len(density)
        for d in range(len(density)):
            if density[d] < param['Ethresh1']:
                modulus[d] = param['Ea1'] + (param['Eb1'] * (density[d] ** param['Ec1']))
            elif (density[d] >= param['Ethresh1']) & (density[d] <= param['Ethresh2']):
                modulus[d] = param['Ea2'] + (param['Eb2'] * (density[d] ** param['Ec2']))
            elif density[d] > param['Ethresh2']:
                modulus[d] = param['Ea3'] + (param['Eb3'] * (density[d] ** param['Ec3']))
            else:
                raise ValueError("Error: modulus for density value: " + repr(density[d]) + " cannot be calculated")                
    else:
        raise IOError("Error: numEparam must be defined as either single or triple")
    return modulus

def checkNecessaryParam(param, fields):
    for f in fields:
        if f not in param.keys():
            raise IOError("Error: " + f + " is not defined in parameters file")

def _get_all_modulus_values(parts):
    """ Create list of all modulus values for all parts in model """
    
    moduli = []
    for p in parts:
        if p.ignore != True:
            moduli.extend(p.moduli)

    return moduli

def _get_mod_intervals(moduli, max_materials):
    """ Find the refined moduli values """

    min_mod = float(min(moduli))
    max_mod = float(max(moduli))
    if len(set(moduli)) < max_materials:
        mod_interval = 0.0
    else:
        mod_interval = (max_mod - min_mod) / float(max_materials)

    return mod_interval
    
def _limit_num_materials(moduli, gapValue, minVal, groupingDensity, Ehigh):
    """ Groups the moduli into bins and calculates the max/mean for each """
    if gapValue == 0:
        return moduli
    else:
        bins = arange(Ehigh, minVal-gapValue, -gapValue).tolist()
        # warn user if there are a lot of bins to calculate
        if len(bins) > 10000:
            print('\n    WARNING:')
            print('    You have specified a very small gap size relative to your maximum modulus')
            print('    So your modulus "bin" size is: ' + repr(len(bins)))
            print('    This will take so long to calculate it may crash your computer ***')
            answ = raw_input('    Would you like to continue (y/n)')
            if (answ == 'n') | (answ == 'N') | (answ == 'no') | (answ == 'No') | (answ == 'NO'):
                sys.exit()
            else:
                print('\n')
        # calculate the modulus values
        binplace = digitize(moduli, bins)
        new_moduli = array([minVal] * len(moduli))
        moduli_array = array(moduli)
        for b in range(len(bins)):
            values = moduli_array[binplace == b].tolist()
            if values != []:
                if groupingDensity == 'mean':
                    value = sum(values) / len(values)
                elif groupingDensity == 'max':
                    value = max(values)
                else:
                    raise IOError("Error: groupingDensity should be 'max' or 'mean' in parameters file")
                new_moduli[binplace == b] = value
        
        return new_moduli.tolist()

def _save_modulus(part, modulus):
    """ Save modulus data in part class """
    
    part.moduli = modulus

    return part
