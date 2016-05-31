#!/usr/bin/python
#
# py_bonemat_abaqus - tests functional
# ====================================
#
# Functional tests for testing py_bonemat_abaqus software
# To run tests, in main py_bonemat_abaqus working directory type:
#    >>> nosetests
#
# Created by Elise Pegg, University of Bath

#-------------------------------------------------------------------------------
# Import modules
#-------------------------------------------------------------------------------
from unittest import TestCase, main
from py_bonemat_abaqus.general import *
from py_bonemat_abaqus.data_import import *
from py_bonemat_abaqus.general import _read_text
from py_bonemat_abaqus.calc import *
from py_bonemat_abaqus.classes import *
from py_bonemat_abaqus.data_output import *
import os
from numpy import size
from tests import create_inp_file, create_ntr_file, create_vtk_file
from tests import create_dcm_files, create_parameters, create_linear_tet_part, create_vtk

#-------------------------------------------------------------------------------
# Is command-line argument check working?
#-------------------------------------------------------------------------------
class can_check_arguments(TestCase):
    def setUp(self):
        self.argv_right = ['parameters.txt','DicomFiles','job.inp']
        self.argv_wrong = ['DicomFiles','parameters.txt','job.inp']
        with open('parameters.txt','w') as oupf:
            oupf.write('test py_bonemat_abaqus parameters file\n')
        with open('job.inp','w') as oupf:
            oupf.write('test abaqus input file\n')
        os.mkdir('DicomFiles')
        
    def test_check_correct(self):
        self.assertTrue(check_argv(self.argv_right),
                        "[" + ", ".join(self.argv_right) + "] is correct usage")
        
    def test_check_incorrect(self):
        self.assertRaises(IOError, check_argv, self.argv_wrong)

    def tearDown(self):
        os.remove('job.inp')
        os.remove('parameters.txt')
        os.rmdir('DicomFiles')

#-------------------------------------------------------------------------------
# Can program import parameters file?
#-------------------------------------------------------------------------------
class can_read_parameters_file(TestCase):
    def setUp(self):
        # define input arguments
        self.param_file = os.path.join('py_bonemat_abaqus','tests','test.txt')
        create_parameters(self.param_file)
        self.param_Ea = -3.842
        self.param = import_parameters(self.param_file)
        
    def test_param_file_exists(self):
        self.assertTrue(os.path.isfile(self.param_file),
                        "File: " + self.param_file + " does not exist")
        
    def test_imports_some_parameters(self):
        self.assertTrue(any(self.param),
                        "File: " + self.param_file + " does contain parameter data")
        
    def test_imports_correct_number_parameters(self):
        self.assertIs(len(self.param), 13,
                      "File: " + self.param_file + " contains 12 parameters, program found: " + repr(self.param))
        
    def test_param_data_import_correct(self):
        self.assertEqual(self.param_Ea, self.param['Ea1'])

    def tearDown(self):
        os.remove(self.param_file)
#-------------------------------------------------------------------------------
# Does program import the mesh?
#-------------------------------------------------------------------------------
class can_import_mesh(TestCase):
    def setUp(self):
        self.abq_mesh_file = os.path.join('py_bonemat_abaqus','tests','test.inp')
        self.ntr_mesh_file = os.path.join('py_bonemat_abaqus','tests','test.ntr')
        create_inp_file(self.abq_mesh_file)
        create_ntr_file(self.ntr_mesh_file)
        self.num_parts = 1
        self.abq_num_elements = 1
        self.ntr_num_elements = 1
        self.ele_type = "linear_tet"
        self.ele_type2 = "quad_tet"
        self.num_ele_nodes = 4
        self.num_ele_nodes_quad = 10
        self.param_file = os.path.join('py_bonemat_abaqus','tests','test.txt')
        create_parameters(self.param_file)
        self.param = import_parameters(self.param_file)
        self.abq_mesh = import_mesh(self.abq_mesh_file, self.param)
        self.ntr_mesh = import_mesh(self.ntr_mesh_file, self.param)
        
    def test_abq_mesh_file_exists(self):
        self.assertTrue(os.path.isfile(self.abq_mesh_file),
                        "File: " + self.abq_mesh_file + " does not exist")
        
    def test_ntr_mesh_file_exists(self):
        self.assertTrue(os.path.isfile(self.ntr_mesh_file),
                        "File: " + self.ntr_mesh_file + " does not exist")
        
    def test_imports_some_mesh_data(self):
        self.assertTrue(len(self.abq_mesh) > 0,
                        "No parts have been imported from: " + self.abq_mesh_file)
        self.assertTrue(len(self.ntr_mesh) > 0,
                        "No parts have been imported from: " + self.ntr_mesh_file)
        
    def test_mesh_part_is_a_part(self):
        self.assertIsInstance(self.abq_mesh[0], part,
                              "Imported mesh from " + self.abq_mesh_file + " is not a 'part' class")
        self.assertIsInstance(self.ntr_mesh[0], part,
                              "Imported mesh from " + self.ntr_mesh_file + " is not a 'part' class")
        
    def test_mesh_has_correct_number_parts(self):
        self.assertEqual(len(self.abq_mesh), self.num_parts,
                      "Imported abaqus mesh file should contain " + repr(self.num_parts) + " parts")
        self.assertEqual(len(self.ntr_mesh), self.num_parts,
                      "Imported neutral mesh file should contain " + repr(self.num_parts) + " parts")
        
    def test_imports_correct_number_elements(self):
        self.assertEqual(len(self.abq_mesh[0].elements), self.abq_num_elements,
                      "Imported part from " + self.abq_mesh_file + " should contain " + repr(self.abq_num_elements) + " elements")
        self.assertEqual(len(self.ntr_mesh[0].elements), self.ntr_num_elements,
                      "Imported part from " + self.abq_mesh_file + " should contain " + repr(self.ntr_num_elements) + " elements")
        
    def test_imported_elements_are_tets(self):
        self.assertIsInstance(self.abq_mesh[0].elements[0], linear_tet,
                              "Imported elements from " + self.abq_mesh_file + "are not 'linear tet' class")
        self.assertIsInstance(self.ntr_mesh[0].elements[0], quad_tet,
                              "Imported elements from " + self.ntr_mesh_file + " are not 'quad tet' class")
        
    def test_imports_eletype_correctly(self):
        self.assertEqual(self.abq_mesh[0].ele_type, self.ele_type,
                         "Mesh imported from " + self.abq_mesh_file + " should be of element type " + self.ele_type)
        self.assertEqual(self.ntr_mesh[0].ele_type, self.ele_type2,
                         "Mesh imported from " + self.ntr_mesh_file + " should be of element type " + self.ele_type2)
        
    def test_elements_have_xyz_info(self):
        self.assertEqual(size(self.abq_mesh[0].elements[0].pts,1), 3,
                      "Nodal data from " + self.abq_mesh_file + " is incomplete, not all x,y,z data imported")
        self.assertEqual(size(self.ntr_mesh[0].elements[0].pts,1), 3,
                      "Nodal data from " + self.ntr_mesh_file + " is incomplete, not all x,y,z data imported")
        
    def test_elements_have_node_info(self):
        self.assertEqual(size(self.abq_mesh[0].elements[0].pts,0), self.num_ele_nodes,
                      "Abq element should have x,y,z, data on all " + repr(self.num_ele_nodes) + " nodes")
        self.assertEqual(size(self.ntr_mesh[0].elements[0].pts,0), self.num_ele_nodes_quad,
                      "Ntr element should have x,y,z, data on all " + repr(self.num_ele_nodes_quad) + " nodes")

    def tearDown(self):
        os.remove(self.abq_mesh_file)
        os.remove(self.ntr_mesh_file)
        
#-------------------------------------------------------------------------------
# Does program import the ct data?
#-------------------------------------------------------------------------------
class can_import_vtk_ct_data(TestCase):
    def setUp(self):
        self.vtk_file = os.path.join('py_bonemat_abaqus','tests','test.vtk')
        create_vtk_file(self.vtk_file)
        self.vtk = import_ct_data(self.vtk_file)

    def test_vtk_file_exists(self):
        self.assertTrue(os.path.isfile(self.vtk_file),
                        "File: " + self.vtk_file + " does not exist")
        
    def test_imports_ct_data_as_vtk_instance(self):
        self.assertIsInstance(self.vtk, vtk_data,
                              "Imported ct data is not a vtk data class")

    def test_imports_x_y_z_vtk_data(self):
        self.assertEqual(len(self.vtk.x), 3,
                      "Not all x grid positions have been imported")
        self.assertEqual(len(self.vtk.y), 3,
                      "Not all y grid positions have been imported")
        self.assertEqual(len(self.vtk.z), 3,
                      "Not all z grid posiions have been imported")
    def test_imports_vtk_lookup(self):
        self.assertEqual(len(self.vtk.lookup), len(self.vtk.x) * len(self.vtk.y) * len(self.vtk.z),
                      "Not all lookup values have been imported")

    def tearDown(self):
        os.remove(self.vtk_file)

class can_import_dicom_ct_data(TestCase):
    def setUp(self):
        self.dcm_dir = os.path.join('py_bonemat_abaqus','tests','dicoms')
        if os.path.exists(self.dcm_dir):
            for f in os.listdir(self.dcm_dir):
                os.remove(os.path.join(self.dcm_dir, f))
        else:
            os.makedirs(self.dcm_dir)
        self.files = [os.path.join(self.dcm_dir,'test'+repr(n)+'.dcm') for n in [0,1,2]]
        create_dcm_files(self.files)
        self.vtk = import_ct_data(self.dcm_dir)

    def test_imports_x_y_z_vtk_data(self):
        self.assertEqual(len(self.vtk.x), 3,
                      repr(len(self.vtk.x)) + " x-grid positions have been imported")
        self.assertEqual(len(self.vtk.y), 3,
                      repr(len(self.vtk.y)) + " y-grid positions have been imported")
        self.assertEqual(len(self.vtk.z), 3,
                      repr(len(self.vtk.z)) + " z-grid positions have been imported")
        
    def test_imports_vtk_lookup(self):
        self.assertEqual(len(self.vtk.lookup), len(self.vtk.x) * len(self.vtk.y) * len(self.vtk.z),
                      repr(len(self.vtk.lookup)) + " lookup values have been imported")
    def tearDown(self):
        for f in self.files:
            os.remove(f)
        os.removedirs(self.dcm_dir)
        
#-------------------------------------------------------------------------------
# Can calculate modulus values?
#-------------------------------------------------------------------------------
class can_calc_modulus(TestCase):
    def setUp(self):
        self.param_file = os.path.join('py_bonemat_abaqus', 'tests', 'test.txt')
        create_parameters(self.param_file)
        self.param = import_parameters(self.param_file)
        self.param2 = self.param
        self.param2['version'] = 2
        self.param3 = self.param
        self.param3['version'] = 3
        self.part = create_linear_tet_part()
        self.vtk = create_vtk()
        self.part = calc_mat_props(self.part, self.param, self.vtk)
        self.part2 = calc_mat_props(self.part, self.param2, self.vtk)
        self.part3 = calc_mat_props(self.part, self.param3, self.vtk)

    def test_v1_calculation_returns_data(self):
        self.assertNotEqual(self.part[0].moduli, [],
                            "Version 1 calculation has returned no values")

    def test_v1_calculation_returns_correct_number_data(self):
        self.assertEqual(len(self.part[0].moduli), len(self.part[0].elements),
                         "Length of V1 calculated modulus values not same as number of elements")

    def test_v2_calculation_returns_data(self):
        self.assertNotEqual(self.part2[0].moduli, [],
                            "Version 2 calculation has returned no values")

    def test_v2_calculation_returns_correct_number_data(self):
        self.assertEqual(len(self.part2[0].moduli), len(self.part[0].elements),
                         "Length of V2 calculated modulus values not same as number of elements")
        
    def test_v3_calculation_returns_data(self):
        self.assertNotEqual(self.part3[0].moduli, [],
                            "Version 3 calculation has returned no values")
        
    def test_v3_calculation_returns_correct_number_data(self):
        self.assertEqual(len(self.part3[0].moduli), len(self.part[0].elements),
                         "Length of V3 calculated modulus values not same as number of elements")

    def tearDown(self):
        os.remove(self.param_file)
        
#-------------------------------------------------------------------------------
# Can write new data to an abaqus input file?
#-------------------------------------------------------------------------------
class can_output_data_to_inp_file(TestCase):
    def setUp(self):
        self.parts = create_linear_tet_part()
        self.ntr_mesh_file = os.path.join('py_bonemat_abaqus','tests','test1.ntr')
        create_ntr_file(self.ntr_mesh_file)
        output_abq_inp(self.parts, self.ntr_mesh_file, 0.35)
        self.inp_mesh_file = os.path.join('py_bonemat_abaqus','tests','test2.inp')
        create_inp_file(self.inp_mesh_file)
        output_abq_inp(self.parts, self.inp_mesh_file, 0.35)
        
    def test_output_files_created(self):
        self.assertTrue(os.path.isfile(outFile(self.ntr_mesh_file)),
                        "File: " + outFile(self.ntr_mesh_file) + " has not been created")
        self.assertTrue(os.path.isfile(outFile(self.inp_mesh_file)),
                        "File: " + outFile(self.inp_mesh_file) + " has not been created")

    def test_output_files_contain_data(self):
        lines = _read_text(outFile(self.ntr_mesh_file))
        self.assertTrue(len(lines) > 0,
                        "File: " + outFile(self.ntr_mesh_file) + " contains no text")
        lines = _read_text(outFile(self.inp_mesh_file))
        self.assertTrue(len(lines) > 0,
                        "File: " + outFile(self.inp_mesh_file) + " contains no text")
        
    def tearDown(self):
        os.remove(self.ntr_mesh_file)
        os.remove(self.inp_mesh_file)
        os.remove(outFile(self.ntr_mesh_file))
        os.remove(outFile(self.inp_mesh_file))
        
def outFile(fle):
    return fle[:-4] + 'MAT.inp'

#-------------------------------------------------------------------------------
# Run tests
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
