#!/usr/bin/python
#
# py_bonemat_abaqus - unit tests
# ==========================
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
from py_bonemat_abaqus.general import _read_text, _remove_spaces, _refine_spaces
from py_bonemat_abaqus.classes import part, linear_tet, quad_tet, linear_wedge, linear_hex
from py_bonemat_abaqus.classes import vtk_data, _calc_lookup
from py_bonemat_abaqus.data_import import *
from py_bonemat_abaqus.data_import import _get_param, _what_mesh_filetype, _confirm_eletype
from py_bonemat_abaqus.data_import import _import_vtk_ct_data, _get_transform_data
from py_bonemat_abaqus.data_import import _apply_transform, _find_part_names, _get_nodes
from py_bonemat_abaqus.calc import *
from py_bonemat_abaqus.calc import _identify_voxels_in_tets, _define_equations, _apply_equations
from py_bonemat_abaqus.calc import  _get_mod_intervals, _limit_num_materials, _assign_mat_props
from py_bonemat_abaqus.data_output import *
import os, binascii
from numpy import mean, array
from numpy.linalg import det
from random import uniform
import dicom, dicom.UID
from dicom.dataset import Dataset, FileDataset
import numpy as np
import datetime, time

#-------------------------------------------------------------------------------
# Check general.py functions
#-------------------------------------------------------------------------------
class correct_check_command_line_inputs(TestCase):
    def setUp(self):
        self.too_few = ['parameters.txt','CTdir']
        self.too_many = ['parameters.txt','CTdir','job.inp','extra']           
        self.wrong_order = ['parameters.txt','job.inp','CTdir']
        self.wrong_paramfile = ['index.jpg','CTdir','job.inp']
        self.wrong_dir = ['parameters.txt','CTdir.txt','job.inp']
        self.existdir = ['parameters.txt', 'NoCTdir','job.inp']
        self.existparam = ['parameters2.txt','job.inp','CTdir']
        self.existvtk = ['parameters.txt','job.inp','CT.vtk']
        self.existinp = ['parameters.txt','job2.inp','CTdir']
        self.right = ['parameters.txt','CTdir','job.inp']
        with open('job.inp','w') as oupf:
            oupf.write('test abaqus input file\n')
        with open('parameters.txt','w') as oupf:
            oupf.write('test py_bonemat_abaqus parameters file\n')
        os.mkdir('CTdir')

    def test_too_few_inputs(self):
        self.assertRaises(IOError, check_argv, self.too_few)

    def test_too_many_inputs(self):
        self.assertRaises(IOError, check_argv, self.too_many)

    def test_inputs_wrong_order(self):
        self.assertRaises(IOError, check_argv, self.wrong_order)

    def test_inputs_wrong_paramfile(self):
        self.assertRaises(IOError, check_argv, self.wrong_paramfile)

    def test_inputs_wrong_dir(self):
        self.assertRaises(IOError, check_argv, self.wrong_dir)

    def test_checks_CTdir_exists(self):
        self.assertRaises(IOError, check_argv, self.existdir)

    def test_checks_param_file_exists(self):
        self.assertRaises(IOError, check_argv, self.existparam)

    def test_checks_vtk_file_exists(self):
        self.assertRaises(IOError, check_argv, self.existvtk)

    def test_checks_inp_file_exists(self):
        self.assertRaises(IOError, check_argv, self.existinp)

    def test_correct_inputs(self):
        self.assertTrue(check_argv(self.right))

    def tearDown(self):
        os.remove('job.inp')
        os.remove('parameters.txt')
        os.rmdir('CTdir')

class check_read_text_function(TestCase):
    def setUp(self):
        # create random string
        self.text1 = binascii.b2a_hex(os.urandom(20))
        self.text2 = ""
        
        # write string to file
        with open('test1.txt','w') as oupf:
            oupf.write(self.text1)
        with open('test2.txt','w') as oupf:
            oupf.write(self.text2)
            
        # read text file with function
        self.read_text1 = _read_text('test1.txt')
        self.read_text2 = _read_text('test2.txt')
        
    def test_reads_random_text(self):
        self.assertEqual(self.text1, self.read_text1,
                         "Read text does not match random input text: " + self.text1)
    def test_reads_file_no_text(self):
        self.assertEqual(self.text2, self.read_text2,
                         "Cannot read file with no text")
    def tearDown(self):
        os.remove('test1.txt')
        os.remove('test2.txt')

class check_remove_spaces_function(TestCase):
    def setUp(self):
        self.text1 = _remove_spaces("This is a test\n")
        self.text2 = _remove_spaces("")
        self.text3 = _remove_spaces("        ")

    def test_removes_spaces(self):
        self.assertEqual(self.text1, "Thisisatest\n",
                         "Does not remove spaces correctly")

    def test_can_remove_no_spaces_from_blank_string(self):
        self.assertEqual(self.text2, "",
                         "Error removing spaces from string with no spaces")
        
    def test_can_remove_spaces_no_text(self):
        self.assertEqual(self.text3, "",
                         "Cannot remove spaces if no text")

class check_refine_spaces_function(TestCase):
    def setUp(self):
        self.text1 = _refine_spaces("This is a  test\n")
        self.text2 = _refine_spaces("")
        self.text3 = _refine_spaces("This is a test\n")
        self.text4 = _refine_spaces("This is a     test\n")

    def test_refines_spaces(self):
        self.assertEqual(self.text1, "This is a test\n",
                         "Does not correctly refine spaces")
    def test_can_refine_blank_string(self):
        self.assertEqual(self.text2, "",
                         "Refine spaces cannot process blank string")
    def test_can_refine_string_no_double_spaces(self):
        self.assertEqual(self.text3, "This is a test\n",
                         "Refine spaces fails if no double spaces")
    def test_can_refine_multiple_spaces(self):
        self.assertEqual(self.text4, "This is a test\n",
                         "Refine spaces cannot refine multiple spaces")
#-------------------------------------------------------------------------------
# Check data_import.py functions
#-------------------------------------------------------------------------------
# parameter import
class check_import_param_function(TestCase):
    def setUp(self):
        self.text1 = ""
        self.text2 = "This has no data\n"
        self.text3 = "bla = -5.85\n"
        self.text4 = "bla = -5e-10\n"
        self.text5 = "bla = -5.8 #bla2 = 5.3\n"
        self.text6 = "bla = True\n"
        self.text7 = "bla = Calibration\n"

    def test_can_process_blank_param_file(self):
        self.assertTrue(len(_get_param(_remove_spaces(self.text1))) == 0,
                        "Error processing blank parameters file")
    def test_can_import_param_file_no_data(self):
        self.assertTrue(len(_get_param(_remove_spaces(self.text2))) == 0,
                        "Error processing parameters file no data")
    def test_can_import_param_data(self):
        self.assertTrue('bla' in _get_param(_remove_spaces(self.text3)).keys(),
                        "Error parameter 'bla' not imported correctly")
    def test_will_not_import_data_after_hash(self):
        self.assertFalse('bla2' in _get_param(_remove_spaces(self.text4)).keys(),
                         "Error parameter 'bla2' after hash has been imported")
    def test_will_import_boolean_from_param_file(self):
        self.assertEqual(_get_param(_remove_spaces(self.text6)), {'bla': True},
                         "Error importing boolean from param file")
    def test_will_import_text_from_param_file(self):
        self.assertEqual(_get_param(_remove_spaces(self.text7)), {'bla': 'Calibration'},
                         "Error importing text from param file")
        
# mesh import
class check_mesh_import_functions(TestCase):
    def setUp(self):
        self.part = create_linear_tet_part()
        self.part2 = create_hex_part()
        self.part3 = create_quad_tet_part()
        self.inp_mesh_file = os.path.join('py_bonemat_abaqus','tests','test.inp')
        self.ntr_mesh_file = os.path.join('py_bonemat_abaqus','tests','test.ntr')
        self.parameter_file = os.path.join('py_bonemat_abaqus','tests','test.txt')
        create_inp_file(self.inp_mesh_file)
        create_ntr_file(self.ntr_mesh_file)        
        create_parameters(self.parameter_file)
        self.param = import_parameters(self.parameter_file)
        self.inp_part = import_mesh(self.inp_mesh_file, self.param)
        self.ntr_part = import_mesh(self.ntr_mesh_file, self.param)
        self.nodes = {'1':[4.0, 5e-08, 3.0], '2': [5.6, 3.0, 4.0]}

    def test_can_import_part_names(self):
        self.assertEqual(_find_part_names('\*Part,name=Test\n')[0],'Test')

    def test_can_import_part_name_with_hash(self):
        self.assertEqual(_find_part_names('\*Part,name=Test-1\n')[0],'Test-1')

    def test_can_import_part_name_with_dot(self):
        self.assertEqual(_find_part_names('\*Part,name=Test.1\n')[0],'Test.1')

    def test_can_determine_mesh_filetype(self):
        self.assertEqual(_what_mesh_filetype(self.inp_mesh_file), '.inp',
                         "Cannot identify abaqus input file from text")
        self.assertEqual(_what_mesh_filetype(self.ntr_mesh_file), '.ntr',
                         "Cannot identify neutral mesh file from text")
        self.assertIsNone(_what_mesh_filetype(self.parameter_file),
                        "Bad error handling of non-mesh file")

    def test_correctly_identifies_eletype(self):
        self.assertEqual(_confirm_eletype([range(5),range(5)]), 'linear_tet')

    def test_eletype_correct_class(self):
        self.assertIsInstance(self.part2[0].elements[0], linear_hex)    

    def test_can_import_inp_file_data_correctly(self):
        self.assertEqual(self.part[0].elements[0].pts, self.inp_part[0].elements[0].pts,
                         "Inp mesh import pts incorrect")
        self.assertEqual(self.part[0].elements[0].indx, self.inp_part[0].elements[0].indx,
                         "Inp mesh import element index incorrect")
        self.assertEqual(self.part[0].elements[0].nodes, self.inp_part[0].elements[0].nodes,
                         "Inp mesh import element nodes incorrect")

    def test_can_import_mulitple_node_def_one_part(self):
        self.assertEqual(_get_nodes('*Node\n1,4.0,5e-8,3\n*Node\n2,5.6,3.0,4.0\n*Element'),
                         self.nodes)
                         
        
    def test_can_import_ntr_file_data_correctly(self):
        self.assertEqual(self.part3[0].elements[0].pts, self.ntr_part[0].elements[0].pts,
                         "Ntr mesh import pts incorrect")
        self.assertEqual(self.part3[0].elements[0].indx, self.ntr_part[0].elements[0].indx,
                         "Ntr mesh import element index incorrect")
        self.assertEqual(self.part3[0].elements[0].nodes, self.ntr_part[0].elements[0].nodes,
                         "Ntr mesh import element nodes incorrect")

    def tearDown(self):
        os.remove(self.inp_mesh_file)
        os.remove(self.ntr_mesh_file)
        os.remove(self.parameter_file)

class check_node_transformation_calculations(TestCase):
    def setUp(self):
        self.transf_str = '*Instance,name=Part-1-1,part=Part-1\n0,0,5\n*EndInstance\n'
        self.transf_str2 = '*Instance,name=Part-1-1,part=Part-1\n0,0,5\n0,0,5,0,0,6,90\n*EndInstance\n'
        self.transf_str3 = '*Instance,name=Part-1-1,part=Part-1\n*EndInstance\n*Instance,name=Part-2-1,part=Part-2\n0,0,5\n*EndInstance\n'
        self.trans = [[0.,0.,5]]
        self.trans2 = [[0.,0.,5],[0.,0.,5.,0.,0.,6.,90]]
        self.nodes, self.nodes2, self.nodes3 = {},{},{}
        self.nodes['1'] = [1.,1.,0.]
        self.nodes2['1'] = [1.,1.,5]
        self.nodes3['1'] = [-1.,1.,5.]
                
    def test_can_get_translation_transformation_data(self):
        self.assertEqual(_get_transform_data(self.transf_str, 'Part-1'), self.trans)

    def test_can_get_rotation_transformation_data(self):
        self.assertEqual(_get_transform_data(self.transf_str2, 'Part-1'), self.trans2)

    def test_can_get_transformation_data_multiple_parts(self):
        self.assertEqual(_get_transform_data(self.transf_str3, 'Part-2'), self.trans)

    def test_translation_transform_correctly_applied(self):
        self.assertEqual(round(_apply_transform(self.nodes, self.trans)['1'][0],5),
                         round(self.nodes2['1'][0],5))

    def test_rotation_transform_correctly_applied(self):
        self.assertEqual(round(_apply_transform(self.nodes, self.trans2)['1'][0],5),
                         round(self.nodes3['1'][0],5))
        

# ct vtk data import

class check_can_import_vtk_data_file(TestCase):
    def setUp(self):
        self.vtk_fle = os.path.join('py_bonemat_abaqus','tests','test.vtk')
        create_vtk_file(self.vtk_fle)
        x = [-1., 0., 1.]
        lookup = [0, 5, 10, 5, 10, 15, 10, 15, 20,
                  5, 10, 15, 25, 15, 20, 15, 20, 25,
                  10, 15, 20, 15, 20, 25, 20, 25, 30]
        self.vtk = vtk_data(x,x,x,lookup)
            
    def test_can_import_x_vtk_data_correctly(self):
        self.assertEqual(_import_vtk_ct_data(self.vtk_fle).x, self.vtk.x,
                         "Imported vtk x data does not equal that specified")
        
    def test_can_import_y_vtk_data_correctly(self):
        self.assertEqual(_import_vtk_ct_data(self.vtk_fle).y, self.vtk.y,
                         "Imported vtk y data does not equal that specified")
        
    def test_can_import_z_vtk_data_correctly(self):
        self.assertEqual(_import_vtk_ct_data(self.vtk_fle).z, self.vtk.z,
                         "Imported vtk z data does not equal that specified")
        
    def test_can_import_lookup_vtk_data_correctly(self):
        self.assertEqual(_import_vtk_ct_data(self.vtk_fle).lookup.tolist(), self.vtk.lookup.tolist(),
                         "Imported vtk lookup data does not equal that specified")

    def tearDown(self):
        os.remove(self.vtk_fle)

# dicom data import

class check_can_import_dicom_files(TestCase):
    def setUp(self):
        self.path = os.path.join('py_bonemat_abaqus','tests','dicom')
        os.mkdir(self.path)
        create_dcm_files([os.path.join(self.path,'test1.dcm'),
                          os.path.join(self.path,'test2.dcm'),
                          os.path.join(self.path,'test3.dcm')])
        self.vtk = import_ct_data(self.path)
        self.lookup = [0, 5, 10, 5, 10, 15, 10, 15, 20,
                  5, 10, 15, 25, 15, 20, 15, 20, 25,
                  10, 15, 20, 15, 20, 25, 20, 25, 30]
    def test_can_import_dicom_lookup_correctly(self):
        self.assertEqual(self.vtk.lookup.tolist(), self.lookup,
                         self.vtk.lookup.tolist()) #"VTK data imported from dicoms is incorrect :"
    def tearDown(self):
        os.remove(os.path.join(self.path,'test1.dcm'))
        os.remove(os.path.join(self.path,'test2.dcm'))
        os.remove(os.path.join(self.path,'test3.dcm'))
        os.rmdir(self.path)
                     
#-------------------------------------------------------------------------------
# Check classes.py functions
#-------------------------------------------------------------------------------
# part
class check_adds_elements_to_part(TestCase):
    def setUp(self):
        self.part = create_linear_tet_part()
        self.pts = [[-1., 1., -1.],
                [-1., -1., 1.],
                [1., -1., -1.],
                [1., 1. , 1.],
                [-1., 0., 0.],
                [0., -1., 0.],
                [0., 0., -1.],
                [0., 1., 0.],
                [0., 0., 1.],
                [1., 0., 0.]]
        self.nodes = [repr(n+1) for n in range(10)]

    def test_element_is_stored_in_part(self):
        self.part[0].add_element(linear_tet(1, self.pts, self.nodes))
        self.assertEqual(len(self.part[0].elements), 2,
                         "Elements not stored correctly in part: should have 2 elements, but has: " + repr(len(self.part[0].elements)))
# vtk data
class check_vtk_class_find_grid_works(TestCase):
    def setUp(self):
        self.vtk = create_vtk()
        self.pt = [[0.5],[0.5],[0.5]]
        self.correct_grid = [[1, 2],[1, 2],[1, 2]]

    def test_finds_correct_grid_data(self):
        self.assertEqual(self.vtk.find_grid(self.pt), self.correct_grid,
                    "Does not locate correct grid surrounding a point in CT data")

    def test_find_grid_works_zero(self):
        self.assertEqual(self.vtk.find_grid([[0.],[0.],[0,]]), [[1, 1],[1, 1],[1, 1]],
                         "find_grid function should output [[1, 1],[1, 1],[1, 1]] when pt=(0,0,0)")
        self.assertNotEqual(self.vtk.find_grid([[0.],[0.],[0,]]), self.correct_grid,
                            "When pt=(0,0,0), the find_grid function should not output [[1, 2],[1, 2],[1, 2]]")

    def test_box_function_works(self):
        self.assertEqual(self.vtk.find_grid([[0.],[0.],[0,]], True), self.correct_grid,
                         "Box function part of 'find_grid' does not work correctly")

class check_lookup_for_vtk(TestCase):
    def setUp(self):
        self.vtk = create_vtk([-1, 0, 1])

    def test_lookup_indx(self):
        self.assertEqual(_calc_lookup(1,1,1,self.vtk.dimen),13)

    def test_lookup_value(self):
        self.assertEqual(self.vtk.lookup[_calc_lookup(1, 1, 1, self.vtk.dimen)], 15)

    def test_lookup_value2(self):
        self.assertEqual(self.vtk.lookup[_calc_lookup(0, 0, 0, self.vtk.dimen)], 0)

    def test_lookup_value3(self):
        self.assertEqual(self.vtk.lookup[_calc_lookup(2, 2, 2, self.vtk.dimen)], 30)
        
    def test_lookup_value4(self):
        self.assertEqual(self.vtk.lookup[_calc_lookup(0, 1, 1, self.vtk.dimen)], 25)


class check_vtk_interpolate_scalar_function(TestCase):
    def setUp(self):
        self.vtk = create_vtk([-1.,0.,1.])
        self.vtk2 = create_vtk([0.,1.,2.])
        self.rhoQCT = lambda a:a
        self.rhoAsh = lambda b:b
        self.modulus = lambda c:c

    def test_trilinear_interpolation1(self):
        self.assertEqual(self.vtk.interpolateScalar([0.5, 0.5, 0.5], self.modulus, self.rhoAsh, self.rhoQCT), 22.5)

    def test_trilinear_interpolation2(self):
        self.assertEqual(self.vtk.interpolateScalar([0.0,0.0,1.0], self.modulus, self.rhoAsh, self.rhoQCT), 20.)

    def test_trilinear_interpolation3(self):
        self.assertEqual(self.vtk.interpolateScalar([0.25,0.,0.], self.modulus, self.rhoAsh, self.rhoQCT), 16.25)

    def test_trilinear_interpolation4(self):
        self.assertEqual(self.vtk.interpolateScalar([0.125,0.375,0.125], self.modulus, self.rhoAsh, self.rhoQCT), 18.125)

    def test_trilinear_interpolation5(self):
        self.assertEqual(self.vtk2.interpolateScalar([1.5,0,0], self.modulus, self.rhoAsh, self.rhoQCT), 7.5)

    def test_trilinear_interpolation_works_with_zero(self):
        self.assertEqual(self.vtk.interpolateScalar([0.,0.,0.], self.modulus, self.rhoAsh, self.rhoQCT), 15.)

# linear tetrahedron
class check_linear_tet_calculations(TestCase):
    def setUp(self):
        self.pts = [[0.,0.,0.],
                    [1.,0.,0.],
                    [0.,1.,0.],
                    [0.,0.,1]]
        self.tet = linear_tet(1, self.pts, range(4))
        self.vtk = create_vtk([-1.,0,1])
        self.vtk2 = create_vtk([-1,0,1],[5]*27)
        self.vtk3 = create_vtk([0,1,2],[0, 5, 10]*9)

    def test_linear_tet_calculates_jacobian_determinant(self):
        self.assertEqual(det(self.tet.jacobian()), 1.)

    def test_linear_tet_finds_points_in_tet(self):
        self.assertTrue(self.tet.in_tet([.1,.1,.1]))

    def test_linear_tet_does_not_find_points_outside_tet(self):
        self.assertFalse(self.tet.in_tet([2,2,2]))

    def test_linear_tet_finds_points_on_tet_surface(self):
        self.assertTrue(self.tet.in_tet([0.5,0,0.5]))

    def test_linear_tet_calculates_integral_correctly(self):
        self.assertEqual(round(self.tet.integral(4, self.vtk),5), round(18.75,5))

    def test_linear_tet_calculates_integral_uniform_field(self):
        self.assertEqual(round(self.tet.integral(4, self.vtk2),5), round(5,5))

    def test_linear_tet_calculates_integral_field_variation_one_dim(self):
        self.assertEqual(round(self.tet.integral(4, self.vtk3),5), round(1.25,5))

    def test_linear_tet_calculates_volume_correctly(self):
        self.tet.integral(4, self.vtk)
        self.assertEqual(round(self.tet.volume,5), round(1.0/6.0, 5))


# quadrilateral tetrahedron
class check_quad_tet_calculations(TestCase):
    def setUp(self):
        self.pts = [[0.,0.,0.],
                    [1.,0.,0.],
                    [0.,1.,0.],
                    [0.,0.,1],
                    [0.5,0.,0.],
                    [0.5, 0.5, 0.],
                    [0., 0.5, 0.],
                    [0., 0., 0.5],
                    [0.5, 0., 0.5],
                    [0., 0.5, 0.5]]
        self.tet = quad_tet(1, self.pts, range(10))
        self.vtk = create_vtk([-1.,0,1])
        self.vtk2 = create_vtk([-1,0,1],[5]*27)
        self.vtk3 = create_vtk([0,1,2],[0, 5, 10]*9)

    def test_quad_tet_calculates_jacobian_determinant(self):
        self.assertEqual(round(det(self.tet.jacobian(0.625,0.125,0.125,0.125)),5), round(1.0,5))

    def test_quad_tet_calculates_integral_correctly(self):
        self.assertEqual(round(self.tet.integral(4, self.vtk),5), round(18.75,5))

    def test_quad_tet_calculates_integral_uniform_field(self):
        self.assertEqual(round(self.tet.integral(4, self.vtk2),5), round(5,5))

    def test_quad_tet_calculates_integral_one_dim(self):
        self.assertEqual(round(self.tet.integral(4, self.vtk3),5), round(1.25,5))

    def test_quad_tet_calculates_volume_correctly(self):
        self.tet.integral(4, self.vtk)
        self.assertEqual(round(self.tet.volume,5), round(1./6.,5))
        
# Linear wedge element
class check_linear_wedge_calculations(TestCase):
    def setUp(self):
        self.pts = [[0.,0.,0.],
                    [1.,0.,0.],
                    [1.,1.,0.],
                    [0.,0.,1.],
                    [1.,0.,1.],
                    [1.,1.,1.]]
        self.wedge = linear_wedge(1, self.pts, range(6))
        self.vtk = create_vtk([-1.,0.,1.])
        self.vtk2 = create_vtk([-1,0,1],[5]*27)
        self.vtk3 = create_vtk([0,1,2],[0, 5, 10]*9)

    def test_linear_wedge_calculates_integral_correctly(self):
        self.assertEqual(round(self.wedge.integral(4, self.vtk),5), round(21.875,5))

    def test_linear_wedge_calculates_integral_uniform_field(self):
        self.assertEqual(round(self.wedge.integral(4, self.vtk2),5), round(5,5))

    def test_linear_wedge_calculates_integral_one_dim(self):
        self.assertEqual(round(self.wedge.integral(4, self.vtk3),5), round(17.5/6.,5))

    def test_linear_wedge_calculates_volume_correctly(self):
        self.wedge.integral(4, self.vtk)
        self.assertEqual(round(self.wedge.volume,5), round(1./2.,5))

# Linear hexahedral element
class check_linear_hex_calculations(TestCase):
    def setUp(self):
        self.pts = [[0.,0.,0.],
                    [1.,0.,0.],
                    [1.,1.,0.],
                    [0.,1.,0.],
                    [0.,0.,1.],
                    [1.,0.,1.],
                    [1.,1.,1.],
                    [0.,1.,1.]]
        self.hex = linear_hex(1, self.pts, range(8))
        self.vtk = create_vtk([-1.,0.,1.])
        self.vtk2 = create_vtk([-1., 0., 1.],[5]*27)
        self.vtk3 = create_vtk([-1., 0., 1.],[0, 5, 10]*9)

    def test_linear_hex_calculates_integral_correctly(self):
        self.assertEqual(round(self.hex.integral(4, self.vtk),5), round(22.5,5))

    def test_linear_hex_calculates_integral_uniform_field(self):
        self.assertEqual(round(self.hex.integral(4, self.vtk2),5), round(5,5))

    def test_linear_hex_calculates_integral_one_dim(self):
        self.assertEqual(round(self.hex.integral(4, self.vtk3),5), round(7.5,5))

    def test_linear_hex_calculates_volume_correctly(self):
        self.hex.integral(4, self.vtk)
        self.assertEqual(round(self.hex.volume,5), round(1.,5))
#-------------------------------------------------------------------------------
# Check calc.py functions
#-------------------------------------------------------------------------------
class check_defines_equations_correctly(TestCase):
    def setUp(self):
        self.param = {'integration': 'E',
                      'rhoQCTa': 0.,
                      'rhoQCTb': 1.,
                      'calibrationCorrect': False,
                      'numEparam': 'single',
                      'minVal': 0.0001,
                      'Ea1': 0.,
                      'Eb1': 1.,
                      'Ec1': 1.}        
        
    def test_rhoQCT_equ_correct(self):
        hu = uniform(0,2000)
        self.param['rhoQCTa'] = uniform(0,2000)
        self.param['rhoQCTb'] = uniform(0,2000)
        rhoQCT, rhoAsh, modulus = _define_equations(self.param)
        self.assertEqual(rhoQCT(hu), self.param['rhoQCTa'] + (hu*self.param['rhoQCTb']))
        
    def test_rhoAsh_equ_correct(self):
        hu = uniform(0,2000)
        self.param['rhoQCTa'] = uniform(0,2000)
        self.param['rhoQCTb'] = uniform(0,2000)
        rhoQCT, rhoAsh, modulus = _define_equations(self.param)
        self.assertEqual(rhoAsh(rhoQCT(hu)), self.param['rhoQCTa'] + (hu*self.param['rhoQCTb']))
        
    def test_rhoAsh_equ_correct2(self):
        hu = uniform(0,2000)
        self.param['calibrationCorrect'] = True
        self.param['numCTparam'] = 'single'
        self.param['rhoAsha1'] = uniform(0,2000)
        self.param['rhoAshb1'] = uniform(0,2000)
        rhoQCT, rhoAsh, modulus = _define_equations(self.param)
        self.assertEqual(rhoAsh(rhoQCT(hu)), self.param['rhoAsha1'] + (hu*self.param['rhoAshb1']))
        
    def test_rhoAsh_equ_correct3(self):
        hu = uniform(0,2000)
        self.param['calibrationCorrect'] = True
        self.param['numCTparam'] = 'triple'
        self.param['rhoAsha1'] = uniform(0,2000)
        self.param['rhoAshb1'] = uniform(0,2000)
        self.param['rhoAsha2'] = uniform(0,2000)
        self.param['rhoAshb2'] = uniform(0,2000)
        self.param['rhoAsha3'] = uniform(0,2000)
        self.param['rhoAshb3'] = uniform(0,2000)
        tmp = [uniform(0,2000), uniform(0,2000)]
        self.param['rhoThresh1'] = min(tmp)
        self.param['rhoThresh2'] = max(tmp)
        self.param['rhoQCTa'] = 0.
        self.param['rhoQCTb'] = 1.
        rhoQCT, rhoAsh, modulus = _define_equations(self.param)
        if rhoQCT(hu) < self.param['rhoThresh1']:
            res = self.param['rhoAsha1'] + (self.param['rhoAshb1'] * rhoQCT(hu))
        elif (rhoQCT(hu) >= self.param['rhoThresh1']) & (rhoQCT(hu) <= self.param['rhoThresh2']):
            res = self.param['rhoAsha2'] + (self.param['rhoAshb2'] * rhoQCT(hu))
        else:
            res = self.param['rhoAsha3'] + (self.param['rhoAshb3'] * rhoQCT(hu))
        self.assertEqual(rhoAsh(rhoQCT(hu)), res)
        
#    def test_modulus_equ_correct(self):
#        hu = uniform(-2000,2000)
#        self.param['rhoQCTa'] = 0.
#        self.param['rhoQCTb'] = 1.
#        self.param['calibrationCorrect'] = False
#        self.param['Ea1'] = uniform(-200,200)
#        self.param['Eb1'] = uniform(0,200)
#        self.param['Ec1'] = uniform(-10,10)
#        print(self.param)
#        rhoQCT, rhoAsh, modulus = _define_equations(self.param)
#        if rhoAsh(rhoQCT(hu)) <= 0:
#            res = 0.0001
#        else:
#            res = self.param['Ea1'] + ((self.param['Eb1'] * hu) ** self.param['Ec1'])
#        self.assertEqual(modulus(rhoAsh(rhoQCT(hu))), res)

class check_v1_correctly_calculates_if_voxel_in_tet(TestCase):
    def setUp(self):
        self.part = create_linear_tet_part()
        self.vtk = create_vtk()
        self.correct_voxels = [2, 4, 6, 10, 12, 13, 14, 16, 18, 22, 26]
        self.voxels = _identify_voxels_in_tets(self.part[0], self.vtk)
        self.vtk_large = create_vtk([-5, 5, 15])
        self.correct_voxels_large = [0, 9, 3, 12, 1, 10, 4, 13]
        self.voxels_large = _identify_voxels_in_tets(self.part[0], self.vtk_large)

    def test_has_found_voxels(self):
        self.assertTrue(len(self.voxels) > 0,
                        "No voxels found")
    def test_if_found_voxels_correct(self):
        self.assertEqual(sorted(self.voxels[0]), sorted(self.correct_voxels),
                         "Incorrect voxels identified in tet" + repr(self.voxels) )
    def test_finds_voxels_if_tet_within_voxel(self):
        self.assertTrue(len(self.voxels_large) > 0,
                        "No voxels found when element smaller than voxels")
    def test_voxels_correct_if_tet_within_voxel(self):
        self.assertEqual(self.voxels_large[0], self.correct_voxels_large,
                         "Incorrect voxels found when element smaller than voxel" + repr(self.voxels_large))

class check_v1_correctly_calculates_modulus(TestCase):
    def setUp(self):
        self.vtk = create_vtk()
        self.voxels = [[1, 2], [6, 10, 12], [13, 16, 18, 22], [26]]        
        self.param = {'integration': 'E',
                      'rhoQCTa': 0.,
                      'rhoQCTb': 1.,
                      'calibrationCorrect': False,
                      'numEparam': 'single',
                      'minVal': 0.0001,
                      'Ea1': 0.,
                      'Eb1': 1.,
                      'Ec1': 1.}  
        self.rhoQCT_equ, self.rhoAsh_equ, self.modulus_equ = _define_equations(self.param)  
        self.moduli = [[_apply_equations(self.vtk.lookup[i], 
                                     self.rhoQCT_equ, 
                                     self.rhoAsh_equ, 
                                     self.modulus_equ) for i in v] for v in self.voxels]  
        self.mean_moduli = [mean(m) for m in self.moduli]
        
    def test_correct_number_hu_values(self):
        self.assertEqual(len(self.moduli), len(self.voxels),
                         "Number of modulus values (" + repr(len(self.moduli)) \
                         + " different to number of elements (" + repr(len(self.voxels)) + ")")

    def test_are_modulus_values_correct(self):
        self.assertEqual(self.mean_moduli, [7.5, 15, 16.25, 30],
                         "Mean HU values should equal [7.5, 15, 16.25, 30], but equal: "+repr(self.mean_moduli))                
 
class check_refines_modulus_correctly(TestCase):
    def setUp(self):
        self.moduli = [uniform(0,1000) for n in range(300)]
        self.same_moduli = [10 for n in range(10)]
        self.max_materials = 100

    def test_modulus_interval_calculation_works(self):
        self.assertEqual(_get_mod_intervals(range(11), 5), 2.0,
                         "_get_mod_intervals function correct for modulus range(11), maxMat=5")
        self.assertEqual(_get_mod_intervals([10,10,10,10],5), 0.0,
                         "_get_mod_intervals function cannot cope with same greyscale")

    def calc_moduli(self, moduli, max_materials):
        mod_interval = _get_mod_intervals(moduli, max_materials)
        self.moduli_new = _limit_num_materials(moduli, mod_interval, min(moduli), 'max')

        return self

    def test_number_new_moduli_no_more_than_max_materials(self):
        self.calc_moduli(self.moduli, self.max_materials)
        extra_mat = len(set(self.moduli_new)) - self.max_materials
        self.assertTrue(extra_mat < 0,
                         "Refine modulus creates " + repr(extra_mat) + " more materials than specified")

    def test_refined_moduli_max_same_previous(self):
        self.calc_moduli(self.moduli, self.max_materials)
        self.assertEqual(round(max(self.moduli_new),5), round(max(self.moduli),5),
                         "Modulus refinement changed maximum; refined:" + repr(max(self.moduli_new)) + " original:" + repr(max(self.moduli)))

    def test_can_cope_with_all_same_moduli(self):
        self.calc_moduli(self.same_moduli, self.max_materials)
        extra_mat = len(set(self.moduli_new)) - self.max_materials
        self.assertTrue(extra_mat < 0, "Modulus calculation fails if all same grayscale")

    def test_can_cope_with_one_moduli(self):
        self.assertEqual(_limit_num_materials([50], 60, 0, 'max'), [50])
        
class check_v2_correctly_calculates_modulus(TestCase):
    def setUp(self):
        self.vtk = create_vtk() 
        self.param = {'integration': 'HU',
                      'rhoQCTa': 2.,
                      'rhoQCTb': .5,
                      'calibrationCorrect': False,
                      'numEparam': 'single',
                      'minVal': 0.0001,
                      'intSteps': 4,
                      'Ea1': 1.2,
                      'Eb1': 4.6,
                      'Ec1': 2.6}  
        self.part = create_hex_part()
        self.part2 = _assign_mat_props(self.part[0], self.param, self.vtk)
    
    def test_if_calculates_v2_modulus_correctly(self):
        self.assertEqual(round(self.part2.moduli[0],1), 3807.6)
        
class check_v3_correctly_calculates_modulus(TestCase):
    def setUp(self):
        self.vtk = create_vtk() 
        self.param = {'integration': 'E',
                      'rhoQCTa': 2.,
                      'rhoQCTb': .5,
                      'calibrationCorrect': False,
                      'numEparam': 'single',
                      'minVal': 0.0001,
                      'intSteps': 4,
                      'Ea1': 1.2,
                      'Eb1': 4.6,
                      'Ec1': 2.6}  
        self.part = create_hex_part()
        self.part2 = _assign_mat_props(self.part[0], self.param, self.vtk)
    
    def test_if_calculates_v2_modulus_correctly(self):
        self.assertEqual(round(self.part2.moduli[0],1), 4018.8)  

class check_v3_correctly_calculates_modulus_if_hu_zero(TestCase):
    def setUp(self):
        self.vtk = create_vtk(x = [-1., 0., 1.], lookup = [-5, 5, 10, 5, 10, 15, 10, 15, 20,
                                                            5, 10, 15, 25, 15, 20, 15, 20, 25,
                                                            10, 15, 20, 15, 20, 25, 20, 25, 30])
        self.param = {'integration': 'E',
                      'rhoQCTa': 2.,
                      'rhoQCTb': .5,
                      'calibrationCorrect': False,
                      'numEparam': 'single',
                      'minVal': 0.0001,
                      'intSteps': 4,
                      'Ea1': 1.2,
                      'Eb1': 4.6,
                      'Ec1': 2.6}  
        self.part = create_hex_part()
        self.part2 = _assign_mat_props(self.part[0], self.param, self.vtk)
    
    def test_if_calculates_v2_modulus_correctly(self):
        self.assertEqual(round(self.part2.moduli[0],1), 4018.8)    
        
#-------------------------------------------------------------------------------
# Check data_output.py functions
#-------------------------------------------------------------------------------
class check_outputs_valid_abaqus_input_file(TestCase):
    def setUp(self):
        self.part = create_quad_tet_part()
        self.ntr_mesh_file = os.path.join('py_bonemat_abaqus','tests','test1.ntr')
        self.output_file = os.path.join('py_bonemat_abaqus','tests','test1MAT.inp')
        create_ntr_file(self.ntr_mesh_file)
        output_abq_inp(self.part, self.ntr_mesh_file, 0.35)
        with open(self.output_file,'r') as inpf:
            lines = inpf.read()
        self.outputtext = lines
        self.nodelines = """*Node
      1,         -1.0,          1.0,         -1.0
      2,         -1.0,         -1.0,          1.0
      3,          1.0,         -1.0,         -1.0
      4,          1.0,          1.0,          1.0
      5,         -1.0,          0.0,          0.0
      6,          0.0,         -1.0,          0.0
      7,          0.0,          0.0,         -1.0
      8,          0.0,          1.0,          0.0
      9,          0.0,          0.0,          1.0
     10,          1.0,          0.0,          0.0"""
        self.elementlines = """*Element, type=C3D10
     1,      1,      2,      3,      4,      5,      6,      7,      8,      9,     10"""
        self.materiallines = """*Material, name=BoneMat_1
*Elastic
1.0, 0.35"""
        
    def test_nodes_written_correctly(self):
        self.assertTrue(self.nodelines in self.outputtext,
                        "Nodes written incorrectly to output file")

    def test_elements_written_correctly(self):
        self.assertTrue(self.elementlines in self.outputtext,
                        "Elements written incorrectly to outputfile\n")

    def test_materials_written_correctly(self):
        self.assertTrue(self.materiallines in self.outputtext,
                        "Materials written incorrectly to outputfile")

    def tearDown(self):
        os.remove(self.ntr_mesh_file)
        os.remove(self.output_file)        
#-------------------------------------------------------------------------------
# Functions to create files and objects
#-------------------------------------------------------------------------------
def create_linear_tet_part():
    p = part('Part1','C3D4','linear_tet')
    pts = [[-1., 1., -1.],
           [-1., -1., 1.],
           [1., -1., -1.],
           [1., 1. , 1.]]
    nodes = [repr(n+1) for n in range(4)]
    p.add_element(linear_tet(1, pts, nodes))
    p.moduli = [1.]
    return [p]

def create_quad_tet_part():
    p = part('Part1','C3D10','quad_tet')
    pts = [[-1., 1., -1.],
           [-1., -1., 1.],
           [1., -1., -1.],
           [1., 1. , 1.],
           [-1., 0., 0.],
           [0., -1., 0.],
           [0., 0., -1.],
           [0., 1., 0.],
           [0., 0., 1.],
           [1., 0., 0.]]
    nodes = [repr(n+1) for n in range(10)]
    p.add_element(quad_tet(1, pts, nodes))
    p.moduli = [1.]
    return [p]

def create_wedge_part():
    p = part('Part1','C3D6','linear_wedge')
    pts = [[0.,0.,0.],
          [1.,0.,0.],
          [1.,1.,0.],
          [0.,0.,1.],
          [1.,0.,1.],
          [1.,1.,1.]]
    nodes = [repr(n+1) for n in range(6)]
    p.add_element(linear_wedge(1, pts, nodes))
    p.moduli = [1.]
    return [p]

def create_hex_part():
    p = part('Part1','C3D8','linear_hex')
    pts = [[0.,0.,0.],
           [1.,0.,0.],
           [1.,1.,0.],
           [0.,1.,0.],
           [0.,0.,1.],
           [1.,0.,1.],
           [1.,1.,1.],
           [0.,1.,1.]]
    nodes = [repr(n+1) for n in range(8)]
    p.add_element(linear_hex(1, pts, nodes))
    p.moduli = [1.]
    return [p]
    
def create_hex_mesh():
    p = part('Part1','C3D8','linear_hex')
    pts = [[0.,0.,0.],
               [1.,0.,0.],
               [1.,1.,0.],
               [0.,1.,0.],
               [0.,0.,1.],
               [1.,0.,1.],
               [1.,1.,1.],
               [0.,1.,1.]]
    nodes = []
    for i in [-1,0]:
        for j in [-1,0]:
            for k in [-1,0]:
                for pt in pts:
                    n = [pt[0]+i,pt[1]+j,pt[2]+k]
                    if n not in nodes:
                        nodes.append([n[0],n[1],n[2]])
    count = 0                  
    for i in [-1,0]:
        for j in [-1,0]:
            for k in [-1,0]:
                count += 1
                nodes_c = [nodes.index([pt[0]+i,pt[1]+j,pt[2]+k])+1 for pt in pts]
                pts_c = [[pt[0]+i,pt[1]+j,pt[2]+k] for pt in pts]
                p.add_element(linear_hex(count, pts_c, nodes_c))
        p.moduli = [1.,1.,1.,1.,1.,1.,1.,1.]
    return [p]                
            

def create_vtk(x = [-1., 0., 1.], lookup = [0, 5, 10, 5, 10, 15, 10, 15, 20,
              5, 10, 15, 25, 15, 20, 15, 20, 25,
              10, 15, 20, 15, 20, 25, 20, 25, 30]):
    
    return vtk_data(x,x,x,lookup)

def create_inp_file(filename, type='tet'):
    if type == 'tet':
        part = create_linear_tet_part()
    elif type == 'quad':
        part = create_quad_tet_part()
    elif type == 'wedge':
        part = create_wedge_part()
    elif type == 'hex':
        part = create_hex_part()
    elif type == 'mesh':
        part = create_hex_mesh_part()
    path = os.path.split(filename)[0]
    tmp_inp_filename = os.path.join(path,'test.ntr')
    output_abq_inp(part,tmp_inp_filename, 0.35)
    os.rename(os.path.join(path,'testMAT.inp'),
              filename)

def create_parameters(filename):
    param = """# parameters.txt
integration = E
gapValue = 1
groupingDensity = max
intSteps = 4
rhoQCTa = -9.497890384819272
rhoQCTb = 0.1377430297890385
calibrationCorrect = False
numEparam = single
Ea1 = -3.842
Eb1 = 2
Ec1 = 1
minVal = 0.001
poisson = 0.35
"""
    with open(filename, 'w') as oupf:
        oupf.write(param)
        
def create_ntr_file(filename):
    ntr = """25       0       0       1       0       0       0       0       0
PATRAN Neutral, HyperMesh Template PATRAN/GENERAL
26       0       0       1      10       1       1       1       0
  07-27-201510:28:00
 1       1       0       2       0       0       0       0       0
-1.000000000E+001.0000000000E+00-1.000000000E+00
1G       6       0       0  000000
 1       2       0       2       0       0       0       0       0
-1.000000000E+00-1.000000000E+001.0000000000E+00
1G       6       0       0  000000
 1       3       0       2       0       0       0       0       0
1.0000000000E+00-1.000000000E+00-1.000000000E+00
1G       6       0       0  000000
 1       4       0       2       0       0       0       0       0
1.0000000000E+001.0000000000E+001.0000000000E+00
1G       6       0       0  000000
 1       5       0       2       0       0       0       0       0
-1.000000000E+000.000000000E+000.000000000E+00
1G       6       0       0  000000
 1       6       0       2       0       0       0       0       0
0.000000000E+00-1.000000000E+000.000000000E+00
1G       6       0       0  000000
 1       7       0       2       0       0       0       0       0
0.000000000E+000.000000000E+00-1.000000000E+00
1G       6       0       0  000000
 1       8       0       2       0       0       0       0       0
0.000000000E+001.0000000000E+000.000000000E+00
1G       6       0       0  000000
 1       9       0       2       0       0       0       0       0
0.000000000E+000.000000000E+001.0000000000E+00
1G       6       0       0  000000
 1      10       0       2       0       0       0       0       0
1.0000000000E+000.000000000E+000.000000000E+00
1G       6       0       0  000000
 2       1       5       2       0       0       0       0       0
      10       1       4       0 0.000000000E+00 0.000000000E+00 0.000000000E+00
       1       2       3       4       5       6       7       8       9      10
99       0       0       1       0       0       0       0       0
"""
    with open(filename,'w') as oupf:
        oupf.write(ntr)

        
def create_vtk_file(filename):
     vtk = """# vtk DataFile Version 3.0
vtk output
ASCII
DATASET RECTILINEAR_GRID
DIMENSIONS 3 3 3
X_COORDINATES 3 double
-1.0000 0.0000 1.0000
Y_COORDINATES 3 double
-1.0000 0.0000 1.0000
Z_COORDINATES 3 float
-1.0000 0.0000 1.0000
CELL_DATA 8
POINT_DATA 27
SCALARS scalars short
LOOKUP_TABLE default
0 5 10 5 10 15 10 15 20
5 10 15 25 15 20 15 20 25
10 15 20 15 20 25 20 25 30"""
     with open(filename, 'w') as oupf:
         oupf.write(vtk)

def create_dcm_files(filenames, orientation = ['1.000000','0.000000','0.000000','0.000000','1.000000','0.000000'],
    pixel_array = array([[[0,5,10],[5,10,15],[10,15,20]],
                         [[5,10,15],[25,15,20],[15,20,25]],
                         [[10,15,20],[15,20,25],[20,25,30]]])):
    origins = [[-1,-1,-1],[-1,-1,0],[-1,-1,1]]
    for n in [0,1,2]:
        write_dicom(pixel_array[n], filenames[n], origins[n], origins, orientation)
        
def write_dicom(pixel_array, filename, origin, origins, orientation):   
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = 'Secondary Capture Image Storage'
    file_meta.MediaStorageSOPInstanceUID = '1.3.6.1.4.1.9590.100.1.1.111165684411017669021768385720736873780'
    file_meta.ImplementationClassUID = '1.3.6.1.4.1.9590.100.1.0.100.4.0'
    ds = FileDataset(filename, {},file_meta = file_meta,preamble="\0"*128)
    ds.ImageType= 'ORIGINAL\PRIMARY\AXIAL\CT'
    ds.SOPClassUID = 'Secondary Capture Image Storage'
    ds.SOPInstanceUID =    '1.3.6.1.4.1.9590.100.1.1.111165684411017669021768385720736873780'
    ds.StudyDate = str(datetime.date.today()).replace('-','')
    #ds.ImageDate = str(datetime.date.today()).replace('-','')
    ds.StudyTime = str(time.time())
    #ds.ImageTime = str(time.time())
    ds.AccessionNumber = '153745645'
    ds.Modality = 'CT'
    ds.Manufacturer = 'PY_BONEMAT_ABAQUS'
    ds.InstitutionName = 'UNIVERISTY_OF_BATH'
    ds.ReferringPhysiciansName = 'NOTAPPLICABLE'
    ds.ManufacturersModelName = 'VERSION1-0-9'
    ds.ReferencedImageSequence = ''
    ds.PatientsName = 'SUBJECT001'
    ds.PatientID = ''
    ds.PatientsBirthDate = ''
    ds.PatientsSex = 'F'
    ds.SliceThickness = 2.7
    ds.KVP = 120
    ds.SpacingBetweenSlices = 1
    ds.DataCollectionDiameter = 430
    ds.ReconstructionDiameter = 430.00
    ds.DistanceSourceToDetector = 1093
    ds.DistanceSourceToPatient = 630.
    ds.GantryDetectorTilt = 0.0
    ds.ScanArc = 403
    ds.XRayTubeCurrent = 200
    ds.Exposure = 135
    ds.FilterType = 'F'
    ds.PatientPosition = 'HFS'
    ds.StudyInstanceUID = '1.3.6.1.4.1.9590.100.1.1.124313977412360175234271287472804872093'
    ds.SeriesInstanceUID = '1.3.6.1.4.1.9590.100.1.1.111165684411017669021768385720736873780.1'
    #ds.ReferenceSOPClassUID = 'Secondary Capture Image Storage'
    #ds.ReferencedSOPInstanceUID = '1.3.6.1.4.1.9590.100.1.1.111165684411017669021768385720736873780'
    #ds.ContentDate = str(datetime.date.today()).replace('-','')
    #ds.ContentTime = str(time.time())
    ds.StudyID = '15346254'
    ds.SeriesNumber = '8733'
    ds.AcquisitionNumber = 1
    ds.ImageNumber = origins.index(origin) + 1
    ds.SliceLocation = origin[2]
    ds.SecondaryCaptureDeviceManufctur = 'Python 2.7.3'   
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.HighBit = 15
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.SmallestImagePixelValue = '\\x00\\x00'
    ds.LargestImagePixelValue = '\\xff\\xff'
    ds.Columns = pixel_array.shape[0]
    ds.Rows = pixel_array.shape[1]
    ds.PatientOrientation = ['L','P']
    ds.ImagePositionPatient = origin
    ds.ImageOrientationPatient = orientation
    ds.PixelSpacing = [1.0, 1.0]
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = 0.0
    if pixel_array.dtype != np.uint16:
        pixel_array = pixel_array.astype(np.uint16)
    ds.PixelData = pixel_array.tostring()

    ds.save_as(filename)
    return

# Run tests
if __name__ == "__main__":
    main()
