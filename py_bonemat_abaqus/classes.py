#!/usr/bin/python
#
# py_bonemat_abaqus - class definitions
# =====================================
#
# Created by Elise Pegg, University of Bath

__all__ = ['vtk_data','linear_tet','quad_tet','linear_wedge','linear_hex','part']

#-------------------------------------------------------------------------------
# Import modules
#-------------------------------------------------------------------------------
from numpy.linalg import det
from numpy import mean, arange, matrix, array
from copy import deepcopy
from bisect import bisect_left, bisect_right
from itertools import product

#-------------------------------------------------------------------------------
# Part class
#-------------------------------------------------------------------------------
class part:
    """ Part class which represents a mesh"""
    
    __slots__ = ("name", "elements", "ele_name", "ele_type", "moduli", "transform", "ignore")
    
    def __init__(self, name, ele_name, ele_type, transform=[[0.,0.,0]], ignore = False):
        self.name = name
        self.ele_type = ele_type
        self.ele_name = ele_name
        self.elements = []
        self.moduli = []
        self.transform = transform
        self.ignore = ignore

    def add_element(self, ele):
        # add element to part
        self.elements.append(ele)

#-------------------------------------------------------------------------------
# CT data class
#-------------------------------------------------------------------------------
class vtk_data:
    """ Vtk ct class """
    
    __slots__ = ("x", "y", "z", "dimen", "vox", "lookup")
    
    def __init__(self, x, y, z, lookup):
        self.x = x
        self.y = y
        self.z = z
        self.dimen = [len(self.x), len(self.y), len(self.z)]
        self.vox = set([])
        self.lookup = array(lookup)

    def get_voxels(self, tet):
        # define variables
        voxels = []
        in_element = False

        # check element is within voxel range
        tet_pts = zip(*[tet.pts[0], tet.pts[1], tet.pts[2], tet.pts[3]])
        in_vox = True
        if (min(tet_pts[0]) < self.x[0]):
            raise ValueError('Mesh Error: Element(s) outside of CT data range. X coord min='+repr(min(tet_pts[0])))
        if (max(tet_pts[0]) > self.x[-1]):
            raise ValueError('Mesh Error: Element(s) outside of CT data range. X coord max='+repr(max(tet_pts[0])))
        if (min(tet_pts[1]) < self.y[0]):
            raise ValueError('Mesh Error: Element(s) outside of CT data range. Y coord min='+repr(min(tet_pts[1])))
        if (max(tet_pts[1]) > self.y[-1]):
            raise ValueError('Mesh Error: Element(s) outside of CT data range. Y coord max='+repr(max(tet_pts[1])))
        if (min(tet_pts[2]) < self.z[0]):
            raise ValueError('Mesh Error: Element(s) outside of CT data range. Z coord min='+repr(min(tet_pts[2])))
        if (max(tet_pts[2]) > self.z[-1]):
            raise ValueError('Mesh Error: Element(s) outside of CT data range. Z coord max='+repr(max(tet_pts[3])))

        # find grid data
        [[xstart, xend], [ystart, yend], [zstart, zend]] = self.find_grid(tet_pts)

        # test if element is smaller than voxel
        if (xstart == xend - 1):
            if (ystart == yend - 1):
                if (zstart == zend - 1):
                    in_element = True
                    ele_centroid = mean(tet_pts,0).tolist()
                    x = [bisect_right(self.x, ele_centroid[0])-1, bisect_left(self.x, ele_centroid[0])]
                    y = [bisect_right(self.y, ele_centroid[1])-1, bisect_left(self.y, ele_centroid[1])]
                    z = [bisect_right(self.z, ele_centroid[2])-1, bisect_left(self.z, ele_centroid[2])]
                    eight_vox = _xyz_comb(x, y, z)
                    for v in eight_vox:
                        voxel_lookup = _calc_lookup(v[0], v[1], v[2], self.dimen)
                        self.vox.add(voxel_lookup)
                        voxels.append(voxel_lookup)
                    
        # for each voxel test if in tet, if is return lookup
        if in_element == False:
            for x in range(xstart, xend + 1):
                for y in range(ystart, yend + 1):
                    for z in range(zstart, zend + 1):
                        if _calc_lookup(x,y,z,self.dimen) not in self.vox:
                            if tet.in_tet([self.x[x],self.y[y],self.z[z]]):
                                voxel_lookup = _calc_lookup(x,y,z,self.dimen)
                                self.vox.add(voxel_lookup)
                                voxels.append(voxel_lookup)
        return voxels
    
    def find_grid(self, pts, box = False):
        # calculate grid 
        xstart = bisect_right(self.x, min(pts[0]))-1
        xend = bisect_left(self.x, max(pts[0]))
        ystart = bisect_right(self.y, min(pts[1]))-1
        yend = bisect_left(self.y, max(pts[1]))
        zstart = bisect_right(self.z, min(pts[2]))-1
        zend = bisect_left(self.z, max(pts[2]))            
        # if need start and end to be different (box = true) add one to end
        if box:
            if xstart == xend:
                if xend == (len(self.x)-1):
                    xstart -= 1
                else:
                    xend += 1
            if ystart == yend:
                if yend == (len(self.y)-1):
                    ystart -= 1
                else:
                    yend += 1
            if zstart == zend:
                if zend == (len(self.z)-1):
                    zstart -= 1
                else:
                    zend += 1
                
        return [[xstart, xend], [ystart, yend], [zstart, zend]]
        
    def interpolateScalar(self, xyz, modulus, rhoAsh, rhoQCT):
        # calculate CT box surrounding the point
        [xi, yi, zi] = self.find_grid([[xyz[0]], [xyz[1]], [xyz[2]]], True) # index values
        box = [[xi[0], yi[0], zi[0]],
               [xi[0], yi[0], zi[1]],
               [xi[0], yi[1], zi[0]],
               [xi[0], yi[1], zi[1]],
               [xi[1], yi[0], zi[0]],
               [xi[1], yi[0], zi[1]],
               [xi[1], yi[1], zi[0]],
               [xi[1], yi[1], zi[1]]]
        # define the origin
        origin_indx = 0
        origin = [self.x[xi[origin_indx]],
                  self.y[yi[origin_indx]],
                  self.z[zi[origin_indx]]]
        # for each corner of the box, find the scalar value
        scalars = [self.lookup[_calc_lookup(b[0], b[1], b[2], self.dimen)] for b in box]
        # apply modulus calculation (only changes value if V3)
        scalars = [modulus(rhoAsh(rhoQCT(s))) for s in scalars]
        # calculate the dimensions of the box
        differences = [self.x[xi[1]] - origin[0],
                       self.y[yi[1]] - origin[1],
                       self.z[zi[1]] - origin[2]]
        # calculate the relative position (%) of the co-ordinate within the box
        rel_xyz = [(xyz[n] - origin[n])/differences[n] for n in [0,1,2]]    
        # interpolate scalar
        c00 = (rel_xyz[0] * (scalars[4] - scalars[0])) + scalars[0]
        c01 = (rel_xyz[0] * (scalars[5] - scalars[1])) + scalars[1]
        c11 = (rel_xyz[0] * (scalars[7] - scalars[3])) + scalars[3]
        c10 = (rel_xyz[0] * (scalars[6] - scalars[2])) + scalars[2]
        c0 =  (rel_xyz[1] * (c10 - c00)) + c00
        c1 =  (rel_xyz[1] * (c11 - c01)) + c01
        c =   (rel_xyz[2] * (c1 - c0)) + c0
        return c
    
#-------------------------------------------------------------------------------
# Miscellaneous functions
#-------------------------------------------------------------------------------
def _calc_lookup(x,y,z,dimen):
    return x + (y*dimen[0]) + (z*dimen[0]*dimen[1])   

def _xyz_comb(x, y, z):
    """ Find voxels within x, y, z ranges """
    combinations = []
    for i in x:
        for j in y:
            for k in z:
                combinations.append([i,j,k])
    return combinations

#-------------------------------------------------------------------------------
# Element classes
#-------------------------------------------------------------------------------
# linear tetrahedron (C3D4)
class linear_tet:
    """ Linear tetrahedral element class """
    
    __slots__ = ("indx", "pts", "nodes", "volume")
    
    def __init__(self, indx, pts, nodes):
        self.indx = indx
        self.pts = pts
        self.nodes = nodes
        self.volume = 0
       
    def jacobian(self):
        # create base matrix from vertices for in_tet test
        return [[1, 1, 1, 1],
               [self.pts[0][0], self.pts[1][0], self.pts[2][0], self.pts[3][0]],
               [self.pts[0][1], self.pts[1][1], self.pts[2][1], self.pts[3][1]],
               [self.pts[0][2], self.pts[1][2], self.pts[2][2], self.pts[3][2]]]
        
    def in_tet(self, pt):
        # test if point is within tet
        test = True
        for n in [0, 1, 2, 3]:
            tmp_mtx = deepcopy(self.jacobian())
            tmp_mtx[1][n] = pt[0]
            tmp_mtx[2][n] = pt[1]
            tmp_mtx[3][n] = pt[2]
            if det(tmp_mtx) < 0:
                test = False
        return test

    def integral(self, numSteps, vtk, rhoQCT = lambda a:a, rhoAsh = lambda b:b, modulus = lambda c:c):
        # calculate integral co-ordinates for the element
        step = 1.0 / numSteps
        volume = 0
        integral = 0
        count = 0
        # iterate through tetrahedral volume, where:
        #    l = natural co-ordinate 1
        #    r = natural co-ordinate 2
        #    s = natural co-ordinate 3
        #    t = natural co-ordinate 4
        for t in arange(step / 2., 1, step):
            for s in arange(step / 2., 1 - t, step):
                for r in arange(step / 2., 1 - s - t, step):                  
                    count += 1
                    # find jacobian
                    detJ = det(self.jacobian())
                    # add up volume
                    volume += (detJ / 6.)
                    # calculate shape functions
                    w = [1 - r - s - t,
                         r,
                         s,
                         t]
                    # find co-ordinate for each iteration using shape functions
                    x = [0,0,0]
                    for n in [0,1,2]:
                        for i in range(4):
                            x[n] += w[i]*self.pts[i][n]
                    # for each co-ordinate, find corresponding CT co-ordinate
                    integral += (vtk.interpolateScalar(x, modulus, rhoAsh, rhoQCT)) * (detJ / 6.)

        self.volume = volume / count
        return integral / volume
    
# Quadratic tetrahedron (C3D10)
class quad_tet:
    """ Quadratic tetrahedral element class """
    
    __slots__ = ("indx", "pts", "nodes", "volume")
    
    def __init__(self, indx, pts, nodes):
        self.indx = indx
        self.pts = pts
        self.nodes = nodes
        self.volume = []

    def jacobian(self, l, r, s, t):
        # create jacobian for integration methods
        p = self.pts
        J = [[((p[0][i] * ((4*l)-1)) + (p[4][i] * 4*r) + (p[6][i] * 4*s) + (p[7][i] * 4*t)),                   #{dN/dl}
              ((p[1][i] * ((4*r)-1)) + (p[4][i] * 4*l) + (p[5][i] * 4*s) + (p[8][i] * 4*t)),                   #{dN/dr}
              ((p[2][i] * ((4*s)-1)) + (p[5][i] * 4*r) + (p[6][i] * 4*l) + (p[9][i] * 4*t)),                   #{dN/ds}
              ((p[3][i] * ((4*t)-1)) + (p[7][i] * 4*l) + (p[8][i] * 4*r) + (p[9][i] * 4*s))] for i in [0,1,2]] #{dN/dt}

        # make the matrix square
        J = [[1.,1.,1.,1.]] + J

        ## Jacobian used by original BoneMat Software (possibly erroneous)
        ## ct = 1-t
        ## lt = 4 * t
        ## cs = 1-s
        ## ls = 4 *s
        ## l = 4 * (1 -r - s - t)
        ## lr = 4 * r
        ## J = [[(p[0][i] * (1 - l)) + (p[1][i] * (lr-1)) + (p[4][i] * (l - lr)) + ((p[5][i] - p[6][i])*ls) + ((p[8][i]-p[7][i]) * lt),
        ##       (p[0][i] * (1 - l)) + (p[2][i] * (ls-1)) + (p[6][i] * (l - ls)) + ((p[3][i] - p[4][i])*lr) + ((p[9][i]-p[7][i]) * lt),
        ##       (p[0][i] * (1 - l)) + (p[3][i] * (lt-1)) + (p[7][i] * (l - lt)) + ((p[8][i] - p[4][i])*lr) + ((p[9][i]-p[6][i]) * ls)] for i in [0,1,2]]
              

        return J


    def integral(self, numSteps, vtk, rhoQCT = lambda a:a, rhoAsh = lambda b:b, modulus = lambda c:c):
        # calculate integral co-ordinates for the element
        p = self.pts
        step = 1.0 / numSteps
        volume = 0
        integral = 0
        count = 0
        # iterate through tetrahedral volume, where:
        #    l = natural co-ordinate 1
        #    r = natural co-ordinate 2
        #    s = natural co-ordinate 3
        #    t = natural co-ordinate 4
        for t in arange(step / 2., 1, step):
            for s in arange(step / 2., 1 - t, step):
                for r in arange(step / 2., 1 - s - t, step):
                    count += 1
                    l = 1 - r - s - t
                    # find jacobian
                    detJ = det(self.jacobian(l, r, s, t))
                    # add up volume
                    volume += (detJ / 6.)
                    # calculate shape functions
                    w = [(2 * l - 1) * l,
                         (2 * r - 1) * r,
                         (2 * s - 1) * s,
                         (2 * t - 1) * t,
                         4 * l * r,
                         4 * r * s,
                         4 * l * s,
                         4 * l * t,
                         4 * r * t,
                         4 * s * t]
                    # find co-ordinate for each iteration using shape functions
                    x = [0,0,0]
                    for n in [0,1,2]:
                        for i in range(10):
                            x[n] += w[i]*p[i][n]
                    # for each co-ordinate, find corresponding CT co-ordinate
                    integral += (vtk.interpolateScalar(x, modulus, rhoAsh, rhoQCT)) * (detJ / 6.)

        self.volume = volume / count
        return integral / volume              
        
# Wedge element
class linear_wedge:
    """ Linear wedge element class """
    
    __slots__ = ("indx", "pts", "nodes", "volume")
    
    def __init__(self, indx, pts, nodes):
        self.indx = indx
        self.pts = pts
        self.nodes = nodes
        self.volume = []

    def jacobian(self, r, s, t):
        # create jacobian for integration methods
        p = self.pts
        J = [[(p[1][i]-p[0][i])*(1-t) + (p[4][i]-p[3][i])*t,
              (p[2][i]-p[0][i])*(1-t) + (p[5][i]-p[3][i])*t,
              (p[4][i]-p[1][i])*r + (p[5][i]-p[2][i])*s + (p[3][i]-p[0][i])*(1-r-s)] for i in [0,1,2]] 
        
        return J
           
    def integral(self, numSteps, vtk, rhoQCT = lambda a:a, rhoAsh = lambda b:b, modulus = lambda c:c):
        # calculate integral co-ordinates for the element
        p = self.pts
        step = 1.0 / numSteps
        volume = 0
        integral = 0
        count = 0
        # iterate through tetrahedral volume, where:
        #    l = natural co-ordinate 1
        #    r = natural co-ordinate 2
        #    s = natural co-ordinate 3
        #    t = natural co-ordinate 4
        for t in arange(step / 2., 1., step):
            for s in arange(step / 2., 1., step):
                for r in arange(step / 2., 1. - s, step):
                    count += 1
                    # find jacobian
                    detJ = det(self.jacobian(r, s, t))
                    # add up volume
                    volume += detJ / 2.
                    # calculate shape functions
                    w = [(1 - r - s) * (1 - t),
                         s * (1 - t),
                         r * (1 - t),
                         (1 - r - s) * t,
                         s * t,
                         r * t]
                    
                    # find co-ordinate for each iteration using shape functions
                    x = [0,0,0]
                    for n in [0,1,2]:
                        for i in range(6):
                            x[n] += w[i]*p[i][n]
                    # for each co-ordinate, find corresponding CT co-ordinate
                    integral += (vtk.interpolateScalar(x, modulus, rhoAsh, rhoQCT)) * (detJ / 2.)

        self.volume = volume / count
        
        return integral / volume              
        
# Hexahedral element
class linear_hex:
    """ Hexahedral element class """
    
    __slots__ = ("indx", "pts", "nodes", "jacobian")
    
    def __init__(self, indx, pts, nodes):
        self.indx = indx
        self.pts = pts
        self.nodes = nodes
        self.volume = []

    def jacobian(self, r, s, t):
        # create jacobian for integration methods
        p = self.pts
        J  = [[(p[1][i]-p[0][i])*(1-s)*(1-t) + (p[2][i]-p[3][i])*(1+s)*(1-t) + (p[5][i]-p[4][i])*(1-s)*(1+t) + (p[6][i]-p[7][i])*(1+s)*(1+t),
               (p[3][i]-p[0][i])*(1-r)*(1-t) + (p[2][i]-p[1][i])*(1+r)*(1-t) + (p[7][i]-p[4][i])*(1-r)*(1+t) +(p[6][i]-p[5][i])*(1+r)*(1+t),
               (p[4][i]-p[0][i])*(1-r)*(1-s) + (p[5][i]-p[1][i])*(1+r)*(1-s) + (p[6][i]-p[2][i])*(1+r)*(1+s) +(p[7][i]-p[3][i])*(1-r)*(1+s)] for i in [0,1,2]]

        return J
        
    def integral(self, numSteps, vtk, rhoQCT = lambda a:a, rhoAsh = lambda b:b, modulus = lambda c:c):
        # calculate integral co-ordinates for the element
        p = self.pts
        step = 1.0 / numSteps
        volume = 0
        integral = 0
        count = 0
        for t in arange(-1 + step, 1, step * 2):
            for s in arange(-1 + step, 1, step * 2):
                for r in arange(-1 + step, 1, step * 2):
                    count += 1
                    # find jacobian
                    detJ = det(self.jacobian(r, s, t))
                    # add up volume
                    volume += (detJ / 8.)
                    # calculate shape functions
                    w = [((1 - r) * (1 - s) * (1 - t)) / 8.,
                         ((1 + r) * (1 - s) * (1 - t)) / 8.,
                         ((1 + r) * (1 + s) * (1 - t)) / 8.,
                         ((1 - r) * (1 + s) * (1 - t)) / 8.,
                         ((1 - r) * (1 - s) * (1 + t)) / 8.,
                         ((1 + r) * (1 - s) * (1 + t)) / 8.,
                         ((1 + r) * (1 + s) * (1 + t)) / 8.,
                         ((1 - r) * (1 + s) * (1 + t)) / 8.,]
                    # find co-ordinate for each iteration using shape functions
                    x = [0,0,0]
                    for n in [0,1,2]:
                        for i in range(8):
                            x[n] += w[i]*p[i][n]
                    # for each co-ordinate, find corresponding CT co-ordinate
                    integral += (vtk.interpolateScalar(x, modulus, rhoAsh, rhoQCT)) * (detJ / 8.)
                    
        self.volume = volume / (count * 8) 

        return integral / volume             
