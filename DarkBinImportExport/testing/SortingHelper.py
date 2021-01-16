import bpy
import bmesh
import numpy as np
from mathutils import Vector
from math import sin, cos, asin, acos, atan2, degrees, radians

from operator import itemgetter, attrgetter
from itertools import product as product_iter, chain, permutations, compress, tee

import time

# (dezimal) percentage that the opt algorithm expects before aborting
REL_OPT_IMPROVEMENT = 0.05
#Note: Faces with absolute theta angles -90 and 90
# with a representative normal that is tilted - if something ever comes up.

"""
typedef struct mds_sphere {
   mxs_vector cen;
   float rad;
} mds_sphere;
"""

class ExportFace:
    """
    Instance holds data important for exporting.
    Classmethods relations between faces
    """
    LookUpFaces  = {}       # mesh specific but maybe not complete!
    LookUpMatrix = {}
    _FacesNotSeen = {}
    _FacesNotSeeing = {}
    _FacesNormal = {}
    _InitialLookUpMatrix = {}
    activeMesh = None
    
    LookUpSideInformation = {}
    
    @classmethod
    def allFacesInObject(cls):
        """
        Get all faces in the object.
        Important may NOT BE COMPLETE
        as not all subobjects has been processed.
        """
        return chain(cls.LookUpFaces.values()) # Iterable!
    
    @classmethod
    def allFacesInMesh(cls, mesh):
        return cls.LookUpFaces[mesh]
    
    @classmethod
    def get(cls, idx, mesh=None):
        """
        Get ExportFace object with the given index or indices
        in the given mesh
        """
        if mesh == None:
            mesh = cls.activeMesh
        if type(idx) == int:
            return cls.LookUpFaces[mesh][idx]
        return [cls.LookUpFaces[mesh][i] for i in idx]

# ============================================================
    
    @staticmethod
    def FaceVisible(f1, f2):
        """
        Checks if a face f2 is visible through a face f1. Ignoring obstacles.
        This is object internally.
        """        
        # Here we could determine if a split would be good if only one vertex is visible.
        # f1.d is positive as it was calulated with positive normal
        # Plane equation: pos @ n - d = 0
        r1 = np.dot(f2.vert_pos, -f1.normal) + f1.d
        r2 = np.dot(f1.vert_pos,  f2.normal) - f2.d

        # Need to cope with numerical Xe-7 close to 0 errors.
        # should also cope for double sided but maybe np.array_equal/allclose could be needed
        return (r1 > 1e-2).any() and (r2 > 1e-2).any() 
    
    @staticmethod
    def FaceVisibleNP(verts, normals, ds):
        """
        Checks if a face f2 is visible through a face f1. Ignoring obstacles.
        This is object internally.
        """
        size=len(ds)
        r1 = np.empty((size, size), dtype=int)
        r2 = r1.copy()
        for i, vs in enumerate(verts):
            for j, n in enumerate(normals):
                res = vs.dot(n)
                # Here we could determine if a split would be good if only one vertex is visible.
                # f1.d is positive as it was calulated with positive normal
                # Plane equation: pos @ n - d = 0
                r2[i,j] = ( res - ds[j] > 1e-2).any()
                r1[j,i] = (-res + ds[j] > 1e-2).any()
                if (i == 6) and (j == 18):
                    print("CHECK ME", i,j, r1[j,i], r2[i,j])
                
            #print("Processing face", i,"/", size, end='\r')
        #print("Processing face", size,"/", size)
        print(r2[4:8,15:22],"\n", r1[4:8,15:22], "\n\n")
        print(r2[4:8,15:22] & r1[4:8,15:22])
        print("==")
        print(r2[4:8,15:22] & ~r2.T[4:8,15:22])
        # Sounds cool doesnt work.
        # There are situations where you can see through front AND Back => This wrong
        return r2 & ~r2.T
    
    @classmethod
    def _FaceVisibleNPworks(cls, verts, normals, ds, mesh):
        """
        Checks if a face f2 is visible through a face f1. Ignoring obstacles.
        This is object internally.
        """
        size=len(ds)
        r1 = np.empty((size, size), dtype=int)
        r2 = r1.copy()
        for i, vs in enumerate(verts):
            for j, n in enumerate(normals):
                res = vs.dot(n)
                # Here we could determine if a split would be good if only one vertex is visible.
                # f1.d is positive as it was calulated with positive normal
                # Plane equation: pos @ n - d = 0
                r2[i,j] = ( res - ds[j] > 1e-2).any()
                r1[j,i] = (-res + ds[j] > 1e-2).any()
                    
            print("Processing face", i,"/", size, end='\r')
        print("Processing face", size,"/", size)

        cls.LookUpSideInformation[mesh]=r2
        return r1 & r2        
    
    @classmethod
    def _makeNP(cls, mesh):
        new_faces = cls.LookUpFaces[mesh]
        data = np.array([[f.vert_pos, f.normal, f.d] for f in new_faces])
        data = [0]*len(new_faces)
        data2=data.copy()
        data3=data.copy()
        for i,f in enumerate(new_faces):
            data[i]=f.vert_pos
            data2[i]=f.normal
            data3[i]=f.d
        V=cls._FaceVisibleNPworks(data,data2,data3, mesh)
        #V2 = cls.FaceVisibleNP(data,data2,data3)
        for idx, row in enumerate(V):
            new_faces[idx].sees.extend(f for f,m in zip(new_faces, row) if m)
            col = V[:, idx]
            new_faces[idx].seen_by.extend(f for f,m in zip(new_faces, col) if m)

        #print("NEW:", (V==V2).astype(int))
        #raise
        #V=cls.FaceVisibleNPworks(data,data2,data3)
        #V=cls.FaceVisibleNPworks(data[:,0],data[:,1],data[:,2])
        return V  
    
    # Non vectorized code
    """
    new_faces = cls.LookUpFaces[mesh]
    mat = cls.LookUpMatrix[mesh] = np.zeros((len(mesh.faces), len(mesh.faces)), dtype=bool)
    for f1 in new_faces:
        for f2 in new_faces:
            if f1.index == f2.index:
                continue
            visible = cls.FaceVisible(f1, f2)
            if visible:
                mat[f1.index, f2.index] = visible
                f1.sees.append(f2)
                f2.seen_by.append(f1)
        print("Processing face", f1.index,"/", len(mesh.faces), end='\r')
    tac = time.perf_counter()
    """
    
    @classmethod
    def doVisibilityChecks(cls, mesh):
        tic = time.perf_counter()
        V = cls.LookUpMatrix[mesh] = cls._makeNP(mesh)
        toc = time.perf_counter()
        print(f"{toc-tic:0.4f}")
        """
        new_faces = cls.LookUpFaces[mesh]
        mat = cls.LookUpMatrix[mesh] = np.zeros((len(mesh.faces), len(mesh.faces)), dtype=bool)
        for f1 in new_faces:
            for f2 in new_faces:
                if f1.index == f2.index:
                    continue
                visible = cls.FaceVisible(f1, f2)
                if visible:
                    mat[f1.index, f2.index] = visible
                    f1.sees.append(f2)
                    f2.seen_by.append(f1)
            #print("Processing face", f1.index,"/", len(mesh.faces), end='\r')
        tac = time.perf_counter()
        print(f"{tac-toc:0.4f}")
        print("Equal", np.array_equal(V, mat))
        """
        cls._InitialLookUpMatrix[mesh] = V.copy()
        return V
    
    @classmethod
    def buildFaces(cls, mesh, vis_check=False):
        cls.activeMesh = mesh
        for f in mesh.faces:
            ExportFace(f, mesh) # Goes into LookUpFaces
        if vis_check:
            # Slow for many faces
            cls.doVisibilityChecks(mesh)
        return cls.LookUpFaces[mesh]

# ============================================================
    
    @classmethod
    def updateIndices(cls, mesh, order, only_sort=False):
        faces = cls.LookUpFaces[mesh]
        # Reset, that is new
        faces.sort(key=attrgetter('old_index'))
        if only_sort:
            backup = faces.copy()
            # A FEW THINGS WILL NOT WORK PROPERLY AFTER THIS
            for i, idx in enumerate(order):
                faces[i] = backup[idx]
        else:
            for i, idx in enumerate(order):
                faces[idx].index = i
            faces.sort(key=attrgetter('index'))         
    
    @classmethod
    def updateMatrix(cls, mesh, order):
        # Order describes how the old -> new face numbers have to be aranged.
        # Need to fall back to the initial lookup matrix
        org_mat = cls._InitialLookUpMatrix[mesh]
        # And create new one
        new_mat = cls.LookUpMatrix[mesh] = np.ndarray(org_mat.shape, dtype=int)

        for new, old in enumerate(order):
            # Update n-th row and column
            new_mat[new] = org_mat[old][order]
            new_mat[:,new] = org_mat[:,old][order]
    
    @classmethod
    def refresh_faces(cls, mesh, order):
        """ 
        Needs to run after bmesh faces become invalid
        """
        cls.updateIndices(mesh, order)
        faces = cls.LookUpFaces[mesh]
        for f in faces:
            f.refresh_face(mesh)
        if mesh in cls.LookUpMatrix:
            cls.updateMatrix(mesh, order)
    
    def refresh_face(self, mesh):
        self.bmesh_face = mesh.faces[self.index]

# ============================================================
    @classmethod
    def getConflictsForFaces(cls, mesh, faces, filter=3):
        """
        Faces that see one with higher indices
        filter bitwise
        1: sees higher index
        2: seen by lower index
        3: combined
        """
        assert (filter < 4 and filter > 0), "Filter must be between 1 and 3."
        mat = cls.LookUpMatrix[mesh]
        if faces != cls.LookUpFaces[mesh]:
            face_indices = [f.index for f in faces]
            reduced_mat = mat[np.ix_(face_indices, face_indices)]
        else:
            reduced_mat = mat
        
        reduced = np.triu(reduced_mat)
        if filter & 1:
            conflict = reduced.any(axis=0)
        else:
            conflict = np.zeros((mat.shape[0],), dtype=bool)
        if filter & 2:
            conflict |= reduced.any(axis=1)
        
        print("Conflict FACES")
        print("Can see", [b.index for a, b in zip(reduced.any(axis=0), faces) if a])
        print("Is Seen", [b.index for a, b in zip(reduced.any(axis=1), faces) if a])
        
        return [b for a, b in zip(conflict, faces) if a]
        
    
    @classmethod
    def getConflicts(cls, mesh, filter=3):
        faces = cls.LookUpFaces[mesh]
        return cls.getConflictsForFaces(mesh, faces, filter)
    
    @classmethod
    def getConflictsForNode(cls, node, filter=1):
        # Sees wrong
        return cls.getConflictsForFaces(node.mesh, node.faces, filter)
    
    # Bool / See / Seen functions
    
    @classmethod
    def getFacesNotSeen(cls, mesh):
        return [f for f in cls.LookUpFaces[mesh] if not f.is_seen()]
        # Possible cache option
        rv = cls._FacesNotSeen.get(mesh, None)
        if rv != None:
            return rv
        rv = cls._FacesNotSeen[mesh] = [f for f in cls.LookUpFaces[mesh] if not f.is_seen()]
        return rv

    @classmethod
    def getFacesNotSeeing(cls, mesh):
        return [f for f in cls.LookUpFaces[mesh] if not f.can_see()]    
    
    @classmethod
    def getFacesNormal(cls, mesh):
        return [f for f in cls.LookUpFaces[mesh] if f.can_see() and f.is_seen()] 
    
    @classmethod
    def getFacesFiltered(cls, mesh, filter=None):
        """
        Returns faces NotSeen + Normal + NotSeein as Tuple
        if a filter is given, restrict to these faces
        """
        if filter is not None:
            faces = filter
        else:
            faces = cls.LookUpFaces[mesh]
        
        NotSeeing = []
        Normal = faces.copy()
        NotSeen = []
        for f in faces:
            if not f.is_seen(filter):
                Normal.remove(f) # pop would be cooler
                NotSeen.append(f)
            elif not f.can_see(filter):
                Normal.remove(f)
                NotSeeing.append(f)
        return NotSeeing, Normal, NotSeen
    
    #
    
    def is_seen(self, filter=None):
        if not filter:
            return bool(self.seen_by)
        # Only take faces in filter into account
        return bool([f for f in self.seen_by if f in filter])
    
    def can_see(self, filter=None):
        if not filter:
            return bool(self.sees)
        return bool([f for f in self.sees if f in filter])
    #
    
    def calc_center_bounds(self):
        # Cacheing for speed
        return self.center_bounds
    #
    
    def __init__(self, face, mesh):
        self.bmesh_face = face
        self.old_index = face.index
        self.index = face.index
        self.mesh = mesh
        # cache them
        self.center_bounds = face.calc_center_bounds()
        self.d = face.normal @ self.center_bounds
        self.normal = np.array(face.normal)
        self.vert_pos = np.array([v.co for v in face.verts])
        self.sees = []
        self.seen_by = []
        ExportFace.LookUpFaces.setdefault(mesh, []).append(self)
    
    def __repr__(self):
        return str(self.index)    
        
    def __getattr__(self, key):
        # redirect to bmesh face if not present
        return getattr(self.bmesh_face, key)



# ========================================================================================

local_bbox_center = None   
def getBBoxCenter(obj):
    if local_bbox_center:
        return local_bbox_center
    local_bbox_center = 0.125 * sum((Vector(b) for b in obj.bound_box), Vector())
    return local_bbox_center
    
def vectortocenter(pos):
    return local_bbox_center - pos

def disttocenter_face(f):
    return vectortocenter(f.calc_center_median()).length

def disttocenter_vert(v):
    return vectortocenter(v.co).length


      
# //////////////////////////////////////////////////////////

def PolarAngles(v):
    r = 1 
    # Yay atan2 loves -0.0
    if v[0] == -0.0:
        v[0] = 0.0
    # Can get rid of round and int when in range() method gets replaced by a float variant.
    # NOTE: Theta the co-angle from the X-Axis [-90,90]
    return int(round(degrees(asin(v[2] / r)))), int(round(degrees(atan2(v[1], v[0]))))

# Unused
def vertex_radius(v):
    return v.co.length

def CalcNormal(theta, phi):
    theta = radians(90.0 - theta) # Get normal theta
    phi = radians(phi)
    return Vector([sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)])



# ==============================================================================
# Sorter
# ==============================================================================

def ZDistSorter(groups, local_ZDists):
    """
    This prevents a biased sorting by pure ZDist if more than one face
    in other groups also have that distance.
    """
    done = False
    group_list = []
    value_list = []
    for k, v in groups.items():
        if k in local_ZDists:
            group_list.append(v.copy())
            value_list.append(local_ZDists[k].copy())
    ordered = []
    prevent_bias = False
    while not done:
        done = True
        chosen_group_idx = None
        n_min = 10000
        for i, ZDists in enumerate(value_list):
            if not len(ZDists):
                continue
            done = False
            n = ZDists[0]
            if n < n_min:
                n_min = n
                chosen_group_idx = i
        if not done:
            next_face = group_list[chosen_group_idx].pop(0)
            # DEBUG
            ordered.append(next_face.index)
            next_face.index = len(ordered) - 1  # New index
            value_list[chosen_group_idx].pop(0) # Remove ZDist value for face
            if prevent_bias:
                # Move the currently chosen list to the end, in case multiple
                # groups have the same radius and one group gets drained
                group_list.append(group_list.pop(chosen_group_idx))
                value_list.append(value_list.pop(chosen_group_idx))             
    return ordered


def _make_division_angles(divisions, sphere_hat):
    t_divisions_adjust = -1 # Plus or -1
     
    # For the theta angle there should always be an uneven division
    # Else the split would be directly in the XY plane might be troublesome.
    if divisions % 2 == 1:
        # uneven
        t_divisions = divisions
    else:
        t_divisions = divisions + t_divisions_adjust # Make uneven

    p_step = 360 // divisions
    t_step = 180 // t_divisions

    # Out of range objects we build the possible angle combinations we can have:
    p_angles = []
    t_angles = []
    division_angles = (p_angles, t_angles)
    cur_p = -180 - p_step // 2 
    cur_t = -90
    # First and last vision are halfed cause of the -180 / 180 step
    for i in range(divisions):
        p_angles.append(range(cur_p, cur_p + p_step))
        if i == t_divisions + t_divisions_adjust:
            t_step += 1     # so range(, 91) included 90
        if i < t_divisions:
            t_angles.append(range(cur_t, cur_t + t_step))
        cur_t += t_step
        cur_p += p_step
    
    # Fix groups:
    p_angles.append(range(cur_p, cur_p + p_step)) # Add second half of first.
    if t_divisions > divisions:  # t_divisions are always uneven or +1
        t_angles.append(range(cur_t, cur_t + t_step + 1))   
    return division_angles, t_step, p_step, t_divisions


def normal_filter2(mesh, divisions=4, sphere_hat=True, filter=None):
    """
    This sorts all faces with similar normals into division² groups.
    Each spherical angle [-180, 180]° around Z-Axis (phi), 
    and [-90, 90]° around X-Axis (theta) is divided into n segments.
    
    Theta divisions are always uneven.
    
    With Sphere Hat instead of having multiple sides at the top and bottom
    will combine them to a single Z and -Z Axis.
    """
    division_angles, t_step, p_step, t_divisions = _make_division_angles(divisions, sphere_hat)
    if filter is None:
        faces = ExportFace.allFacesInMesh(mesh)
    else:
        faces = filter
    
    groups = { (p, t):[] for t in range(t_divisions) for p in range(divisions) }
    
    # Now sort all faces into these groups
    for f in faces:
        n = f.normal
        theta, phi = PolarAngles(n)
        ##print("="*20+"\n",f.index, "\n", theta, phi)
        ##print(f.normal)
        
        t_idx = (theta +90) // t_step
        p_idx = (phi + 180) // p_step
        
        if p_idx == divisions:      # As first and last are the same group just halfed.
            p_idx = 0  
        if t_idx == t_divisions:    # Would give one idx to high
            t_idx = t_divisions - 1
        
        if sphere_hat and (t_idx == t_divisions - 1 or t_idx == 0):
            # Don't separate by p_idx for highest/lowst theta angles
            p_idx = 0
        # Store them in their group and also in a lookup table
        groups[(p_idx, t_idx)].append(f)
        f.group = (p_idx, t_idx)
    
    
    # Helper function the get the group of a unspecific angle    
    def group_lookup(p, t, sphere_hat=True):
        for idx, a in enumerate(division_angles[0]): # p_angles
            if int(p) in a:
                p_idx = idx
                break
        for idx, a in enumerate(division_angles[1]): # t_angles
            if int(t) in a:
                t_idx = idx
                if sphere_hat:
                    if idx == 0 or a == division_angles[1][-1]:
                        p_idx = 0
                break
        return (p_idx, t_idx)
    return groups, group_lookup, division_angles

def order_normal_after_sort(faces):
    pass

def order_normal_space(faces, groups, group_lookup, division_angles, sphere_hat=True, mode='ZDist'):
    ##print(division_angles)
    Z_Dists = []
    local_ZDists={} # same structure as groups
    # as multiple groups would do the same thing need a flag indicator.
    sphere_hat_done = 0
    mainNormals={}
    
    # Sort all by their representative vector - division² cycles.
    for p_range, t_range in product_iter(*division_angles):
        # As the -180 + 180 gap is covered twice in angles.
        if p_range == division_angles[0][-1]:
            continue
        if sphere_hat:
            if t_range == division_angles[1][0]:
                if sphere_hat_done & 1:
                    continue
                t_main = -90
                p_main = 0
                sphere_hat_done |= 1
            elif t_range == division_angles[1][-1]:
                if sphere_hat_done & 2:
                    continue
                t_main = 90
                p_main = 0
                sphere_hat_done |= 2
            else:
                p_main = (p_range[-1] + p_range[0]) / 2 + 0.5
                t_main = (t_range[-1] + t_range[0]) / 2 + 0.5
        else:       
            # It's 89.5 - does it make a difference?
            p_main = (p_range[-1] + p_range[0]) / 2 + 0.5
            t_main = (t_range[-1] + t_range[0]) / 2 + 0.5
        
        # Get in which group the theoretical angle would be:
        gr_key = group_lookup(p_main, t_main)
        similar_faces = groups[gr_key]
        if len(similar_faces) == 0:
            continue
        
        # As theta is caped at 90 these last normals are uneven.
        # Could calculate the average normal this is the theoretical one
        main_normal = CalcNormal(t_main, p_main)
        mainNormals[gr_key] = main_normal
        ##print("main normal", main_normal, main_normal.length)
        
        # ----------------------------------------------------------
        ## STEP 1 - SORT PER GROUP ##
        
        # Now we to sort them based on the representative main normal
        # Their position vector will be divides into 
        # pos = n_scale * main_normal + b 
        # where b is othogonal to the normal. Then sort by n
        # Then they will be sorted from back to front
        
        # ----------------------------------------------------------
        local_ZDists[gr_key] = []
        def pos_sort(f):
            # Still in loop
            pos = 0
            # Both ideas vert vs. center have advantages
            # Problem by verts is if verts are in the same plane
            # but faces should be different.
            for v in f.verts:
                if not pos or v.co.length > pos.length:
                    pos = v.co
            # I would prefer verts
            pos = f.calc_center_median()
            # n.length is not exact 1 so this is numerical, maybe round it?
            n_scale = round(main_normal @ pos, 7)
            # Experimental scale - do further out are
            
            ##print(f.index, "main:", main_normal, "N:", n)
            local_ZDists[gr_key].append(n_scale)
            # Now we also use this for a later use
            f.ZDist = n_scale
            return n_scale
        # This is to make the sorting more constant for equivalent faces
        first  = [pos_sort(f) for f in similar_faces]
        second = [f.calc_center_median().length for f in similar_faces]
        # This is still not perfect if two faces share this [0] vertex
        third  = [f.verts[0].co[0] for f in similar_faces]
        sorter = np.lexsort((third, second, first))
        similar_faces = [similar_faces[i] for i in sorter]
        groups[gr_key] = similar_faces
        local_ZDists[gr_key].sort() # small n to great, equivalent faces hae equivalent n
        #print([[f.index, f.ZDist] for i, f in enumerate(similar_faces)],"")
    
    # ----------------------------------------------------------
    ## STEP 2 - SORT ALL ##
    # From Step 1 we know how they have to be ordered in a group
    # Now these relative orders have to been combined by another algorithm into one list
    # Always one bottom element from all groups has to been added
    
    # I'm still looking a good algorithm
    # possible could be position, currently doing furthest vertex
    # swaped out by local Z-Dist of the groups
    # ----------------------------------------------------------
    if mode == 'ZDist':
        """
        ZDist is basically also a radius sort with negative values
        for faces that should be furth in the back.
        """
        #Z_Dists.sort(key=lambda f: f[1])
        # find same ns
        #ordered = [f[0] for f in Z_Dists]
        ordered = ZDistSorter(groups, local_ZDists)

    elif mode == 'Radius':
        ordered = []
        #faces.sort(key= lambda f: f.calc_center_median.length)
        done = False
        group_list = [*groups.values()]
        while not done:
            done = True
            chosen_group_idx = None
            r_max = 0
            for i, group in enumerate(group_list):
                if not len(group):
                    continue
                done = False
                # METHOD Furthest vertex
                #r = sorted(group[0].verts, key=vertex_radius)[-1].co.length
                # METHOD Face median center
                r = group[0].calc_center_median().length
                if r > r_max:
                    r_max = r
                    chosen_group_idx = i
            if not done:
                ordered.append(group_list[chosen_group_idx].pop(0).index)
                # Move the currently chosen list to the end, in case multiple
                # groups have the same radius and one group gets drained
                group_list.append(group_list.pop(chosen_group_idx))
    else:
        raise Exception("Invalid Mode")
    
    return ordered


# ========================================================================================
# KOpt Post optimization
# ========================================================================================


def makeAccessMat(mat):
    """
    Gives fast access to the rows and columns
    in the upper right triangle matrix.
    """
    rows = [m[i+1:] for i,m in enumerate(mat)]
    cols = [m[:i] for i,m in enumerate(mat.T)]
    return [rows, cols]

# 5 better and faster than 4, in Step1
# 6 faster and little less good than 5
OptTimes = []
WholeOptTimes = []
def kOpt(mat, mesh, order, max_runs=4):

    """
    Post Optimization.
    Can be run multiple times for further improvement.
    """
    from DarkBinImportExport.testing import SortingAlgorithms
    access = makeAccessMat(mat)
    mat2=mat.copy()
    access2 = makeAccessMat(mat2)

    OptTimes2 = []
    print("start cost", np.sum(np.triu(mat)), sum(a.sum() for a in access[0]))
    if mat.shape[0] < 150:
        optfunc = SortingAlgorithms.kOptKernel4
        args = (mat2, mesh, order, access2, 100, 100)
        SortingAlgorithms.REL_OPT_IMPROVEMENT = 0.00 # As good as algorithm allows.
        max_runs+=3
    elif mat.shape[0] < 300:
        optfunc = SortingAlgorithms.kOptKernel4
        args = (mat2, mesh, order, access2, 100, 100)
        SortingAlgorithms.REL_OPT_IMPROVEMENT = 0.03 # Tolerance between two runs.
        max_runs+=2
    elif mat.shape[0] < 450:
        optfunc = SortingAlgorithms.kOptKernel4
        args = (mat2, mesh, order, access2, 55, 20)
        max_runs += 1
        SortingAlgorithms.REL_OPT_IMPROVEMENT = 0.05
    elif mat.shape[0] < 650:
        optfunc = SortingAlgorithms.kOptKernel4
        args = (mat2, mesh, order, access2, 38, 8)
        SortingAlgorithms.REL_OPT_IMPROVEMENT = 0.06
    elif mat.shape[0] < 900:
        optfunc = SortingAlgorithms.kOptKernel4
        args = (mat2, mesh, order, access2, 30, 6)
        SortingAlgorithms.REL_OPT_IMPROVEMENT = 0.07
    else:
        optfunc = SortingAlgorithms.kOptKernel4
        args = (mat2, mesh, order, access2, 25, 5)
        SortingAlgorithms.REL_OPT_IMPROVEMENT = 0.08
    counter=0
    while counter < max_runs:
        counter += 1
        tic = time.perf_counter()
        stop, mat = optfunc(*args)
        toc = time.perf_counter()
        OptTimes.append((mat.shape[0], toc-tic, np.sum(np.triu(mat))))
        print("kOpt:", counter)
        if stop:
            break
    print(OptTimes)
    #print(OptTimes2)
    #print(Swaps)
    #print(Swaps2)
    #print(Swaps1)
    #raise
    return stop, mat

# Not so usefull these two
def makeCostMatrix(mat):
    # old - would need to be updated each change with all external costs.
    """
    # Get Badness
    # Sum up row on the right and column above
    # if face would have index j instead of i.
    # Internal costs: in the row. How many wrong faces are seen
    # External costs: column: Seen by faces with lower index.
    # External costs might be simply removed by bsp split
    """
    cost_mat = np.ndarray(mat.shape, dtype=int)
    for i in range(mat.shape[0]):
       r_val = np.sum(mat[i])
       c_val = 0
       cost_mat[0,i] = r_val
       for j in range(1, mat.shape[0]):
           r_val -= mat[i  , j-1]
           c_val += mat[j-1,   i]
           cost_mat[j,i] = r_val + c_val
    return cost_mat

def makeCurrentCost(mesh, all=False):
    # Splits it up in internal(can see) and external costs(is seen)
    if all:
        faces = ExportFace.LookUpFaces[mesh]
    else:
        faces = ExportFace.getFacesNormal(mesh)
    #(Total cost, Internal costs)
    costs = [[0, 0] for _ in range(len(faces))]
    for i, f in enumerate(faces):
        # Internal costs+External. These count active
        for f2 in f.sees:
            if f2.index > f.index:
                costs[i][0] += 1
                costs[i][1] += 1
        # External costs. These are needed for comparission if a swap is good
        for f2 in f.seen_by:
            if f2.index < f.index:
                costs[i][0] += 1
    print("Current Cost mat", costs)
    return costs

# 

def getReducedLookUpMatrix(mesh, filter=None, return_faces=False):
    """
    Remove only 0 rows and columns from the lookup matrix and coresponding in order.
    Filter can be used to only work with the provides ExportFaces
    """
    back_faces, normal_faces, front_faces = ExportFace.getFacesFiltered(mesh, filter)
    
    not_both = [f.index for f in normal_faces]
    reduced = ExportFace.LookUpMatrix[mesh][np.ix_(not_both, not_both)]
    
    if (np.sum(reduced, axis=1) == 0).any() or (np.sum(reduced, axis=0) == 0).any():
        print("ANOTHER 0 Col")
        reduced, back_faces2, normal_faces2, front_faces2 = getReducedLookUpMatrix(mesh, filter=normal_faces, return_faces=True)
        back_faces.extend(back_faces2)
        front_faces2.extend(front_faces)
        front_faces = front_faces2           
        normal_faces = normal_faces2

    # Need these to filter order. Original indices
    if return_faces:
        return (reduced, back_faces, normal_faces, front_faces)
    # Give back old indices for order
    back_faces_old = [f.old_index for f in back_faces]
    front_faces_old = [f.old_index for f in front_faces]
    # This is equal to order with filter and without the front+back
    reduced_order = [f.old_index for f in normal_faces]
    return (reduced,
            back_faces_old,
            reduced_order,
            front_faces_old)


def AfterSorter(mesh, ordered, opt_level=1):
    """
    kOpt post optimization
    """
    # This is the matrix we want to optimize
    # Remove faces that are in the front and back from the list
    reduced_mat, back, reduced_order, front = getReducedLookUpMatrix(mesh)
    #print("Lookup Reduced", stack_both_sides(reduced_mat, [f.old_index for f in faces]).astype(int), sep="\n")
    # Reduced cost mat
    
    #print("Real cost", np.sum(np.triu(mat)), "Reduced mat cost", np.sum(np.triu(reduced_mat)))
    #print("Initial Cost mat\n", stack_both_sides(cost_mat, [f.old_index for f in faces]).astype(int))
    

    
    #k33 = lambda: k3(mat, mesh, ordered)
    #k22 = lambda: k2(mat, mesh, ordered)
    #from timeit import timeit
    #foo2 = lambda: fo2(mat, mesh, ordered)
    #foo3 = lambda: fo3(mat, mesh, ordered)
    #print(timeit(foo2), number=2)
    #print(timeit(foo3), number=2)
    #raise    
    
    # reduced_order is changed while keeping reference!    
    stop, reduced_mat = kOpt(reduced_mat, mesh, reduced_order)
        
    # Update Matrix:
    ordered = back + reduced_order + front
    ExportFace.updateIndices(mesh, ordered)
    ExportFace.updateMatrix(mesh, ordered)
    #print("End cost after order", np.sum(np.triu(ExportFace.LookUpMatrix[mesh])))
    return ordered

# ========================================================================================
def SortByNormals(bm, divisions=4, mode='ZDist', sphere_hat=True, opt_type=6, opt_level=1):
    """
    Returns
    """
    faces = ExportFace.buildFaces(bm, vis_check=opt_type >= 3)
    np.set_printoptions(threshold=5000, linewidth=60000)
    print("Initial LookupMatrix \n", stack_both_sides(ExportFace.LookUpMatrix[bm], [f.index for f in faces]).astype(int))
    if opt_type != 3:
        #
        results = normal_filter2(bm, divisions=divisions)
        ordered = order_normal_space(faces, *results, mode='ZDist', sphere_hat=sphere_hat)
    # Groups currently holds bmesh faces
    # reordering them will kill them, storing only face numbers
        groups = results[0]
        ExportFace.updateIndices(bm, ordered)
        ExportFace.updateMatrix(bm, ordered)
    else:
        groups = None
        ordered = [f.old_index for f in faces]
    # Need to run RefreshGroups after reordering
    np.save("HighEndPresorted2.npy", ExportFace.LookUpMatrix[bm])
    #ordered2 = ordered.copy()
    #print("order before", ordered)
    #print("Initial cost before ZDist", np.sum(np.triu(ExportFace.LookUpMatrix[bm])))

    #print("New cost after ZDist", np.sum(np.triu(ExportFace.LookUpMatrix[bm])))
    #print("Lookup Matrix ZDIST\n", stack_both_sides(ExportFace.LookUpMatrix[bm], [f.old_index for f in faces]).astype(int))
        
    if opt_type >= 6:
        tic = time.perf_counter()
        # order is taken into account via indices and the Matrix
        ordered = AfterSorter(bm, opt_level)
        toc = time.perf_counter()
        needed = toc-tic
        print("After Opt Time:", f"{needed:0.4f}")
        WholeOptTimes.append((len(faces),needed))
        
    #print("Lookup Matrix After Opt\n",stack_both_sides(ExportFace.LookUpMatrix[bm], ordered))
    #print(np.vstack(([f.old_index for f in faces], np.sum(np.triu(ExportFace.LookUpMatrix[bm]), axis=1))))

    #print("order after", ordered)
    return ordered, groups

# ==================================================================================
def stack_both_sides(mat, idx):
    """
    For testing, attach indices to left and top of matrix
    """
    disp = np.column_stack((idx, mat)) 
    # [-1] upper left corner no info
    disp = np.row_stack(([-1] + idx, disp))
    return disp

def create_facemap(obj, fmap_name):
    """
    Create a face map for overview
    """
    if fmap_name in obj.face_maps:
        fmap = obj.face_maps[fmap_name]
    else:
        fmap = obj.face_maps.new(name=fmap_name)
    # Remove faces if present
    return fmap


def UseInBlender():
    """
    For standalone in Blender sorting without Nodes
    """
    opt_level = 1
    
    obj = bpy.context.object
    bpy.ops.object.mode_set(mode='OBJECT')
    me = obj.data
    bm = bmesh.new()
    bm.from_object(obj, bpy.context.evaluated_depsgraph_get())
    print("Mesh is", bm)
    ordered, groups = SortByNormals(bm, opt_type=6, opt_level=opt_level)

    if True:
        try:
            from . import _globals
            _globals.ExportFace  = ExportFace   
            from . import Nodes
        except ImportError:
            from DarkBinImportExport.testing import _globals
            _globals.ExportFace   = ExportFace   
            from DarkBinImportExport.testing import Nodes
        Nodes.ExportFace = ExportFace
        from importlib import reload
        reload(Nodes)
        print("reordering")
        from DarkBinImportExport.io_scene_dark_bin import reorder_faces
        
        do_reorder = True
        if do_reorder:
            def reorder_faces2(bm, order):
                bm.faces.ensure_lookup_table()
                faces = bm.faces
                for new, old in enumerate(order):
                    faces[old].index = new
                bm.faces.sort()
                bm.faces.ensure_lookup_table()
            print("Mesh is", bm)
            reorder_faces2(bm, ordered)
            ExportFace.refresh_faces(bm, ordered) # Needs to be called, bmesh data invalid after reorder.
            bm.to_mesh(me)
            obj.data.update()
        
        if opt_level:
            NodeMaker = Nodes.NodeMaker
            Nodes.ExportFace = ExportFace
            print("Mesh is", ExportFace.activeMesh)
            MainNode, new_order = NodeMaker(bm, sphere_hat=True)
            
            #reorder_faces(bm, new_order)
            #ExportFace.refresh_faces(bm, new_order) # Needs to be called, bmesh data invalid after reorder.
            for f in MainNode.faces:
                bm.faces[f.index].select=True
            
            ordered = new_order
            reorder_faces(bm, ordered)
            ExportFace.refresh_faces(bm, ordered)
            bm.to_mesh(me)
            obj.data.update()
            
            # Create Face maps
            obj.face_maps.clear()
            
            fmap = create_facemap(obj, "MainNode")
            fmap.add([f.index for f  in MainNode.faces])
            fmap = create_facemap(obj, "FrontNode")
            fmap.add([f.index for f  in MainNode.node_front.faces])
            fmap = create_facemap(obj, "BehindNode")
            fmap.add([f.index for f  in MainNode.node_behind.faces])
            counter=0
            for node in MainNode:
                counter+=1
                if node == MainNode:# or node == MainNode.node_behind or node == MainNode.node_front:
                    continue
                print("\n", "="*30,"\n")
                print("conf node", node)
                print(node.position)
                if type(node) == Nodes.SplitNode:
                    if not node.index:
                        node.index = counter
                    if not node.position:
                        node.position ="NA"
                    fmap = create_facemap(obj, str(node.index)+"_"+node.position+"Node_behind")
                    fmap.add([f.index for f  in node.node_behind.faces])
                    fmap = create_facemap(obj, str(node.index)+"_"+node.position+"Node_front")
                    fmap.add([f.index for f  in node.node_front.faces])
            print("Num Nodes:", counter)
            
            
                    
            
            
    bpy.ops.object.mode_set(mode='EDIT')

#groups = {idx:[] for idx in range(divisions ** 2)}

if __name__ == "__main__":
    UseInBlender()

