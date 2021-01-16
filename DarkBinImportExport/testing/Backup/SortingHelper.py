import bpy
import bmesh
import numpy as np
from struct import pack, unpack, calcsize
from mathutils import Vector
from math import sin, cos, asin, acos, atan2, degrees, radians

from operator import itemgetter, attrgetter
from itertools import product as product_iter, chain, permutations, compress

# (dezimal) percentage that the opt algorithm expects before aborting
REL_OPT_IMPROVEMENT = 0.08 
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
            mesh = activeMesh
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
    
    @classmethod
    def doVisibilityChecks(cls, mesh):
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
        cls._InitialLookUpMatrix[mesh] = mat.copy()
    
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
    def updateIndices(cls, mesh, order):
        faces = cls.LookUpFaces[mesh]
        for i, idx in enumerate(order):
            faces[idx].index = i
        faces.sort(key=attrgetter('index'))         
    
    @classmethod
    def updateMatrix(cls, mesh, order, _reset=False):
        # Order describes how the old -> new face numbers have to be aranged.
        # Need to fall back to the initial lookup matrix
        mat = cls._InitialLookUpMatrix[mesh]
        # And create new one
        new_mat = cls.LookUpMatrix[mesh] = np.ndarray(mat.shape, dtype=bool)
        for new, old in enumerate(order):
            # Update n-th row and column
            new_mat[new] = mat[old][order]
            new_mat[:,new] = mat[:,old][order]
    
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
    def getConflicts(cls, mesh, filter=3):
        """
        Faces that see one with higher indices
        filter bitwise
        1: sees higher index
        2: seen by lower index
        3: combined
        """
        assert (filter < 4 and filter > 0), "Filter must be between 1 and 3."
        faces = cls.LookUpFaces[mesh]
        mat = cls.LookUpMatrix[mesh]
        reduced = np.triu(mat)
        conflict = np.zeros(mat.shape[0], dtype=bool)
        if filter & 1:
            conflict = reduced.any(axis=0)
        if filter & 2:
            conflict |= reduced.any(axis=1)
        
        print("Conflict FACES")
        print("Can see", [b.index for a, b in zip(reduced.any(axis=0), faces) if a])
        print("Is Seen", [b.index for a, b in zip(reduced.any(axis=1), faces) if a])
        
        return [b for a, b in zip(conflict, faces) if a]
    
    @classmethod
    def getConflictsForNode(cls, node):
        mesh = node.mesh
        node_face_indices = node.faces
        node_faces = [cls.LookUpFaces[mesh][i] for i in node_face_indices]
        
        node_mat = cls.LookUpMatrix[mesh][np.ix_(node_face_indices, node_face_indices)]
        reduced = np.triu(node_mat)
        conflict = reduced.any(axis=0)
        print("Conflicts can see for Node:", conflict)
        return [b for a, b in zip(conflict, node_faces) if a]
    
    # Bool / See / Seen functions
    
    @classmethod
    def getFacesNotSeen(cls, mesh):
        return [f for f in cls.LookUpFaces[mesh] if not f.is_seen()]
        # Cache option
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
    
    def is_seen(self):
        return bool(self.seen_by)
    
    def can_see(self):
        return bool(self.sees)
    
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
        self.normal = face.normal
        self.vert_pos = [v.co for v in face.verts]
        self.sees = []
        self.seen_by = []
        ExportFace.LookUpFaces.setdefault(mesh, []).append(self)
        
        
    def __getattr__(self, key):
        return getattr(self.bmesh_face, key)

 

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

def SimplifyGroups(groups, ordered):
    """
    Needs to run before making a mesh invalid
    """
    for group in groups.values():
        for i, f in enumerate(group):
            group[i] = ordered.index(f.index) # New face number from old.

def RestoreGroups(groups, mesh):
    """
    Needs to run after making mesh invalid
    """
    for group in groups.values():
        for i, f in enumerate(group):
            group[i] = mesh.faces[group[i]]

# ========================================================================================
# Nodes
# ========================================================================================

class NodeIterator:
    """
    For itterating the Node class objects
    """
    def __init__(self, node, conflict_nodes=False):
        """
        Goes self, 
        """
        self._node = node
        # member variable to keep track of current index
        self._index = 0
        if type(node) == SplitNode:
            self._checknodes = [node, node.node_behind, node.node_front]
        else:
            self._checknodes = [node]
        self.conflict_nodes = conflict_nodes
                
             
    def __next__(self):
        if self._index == len(self._checknodes):
            # End of Iteration
            raise StopIteration
        next = self._checknodes[self._index]
        if isinstance(next, SplitNode) and next != self._node:
            self._checknodes.insert(self._index+1, next.node_behind)
            self._checknodes.insert(self._index+2, next.node_front)
        self._index += 1
        if self.conflict_nodes:
            if next.hasConflicts():
                return next
            else:
                return self.__next__()    
        return next
# ========================================================================================

class Node(object):
    type = 0
    size = '<BffffH'
    
    def hasConflicts(self, dump=False):
        mat = ExportFace.LookUpMatrix[self.mesh]
        # The upper triangle matrix (lower is 0)
        # Which represents faces that can see ones with higher indices
        # From there slice the rows and cols represeting the faces
        node_conflicts = np.triu(mat)[np.ix_(self.faces, self.faces)]
        if dump:
            np.set_printoptions(threshold=5000, linewidth=60000)
            disp = stack_both_sides(node_conflicts, self.faces)
            print("Conflicts in Node", self.faces,"\n", disp.astype(int))
            
        if node_conflicts.any():
            # No conflict, no further actions necessary
            return True
        # Now option to try to reorder faces again or split again
        return False
    
    def __init__(self, faces, parent, mesh=None, sphere=None, position=None):
        self.parent_node = parent
        self.sphere = sphere
        self.position = position
        
        if faces:
            # Faces as index not BMesh.Faces
            if (type(faces[0]) != int):
                faces = [f.index for f in faces]
            self.faces = faces
        else:
            self.faces = []

        if mesh:
            self.mesh = mesh
        else:
            self.mesh = parent.mesh # So main needs to be initilaized with it at least.
    
    
    def transform(self, n_first_faces, split_position=None, node_front=None, node_behind=None):
        new = SplitNode(self.faces, n_first_faces, 
                        parent=self.parent_node, 
                        sphere=self.sphere, 
                        node_front=None, node_behind=None,
                        mesh=self.mesh)
        if not split_position:
            if self.parent_node.node_behind == self:
                self.parent_node.node_behind = new
            else:
                self.parent_node.node_front = new
        else:
            if split_position == 'front':
                self.parent.node_front = new
            else:
                self.parent.node_behind = new
        return new
     
       
    def calcsize(self):
        return calcsize(self.size) + len(self.faces)*2
    
    _ignore_keys = ['position']
    
    def check(self, generation=0):
        """
        For debugging to check if all Nodes are set.
        """
        rv = True
        numfaces = len(self.faces) if type(self) != SubobjectNode else 0
        ##DEBUG
        #print(self, "having", numfaces, "faces")
        for k, v in self.__dict__.items():
            if v == None and k not in self.__class__._ignore_keys:
                print(k, "is not set on", self, "Generation:", generation)
                rv = False
            if k != "parent_node" and isinstance(v, Node):
                result = v.check(generation - 1)
                n_childfaces = v.numfaces
                if not result:
                    rv = False
                numfaces += n_childfaces
        self.numfaces = numfaces
        return rv
 
    def makesphere(self):
        try:
            from ..io_scene_dark_bin import get_local_bbox_data, encode_sphere
        except ImportError:
            from DarkBinImportExport.io_scene_dark_bin import get_local_bbox_data, encode_sphere
        if "sphere" in self.__dict__ and self.sphere != None:
            return self.sphere
        # Makesphere
        faces = self.mesh.faces
        # Self.faces are only ints.
        self.verts = [v for idx in self.faces for v in faces[idx].verts]
        if len(self.verts) == 0:
            self.sphere = pack('<ffff', 0.0, 0.0, 0.0 ,0.0)
        else:
            bbox = get_local_bbox_data(self)
            self.sphere = encode_sphere(bbox)
        return self.sphere
 
    def pack(self):
        return pack('<B', self.type) \
                    + self.makesphere() \
                    + pack('<H',len(self.faces))\
                    + b''.join(self.faces)
    
    def packfaces(self):
        #TODO
        pass
    
    def getConflicts(self):
        rv = []
        iter = NodeIterator(self, conflict_nodes=True)
        while True:
            try:
                rv.append(next(iter))
            except StopIteration:
                break
        return rv
    
    def __iter__(self):
        return NodeIterator(self)



class SubobjectNode(Node):
    type = 4
    size = '<BH'
    def __init__(self, subobj_idx, parent):
        self.index = subobj_idx
        self.parent_node = parent

    def calcsize(self):
        return 0    # This is not positioned in the tree.
    
    def pack(self):
        return b'' # Also not packing into tree

    
class SplitNode(Node):
    type = 1
    size = '<BffffHHfHHH'
    do_split = False

    def __init__(self, faces, n_first_faces, parent, mesh=None, node_front=None, node_behind=None,
                    split_face=None, d=None, sphere=None):
        assert n_first_faces <= len(faces)
        
        super().__init__(faces, parent, mesh=mesh, sphere=sphere)
        
        self.n_first_faces = n_first_faces
        self.node_front = node_front
        self.node_behind = node_behind
        self.d = d
        if split_face:
            makenorm(split_face)
        else:
            self.norm = None
        
    def makesimple():
        return Node(self.faces, self.parent)

    def makenorm(self, f, faces):
        # Relies on nemyax encoded normal idx
        ext_f = faces.layers.string.verify()
        self.norm = unpack('<H', f[ext_f][:2])
     
    def pack(self):
        assert self.check()
        if type(self.norm)!= bytes:
            self.norm = pack('<H', self.norm)
        if type(self.d)!= bytes:
        # NOTE Split planes seam to be packed with -d
        # And/or probably later also work with the normal reversed
            self.d = pack('<f', -self.d)
        return (pack('<B', self.type)
                + self.makesphere()
                + pack('<H', self.n_first_faces)
                + self.norm
                + self.d
                + pack('<HHH', self.node_behind_off, 
                               self.node_front_off, 
                               len(self.faces) - self.n_first_faces)
                + b''.join(self.faces))
      
# //////////////////////////////////////////////////////////

def PolarAngles(v):
    r = 1 
    # Yay atan2 loves -0.0
    if v.x == -0.0:
        v.x = 0.0
    # Can get rid of round and int when in range() method gets replaced by a float variant.
    # NOTE: Theta the co-angle from the X-Axis [-90,90]
    return int(round(degrees(asin(v.z / r)))), int(round(degrees(atan2(v.y, v.x))))

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
    in other groups alos have that distance.
    """
    done = False
    group_list = []
    value_list = []
    for k,v in groups.items():
        if k in local_ZDists:
            group_list.append(v.copy())
            value_list.append(local_ZDists[k].copy())
    ordered = []  
    while not done:
        done = True
        chosen_group_idx = None
        n_min = 1000
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
            next_face.index = len(ordered) - 1 # New index
            value_list[chosen_group_idx].pop(0)
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


def normal_filter2(mesh, divisions=4, sphere_hat=True):
    """
    This sorts all faces with similar normals into division² groups.
    Each spherical angle [-180, 180]° around Z-Axis (phi), 
    and [-90, 90]° around X-Axis (theta) is divided into n segments.
    
    Theta divisions are always uneven.
    
    With Sphere Hat instead of having multiple sides at the top and bottom
    will combine them to a single Z and -Z Axis.
    """
    division_angles, t_step, p_step, t_divisions = _make_division_angles(divisions, sphere_hat)

    faces = ExportFace.allFacesInMesh(mesh)
    
    groups = { (p, t):[] for t in range(t_divisions) for p in range(divisions) }
    
    # Now sort all faces into these groups
    for f in faces:
        n = f.normal
        theta, phi = PolarAngles(n)
        ##print("="*20+"\n",f.index, "\n", theta, phi)
        ##print(f.normal)
        
        t_idx = (theta+90) // t_step
        p_idx = (phi+180) // p_step
        
        if p_idx == divisions:      # As first and last are the same group just halfed.
            p_idx = 0  
        if t_idx == t_divisions:    # Would give one idx to high
            t_idx = t_divisions - 1
        
        if sphere_hat and (t_idx == t_divisions - 1 or t_idx == 0):
            # Don't separate by p_idx for highest/lowst theta angles
            p_idx = 0
        # Store them in their group and also in a lookup table
        groups[(p_idx, t_idx)].append(f.index)
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

def order_normal_space(faces, groups, group_lookup, division_angles, sphere_hat=True, mode='ZDist', prepare_nodes=True):
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
        face_indices = groups[gr_key]
        if len(face_indices) == 0:
            continue
        similar_faces = [faces[i] for i in face_indices]
        
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
            Z_Dists.append((f.index, n_scale))
            f.ZDist = n_scale
            return n_scale
        # This is to make the sorting more stable for equivalent faces
        first = [pos_sort(f) for f in similar_faces]
        second = [f.calc_center_median().length for f in similar_faces]
        sorter = np.lexsort((second, first))
        similar_faces = [similar_faces[i] for i in sorter]
        
        groups[gr_key] = similar_faces
        local_ZDists[gr_key].sort() # small n to great
    
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


def NodeMaker(mesh, split_node=None, sphere_hat=True):
    """
    render_after and node_front belong together
    """
    ## Try to create nodes ##
    if split_node == None:
        faces = ExportFace.allFacesInMesh(mesh).copy()
    else:
        # Select only waht's in node
        faces = [ExportFace.allFacesInMesh(mesh)[i] for i in split_node.faces]
        # Test
        print("Test second run\n", split_node.faces)

    #split_groups = groups # Wanted to copy but .copy is only first level.
    back = []   # Rendered first
    
    # TODO: Where can this fuck up?
    # For second round need to remove non present
    
    if split_node:
        conflict_faces = ExportFace.getConflictsForNode(split_node)
        # Need to add those who now are not conflicts but can be seen
        # by ones after split.
    else:
        # First run
        conflict_faces = ExportFace.getConflicts(mesh)
        
        conflict_faces.sort(key=lambda f: f.index)
    conflict_faces_ids = [f.index for f in conflict_faces]
    
    print("§CONFLICTS", [f.index for f in conflict_faces], "\n max", max(conflict_faces_ids))
    
    # This should be equal to f is not seen
    # new_front = [f.index for f in faces if (f not in conflict_faces)]

    # Maybe for the future use this one, are they the same?
    # Well they can be seen if no conflict.
    # This is not filtered after a split!
    new_front_faces = [f for f in faces if not f.is_seen()]
    new_front2 = [f.index for f in new_front_faces]
    
    new_front = []
    new_back = []
    checker = ExportFace.LookUpMatrix[mesh][conflict_faces_ids]
    print(checker.shape)
    for f in faces.copy():
        if f.index < conflict_faces[0].index and not checker[:,f.index].any():
            new_front.append(f.index)
            faces.remove(f)
        elif f.index > conflict_faces[-1].index:
            new_back.append(f.index)
            faces.remove(f)
    
    print("Algorithms match\n", new_front2,"\n", new_front,"\n", new_back)
    
    front = new_front
    # Removed faces are not in groups anymore, adding again below

    n_first_faces = len(front)
    front.extend(new_back)
    # most_front.extend(front) # that makes no sense at the moment.
    # Create node objects
    if split_node:
        MainNode = split_node.transform(n_first_faces)
        num_all = len(split_node.faces)
        MainNode.faces.clear()
        MainNode.faces = front
        print("Second frontr", front)
        # TODO: Call nodes!
    else:
        MainNode = SplitNode(front, n_first_faces, parent=-1, mesh=mesh)
    BackNode = MainNode.node_behind = Node(faces=None, parent=MainNode, position='behind')
    FrontNode = MainNode.node_front = Node(faces=None, parent=MainNode, position='front')
    # Now need to fill these two and determine some split
    # Assuming we are sphere hat here but should work else as well
    
    # DEBUG
    d = {}
    for f in faces:
        d.setdefault(f.group, []).append(f.index)
    print(d)
    
    # DETERMINE SPLIT FACE
    print("="*30, "\nFind Split")
    

    # Maybe conflict_faces[-1]?

    # Choosing a good split plane is important!
    # Maybe need to do more here - for this model it fits.
    split_plane = conflict_faces[-1]
    split_plane_idx = split_plane.index
    MainNode.split_face = split_plane.index
    
    print("SplitPlane\n",split_plane_idx)
    split_d = split_plane.calc_center_bounds() @ split_plane.normal
    BackNode.faces.append(split_plane_idx)
    faces.remove(split_plane)
    for face in faces:  # This group biased again.
        n = (face.calc_center_bounds() @ split_plane.normal) - split_d
        n = round(n, 5) # To make up some numerical unevenness close to 0.0.
        #print(face.index, n)
        # This i done the other way, if in front of split plane they are behind
        if  n >= 0:
            FrontNode.faces.append(face.index)
            print(face.index," is in front of", split_plane_idx)
        else:
            BackNode.faces.insert(0, face.index)

    BackNode.faces.sort()
    FrontNode.faces.sort()
    #FrontNode.faces.extend([f.index for f in front_faces])
    # Is filtered internally from all faces.
    MainNode.makesphere()
    BackNode.makesphere()
    FrontNode.makesphere()
 
    print("Main Faces:\n", MainNode.faces)
    print("Front Node:\n", FrontNode.faces)
    print("Back Node:\n", BackNode.faces)
    
    print("Conflicts", MainNode.hasConflicts(), FrontNode.hasConflicts(), BackNode.hasConflicts(True))
    
    # Makes sure all nodes in tree are set correctly
    valid = MainNode.check()
    num_faces = MainNode.numfaces
    if not split_node:
        num_all = len(mesh.faces)
    print("Nodes are ok?:", valid, f"\nNum Faces: {num_faces}/{num_all} ok?:", num_faces==num_all)
    
    return MainNode

# ========================================================================================
# KOpt Post optimization
# ========================================================================================

def makeCurrentCost(mesh, all=False):
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
    print("Costs", costs)
    return costs 

def makeAccessMat(mat):
    rows = [m[i+1:] for i,m in enumerate(mat)]
    cols = [m[:i] for i,m in enumerate(mat.T)]
    return [rows, cols]

def fo4(mat, mesh, order, access):
    start_cost = sum(a.sum() for a in access[0])
    l = len(order)
    cost = start_cost        
    for i in range(l):
        cur_i_cost = sum(access[0][i].tolist()) + sum(access[1][i].tolist())
        for j in range(l):
            if i < j:
                h = i
                g = j
            elif i == j:
                continue
            else:
                g = i
                h = j
            # The intersection is counted twice need to remove.
            cur_j_cost = sum(access[0][j].tolist()) + sum(access[1][j].tolist())
            old_cost = cur_i_cost + cur_j_cost - access[1][g][h]
            # These are bit faster than the internal interesting.
            new_cost_j_int = sum(mat[j,i:].tolist())
            new_cost_j_ext = sum(mat[:i,j].tolist())
            new_cost_i_int = sum(mat[i,j:].tolist()) 
            new_cost_i_ext = sum(mat[:j,i].tolist())
            #new_cost_j_int = mat[j,i:].sum()
            #new_cost_j_ext = mat[:i,j].sum()
            #new_cost_i_int = mat[i,j:].sum() 
            #new_cost_i_ext = mat[:j,i].sum()
            new_cost = new_cost_i_ext+new_cost_i_int+new_cost_j_ext+new_cost_j_int-access[1][g][h]
            #mat[:, i], mat[:, j] = mat[:, j], mat[:, i].copy()
            #mat[i], mat[j] = mat[j], mat[i].copy()
            #new_cost = access[0][i].sum() + access[1][i].sum() + access[0][j].sum() + access[1][j].sum()-access[1][g][h]
  
            if new_cost < old_cost:
                cost -= (old_cost-new_cost)
                order[i], order[j] = order[j], order[i]
                mat[:, i], mat[:, j] = mat[:, j], mat[:, i].copy()
                mat[i], mat[j] = mat[j], mat[i].copy()
                cur_i_cost = sum(access[0][i].tolist()) + sum(access[1][i].tolist())
 
    print("End cost", cost)
    # More efficient:
    return start_cost/cost < 1.0 + REL_OPT_IMPROVEMENT, mat
    return start_cost == cost, mat

def fo5(mat, mesh, order, access):
    """
    Similar to steepest gradient method
    """
    start_cost = sum(a.sum() for a in access[0])
    l = len(order)
    cost = start_cost
    itered = [True]*l
    for i in compress(range(l), itered):
        cur_i_cost = sum(access[0][i].tolist()) + sum(access[1][i].tolist())
        # Look for best improvement
        cost_change = 0
        choosen_j = None
        for j in range(l):
            if i < j:
                h = i
                g = j
            elif i == j:
                continue
            else:
                g = i
                h = j
            # The intersection is counted twice need to remove.
            cur_j_cost = sum(access[0][j].tolist()) + sum(access[1][j].tolist())
            old_cost = cur_i_cost + cur_j_cost - access[1][g][h]
            # These are bit faster than the internal interesting.
            new_cost_j_int = sum(mat[j,i:].tolist())
            new_cost_j_ext = sum(mat[:i,j].tolist())
            new_cost_i_int = sum(mat[i,j:].tolist()) 
            new_cost_i_ext = sum(mat[:j,i].tolist())
            #new_cost_j_int = mat[j,i:].sum()
            #new_cost_j_ext = mat[:i,j].sum()
            #new_cost_i_int = mat[i,j:].sum() 
            #new_cost_i_ext = mat[:j,i].sum()
            new_cost = new_cost_i_ext+new_cost_i_int+new_cost_j_ext+new_cost_j_int-access[1][g][h]
            if new_cost - old_cost < cost_change:
                cost_change = new_cost - old_cost
                choosen_j = j
             
        if choosen_j != None:
            cost += cost_change
            j = choosen_j
            order[i], order[j] = order[j], order[i]
            mat[:, i], mat[:, j] = mat[:, j], mat[:, i].copy()
            mat[i], mat[j] = mat[j], mat[i].copy()
            cur_i_cost = sum(access[0][i].tolist()) + sum(access[1][i].tolist())
            itered[j] = False
 
    print("End cost", cost)
    # More efficient:
    return start_cost/cost < 1.0 + REL_OPT_IMPROVEMENT, mat
    return start_cost == cost, mat


class _compresslist(list):
    def __contains__(self,key):
        print("trying to get", key)
        return key[0] in self

def _compressgen(iterator, list):
      for n in iterator:
          if n[0] not in list:
              yield n

def fo6(mat, mesh, order, access):
    """
    Similar to steepest gradient method
    """
    start_cost = sum(a.sum() for a in access[0])
    l = len(order)
    cost = start_cost
    itered = []
    cur_i_cost = sum(access[0][0].tolist()) + sum(access[1][0].tolist())
    cost_change = 0
    choosen_j = None
    compressor = []
    for i,j in permutations(range(l),2):
        if i in compressor:
            continue
        if j == 0:
            cur_i_cost = sum(access[0][i].tolist()) + sum(access[1][i].tolist())
            # Look for best improvement
            cost_change = 0
            choosen_j = None
            g = i
            h = 0
        if i < j:
            h = i
            g = j
        else:
            h = j
        # The intersection is counted twice need to remove.
        cur_j_cost = sum(access[0][j].tolist()) + sum(access[1][j].tolist())
        old_cost = cur_i_cost + cur_j_cost - access[1][g][h]
        # These are bit faster than the internal interesting.
        new_cost_j_int = sum(mat[j,i:].tolist())
        new_cost_j_ext = sum(mat[:i,j].tolist())
        new_cost_i_int = sum(mat[i,j:].tolist()) 
        new_cost_i_ext = sum(mat[:j,i].tolist())
        #new_cost_j_int = mat[j,i:].sum()
        #new_cost_j_ext = mat[:i,j].sum()
        #new_cost_i_int = mat[i,j:].sum() 
        #new_cost_i_ext = mat[:j,i].sum()
        new_cost = new_cost_i_ext+new_cost_i_int+new_cost_j_ext+new_cost_j_int-access[1][g][h]
        if new_cost - old_cost < cost_change:
            cost_change = new_cost - old_cost
            choosen_j = j
            
        if j == l-1 and choosen_j != None:
            cost += cost_change
            j = choosen_j
            order[i], order[j] = order[j], order[i]
            mat[:, i], mat[:, j] = mat[:, j], mat[:, i].copy()
            mat[i], mat[j] = mat[j], mat[i].copy()
            cur_i_cost = sum(access[0][i].tolist()) + sum(access[1][i].tolist())
            compressor.append(j)
 
    print("End cost", cost)
    # More efficient:
    return start_cost/cost < 1.0 + REL_OPT_IMPROVEMENT, mat
    return start_cost == cost, mat


def kOpt(mat, mesh, order, access=None):
    """
    Post Optimization.
    Can be run multiple times for further improvement.
    
    Enabling deep_check is NOT advised it takes A TON
    of time O(n!). 
    Acceptable for small < 100 polygon models.
    """
    return fo5(mat, mesh, order, access)
    real_cost = np.sum(np.triu(mat))
    start_cost = real_cost
    working_mat = mat
    # Carefull order?
    #cost_lookup = makeCurrentCost(mesh, all=False)
    l = len(order)
    if False:
        for i in range(l):
            for j in range(l):
                if i == j:
                    continue
                old_cost = cost_lookup[i][0] + cost_lookup[j][0]
                new_cost_j_int = np.sum(working_mat[j,i:])
                new_cost_j_ext = np.sum(working_mat[:i,j])
                new_cost_i_int = np.sum(working_mat[i,j:]) 
                new_cost_i_ext = np.sum(working_mat[:j,i])
                if (new_cost_j_int + new_cost_j_ext + new_cost_i_ext + new_cost_i_int ) < old_cost:
                    print("="*4)
                    normal = (sum(working_mat[i,i:]), sum(working_mat[:i,i]), sum(working_mat[j,j:]), sum(working_mat[:j,j]))
                    sum_normal = sum(normal)
                    new=(sum(working_mat[j,i:]), sum(working_mat[:i,j]), sum(working_mat[i,j:]), sum(working_mat[:j,i]))
                    sum_new = sum(new)
                    print(sum_normal,"=",normal, "\nNew:", sum_new, "=", new)
                    print(old_cost, " vs ", new_cost_j_int + new_cost_j_ext,"+", new_cost_i_ext + new_cost_i_int, "=", new_cost_j_int + new_cost_i_ext + new_cost_i_int + new_cost_j_ext)
                    # TODO: Do not substract external costs
                    real_cost -= (old_cost - (new_cost_j_int+new_cost_i_int))
                    # These need to be updated, now invalid indices
                    cost_lookup[i][0] = new_cost_j_int + new_cost_j_ext
                    cost_lookup[i][1] = new_cost_j_int
                    cost_lookup[j][0] = new_cost_i_int + new_cost_i_ext
                    cost_lookup[j][1] = new_cost_i_int
                    working_mat[i], working_mat[j] = working_mat[j].copy(), working_mat[i].copy()
                    working_mat[:, i], working_mat[:, j] = working_mat[:, j].copy(), working_mat[:, i].copy()
                    order[i], order[j] = order[j], order[i]
                    
                    new_real=sum(np.triu(working_mat))
                    print("math correct", real_cost==new_real)
                    print("Reducing Errors", start_cost, "->", real_cost, "should be", new_real, end="  \r")
            raise
        if start_cost != real_cost:
            print("\nEnd mat\n", stack_both_sides(working_mat, order))
            return False,  working_mat
        return start_cost == cost, working_mat
    cost = start_cost        
    # Good result on performance for low poly objects.
    for i in range(mat.shape[0]):
        for j in range(mat.shape[0]):
            if i == j:
                continue
            j_col = working_mat[:, j]
            i_col = working_mat[:, i]
            i_row = working_mat[i]
            j_row = working_mat[j]
            new_cost = np.sum(j_row[j:]) + np.sum(i_row[i:]) + np.sum(j_col[:j]) + np.sum(i_col[:j])
            if new_cost < cost:
                cost = new_cost
                working_mat[:, i], working_mat[:, j] = j_col, i_col.copy()
                working_mat[i], working_mat[j] = j_row, i_row.copy()
                order[i], order[j] = order[j], order[i]
        #print("Updated real cost", cost, sum(np.triu(working_mat)),"\n")   
        print("Improved Error", i,"/", mat.shape[0]-1, end='\r') 
    print("\nErrors minimized.", cost)
    
    print("End Cost mat\n", stack_both_sides(working_mat, order).astype(int))
    return start_cost == cost, working_mat

def makeCostMatrix(mat):
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
 
def reduceLookUpMatrix(mesh, order):
    # Insert at start if there are any
    back_faces = ExportFace.getFacesNotSeeing(mesh)
    back_faces_new = [f.index for f in back_faces]
    # Insert at end
    front_faces = ExportFace.getFacesNotSeen(mesh)
    front_faces_new = [f.index for f in front_faces]
    # Remove these rows and columns from the lookup mat
    both = back_faces_new + front_faces_new
    reduced = np.delete(np.delete(ExportFace.LookUpMatrix[mesh], both, axis=1), both, axis=0)
    
    # Need these to filter order. Original indices
    back_faces_old = [f.old_index for f in back_faces]
    front_faces_old = [f.old_index for f in front_faces]
    return (reduced,
            back_faces_old,
            [idx for idx in order if idx not in front_faces_old+back_faces_old],
            front_faces_old)

def AfterSorter(mesh, ordered, opt_level=1):
    """
    kOpt post optimization
    """
    # This is the matrix we want to optimize
    mat = ExportFace.LookUpMatrix[mesh]
    ordered_before = ordered.copy()
    # Remove faces that are in the front and back from the list
    reduced_mat, back, reduced_order, front = reduceLookUpMatrix(mesh, ordered)
    faces = ExportFace.getFacesNormal(mesh)
    print("Lookupnew", stack_both_sides(reduced_mat, [f.old_index for f in faces]).astype(int), sep="\n")
    # Reduced cost mat
    cost_mat = makeCostMatrix(reduced_mat)

    print("Real cost", np.sum(np.triu(mat)), "Reduced mat cost", np.sum(np.triu(reduced_mat)))
    print("Initial Cost mat\n", stack_both_sides(cost_mat, [f.old_index for f in faces]).astype(int))
    
    cost = np.sum(np.diag(cost_mat))
    counter = 0
    
    T = makeAccessMat(reduced_mat)
    #k33 = lambda: k3(mat, mesh, ordered)
    #k22 = lambda: k2(mat, mesh, ordered)
    #from timeit import timeit
    #foo2 = lambda: fo2(mat, mesh, ordered)
    #foo3 = lambda: fo3(mat, mesh, ordered)
    #print(timeit(foo2), number=2)
    #print(timeit(foo3), number=2)
    #raise
    while counter < 10:
        counter += 1
        
        stop, reduced_mat = kOpt(reduced_mat, mesh, reduced_order, access=T)
        
        print("kOpt:", counter)
        if stop:
            break
    
    # Update Matrix:
    ordered = back + reduced_order + front
    ExportFace.updateIndices(mesh, ordered)
    ExportFace.updateMatrix(mesh, ordered)
    return ordered

# ========================================================================================

def SortByNormals(bm, divisions=4, mode='ZDist', make_nodes=True, sphere_hat=True, opt_type=4, opt_level=1):
    """
    Returns
    """
    faces = ExportFace.buildFaces(bm, vis_check=opt_type >= 3)
    #print("Initial LookupMatrix \n", stack_both_sides(ExportFace.LookUpMatrix[bm], [f.index for f in faces]).astype(int))
    results = normal_filter2(bm, divisions=divisions)
    ordered = order_normal_space(faces, *results, mode='ZDist', prepare_nodes=make_nodes, sphere_hat=sphere_hat)
    # Groups currently holds bmesh faces
    # reordering them will kill them, storing only face numbers
    groups = results[0]
    # Need to run RefreshGroups after reordering
     
    #ordered2 = ordered.copy()
    #print("order before", ordered)
    #print("LookupMatrix Before (ZDist numbers)\n", stack_both_sides(ExportFace.LookUpMatrix[bm], [f.index for f in faces]).astype(int))
    ExportFace.updateIndices(bm, ordered)
    ExportFace.updateMatrix(bm, ordered)
    
    print("LookupMatrix\n", stack_both_sides(ExportFace.LookUpMatrix[bm], [f.old_index for f in faces]).astype(int))
    if opt_type >= 6:
        ordered = AfterSorter(bm, ordered, opt_level)
    
    print("After opt")
    print(stack_both_sides(ExportFace.LookUpMatrix[bm], ordered))
    #print("order after", ordered)
        
    return ordered, groups, NodeMaker

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


import time
if __name__ == "__main__":
    """
    For standalone in Blender sorting without Nodes
    """
    opt_level = 1
    
    obj = bpy.context.object
    bpy.ops.object.mode_set(mode='OBJECT')
    me = obj.data
    bm = bmesh.new()
    bm.from_object(obj, bpy.context.evaluated_depsgraph_get())
    tic = time.perf_counter()
    ordered, groups, NodeMaker = SortByNormals(bm, make_nodes=True, opt_type=6, opt_level=opt_level)
    toc = time.perf_counter()

    if True:
        print("reordering", f"{toc - tic:0.4f}")
        from DarkBinImportExport.io_scene_dark_bin import reorder_faces
        reorder_faces(bm, ordered)
        ExportFace.refresh_faces(bm, ordered) # Needs to be called, bmesh data invalid after reorder.

        if opt_level:
            MainNode = NodeMaker(bm, sphere_hat=True)
            for f in MainNode.faces:
                bm.faces[f].select=True
        
        bm.to_mesh(me)
        obj.data.update()
        
        if opt_level:
            # Create Face maps
            obj.face_maps.clear()
            fmap = create_facemap(obj, "MainNode")
            fmap.add([idx for idx in MainNode.faces])
            fmap = create_facemap(obj, "FrontNode")
            fmap.add([idx for idx in MainNode.node_front.faces])
            fmap = create_facemap(obj, "BehindNode")
            fmap.add([idx for idx in MainNode.node_behind.faces])
            
            for node in MainNode.getConflicts():
                print("\n", "="*30,"\n")
                print("conf node", node, node.position)
                split_node = NodeMaker(bm, node, sphere_hat=True)
                
                fmap = create_facemap(obj, node.position+"Node_behind")
                fmap.add([idx for idx in split_node.node_behind.faces])
                fmap = create_facemap(obj, node.position+"Node_front")
                fmap.add([idx for idx in split_node.node_front.faces])
                # iteration is not updated
                break
        
    bpy.ops.object.mode_set(mode='EDIT')

#groups = {idx:[] for idx in range(divisions ** 2)}

