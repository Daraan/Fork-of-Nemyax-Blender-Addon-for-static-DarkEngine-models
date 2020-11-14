import bpy
import bmesh
import numpy as np
from struct import pack, unpack, calcsize
from mathutils import Vector
from math import sin, cos, asin, acos, atan2, degrees, radians

from operator import itemgetter
from itertools import product as product_iter, chain


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
    LookUpCanSee = {}
    LookUpSeenBy = {}
    LookUpFaces  = {}       # mesh specific but maybe not complete!
    LookUpConflicts = {}
    LookUpCanSeeConflict = {}
    LookUpIsSeenConflict = {}
    # This will be better
    # TODO: clean up / integreate unnecessary dicts
    LookUpMatrix = {}
    activeMesh = None
    
    @classmethod
    def allFacesInObject(cls):     # May not be complete!
        return chain(cls.LookUpFaces.values()) # Iterable!
    
    @classmethod
    def allFacesInMesh(cls, mesh):
        return cls.LookUpFaces[mesh]
    
    @classmethod
    def buildFaces(cls, mesh):
        activeMesh = mesh
        mat = cls.LookUpMatrix[mesh] = np.zeros((len(mesh.faces), len(mesh.faces)), dtype=bool)        
        can_see = cls.LookUpCanSee[mesh] = {}
        seen_by = cls.LookUpSeenBy[mesh] = {f2:[] for f2 in mesh.faces}
        for f1 in mesh.faces:
            f_new = ExportFace(f1, mesh)  # Make new instance, auto registers
            for f2 in mesh.faces:
                if f1 == f2:
                    continue
                f1 = f_new
                visible = cls.FaceVisible(f1, f2)
                if visible:
                    mat[f1.index, f2.index] = visible
                    can_see.setdefault(f1, []).append(f2)
                    seen_by[f2].append(f1)
                    if f2.index > f1.index:
                        # If a face is seen and the front one has a higher order
                        # it will not matter but the other way need to take care
                        # Will be added ,pre than once.
                        cls.LookUpConflicts.setdefault(mesh, set()).add(f1)
                        cls.LookUpConflicts[mesh].add(f2)
        
        # Fix bmesh entriesfrom f2
        for bm_f2 in list(seen_by):
            for f in cls.LookUpFaces[mesh]:
                if f.bmesh_face == bm_f2:
                    seen_by[f] = seen_by.pop(bm_f2)
                    if bm_f2 in cls.LookUpConflicts[mesh]:
                        cls.LookUpConflicts[mesh].remove(bm_f2)
                        cls.LookUpConflicts[mesh].add(f)
                    break
            else:
                print("SHOULD NOT HAPPEN 1")    
        
        for bm_f2 in cls.LookUpConflicts[mesh]:
            if type(bm_f2) == cls:
                continue # already fixed in above loop
            for f in cls.LookUpFaces[mesh]:
                if f.bmesh_face == bm_f2:
                    cls.LookUpConflicts[mesh].remove(bm_f2)
                    cls.LookUpConflicts[mesh].add(f)
                    break
            else:
                print("SHOULD NOT HAPPEN 2") 
        return cls.LookUpFaces[mesh]
    
    @classmethod
    def get(cls, idx, mesh):
        # mesh.faces[idx] should be faster but if somehow disordered
        if type(idx) == int:
            return cls.LookUpFaces[mesh][idx]
        return np.array(cls.LookUpFaces[mesh])[idx].tolist()
    
    @classmethod
    def __getitem__(cls, idx):
        return LookUpFaces[activeMesh][idx]
    
    @classmethod
    def refresh_faces(cls, mesh):
        """ 
        Needs to run after bmesh faces become invalid
        """
        for f in cls.LookUpFaces[mesh]:
            f.refresh_face(mesh)
        cls.LookUpFaces[mesh].sort(key= lambda f: f.index)
    
    def refresh_face(self, mesh):
        self.bmesh_face = mesh.faces[self.index]
    
    @staticmethod
    def FaceVisible(f1, f2, obj=None):
        """
        Checks if a face f2 is visible through a face f1. Ignoring obstacles.
        This is object internally.
        """
        n1 = -f1.normal
        n2 = f2.normal
        if type(f1) == bpy.types.MeshPolygon:
            assert obj, "Object must be provided. If not bmesh."
            vs1 = [obj.data.vertices[idx] for idx in f1.vertices]
            vs2 = [obj.data.vertices[idx] for idx in f2.vertices]
            ps1 = np.array([v.co for v in vs1])
            ps2 = np.array([v.co for v in vs2])
        else:
            ps1 = np.array([v.co for v in f1.verts])
            ps2 = np.array([v.co for v in f2.verts])

        # d is constant, could jsut use one vertices.
        # slightly variable due to numerical fluctuations.
        d1 = n1 @ f1.calc_center_bounds()
        d2 = n2 @ f2.calc_center_bounds()
        
        # Here we could determine if a split would be good if one one vertex is visible.
        
        r1 = np.dot(ps2, n1) - d1
        r2 = np.dot(ps1, n2) - d2
        # Need to cope with numerical Xe-7 close to 0 errors.
        # should also cope for double sided but maybe np.array_equal/allclose could be needed
        return (r1 > 1e-6).any() and (r2 > 1e-6).any() 
    
    @classmethod
    def getFacesNotSeen(cls, mesh):
        pass

    @classmethod
    def getConflictsForNode(cls, node):
        mesh = node.mesh
        faces = node.faces
        faces = cls.get(faces, mesh)
        all_faces = np.array(cls.allFacesInMesh(mesh))
        mask = np.in1d(all_faces, faces)
        print("Mask", mask)
        reduced = np.triu(cls.LookUpMatrix[mesh][np.ix_(mask, mask)])
        res = reduced.any(axis=0)
        print("reslult", res)
        return np.array(faces)[res].tolist()
        
    #
    
    def can_see(self, f2):
        if type(f2) == int:
            return f2 in ExportFace.LookUpCanSee[self.mesh][self]
        return f2.index in ExportFace.LookUpCanSee[self.mesh][self]
    
    def can_see_any(self):
        return bool(ExportFace.LookUpCanSee[self.mesh][self])
       
    def is_seen(self):
        return bool(ExportFace.LookUpSeenBy[self.mesh][self])
        
    def is_seen_by(self, f2):
        if type(f2) == int:
            return f2 in ExportFace.LookUpSeenBy[self.mesh][self]
        return f2.index in ExportFace.LookUpSeenBy[self.mesh][self]
    
    #
    
    def __init__(self, face, mesh):
        self.bmesh_face = face
        self.old_index = face.index
        self.index = face.index
        self.mesh = mesh
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
            disp = np.column_stack((self.faces, node_conflicts))
            disp = np.row_stack(([np.nan]+self.faces, disp))
            print("Conflicts in Node", self.faces,"\n",disp)
            
            
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
            if (next_face.index == 28
                or next_face.index == 30
                or next_face.index == 25
                or next_face.index == 26):
                print("SORTER", next_face.index, n_min)
            ordered.append(next_face.index)
            next_face.index = len(ordered) - 1 # New index
            value_list[chosen_group_idx].pop(0)
            # Move the currently chosen list to the end, in case multiple
            # groups have the same radius and one group gets drained
            group_list.append(group_list.pop(chosen_group_idx))
            value_list.append(value_list.pop(chosen_group_idx))
    return ordered


def AfterSorter(mesh):
    CanSee, IsSeen, conflict_faces = AfterFilter(mesh)
    # Could use this for better reordering?
    

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
        similar_faces = itemgetter(*face_indices)(faces)
        # NOTE:
        # If this is just 1 face it is not a tuple.
        if type(similar_faces) != tuple:
            similar_faces = [similar_faces] # only 1 element
        else:
            similar_faces = [*similar_faces]
        
        # As theta is caped at 90 these last normals are uneven.
        # Could calculate the average normal this is the theoretical one
        main_normal = CalcNormal(t_main, p_main)
        mainNormals[gr_key] = main_normal
        ##print("main normal", main_normal, main_normal.length)
        
        # ----------------------------------------------------------
        ## STEP 1 - SORT PER GROUP ##
        
        # Now we to sort them based on the representative main normal
        # Their position vector will be divides into 
        # pos = n * main_normal + b 
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
            n = round(main_normal @ pos, 7)
            # Experimental scale - do further out are
            
            ##print(f.index, "main:", main_normal, "N:", n)
            local_ZDists[gr_key].append(n)
            # Now we also use this for a later use
            Z_Dists.append((f.index, n))
            f.ZDist = n
            return n
        similar_faces.sort(key=pos_sort)
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
    
    if prepare_nodes:
        # Have to redo groups as after reordering bmesh is invalid
        
        def NodeMaker(mesh, split_node=None, sphere_hat=True):
            """
            render_after and node_front belong together
            """
            ## Try to create nodes ##
            if split_node == None:
                faces = ExportFace.allFacesInMesh(mesh).copy()
            else:
                # Select only waht's in node
                faces = np.array(ExportFace.allFacesInMesh(mesh))[split_node.faces].tolist()
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
                conflict_faces = list(ExportFace.LookUpConflicts[mesh])
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

        # End NodeMaker
        return ordered, NodeMaker
    else:
        return ordered, None
    

def SortByNormals(bm, divisions=4, mode='ZDist', make_nodes=True, sphere_hat=True):
    """
    Returns
    """
    faces = ExportFace.buildFaces(bm)
    print("Build face", ExportFace[0])
    results = normal_filter2(bm, divisions=divisions)
    ordered, NodeMaker = order_normal_space(faces, *results, mode='ZDist', prepare_nodes=make_nodes, sphere_hat=sphere_hat)
    # Groups currently holds bmesh faces
    # reordering them will kill them, storing only face numbers
    groups = results[0]
    # Need to run RefreshGroups after reordering
    return ordered, groups, NodeMaker


def create_map(obj, fmap_name):
    if fmap_name in obj.face_maps:
        fmap = obj.face_maps[fmap_name]
    else:
        fmap = obj.face_maps.new(name=fmap_name)
    # Remove faces if present
    return fmap

if __name__ == "__main__":
    """
    For standalone in Blender sorting without Nodes
    """
    obj = bpy.context.object
    bpy.ops.object.mode_set(mode='OBJECT')
    me = obj.data
    bm = bmesh.new()
    bm.from_object(obj, bpy.context.evaluated_depsgraph_get())
    ordered, groups, NodeMaker = SortByNormals(bm, make_nodes=True)
 
    if True:
        print("reordering")
        from DarkBinImportExport.io_scene_dark_bin import reorder_faces
        reorder_faces(bm, ordered)
        ExportFace.refresh_faces(bm) # Needs to be called, bmesh data invalid after reorder.
        MainNode = NodeMaker(bm, sphere_hat=True)
        for f in MainNode.faces:
            bm.faces[f].select=True
        
        bm.to_mesh(me)
        obj.data.update()
        
        # Create Face maps
        obj.face_maps.clear()
        fmap = create_map(obj, "MainNode")
        fmap.add([idx for idx in MainNode.faces])
        fmap = create_map(obj, "FrontNode")
        fmap.add([idx for idx in MainNode.node_front.faces])
        fmap = create_map(obj, "BehindNode")
        fmap.add([idx for idx in MainNode.node_behind.faces])
        
        for node in MainNode.getConflicts():
            print("\n", "="*30,"\n")
            print("conf node", node, node.position)
            split_node = NodeMaker(bm, node, sphere_hat=True)
            
            fmap = create_map(obj, node.position+"Node_behind")
            fmap.add([idx for idx in split_node.node_behind.faces])
            fmap = create_map(obj, node.position+"Node_front")
            fmap.add([idx for idx in split_node.node_front.faces])
            # iteration is not updated
            break
        
        

        
    bpy.ops.object.mode_set(mode='EDIT')

#groups = {idx:[] for idx in range(divisions ** 2)}

