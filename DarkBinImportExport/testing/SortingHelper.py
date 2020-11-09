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
    LookUpCanSee = {}
    LookUpSeenBy = {}
    LookUpFaces  = {}       # mesh specific but maybe not complete!
    LookUpConflicts = {}
    LookUpCanSeeConflict = {}
    LookUpIsSeenConflict = {}
    
    @classmethod
    def allFacesInObject(cls):     # May not be complete!
        return chain(cls.LookUpFaces.values()) # Iterable!
    
    @classmethod
    def allFacesInMesh(cls, mesh):
        return cls.LookUpFaces[mesh]
    
    @classmethod
    def buildFaces(cls, mesh):
        can_see = cls.LookUpCanSee[mesh] = {}
        seen_by = cls.LookUpSeenBy[mesh] = {f2:[] for f2 in mesh.faces}
        for f1 in mesh.faces:
            f_new = ExportFace(f1, mesh)  # Make new instance, auto registers
            for f2 in mesh.faces:
                if f1 == f2:
                    continue
                f1 = f_new
                if cls.FaceVisible(f1, f2):
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
        return cls.LookUpFaces[mesh][idx]
    
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
def LocalBBoxCenter(obj):
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

class Node(object):
    type = 0
    size = '<BffffH'
    
    def __init__(self, faces, parent, sphere=None):
        if faces:
            # Faces as index not BMesh.Faces
            if (type(faces[0]) != int):
                faces = [f.index for f in faces]
            self.faces = faces
        else:
            self.faces = []
        self.parent_node = parent
        self.sphere = sphere
    
    def transform(self, n_first_faces, split_position, node_front=None, node_behind=None):
        new = SplitNode(self.faces, n_first_faces, 
                        parent=self.parent, 
                        sphere=self.sphere, 
                        node_front=None, node_behind=None)
        if split_position == 'front':
            self.parent.node_front = new
        else:
            self.parent.node_behind = new
        return new
        
    def calcsize(self):
        return calcsize(self.size) + len(self.faces)*2
    
    def check(self, generation=0):
        """
        For debugging to check if all Nodes are set.
        """
        rv = True
        numfaces = len(self.faces) if type(self) != SubobjectNode else 0
        ##DEBUG
        #print(self, "having", numfaces, "faces")
        for k, v in self.__dict__.items():
            if v == None:
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
 
    def getsphere(self, faces=None):
        try:
            from ..io_scene_dark_bin import get_local_bbox_data, encode_sphere
        except ImportError:
            from DarkBinImportExport.io_scene_dark_bin import get_local_bbox_data, encode_sphere
        if "sphere" in self.__dict__ and self.sphere != None:
            return self.sphere
        # Makesphere
        if not faces:
            print(self, "Has no faces")
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
                    + self.getsphere() \
                    + pack('<H',len(self.faces))\
                    + b''.join(self.faces)
    
    def packfaces(self):
        #TODO
        pass
    
    def __iter__(self):
        return NodeIterator(self)


class NodeIterator:
    def __init__(self, node):
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
        
    def __next__(self):
        if self._index == len(self._checknodes):
            # End of Iteration
            raise StopIteration
        next = self._checknodes[self._index]
        if isinstance(next, SplitNode) and next != self._node:
            self._checknodes.insert(self._index+1, next.node_behind)
            self._checknodes.insert(self._index+2, next.node_front)
        self._index += 1
        return next

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

    def __init__(self, faces, n_first_faces, parent, node_front=None, node_behind=None,
                    split_face=None, d=None, sphere=None):
        assert n_first_faces <= len(faces)
        self.faces = faces
        self.n_first_faces = n_first_faces
        self.node_front = node_front
        self.node_behind = node_behind
        self.parent_node = parent
        self.d = d
        self.sphere=sphere
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
                + self.getsphere()
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
            for v in f.verts:
                if not pos or v.co.length > pos.length:
                    pos = v.co
            # n.length is not exact 1 so this is numerical, maybe round it?
            n = round(main_normal @ pos, 5)
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
        
        def NodeMaker(mesh, groups, sphere_hat=True):
            """
            render_after and node_front belong together
            """
            ## Try to create nodes ##
            faces = ExportFace.allFacesInMesh(mesh)

            split_groups = groups # Wanted to copy but .copy is only first level.
            front = []  # Rendered last
            back = []   # Rendered first
            
            """
            # These are safe frontest and backest
            baseNs = {}
            for key in list(split_groups):
                group = split_groups[key]
                try:
                    # This needs at least 2 faces in a group for one to be 'proper'
                    front.append(group.pop(-1).index)
                    back.append(group.pop(0).index)
                    baseNs[key]=[local_ZDists[key].pop(0), local_ZDists[key].pop(-1)]
                    if baseNs[key][0] == baseNs[key][-1]:
                        # First and last face are in the same plane.
                        group.reverse()
                        front.extend([f.index for f in group])
                        front.append(back.pop(-1))
                        group.clear()
                    if len(group) == 0:
                        raise IndexError # dump it, faces were already moved to front+back
                except IndexError:
                    del split_groups[key]
                    if key in local_ZDists:     # These are already reduced could throw 
                        del local_ZDists[key]
            
            print("Original\n", front, back)
            if len(split_groups) == 0:
                print("Model was to simple for node splitting:")
                raw = Node(faces, parent=-1)
                if len(front) and len(back):
                    print("Maybe could have done something")
                raw.getsphere(faces)
                return raw
            
            # Should be save to also append faces with the same n as the first.
            both = [back, front]

            for key, group in split_groups.items():
                # Worst case baseN length is one
                while len(local_ZDists[key]) != 0 and local_ZDists[key][0] == baseNs[key][0]:
                    back.append(group.pop(0).index)
                    local_ZDists[key].pop(0)
                while len(local_ZDists[key]) != 0 and local_ZDists[key][-1] == baseNs[key][-1]:
                    front.append(group.pop(-1).index)
                    local_ZDists[key].pop(-1)
            """
            """
            front.sort()
            back.sort()
            most_front = np.array(front)
            most_back = np.array(back)
            print(front, back)
            
            # TODO: Where can this fuck up?
            """
            conflict_faces = list(ExportFace.LookUpConflicts[mesh])
            conflict_faces.sort(key=lambda f: f.index)
            conflict_faces_ids = [f.index for f in conflict_faces]
            """
            mask = np.in1d(most_front, conflict_faces_ids, assume_unique=True)
            front_conflicts = most_front[mask]
            front = most_front[~mask].tolist() # Remove conflict faces.
            
            mask = np.in1d(most_back, conflict_faces_ids, assume_unique=True)
            back_conflicts = most_back[mask]
            print("BACK FONCTL", back_conflicts)  
            """
            
            # This should be equal to f is not seen
            new_front = [f.index for f in faces if (f not in conflict_faces)]
            
            # Maybe for the future use this one:
            new_front_faces = [f for f in faces if not f.is_seen()]
            
            new_front2 = [f.index for f in new_front_faces]
            print("Algorithms match", new_front2 == new_front)
            
            for f in new_front_faces:
                idx = split_groups[f.group].index(f)
                del split_groups[f.group][idx]
                del local_ZDists[f.group][idx]
            
            for key in list(split_groups):
                if len(split_groups[key]) == 0:
                    del split_groups[key]
                    if key in local_ZDists:
                        del local_ZDists[key]
               
            front = new_front
            # Removed faces are not in groups anymore, adding again below
                  
            
            """
            i = front[-1]
            print("starting at", i)
            while i >= 5:
                if i not in front:
                    print("found gap at", i)
                    missing_face = faces[i]
                    print("missing face", missing_face)
                    # Option A) Fill hole
                    # Option B) Dump Value
                    if missing_face.index not in conflict_faces_ids and missing_face.index not in most_back:
                        front.insert(front.index(last_valid), missing_face.index)
                        print("inserting", missing_face.index, i+1)
                        # Remove from split_groups
                        split_groups[missing_face.group].remove(missing_face)
                        local_ZDists[missing_face.group].remove(missing_face.ZDist)
                        i += 1 # Increase by 1 to check next again.
                else:
                    last_valid = i
                i -= 1
                
            i = back[0]
            while i<= back_conflicts[-1]:
                i += 1
            # Could continue!  
            print("new front\n", front)    
            """
            
            n_first_faces = 0# len(front)
            # most_front.extend(front) # that makes no sense at the moment.
            # Create node objects
            MainNode = SplitNode(front, n_first_faces, parent=-1)
            BackNode = MainNode.node_behind = Node(faces=None, parent=MainNode)
            FrontNode = MainNode.node_front = Node(faces=None, parent=MainNode)
            # Now need to fill these two and determine some split
            # Assuming we are sphere hat here but should work else as well
            
            """
            # Readd removed faces
            for f_idx in front_conflicts:
                f = ExportFace.get(f_idx, mesh)
                split_groups[f.group].append(f)
                local_ZDists[f.group].append(f.ZDist)
            """
            for key, group in split_groups.items():
                print(key, [f.index for f in group])
            
            # DETERMINE SPLIT FACE
            print("="*30, "\nFind Split")
            
            print("§CONFLICTS", conflict_faces_ids, "\n max", max(conflict_faces_ids))
            # Maybe conflict_faces[-1]?
            """
            if (len(division_angles[0]) // 2, 
                len(division_angles[1]) // 2) in split_groups:      # Main +X axis
                MainGroup = (len(division_angles[0]) // 2, len(division_angles[1]) // 2)
            elif (len(division_angles[0]) // 2 + len(division_angles[0]) // 4,
                  len(division_angles[1]) // 2) in split_groups:    # Main +Y Axis
                MainGroup = (0,1)      
            elif (0, 0) in split_groups:                            # Main +Z
                MainGroup = (0, 0)
            elif (0, len(division_angles[1]) -1) in split_groups:
                MainGroup = (0, len(division_angles[1]) -1)         # Main -Z
            else:
                MainGroup = next(iter(split_groups.keys()))         # Anything
            
            group = np.array(split_groups[MainGroup], dtype="O")
            ZDist = np.array(local_ZDists[MainGroup], dtype="O") # leave int and float
            #del split_groups[MainGroup]
            #del local_ZDists[MainGroup]
            print("Main Group:", MainGroup, group)
            print("Zs", ZDist)

            behind_faces = group[ZDist<=0]
            front_faces = group[ZDist>0]
            print("Front Back faces")
            print(front_faces, behind_faces)

            #FrontNode.faces.extend(front)
            # But we also need to pack ALL other faces into these groups
            if len(front_faces):
                split_plane_idx = front_faces[-1].index
                split_plane = front_faces[-1]
            else:
                split_plane_idx = behind_faces[0].index
                split_plane = behind_faces[0]
            
            split_plane = group[0]
            """
            # Choosing a good split plane is important!
            # Maybe need to do more here - for this model it fits.
            split_plane = conflict_faces[-1]
            split_plane_idx = split_plane.index
            MainNode.split_face = split_plane.index
            
            print("SplitPlane\n",split_plane)
            split_d = split_plane.calc_center_bounds() @ split_plane.normal
            for group in split_groups.values():
                for face in group:  # This group biased again.
                    n = (face.calc_center_bounds() @ split_plane.normal) - split_d
                    n = round(n, 5) # To make up some numerical unevenness close to 0.0.
                    print(face.index, n)
                    # This i done the other way, if in front of split plane they are behind
                    if  n > 0:
                        FrontNode.faces.append(face.index)
                        print(face.index," is in front of", split_plane_idx)
                    else:
                        BackNode.faces.insert(0, face.index)

            BackNode.faces.sort()
            FrontNode.faces.sort()
            #FrontNode.faces.extend([f.index for f in front_faces])
            # Is filtered internally from all faces.
            MainNode.getsphere(faces)
            BackNode.getsphere(faces)
            FrontNode.getsphere(faces)
 
 
            print("Main Faces:\n", MainNode.faces)
            print("Front Node:\n", FrontNode.faces)
            print("Back Node:\n", BackNode.faces)
            
            # Makes sure all nodes in tree are set correctly
            valid = MainNode.check()
            num_faces = MainNode.numfaces
            print("Nodes are ok?:", valid, f"\nNum Faces: {num_faces}/{len(faces)} ok?:", num_faces==len(faces))
            
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
        MainNode = NodeMaker(bm, groups, sphere_hat=True)
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
        
    bpy.ops.object.mode_set(mode='EDIT')

#groups = {idx:[] for idx in range(divisions ** 2)}

