import numpy as np
from struct import pack, unpack, calcsize

# Made a new file this became necessary
try:
    from . import _globals
except ImportError:
    from DarkBinImportExport.testing import _globals
# Setting this later from _globals
ExportFace = None


# ========================================================================================
# Nodes
# ========================================================================================

class _NodeIterator:
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
            from .SortingHelper import stack_both_sides
            disp = stack_both_sides(node_conflicts, self.faces)
            disp = node_conflicts
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
        iter = _NodeIterator(self, conflict_nodes=True)
        while True:
            try:
                rv.append(next(iter))
            except StopIteration:
                break
        return rv
    
    def __iter__(self):
        return _NodeIterator(self)


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

# ==========================================================================================

def NodeMaker(mesh, split_node=None, sphere_hat=True):
    """
    render_after and node_front belong together
    """
    global ExportFace
    ExportFace = _globals.shared["ExportFace"]
    
    ## Try to create nodes ##
    if split_node == None:
        faces = ExportFace.allFacesInMesh(mesh).copy()
    else:
        # Select only what's in node
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
    back_faces, normal_faces, front_faces = ExportFace.getFacesFiltered(mesh, faces)
    new_front = []
    new_back = []
    checker = ExportFace.LookUpMatrix[mesh][conflict_faces_ids]
    print(checker.shape)
    for f in faces.copy():
        if f.index < conflict_faces[0].index and not checker[:,f.index].any():
            new_back.append(f.index)
            faces.remove(f)
        elif f.index > conflict_faces[-1].index:
            new_front.append(f.index)
            faces.remove(f)
    print("Algorithms match\n", new_front2,"\n", new_front,"\n", [f.index for f in front_faces],
            "\nB",new_back,"\nB", [f.index for f in back_faces])
    
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
    # This is numpy
    split_d = split_plane.normal.dot(split_plane.calc_center_bounds())
    BackNode.faces.append(split_plane_idx)
    faces.remove(split_plane)
    for face in faces:  # This group biased again.
        n = split_plane.normal.dot(face.calc_center_bounds()) - split_d # numpy
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

if __name__ == "__main__":
    # THIS FILE NEEDS TO BE SAVED TO HAVE AN EFFECT
    from importlib import reload
    from DarkBinImportExport.testing import SortingHelper
    reload(SortingHelper)
    SortingHelper.UseInBlender()
