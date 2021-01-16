import numpy as np
from struct import pack, unpack, calcsize

# Made a new file this became necessary
try:
    from . import _globals
    from .SortingHelper import stack_both_sides
    from .SortingHelper import getReducedLookUpMatrix
    from .SortingHelper import kOpt
except ImportError:
    print("Using 2nd import")
    from DarkBinImportExport.testing import _globals
    from DarkBinImportExport.testing.SortingHelper import getReducedLookUpMatrix
    from DarkBinImportExport.testing.SortingHelper import stack_both_sides
    from DarkBinImportExport.testing.SortingHelper import kOpt


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
            self.checknodes = [node, node.node_behind, node.node_front]
        else:
            self.checknodes = [node]
        # Alternate lookup
        self._conflict_nodes = conflict_nodes
                
             
    def __next__(self):
        if self._index == len(self.checknodes):
            # End of Iteration
            raise StopIteration
        next_node = self.checknodes[self._index]
        if isinstance(next_node, SplitNode) and next_node != self._node:
            self.checknodes.insert(self._index+1, next_node.node_behind)
            self.checknodes.insert(self._index+2, next_node.node_front)
        self._index += 1
        if self._conflict_nodes:
            if next_node.hasConflicts():
                return next_node
            else:
                return self.__next__()    
        return next_node
# ========================================================================================

class Node(object):
    type = 0
    size = '<BffffH'
    counter = 0
    
    def hasConflicts(self, dump=False):
        mat = ExportFace.LookUpMatrix[self.mesh]
        # The upper triangle matrix (lower is 0)
        # Which represents faces that can see ones with higher indices
        # From there slice the rows and cols represeting the faces
        indices = [f.index for f in self.faces]
        node_conflicts = np.triu(mat[np.ix_(indices, indices)])
        if dump:
            np.set_printoptions(threshold=5000, linewidth=60000)
            disp = stack_both_sides(node_conflicts, indices)
            #print("Conflicts in Node\n", disp.astype(int))
            TestLUM, _,_,_ = getReducedLookUpMatrix(self.mesh, self.faces)
            print("Conflicts in Node\n", disp.astype(int),"\nSame?\n", TestLUM.astype(int))
            
        if node_conflicts.any():
            # No conflict, no further actions necessary
            return True
        # Now option to try to reorder faces again or split again
        return False
    
    def __init__(self, faces, parent, mesh=None, sphere=None, position=None):
        self.parent_node = parent
        self.sphere = sphere
        self.position = position
        self.index = self.__class__.counter
        self.__class__.counter += 1
        if faces:
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
            # detect
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
    
    def check(self, generation=0, ignore=[]):
        """
        For debugging to check if all Nodes are set.
        """
        rv = True
        numfaces = len(self.faces) if type(self) != SubobjectNode else 0
        ##DEBUG
        #print(self, "having", numfaces, "faces")
        for k, v in self.__dict__.items():
            if k not in ignore and v == None and k not in self.__class__._ignore_keys:
                print(k, "is not set on", self, "Generation:", generation)
                rv = False
            if k != "parent_node" and isinstance(v, Node):
                result = v.check(generation - 1, ignore=ignore)
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
        #faces = self.mesh.faces
        # Self.faces are only ints.
        #self.verts = [v for idx in self.faces for v in faces[idx].verts]
        self.verts = [v for f in self.faces for v in f.verts]
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
    
    def getConflictsIter(self):
        return _NodeIterator(self, conflict_nodes=True)
    
    def getConflicts(self):
        iter = _NodeIterator(self, conflict_nodes=True)
        while True:
            try:
                yield next(iter)
            except StopIteration:
                break
    
    def __iter__(self):
        return _NodeIterator(self)

# //////////////////////////////////////

class SubobjectNode(Node):
    type = 4
    size = '<BH'
    def __init__(self, subobj_idx, parent):
        self.index = subobj_idx
        self.parent_node = parent
        self.index = super().counter
        super().counter +=1

    def calcsize(self):
        return 0    # This is not positioned in the tree.
    
    def pack(self):
        return b'' # Also not packing into tree

    
class SplitNode(Node):
    type = 1
    size = '<BffffHHfHHH'
    do_split = False

    def __init__(self, faces, n_first_faces, parent, mesh=None, node_front=None, node_behind=None,
                    split_face=None, d=None, sphere=None, position='NOTSET'):
        assert n_first_faces <= len(faces), "Wrong number of (first) faces passed"
        
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
        # Not tested
        return Node(self.faces, self.parent)

    def makenorm(self, f, faces):
        # Relies on nemyax encoded normal idx
        ext_f = faces.layers.string.verify()
        self.norm = unpack('<H', f[ext_f][:2])
     
    def pack(self):
        assert self.check(), "Node "+ str(self) + " missing data. Can't pack."
        if type(self.norm)!= bytes:
            pass
        self.norm = pack('<H', unpack('<H',self.norm)[0]) #TODO: care
        if type(self.d) != bytes:
            # NOTE Split planes seam to be packed with -d
            # And/or probably later also work with the normal reversed
            self.d = pack('<f', self.d)
        return (pack('<B', self.type)
                + self.makesphere()
                + pack('<H', self.n_first_faces)
                + self.norm
                + self.d
                + pack('<HHH', self.node_behind_off, 
                               self.node_front_off, 
                               len(self.faces) - self.n_first_faces)
                + b''.join(self.faces)) # THESE HAVE BEEN CHANGED IN THE EXPORTER

# ==========================================================================================

def OptNode(mesh, faces, opt_type=6):
    print("Optimizing Node")
    print("befre", faces)
    LookUpMatrix, back_in_node, reduced_order, front_in_node = getReducedLookUpMatrix(mesh, faces)
    # Do another optimization for sub nodes:
    #print("Lookup Matrix For Node\n",stack_both_sides(LookUpMatrix, reduced_order))
    #print("filtered out", back_in_node, "+", front_in_node)
    
    # Do another optimization
    if opt_type >= 6:
        stop, LookUpMatrix = kOpt(LookUpMatrix, mesh, reduced_order)
    
    # Reorder Faces

    old_indices    = [f.old_index for f in faces]
    new_node_order = back_in_node+reduced_order+front_in_node
    faces          = [faces[old_indices.index(idx)] for idx in new_node_order]
    print("after", faces)
    # TODO: Need to update main matrix
    all_faces = ExportFace.LookUpFaces[mesh]
    print("==")
    new_matrix_order = [f.old_index if f not in faces else new_node_order.pop(0) for f in all_faces]
    print("after", old_indices)
    ExportFace.updateIndices(mesh, new_matrix_order)
    ExportFace.updateMatrix(mesh, new_matrix_order)
    return LookUpMatrix, faces, back_in_node, reduced_order, new_matrix_order

all_split_planes = []
def find_split_plane(mesh, faces, LookUpMatrix, back_faces=None):
    # TODO: Maybe use sum of both
    conflict_costs2 = np.sum(np.triu(LookUpMatrix), axis=0)
    conflict_costs  = np.sum(np.triu(LookUpMatrix), axis=1)
    both = conflict_costs+conflict_costs2
    print("Node costs:")
    print(np.vstack(([f.index for f in faces], conflict_costs, conflict_costs2, both)))

    highest_cost_index = np.argmax(both) + (len(back_faces) if back_faces else 0)
    
    print("highest cost at", highest_cost_index, len(faces), LookUpMatrix.shape)
    print( "Face:", faces[highest_cost_index].index, 
            "\nold", faces[highest_cost_index].old_index)
    row = LookUpMatrix[highest_cost_index,highest_cost_index:]
    
    bad_faces = np.array(faces[highest_cost_index:])[row==True]
    #checker2 = conflict_costs2[highest_cost_index:][row==True]
    
    split_plane = faces[highest_cost_index]
    all_split_planes.append(split_plane)
    print("All Splits are", all_split_planes)
    #print(np.vstack(([f.index for f in faces[highest_cost_index:]],row)))
    print(bad_faces)
    #print(checker2)
    if not len(bad_faces):
        return faces[highest_cost_index]
    # Fine one in front of the one with the most wrong seen
    # Which is seen wrongly by the most
    #split_plane = bad_faces[np.argmax(checker2)]

    return split_plane

def NodeMaker(mesh, split_node=True, sphere_hat=True, opt_type=6, max_tree_depth=3,_depth_level=0):
    """
    Render_after and node_front belong together.
    If split_node is True the Main(Split)Node will be created
    If a Node is passed that one is splitted.
    If False/None is passed a RawNode will be returned.
    """
    
    ## Try to create nodes ##
    if isinstance(split_node, Node):
        # Select only what's in node
        faces = split_node.faces
        # Test
        print("Test second run\n", split_node.faces)
        #conflict_faces = ExportFace.getConflictsForNode(split_node, filter=3)
    else:
        faces = ExportFace.allFacesInMesh(mesh).copy()
        #conflict_faces = ExportFace.getConflicts(mesh)
        #conflict_faces.sort(key=lambda f: f.index)
        if not split_node:
            # RawNode as MainNode
            # TODO: Could opt here
            MainNode = Node(faces, parent=-1, mesh=mesh, position='Main')
            MainNode.makesphere()
            return MainNode
        
    #conflict_faces_ids = [f.index for f in conflict_faces]
    #print("§CONFLICTS", [f.index for f in conflict_faces], "\n max", max(conflict_faces_ids) if len(conflict_faces_ids) else "empty")
    
    # This should be equal to f is not seen
    # But cant but seen with low index in front node!
    #new_front = [f for f in faces if (f not in conflict_faces)]

    # Maybe for the future use this one, are they the same?
    # Well they can be seen if no conflict.
    # This is not filtered after a split!
    print("Reducing Data")
    LookUpMatrix, back_faces, normal_faces, front_faces = getReducedLookUpMatrix(mesh, faces, return_faces=True)
    #back_faces, normal_faces, front_faces = ExportFace.getFacesFiltered(mesh, faces)
    faces = normal_faces.copy()
    
    #print("Algorithms match\nOLD", new_front,"\nNEW", [f.index for f in front_faces],
    #        "\nB", [f.index for f in back_faces], "\nB")
    
    front = front_faces
    # Removed faces are not in groups anymore, adding again below
    back = back_faces
    
    n_first_faces = len(back) 
    back.extend(front)        # TODO: Test if totally back faces are okay here

    # most_front.extend(front) # that makes no sense at the moment.
    # Create node objects

    if isinstance(split_node, Node):
        MainNode = split_node.transform(n_first_faces) # This was a raw node until now
        num_all = len(split_node.faces)
        MainNode.faces.clear()
        MainNode.faces = back
        print("Second front", back)
        # TODO: Call nodes!
    else:
        MainNode = SplitNode(back, n_first_faces, parent=-1, mesh=mesh, position='Main')    
    
    BackNode = MainNode.node_behind = Node(faces=None, parent=MainNode, position='behind')
    FrontNode = MainNode.node_front = Node(faces=None, parent=MainNode, position='front')
    # Now need to fill these two and determine some split
    # Assuming we are sphere hat here but should work else as well

    # DETERMINE SPLIT FACE
    print("="*30, "\nFind Split")
    if isinstance(split_node, Node) and True: #or opt_level:
        # opt_type >= 6 for kOpt
        LookUpMatrix, faces, back_in_node, reduced_order, _ = OptNode(mesh, faces, opt_type)
        #print("Opt Matrix For Node\n", stack_both_sides(LookUpMatrix, reduced_order))
        #TestLUM, _,_,_ = getReducedLookUpMatrix(mesh, faces)
        #print("Should be the same\n", stack_both_sides(TestLUM, reduced_order))
        #print([f.old_index for f in normal_faces], "and", reduced_order)
        #print(len(back_faces), len(back_in_node))
    else:
        back_in_node = None

    # Maybe conflict_faces[-1]?

    # Choosing a good split plane is important!
    # Maybe need to do more here - for this model it fits.
    #split_plane = conflict_faces[-1]
    
    #NEW
    split_plane = find_split_plane(mesh, faces, LookUpMatrix, back_in_node)
    
    split_plane_idx = split_plane.index
    MainNode.split_face = split_plane
    
    print("SplitPlane\n", split_plane.old_index)
    # This is numpy
    faces.remove(split_plane)
    for face in faces:  # This group biased again.
        n = split_plane.normal.dot(face.calc_center_bounds()) - split_plane.d # numpy
        n = round(n, 2) # To make up some numerical unevenness close to 0.0.
        # This i done the other way, if in front of split plane they are behind
        if  n > 0:
            FrontNode.faces.append(face)
            #print(face.old_index," is in front of", split_plane.old_index)
        else:
            BackNode.faces.append(face)
    
    #FrontNode.faces.insert(0, split_plane) #TODO would like to move behind, but need to reorder again TODO
    BackNode.faces.append(split_plane)
    if opt_type >= 6:
        _, FrontNode.faces, back_in_node, reduced_order, _  = OptNode(mesh, FrontNode.faces)
        _, BackNode.faces, back_in_node2, reduced_order2, new_matrix_order = OptNode(mesh, BackNode.faces)
    
    # TODO again update
    #Theoretically done in kOptNode
    
    #FrontNode.faces.extend([f.index for f in front_faces])
    # Is filtered internally from all faces.
    MainNode.makesphere()
    BackNode.makesphere()
    FrontNode.makesphere()
 
    print("Main Faces:\n", [f.old_index for f in MainNode.faces])
    print("Front Node:\n", [f.old_index for f in FrontNode.faces])
    print("Back Node:\n",  [f.old_index for f in BackNode.faces])
    
    print("Conflicts", MainNode.hasConflicts(), FrontNode.hasConflicts(), BackNode.hasConflicts())
    if _depth_level < max_tree_depth: #TODO MAKE A CONSTANT
        # NOTE each Node will be split in two(if necessary) and we end up with 2^Depth nodes (or so)
        print("New Nodes at level", _depth_level)
        if FrontNode.hasConflicts():
            # Replacement is done automatically
            _, new_matrix_order = NodeMaker(mesh, FrontNode, max_tree_depth=max_tree_depth, opt_type=opt_type,_depth_level=_depth_level+1)
        if BackNode.hasConflicts():
            _, new_matrix_order = NodeMaker(mesh, BackNode, max_tree_depth=max_tree_depth, opt_type=opt_type, _depth_level=_depth_level+1)
    
    # Makes sure all nodes in tree are set correctly
    valid = MainNode.check(ignore=['norm', 'd'])
    num_faces = MainNode.numfaces # set in check()
    if not isinstance(split_node, Node):
        num_all = len(mesh.faces) # All
    else:
        num_all = len(back_faces) + len(normal_faces)# + len(front_faces) these are in back_faces
    print("Nodes are ok?:", valid, f"\nNum Faces: {num_faces}/{num_all} ok?:", num_faces==num_all)
    if "new_matrix_order" not in locals():
        new_matrix_order = None
    
    # TODO reorder faces again
    return MainNode, new_matrix_order

if __name__ == "__main__":
    # THIS FILE NEEDS TO BE SAVED TO HAVE AN EFFECT
    from importlib import reload
    import DarkBinImportExport.testing.SortingHelper as SortingHelper
    reload(SortingHelper)
    print(SortingHelper.REL_OPT_IMPROVEMENT, 'REL_OPT_IMPROVEMENT')
    from DarkBinImportExport.testing.SortingHelper import *
    #reload(SortingHelper)
    UseInBlender()
    print(WholeOptTimes)
    print(OptTimes)
