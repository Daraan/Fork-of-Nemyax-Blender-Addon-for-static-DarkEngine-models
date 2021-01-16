import numpy as np
from struct import pack, unpack

#enum _mde_node {
MD_NODE_RAW = 0,
MD_NODE_SPLIT = 1,
MD_NODE_CALL = 2,
MD_NODE_VCALL = 3,
MD_NODE_SUBOBJ = 4

class Sphere(object):
    cen = None
    rad = None
    node_type=None
    def __init__(self, bs, node):
        cx, cy, cz, rad = unpack('<ffff', bs)
        self.rad = round(rad, 4)
        self.cen = np.array((round(cx,4), 
                        round(cy,4), 
                        round(cz,4)))
        self.node = node
        self.node_type=node.nd_type
        
    def __repr__(self):
        return f"Sphere(Node:{self.node.position}): r={self.rad:.4f} | {self.cen}"
        

# This is ugly but does what we want - etract nodes and displays them
class Node(object):
    nd_type : -1
    AllNodes = []
    _nd_index = 0
    def __init__(self, **kwargs):
        Node.AllNodes.append(self)
        self.nd_index = Node._nd_index  # Internal
        Node._nd_index += 1             # External counter
        for key, value in kwargs.items():
            if key=="sphere":
                self.sphere = Sphere(value, self)
                continue
            if key=="polys":
                try:
                    from .BinNodeExtractor import binBytes, faceOffset
                except ImportError:
                    from BinNodeExtractor import binBytes, faceOffset
                self.raw_polys = value
                newvalue = []
                for f_off in value:
                    f = unpack('<H', binBytes[faceOffset+f_off:faceOffset+f_off+2])[0]
                    newvalue.append(f)
                key="faces"
                value = newvalue
            self[key] = value
    
    def __setitem__(self, key, value):
        self.__dict__[key] = value
    
    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __getattr__(self, name):
        try:
            if hasattr(Node, name):
                return self[name]
            return self.__dict__[name]
        except KeyError:
            raise AttributeError(name)
    
    def __repr__(self):
        print(self.__class__.__name__, "(",self.position,")")
        for k,v in self.__dict__.items():
            if isinstance(v, Node):
                v = v.__class__.__name__ + " "+ str(v.position)
            if isinstance(k, Node):
                k = k.__class__.__name__ + " "+ str(k.position)
            print(f"{k:<15}", v)
        return ""
    
     
class Node_raw(Node):
    nd_type = MD_NODE_RAW   # 0
    
class Node_sub(Node):
    nd_type = MD_NODE_SUBOBJ # 4
    
class Node_split(Node):
    nd_type = MD_NODE_SPLIT # 1
    def __init__(self, **kwargs):
        try:
            from .BinNodeExtractor import getNormalByIndex
        except ImportError:
            from BinNodeExtractor import getNormalByIndex
        norm = kwargs["norm"]
        self.norm_id = norm
        self.normal = getNormalByIndex(norm)
        #Norms[norm].append(self.normal)
        super().__init__(**kwargs)

class Node_call(Node):
    nd_type = MD_NODE_CALL # 2

class Node_vcall(Node):
    nd_type = MD_NODE_VCALL # 3