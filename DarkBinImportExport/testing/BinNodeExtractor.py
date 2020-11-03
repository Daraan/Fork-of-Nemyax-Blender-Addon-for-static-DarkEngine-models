import numpy as mu
import math
import re
import struct
import os
import glob
from struct import pack, unpack
np = mu

###
### Helper classes
###

class Sphere(object):
    cen = None
    rad = None
    nodend_type=None
    def __init__(self, bs, nd_type=-1):
        cx, cy, cz, rad = unpack('<ffff', bs)
        self.rad = round(rad, 4)
        self.cen = np.array((round(cx,4), 
                        round(cy,4), 
                        round(cz,4)))
        self.nodend_type=nd_type
        
    def __repr__(self):
        return f"Sphere({self.nodend_type}): r={self.rad:.4f} | {self.cen}"
        

# This is ugly but does what we want - etract nodes and displays them

class Node:
    nd_type : -1
    AllNodes = []
    _nd_index = 0
    def __init__(self, **kwargs):
        Node.AllNodes.append(self)
        self.nd_index = Node._nd_index # Internal
        Node._nd_index += 1
        for key, value in kwargs.items():
            if key=="polys":
                self.raw_polys = value
                newvalue = []
                for f_off in value:
                    f = unpack('<H', binBytes[faceOffset+f_off:faceOffset+f_off+2])[0]
                    newvalue.append(f)
                key="faces"
                value = newvalue

            if key!="sphere":
                self[key] = value
            else:
                self.sphere = Sphere(value, self.nd_type)
        
    
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
            print(f"{k:<15}",v)
        return ""
    
     
class Node_raw(Node):
    nd_type = 0
    
class Node_sub(Node):
    nd_type = 4
    
class Node_split(Node):
    nd_type = 1
    def __init__(self, **kwargs):
        norm = kwargs["norm"]
        self.norm_id = norm
        self.normal = getNormalByIndex(norm)
        #Norms[norm].append(self.normal)
        super().__init__(**kwargs)

class Node_call(Node):
    nd_type = 2

class Node_vcall(Node):
    nd_type = 3


#########################
class FaceImported:
    allFaces = []
    def __ini__(self):
        allFaces.append(self)

normOffset = None
faceOffset = None
def getNormalByIndex(i):
    off = normOffset + 12 * i # length of 3 floats
    end = off + 12
    assert end <= faceOffset
    return unpack('<fff', binBytes[off:end])


###
### Import
###


def aka(key, l):
    result = None
    for i in range(len(l)):
        if key == l[i][0]:
            result = (i,l[i])
            break
    return result

def get_uints(bs):
    spec = '<' + str(len(bs) // 4) + 'I'
    return list(unpack(spec, bs))

def get_ushorts(bs):
    spec = '<' + str(len(bs) // 2) + 'H'
    return list(unpack(spec, bs))

def get_floats(bs):
    spec = '<' + str(len(bs) // 4) + 'f'
    return list(unpack(spec, bs))

def get_string(bs):
    s = ""
    for b in bs:
        s += chr(b)
    result = ""
    for c in filter(lambda x: x!='\x00', s):
        result += c
    return result

class SubobjectImported(object):
    def __init__(self, bs, faceRefs, faces, materials, vhots):
        self.name   = get_string(bs[:8])
        self.motion, self.parm, self.min, self.max = unpack('<Biff', bs[8:21])
        self.child, self.next  = unpack('<hh', bs[69:73])
        self.xform = get_floats(bs[21:69])
        curVhotsStart, numCurVhots = get_ushorts(bs[73:77])
        self.vhots = vhots[curVhotsStart:curVhotsStart+numCurVhots]
        facesHere = [faces[addr] for addr in faceRefs]
        matsUsed = {}
        for f in facesHere:
            m = f.binMat
            if m != None:
                matsUsed[m] = materials[m]
        self.faces = facesHere
        self.matsUsed = matsUsed
    def matSlotIndexFor(self, matIndex):
        if matIndex != None:
            return list(self.matsUsed.values()).index(self.matsUsed[matIndex])
    def localMatrix(self):
        if all(map(lambda x: x == 0.0, self.xform)):
            return np.eye(4)
        else:
            matrix = mu.empty((3,3))
            matrix[0][0], matrix[1][0], matrix[2][0] = self.xform[:3]
            matrix[0][1], matrix[1][1], matrix[2][1] = self.xform[3:6]
            matrix[0][2], matrix[1][2], matrix[2][2] = self.xform[6:9]
            matrix[0][3] = self.xform[9]
            matrix[1][3] = self.xform[10]
            matrix[2][3] = self.xform[11]
            return matrix

def prep_materials(matBytes, numMats, file_path):
    materials = {}
    stage1 = []
    stage2 = []
    for _ in range(numMats):
        matName = get_string(matBytes[:16])
        matSlot = matBytes[17]
        stage1.append((matSlot,matName))
        matBytes = matBytes[26:]
    if matBytes: # if there's aux data
        auxChunkSize = len(matBytes) // numMats
        for _ in range(numMats):
            clear, bright = get_floats(matBytes[:8])
            stage2.append((clear,bright))
            matBytes = matBytes[auxChunkSize:]
    else:
        for _ in range(numMats):
            stage2.append((0.0,0.0))
    for i in range(numMats):
        s, n = stage1[i]
        c, b = stage2[i]
        materials[s] = (n,c,b)
    return materials

def prep_vhots(vhotBytes):
    result = []
    while len(vhotBytes):
        result.append((
            unpack('<I', vhotBytes[:4])[0],
            list(get_floats(vhotBytes[4:16]))))
        vhotBytes = vhotBytes[16:]
    return result

def prep_verts(vertBytes):
    floats = list(get_floats(vertBytes))
    verts = []
    i = -1
    while floats:
        i += 1
        x = floats.pop(0)
        y = floats.pop(0)
        z = floats.pop(0)
        verts.append((i,(x,y,z)))
    return verts

def prep_uvs(uvBytes):
    floats = list(get_floats(uvBytes))
    uvs = []
    i = -1
    while floats:
        i += 1
        u = floats.pop(0)
        v = floats.pop(0)
        uvs.append(mu.array((u,v)))
    return uvs

def prep_faces(faceBytes, version):
    garbage = 9 + version # magic 12 or 13: v4 has an extra byte at the end
    faces = {}
    faceAddr = 0
    faceIndex = 0
    while len(faceBytes):
        f_index = unpack('<H', faceBytes[0:2])[0]
        matIndex = unpack('<H', faceBytes[2:4])[0]
        nd_type = faceBytes[4] & 3
        num_verts = faceBytes[5]
        norm = unpack('<H', faceBytes[6:8])[0]
        d = unpack('<f', faceBytes[8:12])[0]

        verts = get_ushorts(faceBytes[12:12+num_verts*2])
        uvs = []
        if nd_type == 3:
            faceEnd = garbage + num_verts * 6
            uvs.extend(get_ushorts(faceBytes[12+num_verts*4:12+num_verts*6]))
        else:
            faceEnd = garbage + num_verts * 4
            matIndex = None
        face = FaceImported()
        face.binVerts = verts
        face.binUVs = uvs
        face.binMat = matIndex
        face.index = f_index
        face.d = round(d,3)
        face.norm_id = norm
        if norm in Norms:
            Norms[norm].append(f_index)
        else:
            Norms[norm] = [f_index]
        face.normal = getNormalByIndex(norm)
        faces[faceAddr] = face
        faceAddr += faceEnd
        faceIndex += 1
        faceBytes = faceBytes[faceEnd:]
    return faces

def node_subobject(bs, pos):
    node = Node_sub(index=unpack('<H', bs[1:3])[0], position=pos)
    return ([],bs[3:], node, pos+3)

def node_vcall(bs, pos):
    node = Node_vcall(position=pos,
                      sphere=bs[1:17], 
                      index=unpack('<H', bs[17:19]))
    return ([],bs[19:], node, pos+19)

def node_call(bs, pos):
    facesStart = 23
    sphere = bs[1:17]
    num_faces1 = unpack('<H', bs[17:19])[0]
    node_call = unpack('<H', bs[19:21])[0]
    num_faces2 = unpack('<H', bs[21:facesStart])[0]
    facesEnd = facesStart + (num_faces1 + num_faces2) * 2
    faces = get_ushorts(bs[facesStart:facesEnd])
    node = Node_call(sphere=sphere, 
                     pgons_before= num_faces1, 
                     node_call= node_call,
                     pgons_after= num_faces2,
                     polys=faces,
                     position=pos)
    return (faces,bs[facesEnd:], node, pos+facesEnd)

Norms = {}
ds = {}

def node_split(bs, pos):
    facesStart = 31
    num_faces1 = unpack('<H', bs[17:19])[0]
    norm = unpack('<H', bs[19:21])[0]
    d = unpack('<f', bs[21:25])[0]
    Norms[norm].append(f"split d={d:.3f}")
    node_behind = unpack('<H', bs[25:27])[0]
    node_front = unpack('<H', bs[27:29])[0]
    num_faces2 = unpack('<H', bs[29:facesStart])[0]
    facesEnd = facesStart + (num_faces1 + num_faces2) * 2
    faces = get_ushorts(bs[facesStart:facesEnd])
    
    node = Node_split(sphere=bs[1:17],
                      pgons_before=num_faces1,
                      norm=norm,
                      d=d,
                      node_behind=node_behind,
                      node_front=node_front,
                      pgons_after=num_faces2,
                      polys=faces,
                      position=pos)
    
    return (faces,bs[facesEnd:],node, pos+facesEnd)

def node_raw(bs, pos):
    facesStart = 19
    num_faces = unpack('<H', bs[17:facesStart])[0]
    facesEnd = facesStart + num_faces * 2
    faces = get_ushorts(bs[facesStart:facesEnd])
    node = Node_raw(sphere=bs[1:17],
                    num = num_faces,
                    polys=faces,
                    position=pos)
    return (faces,bs[facesEnd:],node,pos+facesEnd)


########## /////////////////////////////////////////////////////

def prep_face_refs(nodeBytes):
    faceRefs = []
    position = 0
    while len(nodeBytes):
        nodend_type = nodeBytes[0]
        if nodend_type == 4:
            faceRefs.append([])
            process = node_subobject
        elif nodend_type == 3:
            process = node_vcall
        elif nodend_type == 2:
            process = node_call
        elif nodend_type == 1:
            process = node_split
        elif nodend_type == 0:
            process = node_raw
        else:
            return
        faces, newNodeBytes, node, position= process(nodeBytes, position)
        nodeBytes = newNodeBytes
        faceRefs[-1].extend(faces)
    return faceRefs

def prep_subobjects(subBytes, faceRefs, faces, materials, vhots):
    subs = []
    index = 0
    while len(subBytes):
        sub = SubobjectImported(
            subBytes[:93],
            faceRefs[index],
            faces,
            materials,
            vhots)
        subs.append(sub)
        index += 1
        subBytes = subBytes[93:]
    return subs

if __name__ == "__main__":
    try:
        import bpy
        paths = [r"C:\Spiele\Dark Projekt\MYMODS\UPPERMODS\obj",
                 r"C:\Spiele\Dark Projekt\MODS\ep\obj"]
        file_found = False
        while len(paths) and not file_found:
            file_path = paths.pop(-1) + "\\" + bpy.context.object.users_collection[0].name + ".bin"
            file_found = os.path.exists(file_path)
    except ImportError:
        file_found = False
        
    if not file_found:
        file_path=r"C:\Spiele\Dark Projekt\MYMODS\UPPERMODS\obj\RenderTest2.bin"
        #file_path=r"C:\Users\Daniel\AppData\Local\Temp\untitled.bin"
        #
        file_path=r"C:\Spiele\Dark Projekt\MYMODS\UPPERMODS\obj\RenderTestbsp2.bin"
        #file_path=r"C:\Spiele\Dark Projekt\MYMODS\UPPERMODS\obj\tree3div.bin"
        z = input("Choose file_path:")
        if z != "":
            if len(z) < 20:
                file_path = r"C:\Spiele\Dark Projekt\MYMODS\UPPERMODS\obj\\" +z
            else:
                file_path = z
    else:
        print("FILE FOUND")

    with open(file_path, 'r+b') as binData:
        binBytes = binData.read(-1)
        nd_typeID = binBytes[:4]
        version = unpack('<I', binBytes[4:8])[0]
        name = unpack('<8s', binBytes[8:16])[0].decode('ASCII')

        radius = unpack('<f', binBytes[16:20])
        max_pgon_radius = unpack('<f', binBytes[20:24])
        bmax = unpack('<fff', binBytes[24:36])
        bmin = unpack('<fff', binBytes[36:48])
        bbox = get_floats(binBytes[24:48])
        center = unpack('<fff', binBytes[48:60])
        pgons, num_verts, parms = unpack('<HHH', binBytes[60:66])          
        numMats = binBytes[66]
        vcalls = binBytes[67]
        vhots= binBytes[68]
        num_subs= binBytes[69]
        
        # 9 * 4 = 36
        subobjOffset,\
        matOffset,\
        uvOffset,\
        vhotOffset,\
        vertOffset,\
        lightOffset,\
        normOffset,\
        faceOffset,\
        nodeOffset = get_uints(binBytes[70:106])
        # How many bytes readable
        mod_size   = unpack('<L', binBytes[106:110])
        mat_flags  = unpack('<L', binBytes[110:114])
        amat_off   = unpack('<L', binBytes[118:122])    # this is 16 -> 
        amat_size  = unpack('<L', binBytes[122:126])
        # These seam wrong, cant be > 132!
        mesh_off, \
        submeshlist_off, \
        meshes = unpack('<LLH', binBytes[126:136])
        materials  = prep_materials(binBytes[matOffset:uvOffset], numMats, file_path)
        uvs        = prep_uvs(binBytes[uvOffset:vhotOffset])
        vhots      = prep_vhots(binBytes[vhotOffset:vertOffset])
        verts      = prep_verts(binBytes[vertOffset:lightOffset])
        faces      = prep_faces(binBytes[faceOffset:nodeOffset], version)
        faceRefs  = prep_face_refs(binBytes[nodeOffset:])

        Nodes = Node.AllNodes
        Nodes.sort(key=lambda n: n.nd_index)   

        NodeDict = {}
        for node in Nodes:
            NodeDict[node.position] = node
                
        if True:
            for node in Nodes:
                if node.nd_type == 1:
                    node._node_behind = NodeDict[node.node_behind]
                    node._node_front = NodeDict[node.node_front]
                    NodeDict[node.node_front].parent = node.position
                    NodeDict[node.node_behind].parent= node.position
                elif node.nd_type == 2:
                    NodeDict[node.node_call].parent = node.position
                elif type(node) == Node_sub:   
                    node.main_sub_node = NodeDict[node.position+3]
                    node.main_sub_node.parent = node.position
                    if node.position == 0:
                        node.parent = -1
                print(node,"\n" + ("="*20))
        subobjects = prep_subobjects(
        binBytes[subobjOffset:matOffset],
        faceRefs,
        faces,
        materials,
        vhots)
        object_data = (bbox,subobjects,verts,uvs,materials)
        


    #for i in range((faceOffset-normOffset)//12):
    #    if i not in Norms:
    #        Norms[i] = getNormalByIndex(i)
            
        

    # /////////////////////////////BLENDER///////////////////////////////////////////////
    try:
        import bpy
    except ImportError:
        pass
    else:
        obj = parent_obj = bpy.context.object
        bpy.ops.object.mode_set(mode='OBJECT')
        adjustor = 0
        for n in Nodes:
            if  type(n) == Node_sub and n.parent != -1:
                adjustor += len(obj.data.polygons)
                obj = parent_obj.children[n.index - 1]
        
            if hasattr(n, "faces"):
                if hasattr(n, "pgons_before"):
                    i_max = n.pgons_before
                    vg_name = f"{n.nd_index}_{n.__class__.__name__}_Before"
                else:
                    i_max = len(n.faces)
                    vg_name = f"{n.nd_index}_{n.__class__.__name__}"
                if vg_name in obj.face_maps:
                    vg = obj.face_maps[vg_name]
                else:
                    vg = obj.face_maps.new(name=vg_name)
                vg.add([f-adjustor for f in n.faces[:i_max]])
                if i_max < len(n.faces):
                    vg_name = f"{n.nd_index}_{n.__class__.__name__}_After"
                    if vg_name in obj.face_maps:
                        vg = obj.face_maps[vg_name]
                    else:
                        vg = obj.face_maps.new(name=vg_name)
                    vg.add([f-adjustor for f in n.faces[i_max:]])
        
        # Create Split planes if addon is present
        if True and hasattr(bpy.ops.mesh, "primitive_z_function_surface"):
            for n in Nodes:
                if type(n) == Node_split:
                    d = n.d
                    normal = n.normal
                    if normal[2] == 0:
                        z = 0.01 # Division by 0
                    else:
                        z = normal[2]
                    equation=f"(-{d} - ({normal[1]}*y) -( {normal[0]}*x)) / {z}"
                    bpy.ops.mesh.primitive_z_function_surface(
                        equation=equation,
                        div_x=3, div_y=3, size_x=20, size_y=20)
                    split_plane = bpy.context.object
                    split_plane.name = f"({n.nd_index})_Split: {equation}"
                    
        

    # /////////////////////////JUPYTER//////////////////////////////////////////////////
    try:
        import igraph
    except ImportError:
        pass
    else:
        import numpy as np
        Nodes = Node.AllNodes
        Nodes.sort(key=lambda n: n.nd_index)

        from igraph import Graph
        nr_vertices = len(Nodes)
        #G = Graph.Tree(nr_vertices, 2) # 2 stands for children number
        G = Graph(n=nr_vertices, directed=True)


        #v_label = list(map(str, range(nr_vertices)))
        v_label = [("(" + str(n.nd_index) + ") " + n.__class__.__name__) for n in Node.AllNodes]

        def NodeFacesToStr(n):
            s = ""
            if not hasattr(n, "faces"):
                return s
            s += "\n"
            if hasattr(n, "pgons_before"):
                i_max = n.pgons_before
            else:
                i_max = len(n.faces)
            for i in range(i_max):
                s+= str(n.faces[i]) +", "
            if i_max < len(n.faces):
                s+= " || "
                for i in range(i_max, len(n.faces)):
                    s+= str(n.faces[i]) +", "
            return s[:-2]
            
        labels = [NodeFacesToStr(n) for n in Nodes]
        es_labels = []
        nd_types = [""] * nr_vertices
        nd_normals = [""] * nr_vertices
        nd_normal_ids = [""] * nr_vertices
        nd_d = [""] * nr_vertices

        for node in Nodes:
            for k,v in node.__dict__.items():
                if isinstance(v, Node) and k != "parent":
                    G.add_edges([(node.nd_index, v.nd_index)])
                    nd_types[v.nd_index] = k
            if type(node) == Node_split:
                nd_normals[node.nd_index] = f"{node.normal[0]:.3f}, {node.normal[1]:.3f}, {node.normal[2]:.3f}"
                nd_d[node.nd_index] = f"{node.d:.3f}"
                nd_normal_ids[node.nd_index] =node.norm_id
                
        data = (np.vstack(v_label), 
                np.vstack(nd_types), 
                np.vstack(labels),
                np.vstack(nd_normals),
                np.vstack(nd_d),
                np.vstack(nd_normal_ids)
               )        

        lay = G.layout('rt')

        position = {k: lay[k] for k in range(nr_vertices)}
        Y = [lay[k][1] for k in range(nr_vertices)]
        M = max(Y)

        #es = EdgeSeq(G) # sequence of edges
        E = [e.tuple for e in G.es] # list of edges
        L = len(position)
        Xn = [position[k][0] for k in range(L)]
        Yn = [2*M-position[k][1] for k in range(L)]
        Xe = []
        Ye = []
        for edge in E:
            Xe+=[position[edge[0]][0],position[edge[1]][0], None]
            Ye+=[2*M-position[edge[0]][1],2*M-position[edge[1]][1], None]

        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=Xe,
                           y=Ye,
                           mode='lines',
                            name='edge',
                           line=dict(color='rgb(210,210,210)', width=1),
                           text=es_labels,
                           hoverinfo='text',
                           opacity = 1
                           ))
        fig.add_trace(go.Scatter(x=Xn,
                          y=Yn,
                          mode='markers',
                          name ="",
                          marker=dict(symbol='square',
                                        size=50,
                                        color='#6175c1',    #'#DB4551',
                                        line=dict(color='rgb(50,50,50)', width=1)
                                        ),
                          text=labels,
                          #hoverinfo='text',
                          opacity=0.8,
                          customdata = np.dstack(data),
                          hovertemplate = "%{customdata[0][0]} - %{customdata[0][1]}<br>"
                                         +"-----<br>%{customdata[0][2]}"
                                         +"<br>norm(%{customdata[0][5]}):\t%{customdata[0][3]}"
                                         +"<br>\td:\t\t%{customdata[0][4]}"
                          ))



        def make_annotations(pos, text, font_size=10, font_color='rgb(250,250,250)'):
            L=len(pos)
            if len(text)!=L:
                raise ValueError('The lists pos and text must have the same len')
            annotations = []
            for k in range(L):
                annotations.append(
                    dict(
                        text=v_label[k], # or replace labels with a different list for the text within the circle
                        x=pos[k][0], y=2*M-position[k][1],
                        xref='x1', yref='y1',
                        font=dict(color=font_color, size=font_size),
                        showarrow=False)
                )
            return annotations

        axis = dict(showline=False, # hide axis line, grid, ticklabels and  title
                    zeroline=False,
                    showgrid=False,
                    showticklabels=False,
                    )

        fig.update_layout(title= file_path,
                      annotations=make_annotations(position, v_label),
                      font_size=12,
                      showlegend=False,
                      xaxis=axis,
                      yaxis=axis,
                      margin=dict(l=40, r=40, b=85, t=100),
                      hovermode='closest',
                      plot_bgcolor='rgb(0,0,0)'
                      )
        fig.show()