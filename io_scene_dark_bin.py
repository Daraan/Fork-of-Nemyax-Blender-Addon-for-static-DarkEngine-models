bl_info = {
    "name": "Dark Engine Static Model",
    "author": "nemyax",
    "version": (0, 5, 20201020.7), # Using YMD
    "blender": (2, 83, 7),
    "location": "File > Import-Export",
    "description": "Import and export Dark Engine static model .bin",
    "warning": "inofficial version",
    "wiki_url": "https://sourceforge.net/p/blenderbitsbobs/wiki/Dark%20Engine%20model%20importer-exporter/",
    "tracker_url": "",
    "category": "Import-Export"
}

import bpy
import bmesh
import mathutils as mu
import math
import re
import struct
import os
import glob
from struct import pack, unpack
from bpy.props import (
    StringProperty,
    EnumProperty,
    BoolProperty)
from bpy_extras.io_utils import (
    ExportHelper,
    ImportHelper,
    path_reference_mode)

###
### Import
###

class FaceImported:
    binVerts   = []
    binUVs     = []
    binMat     = None
    bmeshVerts = []

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
            return mu.Matrix.Identity(4)
        else:
            matrix = mu.Matrix()
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
        uvs.append(mu.Vector((u,v)))
    return uvs

def prep_faces(faceBytes, version):
    garbage = 9 + version # magic 12 or 13: v4 has an extra byte at the end
    faces = {}
    faceAddr = 0
    faceIndex = 0
    while len(faceBytes):
        matIndex = unpack('<H', faceBytes[2:4])[0]
        type = faceBytes[4] & 3
        num_verts = faceBytes[5]
        verts = get_ushorts(faceBytes[12:12+num_verts*2])
        uvs = []
        if type == 3:
            faceEnd = garbage + num_verts * 6
            uvs.extend(get_ushorts(faceBytes[12+num_verts*4:12+num_verts*6]))
        else:
            faceEnd = garbage + num_verts * 4
            matIndex = None
        face = FaceImported()
        face.binVerts = verts
        face.binUVs = uvs
        face.binMat = matIndex
        faces[faceAddr] = face
        faceAddr += faceEnd
        faceIndex += 1
        faceBytes = faceBytes[faceEnd:]
    return faces

def node_subobject(bs):
    return ([],bs[3:])

def node_vcall(bs):
    return ([],bs[19:])

def node_call(bs):
    facesStart = 23
    num_faces1 = unpack('<H', bs[17:19])[0]
    num_faces2 = unpack('<H', bs[21:facesStart])[0]
    facesEnd = facesStart + (num_faces1 + num_faces2) * 2
    faces = get_ushorts(bs[facesStart:facesEnd])
    return (faces,bs[facesEnd:])

def node_split(bs):
    facesStart = 31
    num_faces1 = unpack('<H', bs[17:19])[0]
    num_faces2 = unpack('<H', bs[29:facesStart])[0]
    facesEnd = facesStart + (num_faces1 + num_faces2) * 2
    faces = get_ushorts(bs[facesStart:facesEnd])
    return (faces,bs[facesEnd:])

def node_raw(bs):
    facesStart = 19
    num_faces = unpack('<H', bs[17:facesStart])[0]
    facesEnd = facesStart + num_faces * 2
    faces = get_ushorts(bs[facesStart:facesEnd])
    return (faces,bs[facesEnd:])

def prep_face_refs(nodeBytes):
    faceRefs = []
    while len(nodeBytes):
        nodeType = nodeBytes[0]
        if nodeType == 4:
            faceRefs.append([])
            process = node_subobject
        elif nodeType == 3:
            process = node_vcall
        elif nodeType == 2:
            process = node_call
        elif nodeType == 1:
            process = node_split
        elif nodeType == 0:
            process = node_raw
        else:
            return
        faces, newNodeBytes = process(nodeBytes)
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

def parse_bin(binBytes, file_path):
    version = unpack('<I', binBytes[4:8])[0]
    bbox = get_floats(binBytes[24:48])
    numMats = binBytes[66]
    subobjOffset,\
    matOffset,\
    uvOffset,\
    vhotOffset,\
    vertOffset,\
    lightOffset,\
    normOffset,\
    faceOffset,\
    nodeOffset = get_uints(binBytes[70:106])
    materials  = prep_materials(binBytes[matOffset:uvOffset], numMats, file_path)
    uvs        = prep_uvs(binBytes[uvOffset:vhotOffset])
    vhots      = prep_vhots(binBytes[vhotOffset:vertOffset])
    verts      = prep_verts(binBytes[vertOffset:lightOffset])
    faces      = prep_faces(binBytes[faceOffset:nodeOffset], version)
    faceRefs   = prep_face_refs(binBytes[nodeOffset:])
    subobjects = prep_subobjects(
        binBytes[subobjOffset:matOffset],
        faceRefs,
        faces,
        materials,
        vhots)
    return (bbox,subobjects,verts,uvs,materials)

def build_bmesh(bm, sub, verts):
    faces = sub.faces
    for v in verts:
        bm.verts.new(v[1])
        bm.verts.index_update()
    for f in faces:
        bmVerts = []
        for oldIndex in f.binVerts:
            newIndex = aka(oldIndex, verts)[0]
            bm.verts.ensure_lookup_table()
            bmVerts.append(bm.verts[newIndex])
        bmVerts.reverse() # flip normal
        try:
            bm.faces.new(bmVerts)
            f.bmeshVerts = bmVerts
        except ValueError:
            extraVerts = []
            for oldIndex in reversed(f.binVerts):
                sameCoords = aka(oldIndex, verts)[1][1]
                ev = bm.verts.new(sameCoords)
                bm.verts.index_update()
                extraVerts.append(ev)
            bm.faces.new(extraVerts)
            f.bmeshVerts = extraVerts
        bm.faces.index_update()
    bm.faces.ensure_lookup_table()
    for i in range(len(faces)):
        bmFace = bm.faces[i]
        binFace = faces[i]
        mi = sub.matSlotIndexFor(binFace.binMat)
        if mi != None:
            bmFace.material_index = sub.matSlotIndexFor(binFace.binMat)
    bm.edges.index_update()
    return

def assign_uvs(bm, faces, uvs):
    bm.loops.layers.uv.new()
    uvData = bm.loops.layers.uv.active
    for x in range(len(bm.faces)):
        bmFace = bm.faces[x]
        binFace = faces[x]
        loops = bmFace.loops
        binUVs = binFace.binUVs
        binUVs.reverse() # to match the face's vert direction
        for i in range(len(binUVs)):
            loop = loops[i]
            u, v = uvs[binUVs[i]]
            loop[uvData].uv = (u,1-v)
    return

def make_mesh(subobject, verts, uvs):
    faces = subobject.faces
    vertsSubset = []
    for f in faces:
        vertsSubset.extend([aka(v, verts)[1] for v in f.binVerts])
    vertsSubset = list(set(vertsSubset))
    bm = bmesh.new()
    build_bmesh(bm, subobject, vertsSubset)
    assign_uvs(bm, faces, uvs)
    mesh = bpy.data.meshes.new(subobject.name)
    bm.to_mesh(mesh)
    bm.free()
    return mesh

def parent_index(index, subobjects):
    for i in range(len(subobjects)):
        if subobjects[i].next == index:
            return parent_index(i, subobjects)
        elif subobjects[i].child == index:
            return i
    return -1

# NEW adding bbox to new collection if enabled.
def make_bbox(coords, collection):
    bm = bmesh.new()
    v1 = bm.verts.new(coords[:3])
    v2 = bm.verts.new(coords[3:])
    e = bm.edges.new((v1, v2))
    mesh = bpy.data.meshes.new("bbox")
    bm.to_mesh(mesh)
    bm.free()
    bbox = bpy.data.objects.new(name="bbox", object_data=mesh)
    bbox.display_type = 'BOUNDS'
    collection.objects.link(bbox)
    return bbox

# NEW discarding file name extension for import.
# Always creating a generated texture with UV_GRID image if not found
# NEW : Returns image not texture!
def load_img(file_path, img_name, options):
    dir_path = os.path.dirname(file_path)
    # File name without extension
    img_noext, ext = os.path.splitext(img_name.lower())
    # Special for replace#
    if options['fancy_txtrepl'] and img_noext.startswith('replace'):
        try:
            num = int(img_noext[7]) # This could fail
            if not img_name in bpy.data.images:
                # Create fancy texture
                bpy.ops.image.new(name=img_name, generated_type='COLOR_GRID')
            img = bpy.data.images[img_name]
            # This is just visual to differentiate between the textures.
            if img_noext[-1] == "1":
                img.colorspace_settings.name = 'Filmic Log'
            elif img_noext[-1] == "2":
                img.colorspace_settings.name = 'Linear'
            return img
        except ValueError:
            pass # Fall back to standard import
    
    ps       = os.sep
    files    = []
    for n in (glob.glob(dir_path + ps + "*")):
        nl = os.path.split(n)[-1].lower()
        if ((nl == "txt") or (nl == "txt16")) and os.path.isdir(n):
            files.extend(glob.glob(n + ps + "*.???"))
    if not files:
        print("No txt folder or image file found. Using empty image for", img_name)
        bpy.ops.image.new(name=img_name, generated_type='UV_GRID')
        return bpy.data.images[img_name]
    img_file = None
    for f in files:
        fbase = os.path.splitext(os.path.split(f)[-1])[0]
        # Problem this could be blustn.png vs blustn
        # Using material / image without extension.
        if img_noext == fbase.lower():
            img_file = f
            break
    if not img_file:
        print(img_name, "not found in txt subfolders. Using empty image.")
        img = bpy.ops.image.new(name=img_name, generated_type='UV_GRID')
        return bpy.data.images[img_name]
    # Textures are not used anymore still keeping this around for a bit longer
    for t in bpy.data.textures:
        if t.type == 'IMAGE' and t.image != None and t.image.filepath == img_file:
            return t.image
    img = None
    # Already loaded images
    for i in bpy.data.images:
        if i.filepath == img_file:
            img = i
            break
    # Image already present    
    if img:
        return img
    # Making sure that the found texture is not a gif in case one was searched for.
    ext_found = os.path.splitext(img_file.lower())[1]
    if ext_found != ".gif" or not options['convert_gif'] or ext_found != ".gif" :
        if ext == ".gif":
            print(img_name, "was gif but supported format", ext_found, "found in txt folder.")
        return bpy.data.images.load(img_file)   
    # Do gif conversion
    try:
        #Convert to png and import temporary file.
        from gif2png import convert as gif2png
        filename = gif2png(img_file, bpy.app.tempdir+"temp.png")
        try:
            # This should not fail but in case.
            img = bpy.data.images.load(filename)
        except Exception as e:
            print(e, "\nConversion of",img_file,"to", filename,"was successful but can't load file.")
            raise e
    except Exception as e:
        print(e, "Importer could not convert gif:", img_file, "Using blank image.")
        bpy.ops.image.new(name=img_name, generated_type='BLANK')
        img = bpy.data.images[img_name]
    else:
        print(img_name, 'found. Converted and packed as png.')
        img.pack()
        img.name = img_name
        img.source = 'FILE'
        img.filepath = img_file # So the above bpy.data.images check works.
    return img
            
def create_texture(img):
    # Optional textures are not needed anymore but good to manage images.
    if "TEX for " + img.name.lower() in bpy.data.textures:
        return bpy.data.textures["TEX for " + img.name.lower()]
    tex = bpy.data.textures.new(name="TEX for " + img.name.lower(), type='IMAGE')
    tex.image = img
    tex.use_fake_user = True
    return tex

def make_objects(object_data, file_path, options):
    bbox, subobjects, verts, uvs, mats = object_data
    objs = []
    # Making a collection out of the filename, alternative to old layers
    if (options['use_collections']):
        collection = bpy.data.collections.new(os.path.splitext(os.path.split(file_path)[-1])[0])
        bpy.context.scene.collection.children.link(collection)
    else:
        collection = bpy.context.scene.collection
   
    for s in subobjects:
        mesh = make_mesh(s, verts, uvs)
        obj = bpy.data.objects.new(name=mesh.name, object_data=mesh)
        obj.matrix_local = s.localMatrix()
        collection.objects.link(obj)
        
        if (s.motion):
            # For better visualization of the rotation axis
            if s.motion == 1:
                limits = obj.constraints.new(type='LIMIT_ROTATION')
                limits.owner_space = 'LOCAL'
                limits.min_x = s.min
                limits.max_x = s.max
                oldschool_char = "@x"
            elif s.motion == 2:
                limits = obj.constraints.new(type='LIMIT_LOCATION')
                limits.owner_space = 'LOCAL'
                limits.min_x = s.min
                limits.max_x = s.max
                oldschool_char = "@z"
            if options['support_3ds_export']:
                # Create axle for old school support
                # XX must be replaced by parent but we don't have that info yet.
                # TODO: Angles in Gon/4?
                min = int(s.min / math.pi * 50) or "00"
                max = int(s.max / math.pi * 50) or "00"
                mesh = bpy.data.meshes.new(oldschool_char 
                                            + obj.name[2:4] 
                                            + "XX"
                                            + str(max) + str(min))
                axis = bpy.data.objects.new(mesh.name, mesh)
                inv = obj.matrix_world.copy()
                inv.invert()
                # Vector as long as X, to be tested.
                trans = mu.Vector([obj.dimensions[0], 0.0, 0.0]) @ inv
                mesh.from_pydata([obj.matrix_world.translation, obj.matrix_world.translation + trans], [(0,1)], [])
                axis.parent = obj
                axis.show_axis = True
                axis.show_in_front = True
                collection.objects.link(axis)
            else:
                obj.show_axis = True
                
        for v in s.vhots:
            if not options['support_3ds_export']:
                vhot_name = s.name + "-vhot-" + str(v[0])
                vhot = bpy.data.objects.new(vhot_name, None)
                # Purely visual
                vhot.empty_display_type = 'CUBE'
                vhot.empty_display_size = 0.075
            else:
                # Create as a cube to for (future) 3ds to bin
                vhot_name = "@h" + ("0" + str(v[0]) if v[0] < 10 else str(v[0])) + s.name
                mesh = bpy.data.meshes.new(vhot_name)
                vhot = bpy.data.objects.new(vhot_name, mesh)
                # Create as cube
                bm = bmesh.new()
                bmesh.ops.create_cube(bm, size=0.075)
                bm.to_mesh(mesh)
                bm.free()
                vhot.display_type = 'BOUNDS'
                
            collection.objects.link(vhot)
            vhot.parent = obj
            vhot.location = v[1]
            vhot.show_in_front = True
                
        # Only select new object.
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT') # initializes UVmap correctly
        mesh.uv_layers.new()
        for m in s.matsUsed.values():
            bpy.ops.object.material_slot_add()
            mat = None
            mat_name = m[0]
            for a in bpy.data.materials:
                if a.name == mat_name:
                    mat = a
                    break
            if not mat:
                mat = bpy.data.materials.new(mat_name)
                
            # NEW NODE IMPORT
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            # This is the main node for the interface Surface
            shaders = nodes['Principled BSDF'].inputs
            
            # 0 is no alpha value => opaque => alpha = 1
            if m[1] == 0.0:
                shaders["Alpha"].default_value = 1.0
            else:
                shaders["Alpha"].default_value = m[1]
            # For emission use nodes
            if (not shaders['Emission'].is_linked):
                # Need a shader to RGB so it is displayed correctly
                emissionNode = nodes.new('ShaderNodeEmission')
                emissionNode.inputs['Strength'].default_value = m[2]
                converter = nodes.new('ShaderNodeShaderToRGB')
                mat.node_tree.links.new( emissionNode.outputs['Emission'], converter.inputs['Shader'])
                mat.node_tree.links.new( converter.outputs['Color'], shaders['Emission'])
                # Visual stuff
                converter.hide = True
                converter.location = [-150, -170]
                emissionNode.location = [-350, -160]
                emissionNode.width = 200
                emissionNode.label = "Emission: Strength value only"
            
            # Add image to new material
            if (not shaders["Base Color"].is_linked
                or shaders["Base Color"].links[0].from_node.image == None): # This allows reimport after deletion
                if (shaders["Base Color"].is_linked):
                    texNode = shaders["Base Color"].links[0].from_node
                else:
                    texNode = nodes.new('ShaderNodeTexImage')
                    texNode.location = [-350, 350]
                texNode.image = load_img(file_path, mat_name, options) # Always returns an image
                # Might be useful for management.
                create_texture(texNode.image)
                if (texNode.image.type == 'UV_TEST'):
                    texNode.label = "IMAGE NOT FOUND"
                mat.node_tree.links.new( texNode.outputs['Color'], shaders["Base Color"] )
            
            # End of new stuff apply mat to object
            obj.material_slots[-1].material = mat
        # Object Materials done
        objs.append(obj)
    for i in range(len(subobjects)):
        mum = parent_index(i, subobjects)
        if mum >= 0:
            objs[i].parent = objs[mum]
    make_bbox(bbox, collection)
    return {'FINISHED'}

def do_import(file_path, options):
    binData = open(file_path, 'r+b')
    binBytes = binData.read(-1)
    typeID = binBytes[:4]
    if typeID == b'LGMD':
        object_data = parse_bin(binBytes, file_path)
        msg = "File \"" + file_path + "\" loaded successfully."
        result = make_objects(object_data, file_path, options)
    elif typeID == b'LGMM':
        msg = "The Dark Engine AI mesh format is not supported."
        result = {'CANCELLED'}
    else:
        msg = "Cannot understand the file format."
        result = {'CANCELLED'}
    return (msg,result)

###
### Export
###

# Joints have to be named @_##Name the second character
re_joint = re.compile(r"@\D*(?P<number>\d+)(?P<name>.*)", re.IGNORECASE)

# Classes

class Subobject(object):
    def __init__(self,
        name,
        mesh,
        matrix,
        motion_type,
        motion_min,
        motion_max,
        vhots):
        self.name        = name
        self.mesh        = mesh
        self.matrix      = matrix
        self.motion_type = motion_type
        self.min         = motion_min
        self.max         = motion_max
        self.parent      = None
        self.children    = []
        self.index       = 0
        self.joint_id    = -1
        self.next        = None
        self.call        = None
        self.vhots       = vhots
    def get_root(self):
        par = self.parent
        if par:
            return par.get_root()
        return self
    def list_subtree(self):
        result = [self]
        for c in self.children:
            result.extend(c.list_subtree())
        return result
    def set_parent(self, new_par):
        self.parent = new_par
        # EXPERIMENTAL
        # Back to the old method append method. Don't forget to change in Model class
        if (True):
            if new_par.children:
                self.next = new_par.children[-1]
            #new_par.children.insert(0, self) # old
            new_par.children.append(self)
            if len(new_par.children) > 1: # "splits" instead of "call"
                new_par.call = None
            else: # no "splits"
                new_par.call = new_par.children[-1]
        else:
            if new_par.children:
                self.next = new_par.children[0]
            new_par.children.insert(0, self) # old
            if len(new_par.children) > 1: # "splits" instead of "call"
                new_par.call = None
            else: # no "splits"
                new_par.call = new_par.children[0]
        # Reindex entire tree
        flat_tree = self.get_root().list_subtree()
        for s, i in zip(flat_tree, range(len(flat_tree))):
            s.index = i

class Model(object):
    def __init__(self, root_sub, materials, bbox, clear, bright, use_node_image):
        subs   = root_sub.list_subtree()
        meshes = [s.mesh for s in subs]
        vhots  = [s.vhots for s in subs]
        num_vs = num_uvs = num_lts = num_fs = num_ns = 0
        for bm in meshes:
            bm.edges.ensure_lookup_table()
            ext_e = bm.edges.layers.string.active
            num_vs0, num_uvs0, num_lts0, num_ns0, num_fs0 = \
                unpack('<5H', bm.edges[0][ext_e])
            num_vs  += num_vs0
            num_uvs += num_uvs0
            num_lts += num_lts0
            num_ns  += num_ns0
            num_fs  += num_fs0
        self.num_vhots       = deep_count(vhots)
        self.num_faces       = num_fs
        self.num_verts       = num_vs
        self.num_normals     = num_ns
        self.num_uvs         = num_uvs
        self.num_lights      = num_lts
        self.num_meshes      = len(meshes)
        self.num_mats        = len(materials)
        self.bbox            = bbox
        self.max_poly_radius = max([max_poly_radius(m) for m in meshes])
        mat_flags = 0
        if clear:
            mat_flags += 1
        if bright:
            mat_flags += 2
        self.mat_flags = mat_flags
        # Encode verts
        verts_bs = b''
        for bm in meshes:
            ext_v = bm.verts.layers.string.active
            verts_bs += concat_bytes([o[2:] for o
                in sorted(set([v[ext_v] for v in bm.verts]))])
        self.verts_bs = verts_bs
        # Encode UVs
        uvs_bs = b''
        for bm in meshes:
            ext_l = bm.loops.layers.string.active
            uv_set = set()
            for f in bm.faces:
                for l in f.loops:
                    uv_set.add(l[ext_l][:10])
            uvs_bs += concat_bytes([o[2:] for o in sorted(uv_set)])
        self.uvs_bs = uvs_bs
        # Encode lights 
        lights_bs = b''
        for bm in meshes:
            ext_l = bm.loops.layers.string.active
            lt_set = set()
            for f in bm.faces:
                for l in f.loops:
                    lt_set.add(l[ext_l][10:])
            lights_bs += concat_bytes([o[2:] for o in sorted(lt_set)])
        self.lights_bs = lights_bs
        # Encode vhots
        vhot_list = []
        for mi in range(len(vhots)):
            currentVhots = vhots[mi]
            offset = deep_count(vhots[:mi])
            for ai in range(len(currentVhots)):
                id, coords = currentVhots[ai]
                vhot_list.append(concat_bytes([
                    pack('<I', id),
                    encode_floats(coords[:])]))
        self.vhots_bs = concat_bytes(vhot_list)
        # Encode normals
        normals_bs = b''
        for bm in meshes:
            ext_f = bm.faces.layers.string.active
            normals_bs += concat_bytes([o[2:] for o
                in sorted(set([f[ext_f][:14] for f in bm.faces]))])
        self.normals_bs = normals_bs
        # Encode faces
        face_lists = []
        faces_bs = b''
        for bm in meshes:
            ext_f = bm.faces.layers.string.active
            fs = [f[ext_f][14:] for f in bm.faces]
            face_lists.append(fs)
            faces_bs += concat_bytes(fs)
        self.faces_bs = faces_bs
        # Encode nodes
        addr        = 0
        node_sizes  = []
        addr_chunks = []
        num_subs    = len(face_lists)
        for s in subs:
            bfl = face_lists[s.index]
            node_sizes.append(precalc_node_size(s, bfl))
            ext_face_addrs = b''
            for bf in bfl:
                ext_face_addrs += pack('<H', addr)
                addr += len(bf)
            addr_chunks.append(ext_face_addrs)
        nodes = []
        for s in subs:
            node_bs = b''
            si        = s.index
            node_bs  += encode_sub_node(si) # This uses index too
            sphere_bs = encode_sphere(get_local_bbox_data(meshes[si]))
            call      = s.call
            nbf       = len(face_lists[si])
            if call:
                node_bs += encode_call_node(
                    nbf,
                    sum(node_sizes[:call.index]),
                    sphere_bs,
                    addr_chunks[si])
            elif len(s.children) > 1:
                kids = s.children
                node_offs = [sum(node_sizes[:a])
                    for a in range(len(node_sizes))]
                ns = len(kids)
                if ns > 2: # need to reference split nodes
                    offs1 = [node_offs[i.index] for i in kids]
                    offs2 = []
                    pos = node_offs[si] + 34 + nbf * 2
                    for _ in range(ns - 2):
                        offs2.append(pos)
                        pos += 31
                    offs2.append(offs1.pop())
                    f_counts = [nbf] + [0] * (ns - 1)
                    ac_list  = [addr_chunks[si]] + [b''] * (ns - 1)
                    for nf, addr_chunk, off1, off2 in zip(
                        f_counts, ac_list, offs1, offs2):
                        node_bs += encode_split_node(
                            nf, sphere_bs, off1, off2, addr_chunk)
                else:
                    off1, off2 = [node_offs[k.index] for k in kids]
                    node_bs += encode_split_node(
                        nbf, sphere_bs, off1, off2, addr_chunks[si])
            else:
                node_bs += encode_raw_node(nbf, sphere_bs, addr_chunks[si])
            nodes.append(node_bs)
        node_offs = [sum(node_sizes[:i+1]) for i in range(len(node_sizes))]
        node_offs.insert(0, 0)
        self.nodes_bs = concat_bytes(nodes)
        self.node_offsets = node_offs
        # Encode materials
        mat_names = []
        if materials:
            for m in materials:
                if not m: continue
                # Use the texture of the texture node instead of material name
                if use_node_image:
                    image_name = None
                    input_links = m.node_tree.nodes['Principled BSDF'].inputs["Base Color"].links
                    # Input link exists
                    if (len(input_links)):
                        inputNode = input_links[0].from_node
                        # Type is only TexImage (depracted) using bl_idname
                        if (inputNode.bl_idname == 'ShaderNodeTexImage' and inputNode.image):
                            image_name = inputNode.image.name
                    if image_name:
                        mat_names.append(image_name)                        
                    else:
                        print("No image node or image found for", m.name)
                        mat_names.append(m.name)
                else: 
                    mat_names.append(m.name)
        else:
            mat_names.append("oh_bugger")
        final_names = encode_names(mat_names, 16)
        self.materials_bs = concat_bytes(
            [encode_material(final_names[i], i)
                for i in range(len(mat_names))])
        # Encode auxiliary data
        if materials:
            aux_data_bs = b''
            for m in materials:
                shaders = m.node_tree.nodes['Principled BSDF'].inputs
                # Use emission node if present else use Value = max(R, G, B) as emission
                if shaders['Emission'].is_linked:
                    inputNode = shaders['Emission'].links[0].from_node
                    # Rerouted via ShaderToRGB?
                    if (inputNode.bl_idname == 'ShaderNodeShaderToRGB'):
                        inputNode = inputNode.inputs['Shader'].links[0].from_node
                    emission = inputNode.inputs['Strength'].default_value
                else:
                    emission = max(shaders["Emission"].default_value[:-1])
                # Alpha = 1 is fully visible, as is alpha = 0! Low values are more transparent.
                alpha = shaders["Alpha"].default_value
                aux_data_bs += encode_floats([
                    alpha,
                    min([1.0, emission])
                    ])
            self.aux_data_bs = aux_data_bs
        else:
            self.aux_data_bs = bytes(8)
        # Encode subobjects
        subs_bs = b''
        sub_names = encode_names([s.name for s in subs], 8)
        self.names = sub_names
        for sub in subs:
            index = sub.index
            # NEW v6
            # This is the joint number, -1 means it will not move / it moves relative to it's parent
            # regexp catches @s##__ the number
            isjoint = re_joint.match(sub.name)
            jointid = int(isjoint['number']) if isjoint else -1
            vhot_off = deep_count(vhots[:index])
            num_vhots = len(sub.vhots)
            bm = sub.mesh
            bm.edges.ensure_lookup_table()
            ext_e = bm.edges.layers.string.active
            num_vs, num_lts, num_ns = \
                unpack('<HxxHHxx', bm.edges[0][ext_e])
            v_off, lt_off, n_off = \
                unpack('<3H', bm.edges[1][ext_e])
            xform   = sub.matrix
            if sub.children:
                child = sub.children[-1].index
            else:
                child = -1
            if sub.next:
                sibling = sub.next.index
            else:
                sibling = -1
            if len(sub.children) > 1:
                num_nodes = len(sub.children) - 1
            else:
                num_nodes = 1
            subs_bs += concat_bytes([
                sub_names[sub.index],
                pack('b', sub.motion_type),
                # NEW v6: 
                pack('<i', jointid),
                encode_floats([
                    sub.min,
                    sub.max,
                    xform[0][0],
                    xform[1][0],
                    xform[2][0],
                    xform[0][1],
                    xform[1][1],
                    xform[2][1],
                    xform[0][2],
                    xform[1][2],
                    xform[2][2],
                    xform[0][3],
                    xform[1][3],
                    xform[2][3]]),
                encode_shorts([child,sibling]),
                encode_ushorts([
                    vhot_off,
                    num_vhots,
                    v_off,
                    num_vs,
                    lt_off,
                    num_lts,
                    n_off,
                    num_ns,
                    node_offs[sub.index],
                    num_nodes])])
        self.subs_bs = subs_bs

# Utilities

def strip_wires(bm):
    [bm.verts.remove(v) for v in bm.verts if v.is_wire or not v.link_faces]
    [bm.edges.remove(e) for e in bm.edges if not e.link_faces[:]]
    [bm.faces.remove(f) for f in bm.faces if len(f.edges) < 3]
    for seq in [bm.verts, bm.faces, bm.edges]: seq.index_update()
    for seq in [bm.verts, bm.faces, bm.edges]: seq.ensure_lookup_table()
    return bm

def concat_bytes(bs_list):
    return b"".join(bs_list)

def deep_count(deepList):
    return sum([len(i) for i in deepList])

def encode(fmt, what):
    return concat_bytes([pack(fmt, i) for i in what])

def encode_floats(floats):
    return encode('<f', floats)

def encode_uints(uints):
    return encode('<I', uints)

def encode_ints(ints):
    return encode('<i', ints)

def encode_shorts(shorts):
    return encode('<h', shorts)

def encode_ushorts(ushorts):
    return encode('<H', ushorts)

def encode_ubytes(ubytes):
    return encode('B', ubytes)

def encode_misc(items):
    return concat_bytes([pack(fmt, i) for (fmt, i) in items])

def find_common_bbox(ms, bms):
    xs = set()
    ys = set()
    zs = set()
    for pair in zip(ms, bms):
        matrix, bm = pair
        coords = [matrix @ v.co for v in bm.verts]
        [xs.add(c[0]) for c in coords]
        [ys.add(c[1]) for c in coords]
        [zs.add(c[2]) for c in coords]
    return {min:(min(xs),min(ys),min(zs)),max:(max(xs),max(ys),max(zs))}

def find_d(n, vs):
    nx, ny, nz = n
    count = len(vs)
    vx = sum([v[0] for v in vs]) / count
    vy = sum([v[1] for v in vs]) / count
    vz = sum([v[2] for v in vs]) / count
    return -(nx*vx+ny*vy+nz*vz)

def max_poly_radius(bm):
    diam = 0.0
    for f in bm.faces:
        dists = set()
        vs = f.verts
        for v in vs:
            for x in vs:
                dists.add((v.co-x.co).magnitude)
        diam = max([diam,max(list(dists))])
    return diam * 0.5
        
def calc_center(pts):
    n = len(pts)
    x = sum([pt[0] for pt in pts]) / n
    y = sum([pt[1] for pt in pts]) / n
    z = sum([pt[2] for pt in pts]) / n
    return mu.Vector((x,y,z))

def calc_bbox_center(pts):
    max_xyz = mu.Vector((
        max(a[0] for a in pts),
        max(a[1] for a in pts),
        max(a[2] for a in pts)))
    min_xyz = mu.Vector((
        min(a[0] for a in pts),
        min(a[1] for a in pts),
        min(a[2] for a in pts)))
    return min_xyz + ((max_xyz - min_xyz) * 0.5)

# Other functions

def precalc_node_size(sub, fl):
    size_fs = len(fl) * 2
    if sub.call: # subobject node (3), call node (23) and raw node (19+size_fs)
        return 26 + size_fs
    nch = len(sub.children)
    if nch > 1: # subobject node and some split nodes (31 each) + size_fs
        return 3 + size_fs + 31 * (nch - 1)
    return 22 + size_fs # subobject node and raw node

def encode_sub_node(index):
    return pack('<BH', 4, index)

def encode_call_node(nf, off, sphere_bs, addr_chunk):
    return pack('<B16s3H', 2, sphere_bs, nf, off, 0) + addr_chunk

def encode_raw_node(nf, sphere_bs, addr_chunk):
    return pack('<B16sH', 0, sphere_bs, nf) + addr_chunk

def encode_split_node(nf, sphere_bs, n_back, n_front, addr_chunk):
    return pack('<B16sHHf3H',
        1, sphere_bs, nf, 0, 0, n_back, n_front, 0) + addr_chunk

def pack_light(xyz):
    result = 0
    shift = 22
    for f in xyz:
        val = round(f * 256)
        sign = int(val < 0) * 1024
        result |= (sign + val) << shift
        shift -= 10
    return pack('<I', result)

def encode_header(model, offsets):
    radius = (
        mu.Vector(model.bbox[max]) -\
        mu.Vector(model.bbox[min])).magnitude * 0.5
    return concat_bytes([
        b'LGMD\x04\x00\x00\x00',
        model.names[0],
        pack('<f', radius),
        pack('<f', model.max_poly_radius),
        encode_floats(model.bbox[max]),
        encode_floats(model.bbox[min]),
        bytes(12), # relative centre
        encode_ushorts([
            model.num_faces,
            model.num_verts,
            max(0, model.num_meshes - 1)]), # parms
        encode_ubytes([
            max(1, model.num_mats), # can't be 0
            0, # vcalls
            model.num_vhots,
            model.num_meshes]),
        encode_uints([
            offsets['subs'],
            offsets['mats'],
            offsets['uvs'],
            offsets['vhots'],
            offsets['verts'],
            offsets['lights'],
            offsets['normals'],
            offsets['faces'],
            offsets['nodes'],
            offsets['end'],
            model.mat_flags, # material flags
            offsets['matsAux'],
            8, # bytes per aux material data chunk
            offsets['end'], # ??? mesh_off
            0]), # ??? submesh_list_off
        b'\x00\x00']) # ??? number of meshes

def encode_sphere(bbox): # (min,max), both tuples
    xyz1 = mu.Vector(bbox[0])
    xyz2 = mu.Vector(bbox[1])
    halfDiag = (xyz2 - xyz1) * 0.5
    cx, cy, cz = xyz1 + halfDiag
    radius = halfDiag.magnitude
    return encode_floats([cx,cy,cz,radius])

def encode_names(names, length):
    newNames = []
    for n in names:
        trail = 0
        newName = ascii(n)[1:-1][:length]
        while newName in newNames:
            trail += 1
            trailStr = str(trail)
            newName = newName[:(length - len(trailStr))] + trailStr
        newNames.append(newName)
    binNames = []
    for nn in newNames:
        binName = bytes([ord(c) for c in nn])
        while len(binName) < length:
            binName += b'\x00'
        binNames.append(binName)
    return binNames

def encode_material(binName, index):
    return concat_bytes([
        binName,
        b'\x00', # material type = texture
        pack('B', index),
        bytes(4), # ??? "texture handle or argb"
        bytes(4)]) # ??? "uv/ipal"

def build_bin(model):
    mats_chunk     = model.materials_bs
    mats_aux_chunk = model.aux_data_bs
    uv_chunk       = model.uvs_bs
    vhot_chunk     = model.vhots_bs
    vert_chunk     = model.verts_bs
    light_chunk    = model.lights_bs
    normal_chunk   = model.normals_bs
    node_chunk     = model.nodes_bs
    face_chunk     = model.faces_bs
    node_offsets   = model.node_offsets
    face_chunk     = model.faces_bs
    subs_chunk     = model.subs_bs
    offsets = {}
    def offs(cs):
        return [sum([len(c) for c in cs[:i+1]]) for i in range(len(cs))]
    offsets['subs'],\
    offsets['mats'],\
    offsets['matsAux'],\
    offsets['uvs'],\
    offsets['vhots'],\
    offsets['verts'],\
    offsets['lights'],\
    offsets['normals'],\
    offsets['faces'],\
    offsets['nodes'],\
    offsets['end'] = offs([
        bytes(132),
        subs_chunk,
        mats_chunk,
        mats_aux_chunk,
        uv_chunk,
        vhot_chunk,
        vert_chunk,
        light_chunk,
        normal_chunk,
        face_chunk,
        node_chunk])
    header = encode_header(model, offsets)
    return concat_bytes([
        header,
        subs_chunk,
        mats_chunk,
        mats_aux_chunk,
        uv_chunk,
        vhot_chunk,
        vert_chunk,
        light_chunk,
        normal_chunk,
        face_chunk,
        node_chunk])

def get_local_bbox_data(mesh):
    xs = [v.co[0] for v in mesh.verts]
    ys = [v.co[1] for v in mesh.verts]
    zs = [v.co[2] for v in mesh.verts]
    return (
        (min(xs),min(ys),min(zs)),
        (max(xs),max(ys),max(zs)))

def get_mesh(obj, materials): # and tweak materials
    mat_slot_lookup = {}
    for i in range(len(obj.material_slots)):
        maybe_mat = obj.material_slots[i].material
        if maybe_mat and (maybe_mat in materials):
            mat_slot_lookup[i] = materials.index(maybe_mat)
    bm = bmesh.new()
    #bm.from_object(obj, bpy.context.scene)
    # Old code gave depsgraph wanted error just inserted not sure what this is about.
    bm.from_object(obj, bpy.context.evaluated_depsgraph_get())

    strip_wires(bm) # goodbye, box tweak hack
    uvs = bm.loops.layers.uv.verify()
    for f in bm.faces:
        orig_mat = f.material_index
        if orig_mat in mat_slot_lookup.keys():
            f.material_index = mat_slot_lookup[orig_mat]
            for c in f.loops:
                c[uvs].uv[1] = 1.0 - c[uvs].uv[1]
    return bm

def append_bmesh(bm1, bm2, matrix):
    bm2.transform(matrix)
    uvs = bm1.loops.layers.uv.verify()
    uvs_orig = bm2.loops.layers.uv.verify()
    ord = bm1.faces.layers.int.verify()
    ord_orig = bm2.faces.layers.int.verify()
    vs_so_far = len(bm1.verts)
    for v in bm2.verts:
        bm1.verts.new(v.co)
        bm1.verts.index_update()
    for f in bm2.faces:
        try:
            bm1.verts.ensure_lookup_table()
            bm1.faces.ensure_lookup_table()
            nf = bm1.faces.new(
                [bm1.verts[vs_so_far+v.index] for v in f.verts])
            nf[ord] = f[ord_orig]
        except ValueError:
            continue
        for i in range(len(f.loops)):
            nf.loops[i][uvs].uv = f.loops[i][uvs_orig].uv
            nf.material_index = f.material_index
        bm1.faces.index_update()
    bm2.free()
    bm1.normal_update()

def combine_meshes(bms, matrices):
    result = bmesh.new()
    for bm, mtx in zip(bms, matrices):
        append_bmesh(result, bm, mtx)
    return result
    
def get_motion(obj):
    if not obj:
        mot_type = 0
        min = max = 0.0
    else:
        types = ('LIMIT_ROTATION','LIMIT_LOCATION')
        limits = [c for c in obj.constraints if
            c.type in types]
        if limits:
            c = limits.pop()
            mot_type = types.index(c.type) + 1
            min = c.min_x
            max = c.max_x
        else:
            mot_type = 1
            min = max = 0.0
    return (mot_type,min,max)

def categorize_objs(objs):
    custom_bboxes = [o for o in objs if o.name.lower().startswith("bbox")]
    bbox = None
    if custom_bboxes:
        bo = custom_bboxes[0]
        bm = bo.matrix_world
        bmin = bm @ mu.Vector(bo.bound_box[0])
        bmax = bm @ mu.Vector(bo.bound_box[6])
        bbox = {min:tuple(bmin),max:tuple(bmax)}
    for b in custom_bboxes:
        objs.remove(b)
        
    root = [o for o in objs if o.data.polygons and not (o.parent in objs)]
    gen2 = [o for o in objs if o.data.polygons and o.parent in root]
    gen3_plus = [o for o in objs if
        o.data.polygons and
        o.parent in objs and
        not (o.parent in root)]
    return (bbox,root,gen2,gen3_plus)

def shift_box(box_data, matrix):
    return {
        min:tuple(matrix @ mu.Vector(box_data[min])),
        max:tuple(matrix @ mu.Vector(box_data[max]))}

def tag_vhots(generations):
    ids = {}
    id_fix = 0
    for gen in generations:
        for l in gen:
            # l is vhot name and position
            for vhn, _ in l:
                # NEW vhot01 shall be the same as vhot01.001 when you copy an object and not 1001
                id_s = re.search(r"\d+", vhn)
                #id_s = "".join(re.findall("\d", vhn))
                if id_s:
                    id = int(id_s[0]) % (2**32-1)
                    if id in ids.values():
                        print("Warning: " + vhn + " with ID", id ,"already used by another VHot. Rename! Giving it ID", id_fix, "instead.")
                        ids[vhn] = id_fix
                    else:
                        ids[vhn] = id
                else:
                    ids[vhn] = id_fix
                while id_fix in ids.values():
                    id_fix += 1
    # This renames the vhot to its number.
    for gen in generations:
        for l in gen:
            for i in range(len(l)):
                if l[i]:
                    name, pos = l[i]
                    l[i] = (ids[name], pos)
    return generations

def prep_vg_based_ordering(objs, bms):
    lookup1 = {}
    for o, oi in zip(objs, range(len(objs))):
        for vg in o.vertex_groups:
            n = vg.name
            lookup1[(oi,vg.index)] = n
    names = list(set(lookup1.values()))
    names.sort()
    lookup2 = dict(zip(names, range(1, len(names) + 1)))
    for bm, bmi in zip(bms, range(len(bms))):
        ord = bm.faces.layers.int.verify()
        dfm = bm.verts.layers.deform.verify()
        for f in bm.faces:
            shared = set(f.verts[0][dfm].keys())
            for v in f.verts[1:]:
                shared &= set(v[dfm].keys())
            if shared:
                a = sorted(shared)[0]
                if (bmi,a) in lookup1:
                    mark = lookup2[lookup1[(bmi,a)]]
                    f[ord] = mark

def prep_subs(all_objs, materials, use_origin, sorting):
    bbox, root, gen2, gen3_plus = categorize_objs(all_objs)
    # Sort
    # Ordering by their @s__XX name in case of some extreme sub joints are wanted.
    # @s04bb can have child @s03cc. This fixes a index not found error in the gen3 as its not ordered by 03 < 04.
    gen3_plus.sort(key=lambda kv: (kv.name.casefold() if kv.name[0] != "@" else re_joint.match(kv.name)['name']))
    # Doing for gen2 as well as it would be consistent, root meshes get combined and are sorted by name already.
    gen2.sort(key=lambda kv: (kv.name.casefold() if kv.name[0] != "@" else re_joint.match(kv.name)['name']))
    
    root_meshes = [get_mesh(o, materials) for o in root]
    gen2_meshes = [get_mesh(o, materials) for o in gen2]
    gen3_plus_meshes = [get_mesh(o, materials) for o in gen3_plus]
    if sorting == 'vgs':
        prep_vg_based_ordering(root, root_meshes)
        for a, b in zip(gen2, gen2_meshes):
            prep_vg_based_ordering([a], [b])
            do_groups(b)
        for a, b in zip(gen3_plus, gen3_plus_meshes):
            prep_vg_based_ordering([a], [b])
            do_groups(b)
    # Use object's position in blender as model origin
    if not use_origin == 'Custom':
        root_mesh = combine_meshes(root_meshes, [o.matrix_world for o in root])
    else:
        root_mesh = combine_meshes(root_meshes, [o.matrix_local for o in root])
    if sorting == 'vgs':
        do_groups(root_mesh)
    
    # Use custom or calced bbox
    real_bbox = find_common_bbox(
        [mu.Matrix.Identity(4)] +
        [o.matrix_world for o in gen2] +
        [o.matrix_world for o in gen3_plus],
        [root_mesh] + gen2_meshes + gen3_plus_meshes)
    if not bbox:
        bbox = real_bbox
    # Origin shift
    if use_origin == 'World':
        origin_shift = mu.Matrix.Identity(4)
    elif use_origin == 'Center':
        origin_shift = mu.Matrix.Translation(
            (mu.Vector(real_bbox[max]) + mu.Vector(real_bbox[min])) * -0.5)
    else: #use_origin == 'Custom'
        origin_shift =  mu.Matrix.Translation(-root[0].matrix_local.translation)
    
    root_mesh.transform(origin_shift)
    bbox = shift_box(bbox, origin_shift)
    
    if sorting == 'bsp':
        for el in gen2_meshes + gen3_plus_meshes + [root_mesh]:
            do_bsp(el)
    
    vhots = tag_vhots([
        [[(e.name, origin_shift @ e.matrix_world.translation)
            for e in o.children if e.type == 'EMPTY']
                for o in root],
        [[(e.name, e.matrix_local.translation)
            for e in o.children if e.type == 'EMPTY']
                for o in gen2],
        [[(e.name, e.matrix_local.translation)
            for e in o.children if e.type == 'EMPTY']
                for o in gen3_plus]])
    obj_lookup = {}
    root_sub = Subobject(
        root[0].name, root_mesh, mu.Matrix([[0]*4] * 4), 0, 0.0, 0.0, [])
    for rvh in vhots[0]:
        root_sub.vhots.extend(rvh)
    for g2o, g2m, g2vh in zip(gen2, gen2_meshes, vhots[1]):
        mtx = origin_shift @ g2o.matrix_world
        m_type, m_min, m_max = get_motion(g2o)
        sub = Subobject(g2o.name, g2m, mtx, m_type, m_min, m_max, g2vh)
        sub.set_parent(root_sub)
        obj_lookup[g2o] = sub
    for g3po, g3pm, g3pvh in zip(gen3_plus, gen3_plus_meshes, vhots[2]):
        mtx = g3po.matrix_local
        m_type, m_min, m_max = get_motion(g3po)
        sub = Subobject(g3po.name, g3pm, mtx, m_type, m_min, m_max, g3pvh)
        sub.set_parent(obj_lookup[g3po.parent])
        obj_lookup[g3po] = sub
    return root_sub, bbox

# Each bmesh is extended with custom bytestring data used by the exporter.
# Edges #0 and #1 carry custom mesh-level attributes.
#     Custom vertex data layout:
# v[ext_v][0:2]   : bin vert index as '>H' (BE for sorting)
# v[ext_v][2:14]  : vert coords as '<3f'
#     Custom loop data layout:
# l[ext_l][0:2]   : bin UV index as '>H' (BE for sorting)
# l[ext_l][2:10]  : UV coords as '<ff'
# l[ext_l][10:12] : bin light index as '>H' (BE for sorting)
# l[ext_l][12:14] : bin light mat index as '<H'
# l[ext_l][14:16] : bin light vert index as '<H'
# l[ext_l][16:20] : bin light normal as '<I'
#     Custom face data layout:
# f[ext_f][0:2]  : normal index as '>H' (BE for sorting)
# f[ext_f][2:14] : normal as '<3f'
# f[ext_f][14:]  : ready-made mds_pgon struct
#     Custom edge data layout:
#   Edge #0:
# e0[ext_e][0:2]   : number of bin verts as '<H'
# e0[ext_e][2:4]   : number of bin UVs as '<H'
# e0[ext_e][4:6]   : number of bin lights as '<H'
# e0[ext_e][6:8]   : number of normals as '<H'
# e0[ext_e][8:10]  : number of faces as '<H'
#   Edge #1:
# e1[ext_e][0:2]   : vert offset as '<H'
# e1[ext_e][2:4]   : light offset as '<H'
# e1[ext_e][4:6]   : normal offset as '<H'

def extend_verts(offs, bm):
    ext_v = bm.verts.layers.string.verify()
    ext_e = bm.edges.layers.string.verify()
    v_set = set()
    v_off = offs.v_off
    for v in bm.verts:
        xyz = v.co
        xyz_bs = pack('<3f', xyz.x, xyz.y, xyz.z)
        v_set.add(xyz_bs)
        v[ext_v] = xyz_bs
    num_vs = len(v_set)
    v_dict = dict(zip(v_set, range(num_vs)))
    for v in bm.verts:
        xyz_bs = v[ext_v]
        v_idx = pack('>H', v_off + v_dict[xyz_bs])
        v[ext_v] = v_idx + xyz_bs
    bm.edges.ensure_lookup_table()
    bm.edges[0][ext_e] = pack('<H', num_vs)
    bm.edges[1][ext_e] = pack('<H', v_off)
    offs.v_off += num_vs

def extend_loops(offs, bm):
    ext_l = bm.loops.layers.string.verify()
    ext_v = bm.verts.layers.string.active
    ext_e = bm.edges.layers.string.verify()
    uv = bm.loops.layers.uv.active
    lt_set = set()
    uv_set = set()
    for f in bm.faces:
        mat = pack('<H', f.material_index)
        for l in f.loops:
            v = l.vert[ext_v][-13:-15:-1] # BE to LE
            n = pack_light(l.vert.normal)
            lt = mat + v + n
            lt_set.add(lt)
            l[ext_l] = lt
            uv_set.add(l[uv].uv[:])
    lt_off  = offs.lt_off
    uv_off  = offs.uv_off
    num_lts = len(lt_set)
    num_uvs = len(uv_set)
    lt_dict = dict(zip(lt_set, range(num_lts)))
    uv_dict = dict(zip(uv_set, range(num_uvs)))
    for f in bm.faces:
        for l in f.loops:
            lt = l[ext_l]
            lt_idx = pack('>H', lt_off + lt_dict[lt])
            uv_co = l[uv].uv[:]
            uv_co_bs = pack('<ff', uv_co[0], uv_co[1])
            uv_idx = pack('>H', uv_off + uv_dict[uv_co])
            l[ext_l] = uv_idx + uv_co_bs + lt_idx + lt
    bm.edges.ensure_lookup_table()
    bm.edges[0][ext_e] += pack('<HH', num_uvs, num_lts)
    bm.edges[1][ext_e] += pack('<H', lt_off)
    offs.uv_off += num_uvs
    offs.lt_off += num_lts

def extend_faces(offs, bm):
    ext_f = bm.faces.layers.string.verify()
    ext_v = bm.verts.layers.string.active
    ext_l = bm.loops.layers.string.active
    ext_e = bm.edges.layers.string.active
    f_off = offs.f_off
    n_off = offs.n_off
    n_set = set()
    for f in bm.faces:
        n = f.normal
        n_bs = pack('<3f', n.x, n.y, n.z)
        n_set.add(n_bs)
        f[ext_f] = n_bs
    num_ns = len(n_set)
    n_dict = dict(zip(n_set, range(num_ns)))
    for f in bm.faces:
        f_idx = f_off + f.index
        tx = f.material_index
        num_vs = len(f.verts)
        n_bs = f[ext_f]
        n_idx = n_off + n_dict[n_bs]
        d = find_d(f.normal, [v.co[:] for v in f.verts])
        corners = list(reversed(f.loops)) # flip normal
        vs  = concat_bytes([l.vert[ext_v][-13:-15:-1] for l in corners])
        lts = concat_bytes([l[ext_l][-9:-11:-1] for l in corners])
        uvs = concat_bytes([l[ext_l][-19:-21:-1] for l in corners])
        f[ext_f] = pack('>H', n_idx) + n_bs + \
            pack('<HHBBHf', f_idx, tx, 27, num_vs, n_idx, d) + \
            vs + lts + uvs + pack('B', tx)
    num_fs = len(bm.faces)
    bm.edges.ensure_lookup_table()
    bm.edges[0][ext_e] += pack('<HH', num_ns, num_fs)
    bm.edges[1][ext_e] += pack('<H', n_off)
    offs.n_off += num_ns
    offs.f_off += num_fs

def gather_materials(objs):
    ms = set()
    for o in objs:
        mat_indexes = set()
        for f in o.data.polygons:
            mat_indexes.add(f.material_index)
        slots = o.material_slots
        for mi in mat_indexes:
            try:
                ms.add(slots[mi].material)
            except:
                pass
    return list(ms)

class OffsetWrapper(object):
    def __init__(self):
        self.v_off  = 0
        self.uv_off = 0
        self.lt_off = 0
        self.n_off  = 0
        self.f_off  = 0


def get_objs_toexport(export_filter):
    # Also for draw layout
    if (export_filter == 'visible'):
        raw_objs = bpy.context.visible_objects
    elif (export_filter == 'selected'):
        raw_objs = bpy.context.selected_objects
    elif (export_filter == 'collection'):
        # all_objects will also give objects in child collections
        raw_objs = bpy.context.collection.all_objects
    elif (export_filter == 'all'):
        raw_objs = bpy.data.objects
    else:
        return ('ERROR: Invalid export option', {'CANCELLED'}), None  # Can not happen

    # Sort alphabetically. Depending on the filter the order is not constant by default!
    # raw_objs = sorted(raw_objs, key=lambda kv: (kv.name.casefold() if kv.name[0] != "@" else kv.name[1:].casefold())) 
    #raw_objs = sorted(raw_objs, key=lambda kv: (kv.name.casefold() if kv.name[0] != "@" else re.search(r"\d+(.*)", kv.name, flags=re.IGNORECASE)[1]))
    raw_objs = sorted(raw_objs, key=lambda kv: (kv.name.casefold() if kv.name[0] != "@" else re_joint.match(kv.name)['name']))
    return [o for o in raw_objs if o.type == 'MESH']

def do_export(file_path, options):
    objs = get_objs_toexport(options['export_filter'])
    if not objs:
        return ("Nothing to export.",{'CANCELLED'})
    
    materials = gather_materials(objs)
    root_sub, bbox = prep_subs(
        objs,
        materials,
        options['use_origin'],
        options['sorting'])
    offs = OffsetWrapper()
    for s in root_sub.list_subtree():
        extend_verts(offs, s.mesh)
        extend_loops(offs, s.mesh)
        extend_faces(offs, s.mesh)
    model = Model(
        root_sub,
        materials,
        bbox,
        options['clear'],
        options['bright'],
        options['node_texture'])
    try:
        binBytes = build_bin(model)
        f = open(file_path, 'w+b')
        f.write(binBytes)
        msg = "File \"" + file_path + "\" written successfully."
        result = {'FINISHED'}
    except struct.error:
        msg = "The model has too many polygons. Export cancelled.\n"
        result = {'CANCELLED'}
    return (msg, result)

###
### Sorting: common
###

def reorder_faces(bm, order):
    bm0 = bm.copy()
    bm.clear()
    uvs = bm.loops.layers.uv.verify()
    uvs_orig = bm0.loops.layers.uv.verify()
    for v in bm0.verts:
        bm.verts.new(v.co)
        bm.verts.index_update()
    bm0.faces.ensure_lookup_table()
    bm0.verts.ensure_lookup_table()
    bm.verts.ensure_lookup_table()
    for fi in order:
        f = bm0.faces[fi]
        nf = bm.faces.new(
            [bm.verts[v.index] for v in f.verts])
        for i in range(len(f.loops)):
            nf.loops[i][uvs].uv = f.loops[i][uvs_orig].uv
            nf.material_index = f.material_index
        bm.faces.index_update()
    bm.normal_update()
    bm0.free()

###
### Sorting: by vertex group
###

def do_groups(bm):
    ord = bm.faces.layers.int.verify()
    lookup = {}
    ordered = []
    for f in bm.faces:
        mark = f[ord]
        if mark in lookup:
            lookup[mark].append(f.index)
        else:
            lookup[mark] = [f.index]
    for k in sorted(lookup):
        if k > 0:
            ordered.extend(lookup[k])
    if 0 in lookup:
        ordered.extend(lookup[0])
    reorder_faces(bm, ordered)

###
### Sorting: BSP
###

def walk(paths): # incremented indexes in keys, regular in values
    path = (0,) # 0 is used for root (index 0 doesn't occur)
    ps = []
    ns = []
    while paths:
        v = paths[path] + 1
        neg = path + (-v,) # negative for "is behind"
        pos = path + (v,)  # positive for "is in front"
        if neg in paths:
            path = neg
        elif pos in paths:
            path = pos
        else:
            if path[-1] >= 0:
                ps.append(paths.pop(path))
            else:
                ns.append(paths.pop(path))
            path = path[:-1]
    ps.reverse()
    ns.reverse()
    return ps + ns

def do_bsp(bm):
    all_fs = bm.faces[:]
    todo = [set(all_fs)]
    paths = dict(zip([f.index for f in all_fs], [(0,)] * len(all_fs)))
    while todo:
        fs = todo.pop(0)
        f, plane_n, plane_xyz = split_plane(fs, bm)
        idx = f.index + 1 # regular indexes in keys, incremented in values
        fs_inner, fs_outer = slash(bm, fs, plane_xyz, plane_n, f, idx, paths)
        if fs_inner:
            todo.append(fs_inner)
        if fs_outer:
            todo.append(fs_outer)
    paths_swapped = dict([(paths[i],i) for i in paths])
    ordered = walk(paths_swapped)
    return reorder_faces(bm, ordered)

def slash(bm, fs, plane_xyz, plane_n, ref_f, idx, paths):
    g = set()
    par = paths[ref_f.index]
    fs.remove(ref_f)
    g |= fs
    for f in fs:
        for e in f.edges:
            g.add(e)
        for v in f.verts:
            g.add(v)
    fs_orig = set(bm.faces)
    bmesh.ops.bisect_plane(
        bm, geom=list(g), dist=0.05, plane_co=plane_xyz, plane_no=plane_n)
    fs_new = set(bm.faces)
    fs_new -= fs_orig
    fs |= fs_new
    for f in fs:
        paths[f.index] = par
    fs_inner = set()
    fs_outer = set()
    for f in fs:
        fi = f.index
        a = plane_n.dot(f.calc_center_median() - plane_xyz)
        if a >= 0: # play with equation
            paths[fi] = paths[fi] + (idx,)
            fs_outer.add(f)
        else:
            paths[fi] = paths[fi] + (-idx,)
            fs_inner.add(f)
    return fs_inner, fs_outer

def split_plane(fs, bm):
    rated = list(fs)
    c = calc_center([f.calc_center_median() for f in fs])
    rated.sort(key=lambda a: rate(a, c))
    best_f = rated[0]
    new_plane_n = best_f.normal
    new_plane_d = find_d(new_plane_n, [v.co for v in best_f.verts])
    plane_xyz = new_plane_n * -new_plane_d
    return best_f, new_plane_n, plane_xyz

def rate(f, c):
    return (-find_d(f.normal, [(v.co - c) for v in f.verts]),-f.calc_area())

###
### UI
###

class ImportDarkBin(bpy.types.Operator, ImportHelper):
    '''Load a Dark Engine Static Model File'''
    bl_idname = "import_scene.dark_bin"
    bl_label = 'Import BIN'
    bl_options = {'PRESET'}
    filename_ext = ".bin"
    check_extension = True
    
    filter_glob : StringProperty(
        default="*.bin",
        options={'HIDDEN'})
    use_collections : BoolProperty(
        name="Group in new collection",
        default=True,
        description="Group together in a new collection. Else in scene.")
    fancy_txtrepl : BoolProperty(
        name="Use fancy replace0.gif",
        default=True,
        description="Uses special images for replace# image names."
                     +" Deactivate if you have a custom replace# texture")
    convert_gif : BoolProperty(
        name="Convert gifs",
        default=True,
        description="Gifs will be converted to png and packed in the file."\
                     + " Blank image on failure. Not using this option will print warnings")
    support_3ds_export : BoolProperty(
        name="(Experimental) Oldschool VHots+Axis",
        default=False,
        description="To be used with 3ds export and extern bsp.exe." +
                     " NOT COMPATIBLE with static exporter") #,
  
    path_mode : path_reference_mode
    check_extension : True
    path_mode : path_reference_mode # Why double?

    def draw(self, context):
        # Fancy
        self.layout.prop(self, 'use_collections', icon='GROUP')
        self.layout.prop(self, 'fancy_txtrepl', icon='TEXTURE_DATA')
        self.layout.prop(self, 'convert_gif', icon='PACKAGE')
        #
        self.layout.prop(self, 'support_3ds_export', icon='ERROR')

    def execute(self, context):
        options = {
            'use_collections'       :self.use_collections,
            'fancy_txtrepl'         :self.fancy_txtrepl,
            'convert_gif'           : self.convert_gif,
            'support_3ds_export'   : self.support_3ds_export,}
        msg, result = do_import(self.filepath, options)
        print(msg)
        return result


class ExportDarkBin(bpy.types.Operator, ExportHelper):
    '''Save a Dark Engine Static Model File'''
    bl_idname = "export_scene.dark_bin"
    bl_label = 'Export BIN'
    bl_options = {'PRESET'}
    filename_ext = ".bin"
    path_mode : path_reference_mode
    check_extension = True
    path_mode : path_reference_mode # Why double?

    filter_glob : StringProperty(
        default="*.bin",
        options={'HIDDEN'})
    clear : BoolProperty(
        name="Use Alpha",
        default=True,
        description="Use the Alpha value from Material->Surface")
    bright : BoolProperty(
        name="Use Emission",
        default=True,
        description="Use the Emission value from Material->Surface."
                    + " Preferred is Strength from Emission node, else highest R,G or B")
    node_texture : BoolProperty(
            name="Use image instead of material name",
            default=True,
            description="Use the Base Color image instead of the material name"
                        +" for the models textures.")
    sorting : EnumProperty(
        name="Sort",
        items=(
            ("bsp","BSP","".join([
                "May increase the polygon count unpredictably;",
                " use only if you need transparency support"]), 'NLA', 1),
            ("vgs","By vertex group","".join([
                "Follow the alphabetical order of vertex group names"]), 'GROUP_VERTEX', 2),
            ("none","Don't sort","", 'BLANK', 3)),
        default="vgs")                                    
    use_origin : EnumProperty(
        name="Origin",
        items=(
            ("World","World origin","Model origin is at world origin", 'EMPTY_ARROWS',1),
            ("Center","Bounding Box center","Bounding box center.", 'LIGHTPROBE_CUBEMAP', 2),
            ("Custom","Object origin","Use origin like defined in blender.", 'TRANSFORM_ORIGINS', 3)),
        default="Center")
    export_filter : EnumProperty(
        name="Export only",
        items=(
            ("selected","Selected","Only currently selected.", 'SELECT_SET', 1),
            ("visible","Visible","Hidden objects are ignored.", 'HIDE_OFF', 2),
            ("collection", "Active Collection", "Exports objects in the current active collection.", 'GROUP', 4),
            ("all","All","Export all objects in the file.", 'BLENDER' ,8)),
        default="visible")
    
    def draw(self, context):
        # Fancy
        if context.active_object and context.active_object.mode == 'EDIT':
            self.layout.label(text="You are in Edit Mode.\nMesh data maybe not up to date", icon='ERROR')
        for prop in self.__annotations__:
            if not 'options' in self.__annotations__[prop][1]:
                if prop == 'bright':
                    self.layout.prop(self, prop, icon='LIGHT')
                elif prop == 'clear':
                    self.layout.prop(self, prop, icon='XRAY')
                elif prop == 'node_texture':
                    self.layout.prop(self, prop, icon='FILE_IMAGE')
                else:
                    self.layout.prop(self, prop)
        self.layout.separator_spacer()
        self.layout.label(text="-- Objects to be exported --")
        for o in get_objs_toexport(self.export_filter):
            self.layout.label(text=o.name) 
    
    def execute(self, context):
        options = {
            'clear'       :self.clear,
            'bright'      :self.bright,
            'use_origin'  :self.use_origin,
            'sorting'     :self.sorting,
            'export_filter':self.export_filter,
            'node_texture' : self.node_texture}
        msg, result = do_export(self.filepath, options)
        if result == {'CANCELLED'}:
            self.report({'ERROR'}, msg)
        print(msg)
        return result


def menu_func_import_bin(self, context):
    self.layout.operator(
        ImportDarkBin.bl_idname, text="Dark Engine Static Model (.bin)")

def menu_func_export_bin(self, context):
    self.layout.operator(
        ExportDarkBin.bl_idname, text="Dark Engine Static Model (.bin)")

# Blender 2.80+ 
def register():
    bpy.utils.register_class(ImportDarkBin)
    bpy.utils.register_class(ExportDarkBin)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import_bin)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export_bin)

def unregister():
    bpy.utils.unregister_class(ImportDarkBin)
    bpy.utils.unregister_class(ExportDarkBin)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import_bin)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export_bin)
