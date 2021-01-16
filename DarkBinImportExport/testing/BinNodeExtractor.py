import numpy as mu # Choosing mu to show where blenders mathutils can be used
import os
from struct import pack, unpack, calcsize
np = mu

###
### Helper classes
###
try:
    # For blender
    from .NodesImport import *
    # These are objects that can parse binary data into logical structures
    # These are all for meshes
    from .mesh_structs import mms_pgon, mms_smatr, mms_segment, mms_uvn, mms_smatseg, sCreatureLengths as cal_file_parser
except ImportError:
    # For Jupyter
    from NodesImport import * 
    from mesh_structs import mms_pgon, mms_smatr, mms_segment, mms_uvn, mms_smatseg, sCreatureLengths as cal_file_parser


# Constants
MD_PGON_PRIM_MASK  = 0x07 # Filters out PRIM
MD_PGON_COLOR_MASK = 0x60 # Filters out PAL, VCOL

MD_PGON_PRIM_NONE   = 0  # no primitive drawn
MD_PGON_PRIM_SOLID  = 1  # vcolor lookup
MD_PGON_PRIM_WIRE   = 2  #// wire
MD_PGON_PRIM_TMAP   = 3  #// texture map

# TODO: Info currently not imported into blender
MD_PGON_LIGHT_ON   = 0x18 # Pgon unaffected by light

MD_PGON_COLOR_PAL  =0x20  #// palette color
MD_PGON_COLOR_VCOL =0x40  #// vcolor lookup


#########################
class FaceImported(object):
    """
    Holds nearly(?) all the data of a face/polygon that we want.
    For a raw import only a tiny bit of this information is needed
    the rest is more for research.
    """
    allFaces = []
    coplanar = {}
    def __init__(self, index, data, type, num_verts, norm_idx, d, verts, light, uvs=None, matByte=None):
        self.binVerts = verts
        self.binUVs = uvs
        self.binMat = data # could also be color
        self.index = index
        self.d = round(d,1)
        self.norm_id = norm_idx
        self.matByte = matByte
        if norm_idx in Norms:
            Norms[norm_idx].append(index)
        else:
            Norms[norm_idx] = [index]
        self.normal = getNormalByIndex(norm_idx)
        nv = (round(n,1) for n in self.normal)
        nv = (0.0 if n == -0.0 else n for n in nv) # get rid of stupid -0.0
        FaceImported.coplanar.setdefault((*nv, self.d), []).append(self.index)
        FaceImported.allFaces.append(self)


def getNormalByIndex(i):
    """
    Get the normal vector (as tuple) from the associated lookup index.
    """
    global normOffset
    global faceOffset
    off = normOffset + 12 * i # length of 3 floats
    end = off + 12
    assert end <= faceOffset, "Normal index/offset is behind faceOffset"
    return unpack('<fff', binBytes[off:end])


###
### Import
###


def aka(key, verts):
    result = None
    for i in range(len(verts)):
        if key == verts[i][0]:
            result = (i,verts[i])
            break
    return result

# Unpack functions

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
    type="static"
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

def ulong_to_argb(argb):
    a = argb >> 24
    r = (argb >> 16) & 0x00ff
    g = (argb >> 8) & 0x0000ff
    b = argb & 0x000000ff
    return a,r,g,b
    

def prep_materials(matBytes, numMats, file_path):
    materials = {}
    stage1 = []
    stage2 = []
    for _ in range(numMats):
        matName = get_string(matBytes[:16])
        matType = matBytes[16] # TMAP(0) or RGB color(1)
        matSlot = matBytes[17]
        # Always have seen this as 0 for TMAP
        handle_or_argb = unpack("<L", matBytes[18:22])[0]
        if handle_or_argb > 0:
            print(handle_or_argb, ulong_to_argb(handle_or_argb))
        # is float or ulong for argb
        # only have seen uv as 0.
        uv_or_ipal = unpack("<fL", matBytes[22:26]+matBytes[22:26])
        print("mat type", matType, "handle/argb:", handle_or_argb,
                "uv/ipal", uv_or_ipal)
        stage1.append((matSlot,matName))
        matBytes = matBytes[26:]
    if matBytes: # if there's aux data, version 4 models
        # This is 8 or 16 also definied in amat_size
        auxChunkSize = len(matBytes) // numMats
        for _ in range(numMats):
            trans, illum = get_floats(matBytes[:8])
            # Sometimes there is another 2x4 data block for float MaxTU,MaxTV; //mipmap max texel size
            stage2.append((trans, illum))
            matBytes = matBytes[auxChunkSize:]
    else:
        for _ in range(numMats):
            stage2.append((0.0,0.0))
    for i in range(numMats):
        slot, name = stage1[i]
        trans, illum = stage2[i]
        materials[slot] = (name, trans, illum)
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
        #uvs.append(mu.array((u,v)))
        uvs.append((u,v))
    return uvs

def prep_faces(faceBytes, version):
    print("File version is", version)
    pgon_base_size = 13 if version >=4 else 12 # magic 12 or 13: v4 has an extra byte at the end
    faces = {}
    faceAddr = 0
    faceIndex = 0
    while len(faceBytes):
        f_index   = unpack('<H', faceBytes[0:2])[0]
        matIndex  = unpack('<H', faceBytes[2:4])[0]
        pgon_type = faceBytes[4] & MD_PGON_PRIM_MASK  # bit field
        num_verts = faceBytes[5]
        norm_idx  = unpack('<H', faceBytes[6:8])[0]
        d         = unpack('<f', faceBytes[8:12])[0]

        verts = get_ushorts(faceBytes[12:12+num_verts*2])
        light = get_ushorts(faceBytes[12+num_verts*2 : 12+num_verts*4])
        
        uvs = []
        if pgon_type == MD_PGON_PRIM_TMAP: #TMAP=3
            faceEnd = pgon_base_size + num_verts * 6
            uvs.extend(get_ushorts(faceBytes[12+num_verts*4:12+num_verts*6]))
        else:
            # No UV coordinates
            faceEnd = pgon_base_size + num_verts * 4
            matIndex = None
        if version >= 4:
            matByte = faceBytes[faceEnd-1] # Material index, in TMAP cases matIndex -1
        else:
            matByte = None
        # Hello im new
        face = FaceImported(f_index, matIndex, pgon_type, num_verts, norm_idx, round(d,3), verts, light, uvs, matByte)
        faces[faceAddr] = face
        faceAddr += faceEnd
        faceIndex += 1
        faceBytes = faceBytes[faceEnd:]
    return faces

########## /////////////////////////////////////////////////////

def make_node_subobject(bs, pos):
    node = Node_sub(index=unpack('<H', bs[1:3])[0], position=pos)
    return ([],bs[3:], node, pos+3)

def make_node_vcall(bs, pos):
    node = Node_vcall(position=pos,
                      sphere=bs[1:17], 
                      index=unpack('<H', bs[17:19]))
    return ([],bs[19:], node, pos+19)

def make_node_call(bs, pos):
    facesStart = 23
    sphere = bs[1:17]
    num_faces1 = unpack('<H', bs[17:19])[0]
    node_called = unpack('<H', bs[19:21])[0]
    num_faces2 = unpack('<H', bs[21:facesStart])[0]
    facesEnd = facesStart + (num_faces1 + num_faces2) * 2
    faces = get_ushorts(bs[facesStart:facesEnd])
    node = Node_call(sphere=sphere, 
                     pgons_before= num_faces1, 
                     node_called = node_called,
                     pgons_after = num_faces2,
                     polys=faces,
                     position=pos)
    return (faces,bs[facesEnd:], node, pos+facesEnd)

Norms = {}

def make_node_split(bs, pos):
    facesStart = 31
    num_faces1 = unpack('<H', bs[17:19])[0]
    norm_idx = unpack('<H', bs[19:21])[0]
    d = unpack('<f', bs[21:25])[0]
    # This could be a norm that is not associated with a face!
    if norm_idx in Norms:
        Norms[norm_idx].append(f"split d={d:.3f}")
    else:
        Norms["solo " + str(norm_idx)] = [f"split d={d:.3f}"]
    node_behind = unpack('<H', bs[25:27])[0]
    node_front = unpack('<H', bs[27:29])[0]
    num_faces2 = unpack('<H', bs[29:facesStart])[0]
    facesEnd = facesStart + (num_faces1 + num_faces2) * 2
    faces = get_ushorts(bs[facesStart:facesEnd])
    
    node = Node_split(sphere=bs[1:17],
                      pgons_before=num_faces1,
                      norm=norm_idx,
                      d=d,
                      node_behind=node_behind,
                      node_front=node_front,
                      pgons_after=num_faces2,
                      polys=faces,
                      position=pos)
    
    return (faces,bs[facesEnd:],node, pos+facesEnd)

def make_node_raw(bs, pos):
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
            process = make_node_subobject
        elif nodend_type == 3:
            process = make_node_vcall
        elif nodend_type == 2:
            process = make_node_call
        elif nodend_type == 1:
            process = make_node_split
        elif nodend_type == 0:
            process = make_node_raw
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

########## /////////////////////////////////////////////////////
 
# These are the numbers associated with Editor values. 
JointNames = dict(
   NA 			= 0,
   Head			= 1,
   Neck			= 2,
   Abdomen		= 3,
   Butt			= 4,
   LeftShoulder	= 5,
   RightShoulder= 6,
   LeftElbow	= 7,
   RightElbow	= 8,
   LeftWrist	= 9,
   RightWrist	= 10,
   LeftFingers	= 11,
   RightFingers	= 12,
   LeftHip		= 13,
   RightHip		= 14,
   LeftKnee		= 15,
   RightKnee	= 16,
   LeftAnkle	= 17,
   RightAnkle	= 18,
   LeftToe		= 19,
   RightToe		= 20,
   Tail			= 21
) # use list(keys())[joint_id]

JointNamesBiped = {
    "LTOE":0,
    "RTOE":1,
    "LANKLE":2,
    "RANKLE":3,
    "LKNEE":4,
    "RKNEE":5,
    "LHIP":6,
    "RHIP":7,
    "BUTT":8,
    "NECK":9,
    "LSHLDR":10,
    "RSHLDR":11,
    "LELBOW":12,
    "RELBOW":13,
    "LWRIST":14,
    "RWRIST":15,
    "LFINGER":16,
    "RFINGER":17,
    "ABDOMEN":18,
    "HEAD":19
}


class BinParser:
    def parse_bin(self, file_path):
        pass
    
    @staticmethod
    def prep_vector(vertBytes):
        """
        For normals or verts
        """
        floats = get_floats(vertBytes)
        vectors = []
        i = -1
        while floats:
            i += 1
            x = floats.pop(0)
            y = floats.pop(0)
            z = floats.pop(0)
            vectors.append((i,(x,y,z)))
        return vectors
    
    @staticmethod
    def prep_uvs(uvBytes):
        floats = get_floats(uvBytes)
        uvs = []
        i = -1
        while floats:
            i += 1
            u = floats.pop(0)
            v = floats.pop(0)
            #uvs.append(mu.array((u,v)))
            uvs.append((u,v))
        return uvs

    @staticmethod
    def parse_light(norm):
        x = ((norm>>16) & 0xFFC0) /16384.0
        y = ((norm>>6)  &0xFFC0) /16384.0
        z = ((norm<<4)  &0xFFC0) /16384.0
        return (x,y,z)
    
    @staticmethod
    def _get_struct(Bytes, bstruct):
        rv = []
        struct_size = bstruct.size
        while Bytes:
            chunk, Bytes = Bytes[:struct_size], Bytes[struct_size:]
            rv.append(bstruct(chunk))
        return rv
    
    
class StaticModel(BinParser):
    def parse_bin(file_path):
        pass
 
 
class AIMesh(BinParser):
    """
     struct _mms_pgon
    {
       ushort v[3];
       ushort smatr_id; // necessary if doing global sort
       float d;         // plane equation coeff to go with normal
       ushort norm;     // index of pgon normal
       ushort pad;
    } mms_pgon;
    """
    
        
    
    # Subclass as the pgon is a little bit different
    class FaceImported(object):
        """
        Holds nearly(?) all the data of a face/polygon that we want.
        For a raw import only a tiny bit of this information is needed
        the rest is more for research.
        """
        allFaces = []
        counter = 0
        def __init__(self, verts, smatr_id, d, norm_idx, pad, uvs, lights=None, vert_pos=None):
            self.binVerts = verts
            self.binUVs = uvs
            self.uvs = uvs
            self.index = self.__class__.counter
            self.__class__.counter += 1
            
            self.lights = lights=None
            self.vert_pos = vert_pos
            
            self.d = round(d,1)
            self.norm_id = norm_idx
            if norm_idx in Norms:
                Norms[norm_idx].append(self.index)
            else:
                Norms[norm_idx] = [self.index]
            self.normal = AIMesh.getNormalByIndex(norm_idx) # defined in parser
            self.__class__.allFaces.append(self)
    
    
    class _StructImported():
        """
        Regions are the greatest structures
        and associated with a number of SingleMatSegments
        
        Segments are also associated with SingleMatSeg (if more than one needs confirmation)
        
        SingleMatSeg keep reference IDs to their region and Segment
        
        After parsing the structs the build order is
        SingleMatSeg -> Segments -> Regions -> SingleMatSeg(add crossreference)
        """
        all = None
        def __init__(self, struct, faces, verts):
            self.index = self.__class__.counter
            self.__class__.counter += 1
            self.struct = struct
            data = struct.data
            num_faces = data.pgons
            num_verts = data.verts
            self.faces = faces[data.pgon_start:data.pgon_start+num_faces]
            self.verts = verts[data.vert_start:data.vert_start+num_verts]
            self.all.append(self)
            
        def __repr__(self):
            return self.__class__.__name__ + f" (index={self.index})"
    
    
    class RegionImported(_StructImported):
        """
        Association with material
        """
        all = [] 
        counter = 0
        def __init__(self, smatr_struct, faces, verts, maps, smat_segs):
            super().__init__(smatr_struct, faces, verts)
            self.name = smatr_struct.name
            print(self.name, self.index)
            self.material = (smatr_struct.name, smatr_struct.Alpha, smatr_struct.Illum) 
            for f in self.faces:
                f.binMat = self.index
                f.material = self.material
            self.struct = smatr_struct

            # maybe could work direct with segments
            num_smatsegs = smatr_struct.num_smatsegs  # always 1 ??
            self.smat_segs_ids = maps[smatr_struct.map_start : smatr_struct.map_start + num_smatsegs]
            self.smat_segs = [smat_segs[id] for id in self.smat_segs_ids]
        
        @classmethod
        def get_materials(cls):
            materials = {}
            for r in cls.all:
                materials[r.index] = r.material
            cls.materials = materials
            return materials    
        
            
    
    class Segment(_StructImported):
        """
        Association with joints
        """
        all = []
        joint_dict = {}
        
        counter = 0
        def __init__(self, segment_struct, faces, verts, maps, smat_segs):
            super().__init__(segment_struct, faces, verts) # While they have a data field this might be empty.
            self.joint_id = segment_struct.joint_id
            
            # Multiple segments can belong to one joint.
            self.__class__.joint_dict.setdefault(self.joint_id, []).append(self)
            
            name_list = JointNames
            name_list = JointNamesBiped # Need to get these
            self.joint_name = list(name_list.keys())[self.joint_id] # This might be different for non humans?
            
            print(self.joint_id, "=", self.joint_name)
            if segment_struct.flags > 1: # Should not be, but I don't know much.
                print("segment_struct.flags is greater than 1. Not expected", segment_struct.flags)
            self.stretchy = bool(segment_struct.flags)
            self.bbox = segment_struct.bbox # This is a ulong ?? Have always seen this as 0
            
            num_smatsegs = segment_struct.num_smatsegs # always 1 ??
            self.smat_segs_ids = maps[segment_struct.map_start : segment_struct.map_start + num_smatsegs]
            self.smat_segs = [smat_segs[id] for id in self.smat_segs_ids]     
            self.offset = np.zeros(3)

        def __repr__(self):
            return self.__class__.__name__ + f"(index={self.index}, joint={self.joint_id})"

    class SingleMatSeg(_StructImported):
        all = []
        counter = 0
        def __init__(self, smatseg_struct, faces, verts):
            super().__init__(smatseg_struct, faces, verts)
            
            self.matIndex = smatseg_struct.smatr_id # or rather associated region
            self.region_id = self.matIndex
            self.segment_id = smatseg_struct.seg_id
            
        @classmethod
        def add_references(cls):
            for sms in cls.all:
                sms.region = AIMesh.RegionImported.all[sms.region_id]
                sms.segment = AIMesh.Segment.all[sms.segment_id]
    
    
    class SubobjectImported(object):
        type = "aimesh"
        def __init__(self, region):
            # location, as verts are absolute probably 0
            self.region = region
            self.name = region.name
            faces = region.faces
            materials = region.materials # ALL via class but should be identical to single mat
            self.vhots = []
            
            matsUsed = {}
            for f in faces:
                m = f.binMat
                if m != None:
                    matsUsed[m] = materials[m]
            self.matsUsed = matsUsed
            self.faces = faces
            
        def matSlotIndexFor(self, matIndex): # hmm used for building bmesh
            if matIndex != None:
                return list(self.matsUsed.values()).index(self.matsUsed[matIndex])
        
        def localMatrix(self):  # Should get this data somewhere
            return np.eye(4)    # vertices are absolute coordinates
    
    
    class SubobjectFromSegment(object):
        type = "aimesh"
        def __init__(self, matseg):
            # location, as verts are absolute probably 0
            region = matseg.region
            segment = matseg.segment
            self.segment = matseg.segment
            self.region = region
            
            self.name = segment.joint_name
            self.joint_id = segment.joint_id
            faces = matseg.faces
            
            materials = region.materials # ALL via class but should be identical to single mat
            self.vhots = []
            
            matsUsed = {}
            for f in faces:
                m = f.binMat
                if m != None:
                    matsUsed[m] = materials[m]
            self.matsUsed = matsUsed
            self.faces = faces
            self._localMatrix = np.eye(4) # gets overwritten if call file present
            
        def matSlotIndexFor(self, matIndex): # hmm used for building bmesh
            if matIndex != None:
                return list(self.matsUsed.values()).index(self.matsUsed[matIndex])
        
        def localMatrix(self):  # Should get this data somewhere
            return self._localMatrix    # vertices are absolute coordinates


    # As bin_struct names are currently the same as in mms.h
    # Some convenience functions with more sense 
    @classmethod
    def get_segments(cls, Bytes):
        return cls._get_struct(Bytes, mms_segment)
        
    @classmethod
    def get_regions(cls, Bytes):
        return cls._get_struct(Bytes, mms_smatr)
    
    @classmethod
    def get_single_mat_seg(cls, Bytes):
        return cls._get_struct(Bytes, mms_smatseg)
        
    @classmethod
    def get_uv_and_light(cls, Bytes):
        data = cls._get_struct(Bytes, mms_uvn)
        uvs = []
        norms = []
        for uvn in data:
            uvs.append((uvn.u, uvn.v))
            norms.append(uvn.norm)
        return uvs, norms
    
    @classmethod
    def get_pgons(cls, faceBytes):
        return cls._get_struct(faceBytes, mms_pgon)

    @classmethod
    def pgons_to_faces(cls, pgons, all_verts, all_uvs, all_lights):
        for pgon in pgons:
            cls.FaceImported(pgon.v, pgon.smatr_id, pgon.d, pgon.norm, pgon.pad, 
                                #uvs      = [all_uvs[i] for i in pgon.v],
                                #lights   = [all_lights[i] for i in pgon.v],
                                #vert_pos = [all_verts[i] for i in pgon.v])
                                # Only index
                                uvs      = [i for i in pgon.v],
                                lights   = [all_lights[i] for i in pgon.v],
                                vert_pos = [i for i in pgon.v])
        return cls.FaceImported.allFaces
    
    @classmethod
    def regions_to_subobject(cls, region, faces, all_verts):
        pass
    
    @staticmethod
    def get_verts(vertBytes):
        verts = []
        for pos in range(0, len(vertBytes)-12, 12):
            verts.append(unpack("<fff"),vertBytes[pos:pos+12])
        return verts
    
     
    MM_LAYOUT_MAT = 0
    MM_LAYOUT_SEG = 1
    @classmethod
    def parse_bin(cls, binData, mesh_type="Human"):
        if type(binData) == str:
            binData = open(binData, 'r+b')

        binData.seek(0)
        binBytes = binData.read(-1)      # read whole file
        binData.close()
        
        # Read header
        nd_typeID = binBytes[:4]         # name of model b"LGMM"
         
        version = unpack('<I', binBytes[4:8])[0] # 1 or 2
        radius = unpack('<f', binBytes[8:12])[0] # Bound radius
        flags = unpack('<I', binBytes[12:16])[0] # ???
        app_data = unpack('<I', binBytes[16:20])[0] # to be set by app, so it can check it during callbacks. 
        layout = binBytes[20]             # MAT 0 or SEG 1
        segs = binBytes[21]               # number of segments, including stretchy # mms_segment size 20   
        
        single_mat_regions = binBytes[22]
        single_mat_segments = binBytes[23]
        num_pgons, num_verts, num_weights = unpack('<HHH', binBytes[24:30])  
        pad = unpack('<H', binBytes[30:32])[0]  # ???
        
        # offset to mappings from seg smatr to smatsegs
        # // relative to start of the model, used to generate pointers
        # offset to array of pgon normal vectors
        # offset to array of mxs_vectors of vertex positions
        # offset to array of other vertex data - (u,v)'s, normals etc
        map_off, \
        seg_off, \
        single_mat_regions_off, \
        single_mat_segments_off, \
        pgon_off, \
        norm_off, \
        vert_vec_off, \
        vert_uvn_off, \
        weight_off = get_uints(binBytes[32: 32+ 9*4])
        
        faceOffset = pgon_off
        
        def getNormalByIndex(i):
            """
            Get the normal vector (as tuple) from the associated lookup index.
            """
            off = norm_off + 12 * i # length of 3 floats
            end = off + 12
            # differnt order norm after pgon
            assert end <= vert_vec_off, ("Normal index/offset is behind vert_vec_off", end, vert_vec_off)
            return unpack('<fff', binBytes[off:end])
        cls.getNormalByIndex = getNormalByIndex
        
        # These are probably for some cross relations between segments and smat_segs
        maps = [uc for uc in binBytes[map_off:seg_off]]
        print("maps", maps)
        
        segments  = cls.get_segments(binBytes[seg_off:single_mat_regions_off])
        regions   = cls.get_regions(binBytes[single_mat_regions_off:single_mat_segments_off])
        smat_segs = cls.get_single_mat_seg(binBytes[single_mat_segments_off:pgon_off])
        

        pgons = AIMesh.get_pgons(binBytes[pgon_off:norm_off])
        norms = cls.prep_vector(binBytes[norm_off:vert_vec_off])        # for pgons
        verts = cls.prep_vector(binBytes[vert_vec_off:vert_uvn_off])
        print("numv", num_verts, len(verts))
        uvs, lights_compact = cls.get_uv_and_light(binBytes[vert_uvn_off:weight_off])
        
        # lights are compacted normals
        lights = [cls.parse_light(norm) for norm in lights_compact]
        print("lights", len(lights))
        
        weights = get_floats(binBytes[weight_off:])
        print("weights", len(weights))

        """
        for k,v in locals().items():
            if v in [binBytes, lights, verts, norms]:
                continue
            print(k,v)
        """
        print("Regions:", len(regions))
        print(regions)
        print(segments)
        print("\nsmat_segs:", len(smat_segs))
        print(smat_segs)
        
        faces = cls.pgons_to_faces(pgons, verts, uvs, lights)
        addr = [i*mms_pgon.size for i in range(mms_pgon.size)]
        
        faceDict = dict.fromkeys(addr, faces)
        
        # upgrade to classes
        # smat_segs first !
        smat_segs = [cls.SingleMatSeg(r, faces, verts) for r in smat_segs]
        regions   = [cls.RegionImported(r, faces, verts, maps, smat_segs) for r in regions]
        segments  = [cls.Segment(r, faces, verts, maps, smat_segs) for r in segments]
        cls.RegionImported.get_materials()
        cls.SingleMatSeg.add_references() # to regions and segments, which already have their crossreference
        # after class conversion:
        print("\nSegments:", len(segments), "Different joints", len(cls.Segment.joint_dict))
        print(regions)
        print(segments)
        print(smat_segs)
        
        bbox = None # for the moment
        #subs = []
        #for r in regions:
        #    subs.append(cls.SubobjectImported(r))

        subs = []
        for s in smat_segs:
            # sum segments belong to the same joint
            subs.append(cls.SubobjectFromSegment(s))
        
        object_data = (bbox, subs, verts, uvs, cls.RegionImported.materials)

        # ////////////////// CAL FILE /////////////////////////////
        print(".bin file sucessfully parsed") 
        base_name, ext = os.path.splitext(binData.name.lower())
        cal_file = base_name + ".cal"
        if os.path.exists(cal_file):
            cls.parse_cal(cal_file, subs)
        else:
            print("WARNING. No .cal file",cal_file," file found in same folder for AIMesh. Object only partially imported.")
        return object_data
        
        
        def __init__(self, binData, mesh_type="Human"):
            self.data = self.parse_bin(binData, mesh_type="Human")

        def __getattr__(self, key):
            return self.data[key]
    
    # Should create these classes at init for better cleanup
    """
    Torso 0 also has abdomen as 3rd fixed_point.
    
    Each Torso has a number of fixed_points, each point is associated with a limb,
    limbs again have other joints associated with them
    
    """
    
    class Skeleton:
        def __init__(self, torsos):
            for torso in torsos:
                if torso.parent == -1:
                    self.root = torso
                    self.root_joint = torso.joint
                    torso.offset = np.zeros(3)
                else:
                    torso.parent.child = torso
                    off_index = torso.parent.joint_id.index(torso.joint)
                    torso.offset = np.array(torso.parent.offset + torso.parent.vectors[off_index])
                self.child = -1 # can be overwriten by child in next loop
                torso.segment_groups = [AIMesh.Segment.joint_dict[idx] for idx in torso.joint_id]
                for idx, seg_group in enumerate(torso.segment_groups):
                    for seg in seg_group:
                        seg.offset = torso.offset + np.array(torso.vectors[idx])
                limb_off = torso.offset        
                for limb_id in torso.joint_id:
                    #limb_off = torso.offset     
                    limb = AIMesh.Limb.joint_dict.get(limb_id)
                    print("limb", limb)
                    if limb is not None:
                        limb = limb[0]
                        torso.limbs.append(limb)
                        #remap?
                        print("ids", limb.joint_id)
                        # TODO!!!!
                        # Find the bug workouraund with .get
                        limb.segment_groups = [AIMesh.Segment.joint_dict.get(idx, ["NAN", idx]) for idx in limb.joint_id]
                        for idx, seg_group in enumerate(limb.segment_groups):
                            if seg_group[0] == "NAN":
                                print("seg", seg_group[1], "not found")
                                continue
                            limb_off += np.array(limb.vectors[idx])        
                            
                            for seg in seg_group:
                                seg.offset = limb_off
                    else:
                        print(limb_id, "as none")

            # We have torso - limbs - joints
               
    
    class Torso:
        """
        Construct before limbs
        """
        all = []
        counter = 0
        joint_dict = {}
        
        def __init__(self, parsed_torso):
            self.__class__.all.append(self)
            self.dict = parsed_torso
            self.index = self.__class__.counter
            self.__class__.counter += 1
            
            for k,v in parsed_torso.items():
                setattr(self, k, v)
            
            # Make relations to other objects
            if self.parent != -1:
                self.parent, self.parent_id = self.__class__.all[self.parent], self.parent
            else:
                self.parent_id = -1
            
            self.__class__.joint_dict.setdefault(self.joint, []).append(self)
            
            # Filter out not used parsed_torso
            self.joint_id = parsed_torso.joint_id[:parsed_torso.num_fixed_points]
            self.vectors   = parsed_torso.pts[:parsed_torso.num_fixed_points] # The relative position of the torso's joint to the root joint
            self.limbs = []
            self.sub_torsos = []
            
             
    class Limb:
        all = []
        counter = 0
        joint_dict = {}
        
        def __init__(self, parsed_limb):
            self.__class__.all.append(self)
            self.index = self.__class__.counter
            self.__class__.counter += 1
            
            self.dict = parsed_limb
            for k,v in parsed_limb.items():
                setattr(self, k, v)
                
            self.torso = AIMesh.Torso.all[self.torso_id]
            
            # attachment_joint_id already done
            self.attachment_joint
            self.__class__.joint_dict.setdefault(self.attachment_joint, []).append(self)
            self.joint_id = parsed_limb.joint_id[:parsed_limb.num_segments]
            
            self.vectors = []
            for i in range(parsed_limb.num_segments):
                # Create numpy vectors and scale uniform to value
                self.vectors.append(np.array(parsed_limb.seg[i]) * parsed_limb.seg_len[i])
            #self.vectors = parsed_limb.seg[:parsed_limb.num_segments]
            #self.v_scale = parsed_limb.seg_len[:parsed_limb.num_segments]
            
    
    @classmethod
    def parse_cal(cls, cal_file, subobjects):
        with open(cal_file, "rb") as file:
            binBytes = file.read(-1)
            data = cal_file_parser(binBytes)
            print("\n Cal File: \n")
            print(data)
            
            sub_by_joint = {}
            for sub in subobjects:
                sub_by_joint.setdefault(sub.joint_id, []).append(sub)
            
            torsos = [cls.Torso(t) for t in data.pTorsos]
            limbs = [cls.Limb(t) for t in data.pLimbs]
            Skeleton = cls.Skeleton(torsos)
            
            for sub in subobjects:
                sub._localMatrix[:3,3] = sub.segment.offset
                
            print("Torsos and limbs", len(torsos), len(data.pLimbs))
            
        
        
def parse_bin(binData):
    global binBytes
    global normOffset
    global faceOffset
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
    # Version 3 ends here!
    mat_flags  = unpack('<L', binBytes[110:114])
    # Depending on older/newer models this value varies
    # Even if version=4
    # For newer cases this is 16, saw this also with 8 and without amat_size!
    amat_off   = unpack('<L', binBytes[114:118])
    # This might not exist, and if it exists it is strangly = mod_size
    amat_size  = unpack('<L', binBytes[118:122])
    # Some version 4 models end here!
    mesh_off, \
    submeshlist_off, \
    meshes = unpack('<LLH', binBytes[122:132])
    
    
    materials  = prep_materials(binBytes[matOffset:uvOffset], numMats, binData.name)
    uvs        = prep_uvs(binBytes[uvOffset:vhotOffset])
    vhots      = prep_vhots(binBytes[vhotOffset:vertOffset])
    verts      = prep_verts(binBytes[vertOffset:lightOffset])
    faces      = prep_faces(binBytes[faceOffset:nodeOffset], version)
    faceRefs  = prep_face_refs(binBytes[nodeOffset:]) # Is this not to much?

    print("LOCALS", "="*20)
    for k,v in locals().items():
        print(k, v)
    Nodes = Node.AllNodes
    Nodes.sort(key=lambda n: n.nd_index)   

    NodeDict = {}
    for node in Nodes:
        NodeDict[node.position] = node

    for node in Nodes:
        if node.nd_type == MD_NODE_SPLIT:
            node._node_behind = NodeDict[node.node_behind]
            node._node_front = NodeDict[node.node_front]
            NodeDict[node.node_front].parent = node.position
            NodeDict[node.node_behind].parent= node.position
        elif node.nd_type == MD_NODE_CALL:
            NodeDict[node.node_called].parent = node.position
        elif node.nd_type == MD_NODE_SUBOBJ:   
            node.main_sub_node = NodeDict[node.position+3]
            node.main_sub_node.parent = node.position
            if node.position == 0:
                node.parent = -1
    for node in Nodes:
        print(node,"\n" + ("="*20))
    subobjects = prep_subobjects(
        binBytes[subobjOffset:matOffset],
        faceRefs,
        faces,
        materials,
        vhots)
    object_data = (bbox, subobjects, verts, uvs, materials)
    return object_data
    #for i in range((faceOffset-normOffset)//12):
    #    if i not in Norms:
    #        Norms[i] = getNormalByIndex(i)


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
    
    binData = open(file_path, 'r+b')
    typeID = binData.read(4)
    binData.seek(0)
    if typeID == b'LGMD':
        object_data = parse_bin(binData)
    elif typeID == b'LGMM':
        object_data = AIMesh.parse_bin(binData)
    else:
        raise TypeError("Wrong format.")
        
        
        
# /////////////////////////////BLENDER///////////////////////////////////////////////    
# Create split splanes and face groups    
if __name__ == '__main__' and typeID == b'LGMD':
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
                    