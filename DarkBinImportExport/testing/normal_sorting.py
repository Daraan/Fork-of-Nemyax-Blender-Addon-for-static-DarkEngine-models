import bpy
import bmesh
import mathutils as mu
from math import sin, acos, atan2, degrees
import itertools

vector = mu.Vector

def PolarAngles(v):
    r = v.length
    return int(90-degrees(acos(v.z / r))), int(degrees(atan2(v.y, v.x)))

#bpy.ops.mesh.select_similar(type='NORMAL', compare='EQUAL', threshold=1)
bpy.ops.object.mode_set(mode='OBJECT')
obj = bpy.context.object
me = obj.data
bm = bmesh.new()
bm.from_object(obj, bpy.context.evaluated_depsgraph_get())

#bm = bmesh.from_edit_mesh(obj.data)

faces = bm.faces


print("Faces:" , len(faces))
#xfaces.sort(key=lambda f: f.co

#base_vectors = (vector([1,0,0]),
#                vector([0,1,0]),
#                vector([0,0,1]))
                
#base_angles =[(x,y) for x,y in itertools.combinations_with_replacement([-90,0,90],2)]

def PolarAngles(v):
    r = v.length
    return int(90-degrees(acos(v.z / r))), int(degrees(atan2(v.y, v.x)))

glob = 0
X = []
Xm = []
Y = []
Ym = []
Z = []
Zm = []




def normal_filter(f):
    global glob
    n = f.normal
    phi, theta = PolarAngles(n)
    print(phi, theta)
    if theta in range(-45, 46):
        if phi > 66:
            Z.append(f)
        elif phi < -66:
            Zm.append(f)
        else:
            X.append(f)
    elif theta in range(45, 135):
        if phi > 66:
            Z.append(f)
        elif phi < -66:
            Zm.append(f)
        else:
            Y.append(f)
    elif theta in range(-45, -135, -1):
        if phi > 66:
            Z.append(f)
        elif phi < -66:
            Zm.append(f)
        else:
            Ym.append(f)
    else:
        if phi > 66:
            Z.append(f)
        elif phi < -66:
            Zm.append(f)
        else:
            Xm.append(f)

for f in faces:
    normal_filter(f)

X.sort(key=lambda v: v.calc_center_median().x)
Xm.sort(key=lambda v: v.calc_center_median().x, reverse=True)
Y.sort(key=lambda v: v.calc_center_median().y)
Ym.sort(key=lambda v: v.calc_center_median().y, reverse=True)
Z.sort(key=lambda v: v.calc_center_median().z)
Zm.sort(key=lambda v: v.calc_center_median().z, reverse=True)

for f in X:
    f.select = True


ordered = []

for set in [Zm, Ym, Xm, Z, Y, X]:
    print(len(set))
    for f in set:
        ordered.append(f.index)
        
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
    
reorder_faces(bm, ordered)
bm.to_mesh(me)
obj.data.update()

bpy.ops.object.mode_set(mode='EDIT')

if bpy.context.mode == 'EDIT_MESH':
    bmesh.update_edit_mesh(obj.data, True)
