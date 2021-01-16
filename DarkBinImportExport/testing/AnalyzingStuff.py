import bpy
import mathutils as mu
from math import degrees
import bmesh
import numpy as np
from itertools import product
obj = bpy.context.object
faces = obj.data.polygons

obj = bpy.context.object
bpy.ops.object.mode_set(mode='OBJECT')
me = obj.data
bm = bmesh.new()
bm.from_object(obj, bpy.context.evaluated_depsgraph_get())
bm.faces.ensure_lookup_table()
faces = bm.faces

f1 = faces[37]
f2 = faces[57]
f3 = faces[51]
f4 = faces[52]

n1 = f1.normal
n2 = f2.normal

def FaceVisibleSimple(f1, f2):
    """
    Checks if a face f2 is visible through a face 1. Ignoring obstacles.
    """
    n1 = -f1.normal
    n2 = f2.normal
    if type(f1) == bpy.types.MeshPolygon:
        p1 = f1.center
        p2 = f2.center
    else:
        p1 = f1.calc_center_bounds()
        p2 = f2.calc_center_bounds()
    
    d1 = (n1 @ p1)
    d2 = (n2 @ p2)

    r1 = (n1 @ p2) - d1

    r2 = (n2 @ p1) - d2
    return r1 > 0 and r2 > 0 and not p1 == p2 # Double sided and to prevent numerical errors.

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
        d1 = n1 @ vs1[0].co
        d2 = n2 @ vs2[0].co
    else:
        ps1 = np.array([v.co for v in f1.verts])
        ps2 = np.array([v.co for v in f2.verts])

        # d is constant, could jsut use one vertices.
        # slightly variable due to numerical fluctuations.
        d1 = n1 @ f1.calc_center_bounds()
        d2 = n2 @ f2.calc_center_bounds()

    r1 = np.dot(ps2, n1) - d1
    r2 = np.dot(ps1, n2) - d2
    
    #print(r1, r2)
    # Need to cope with numerical Xe-7 close to 0 errors.
    # should also cope for double sided but maybe np.array_equal/allclose could be needed
    return (r1 > 1e-6).any() and (r2 > 1e-6).any() 


print("Angle False\t", FaceVisible(faces[-2], faces[1], obj))
print("Angle False\t", FaceVisible(faces[-2], faces[2], obj))


if True:
    #foo = np.vectorize(FaceVisible, otype='bool')
    #F1 = np.array(bpy.context.object.data.polygons)
    Res = np.zeros((len(bpy.context.object.data.polygons), len(bpy.context.object.data.polygons)), dtype=int)
    for f1, f2 in product(bpy.context.object.data.polygons, repeat=2):
        if f1 == f2:
            continue
        Res[f1.index, f2.index] = FaceVisible(f1, f2, obj)
   
    
    ##
if False:  
        def GetBadness(row, column):
            badness = []
            for i in range(len(column)):
                badness.append(sum(row[i:]) + sum(column[:i]))
            return np.array(badness, dtype=int)
        
        opt_me = np.ndarray((Res.shape), dtype=int)
        for i in range(Res.shape[0]):
            opt_me[:,i] = GetBadness(Res[i], Res[:,i])
    
    print("diag sum", np.sum(np.diag(opt_me)))
    
    #print(opt_me, sum(np.diag(opt_me)))
    if False:
        # Select alt
        sum_ = 0
        for i in range(opt_me.shape[0]):
            col = opt_me[:,i]
            row = np.argmin(col)
            low = col[row]
            sum_ += low
            opt_me = np.delete(opt_me, row, axis=0)
        print(sum_, opt_me)



if False:
    i_len = Res.shape[0] - 1
    F = np.flip(Res, axis=1)
    Rmax = i_len - np.argmax(F, axis=1)
    Cmax = np.argmax(Res, axis=0)
    # Correct false max
    # If True then there is True in Res, else it a false positive max/min entry.
    Rfalse = (Rmax == i_len) & Res[:,-1]
    Cfalse = (Cmax == 0) & Res[0]
    Res[0][Cfalse] = -1
    
    new = np.vstack((range(Res.shape[0]), Rmax, Cmax))
    print(new)


np.set_printoptions(threshold=5000, linewidth=60000)


# Find column / face that is seen wrongly by others

bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='DESELECT')
obj.face_maps["FrontNode"].select =True
bpy.ops.object.face_map_select(True)

selfaces = []
for f in faces:
    if f.select:
        print(f.index)
        selfaces.append(f.index)
        
back_t = np.triu(Res)[np.ix_(selfaces,selfaces)]#[:]
print(back_t)

print("ix", np.ix_(selfaces,selfaces))

sum_tri = np.sum(np.triu(back_t), axis=0)
sum_tri_rows = np.sum(np.triu(back_t), axis=1)
print("upper tri cols", sum_tri, selfaces[np.argmax(sum_tri)])
print("upper tri", sum_tri_rows, selfaces[np.argmax(sum_tri_rows)])