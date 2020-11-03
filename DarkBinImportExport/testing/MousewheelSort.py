import bpy
import bmesh


def SortMeshSelected(movement):
    obj = bpy.context.object
    mesh = obj.data
    bpy.ops.object.mode_set(mode='OBJECT')
    bm = bmesh.new()
    bm.from_object(obj, bpy.context.evaluated_depsgraph_get())
    bm.faces.ensure_lookup_table()

    limits = range(len(bm.faces))
    step = movement//abs(movement)
    for f in bm.faces:
        if f.select:
            if f.index + movement not in limits:
                movement_adj = -f.index if step < 0 else limits[-1] - f.index
            else:
                movement_adj = movement
            for i in range(f.index+step, f.index + movement_adj+step, step):
                if bm.faces[i].select == False:
                    bm.faces[i].index -=step
            f.index += movement_adj

    bm.faces.sort()
    bm.to_mesh(mesh)
    bpy.ops.object.mode_set(mode='EDIT')
    obj.data.update()
    return {'FINISHED'}
    


class MESH_OT_sort_faces(bpy.types.Operator):
    """Tooltip"""
    bl_idname = "mesh.sort_faces"
    bl_label = "Sort faces up down"
    bl_options = {'REGISTER', 'UNDO'}
    
    def invoke(self, context, event):
        if event.type =='LEFT_CTRL':
            if event.value == 'PRESS':
                for area in bpy.context.screen.areas:
                    if area.type == 'VIEW_3D':
                        overlay = area.spaces[0].overlay.show_extra_indices = True
            elif event.value == 'RELEASE':
                for area in bpy.context.screen.areas:
                    if area.type == 'VIEW_3D':
                        overlay = area.spaces[0].overlay.show_extra_indices = False
            return {'FINISHED'}
        
        movement = 1
        if event.type == 'WHEELDOWNMOUSE':
            movement = -1
        if event.alt:
            movement *=10

        rv = SortMeshSelected(movement)
 
        if rv == {'CANCELLED'}:
            self.report({'ERROR'}, "Out of range")
        return rv
    

addon_keymaps = []

def register():
    bpy.utils.register_class(MESH_OT_sort_faces)
    # Add the hotkey
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc:
        km = kc.keymaps.new('Mesh', space_type='EMPTY')
        new_events = [
            km.keymap_items.new(MESH_OT_sort_faces.bl_idname, type='WHEELUPMOUSE', value='ANY', ctrl=True),
            km.keymap_items.new(MESH_OT_sort_faces.bl_idname, type='WHEELDOWNMOUSE', value='ANY', ctrl=True),
            km.keymap_items.new(MESH_OT_sort_faces.bl_idname, type='WHEELUPMOUSE', value='ANY', ctrl=True, alt=True),
            km.keymap_items.new(MESH_OT_sort_faces.bl_idname, type='WHEELDOWNMOUSE', value='ANY', ctrl=True, alt=True),
            km.keymap_items.new(MESH_OT_sort_faces.bl_idname, type='LEFT_CTRL', value='ANY')
        ]
        for kmi in new_events:
            addon_keymaps.append((km, kmi))

def unregister():
    bpy.utils.unregister_class(MESH_OT_sort_faces)
    # Remove the hotkey
    for km, kmi in addon_keymaps:
        km.keymap_items.remove(kmi)
    addon_keymaps.clear()

