bl_info = {
    "name": "Dark Engine Static Model",
    "author": "nemyax",
    "version": (0, 5, 8, 20201025), # Using YMD
    "blender": (2, 83, 7),
    "location": "File > Import-Export",
    "description": "Import and export Dark Engine static model .bin",
    "warning": "inofficial version",
    "wiki_url": "https://sourceforge.net/p/blenderbitsbobs/wiki/Dark%20Engine%20model%20importer-exporter/",
    "tracker_url": "",
    "category": "Import-Export"
}

import bpy
from bpy.props import (
    StringProperty,
    EnumProperty,
    BoolProperty)
from bpy_extras.io_utils import (
    ExportHelper,
    ImportHelper,
    path_reference_mode)

from .testing import *

from .io_scene_dark_bin import get_objs_toexport
from importlib import reload


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
        from . import io_scene_dark_bin as darkIO
        reload(darkIO)
        options = {
            'use_collections'       :self.use_collections,
            'fancy_txtrepl'         :self.fancy_txtrepl,
            'convert_gif'           : self.convert_gif,
            'support_3ds_export'   : self.support_3ds_export,}
        msg, result = darkIO.do_import(self.filepath, options)
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
            ("none", "Don't sort", 'Option to sort it manually. With Mesh->Sort elements', 0),
            ("vgs","By vertex group", "".join([
                "Follow the alphabetical order of vertex group names"]), 'GROUP_VERTEX', 1),
            ("bsp","Make splits","".join([
                "Split intersecting faces. Increases poly count!",
                " use only if you need transparency support"]), 'MOD_BEVEL', 2),
            ("BSP", "BSP only", 'Creates BSP tree without sorting', 'NLA', 3),
            ("zdist", "By normals", "Divides the faces by normals and sorts all groups from back to front." ,'NORMALS_FACE', 4),
            ("zBSP", "Normals+BSP", "Combine normal sort+BSP(recommended)" ,'MOD_NORMALEDIT', 5),
            ("kOpt", "Best Sort", "Apply better sorting algorithm. Very slow for high poly." ,'MOD_HUE_SATURATION', 6),
            ("fully", "Best Sort+BSP", "Combine sorter+BSP(recommended)", 'MOD_DATA_TRANSFER', 7)
            ),
            
        default="zdist")                                    
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
            if o.name == "!TO MANY POLYGONS!":
                self.layout.label(text=o.name, icon='ERROR')
            else:
                self.layout.label(text=o.name) 
    
    def execute(self, context):
        from . import io_scene_dark_bin as darkIO
        reload(darkIO)
        # ID number for the string
        # can deprecate that now
        sort_to_int ={
            "none" :0,
            "vgs"  :1,
            "bsp"  :2,
            "BSP"  :3,
            "zdist":4,
            "zBSP" :5,
            "kOpt" :6,
            "fully":7,
        }[self.sorting]
        options = {
            'clear'       :self.clear,
            'bright'      :self.bright,
            'use_origin'  :self.use_origin,
            'sorting'     :self.sorting,
            'export_filter':self.export_filter,
            'node_texture' : self.node_texture,
            'opt_type'     : sort_to_int,}
        print(options['opt_type'], "opt")
        msg, result = darkIO.do_export(self.filepath, options)
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
    from .testing import MousewheelSort
    MousewheelSort.register()


def unregister():
    bpy.utils.unregister_class(ExportDarkBin)
    bpy.utils.unregister_class(ImportDarkBin)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import_bin)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export_bin)
    from .testing import MousewheelSort
    MousewheelSort.unregister()

