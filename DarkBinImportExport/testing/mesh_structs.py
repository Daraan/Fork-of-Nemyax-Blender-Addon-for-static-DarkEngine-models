try:
    from .extract_bin_struct import *
except ImportError:
    from extract_bin_struct import *
    
# =============================================
# Bin file extraction    

# A face
mms_pgon = bin_struct(v         ="<HHH", 
                      smatr_id  = ushort,      # necessary if doing global sort
                      d         = bfloat,      # plane equation coeff to go with normal
                      norm      = ushort,      # index of pgon normal
                      pad       = ushort
                      )
 
# Faces that belong to a region or similar 
mms_data_chunk = bin_struct(pgons         = ushort, 
                            pgon_start    = ushort,
                            verts         = ushort,
                            vert_start    = ushort,
                            weight_start  = ushort,
                            pad           = ushort)

# mms_ssmatr size 40                       
mms_ssmatr = bin_struct(name    = string_struct("<16c"),  
                        handle  = ulong,   # texture handle or 0bgr
                        uv      = bfloat,
                        type    = uchar,     #type 0 = texture, 1 = virtual color
                        num_smatsegs = uchar,
                        map_start    = uchar,
                        flags        = uchar,
                        # this field only relevent if smatsegs are laid
                        # out in material order, so pgons and verts will be consecutive per material.
                        data = mms_data_chunk 
                        )                                    

# Caps
kUseAlpha               = 0x00000001,   # 32 bit long
kUseSelfIllumination    = 0x00000002,
kAmaBigDude             = 0xFFFFFFFF         

#S = string_struct("<6s")
#print(S(b'hello\x00'))

# mms_smatr size 56
# single mat regions          
mms_smatr = bin_struct(name         = string_struct("<16c"),  # texture name 
                        dwCaps      = uint,
                        Alpha       = bfloat,
                        Illum       = bfloat,
                        dwForRent   = uint,
                        handle      = ulong,   # texture handle or 0bgr
                        uv          = bfloat,    # carefull union/ulong!
                        type        = uchar,                            
                        num_smatsegs= uchar,
                        map_start = uchar,
                        flags = uchar,
                        data = mms_data_chunk 
                        )

# Flags
MMSEG_FLAG_STRETCHY = 0x1  # segment composed of stretchy polygons  
# mms_segment size 20                     
mms_segment = bin_struct(bbox           = ulong,  # hmm ?
                          joint_id      = uchar,
                          num_smatsegs  = uchar,
                          map_start     = uchar,
                          flags         = uchar,
                          data          = mms_data_chunk
                        )

# mms_smatseg size 16  
# Single material segment       
mms_smatseg = bin_struct(
                          data     = mms_data_chunk,
                          smatr_id = ushort,
                          seg_id   = ushort
                        )


mms_polysort_elt = bin_struct(depth  = bfloat,
                              index  = ushort,
                              kind   = uchar, # 1 poly, attach 2 ??
                              pad    = uchar
                        )

mms_uvn = bin_struct(u = bfloat,
                     v = bfloat,
                     norm = ulong)


# =============================================
# Cal file

# From Motion.h
"""
# Define a torso segment as follows:

# parent or not. (e.g. abdomen is relative to pelvis).
# fixed points. (e.g. hips, abdomen, shoulders, neck).


// The pelvis position (relative to the world) changes from frame to frame,
// but the abdomen position (relative to the pelvis) is fixed, so for 
// each abdomen frame all we need to store is a relative rotation,
// not a torso_frame struct.
//
// So if a torso has a parent, its per-frame data is just a quaternion. 
// If not, it's a torso_frame struct.
"""

torso = bin_struct(
   joint    = bint,   # The root joint index of this torso. Init to 0,0,0 for parent == -1 to get zero - positioned skeleton
   parent   = bint,	  # index into torso list (it's parent) or -1.
   
   num_fixed_points = bint,         # count of joints of this torso (maxed to 16)
   joint_id         = (16, bint),   # remap of joints, could be checked for uniqueness for sanity checks)
   pts              = (16, "<fff")  # The relative position of the torso's joint to the root joint
)

limb = bin_struct(
   torso_id     = bint,			# which torso does this limb hang off?
   bend         = uint,	        # Which way does it bend? 0 for arms, 1 for legs. # Sometimes junk
   num_segments = bint,         # count of joints in this limb
   
   attachment_joint = short,    # Joint Where is it attached too
   joint_id         = (16, ushort), # indices of the joints of this limb
   seg              = (16, "<fff"), # relative to the previous limb's joint! # Uniform vectors!  
   seg_len          = (16, "<f")    # Lengths of the segment, to scale vectors
   
)

# From creature.h
# Header and file end
sCreatureLengths = bin_struct(
   version = bint,
   nTorsos = bint,
   nLimbs  = bint,
   pTorsos = ("nTorsos", torso),
   pLimbs  = ("nLimbs", limb),
   # how much limbs off the root torso are scaled from original
   # motion capture skeleton. basically an AI scale (compared to the motion capture actor)
   primScale = bfloat  
)


if __name__ == '__main__':
    file_path = r"C:\Spiele\Dark Projekt\RES\robotwor.cal"
    
    __sCreatureHeader = bin_struct(version = uint,
       nTorsos = bint,
       nLimbs = bint
    )
    
    with open(file_path, "rb") as f:
        binBytes = f.read(-1)
        if False:
            header = __sCreatureHeader(binBytes)
            print("Header\n", header)
            
            sCreatureMain = bin_struct(
               pTorsos = (header.nTorsos, torso),
               pLimbs = (header.nLimbs, limb),
               primScale = bfloat   # how much limbs off the root toro are scaled from original # motion capture skeleton.
            )
            main = sCreatureMain(binBytes[header._size:])
            print(main)
            print("\n====================================\n")


        print("size = ,",sCreatureLengths.size)
        print(sCreatureLengths(binBytes))
