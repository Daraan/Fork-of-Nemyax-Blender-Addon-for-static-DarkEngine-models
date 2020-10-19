import numpy as np
import zlib
import struct
from gif2numpy import convert as gif2numpy

def _RGBtoHEX(pixel):
    # RGBA, extra int because of numpy unit, int32 problems     
    return ((0xFF << 24) +           # Alpha
            (int(pixel[0]) << 16) + # R
            (int(pixel[1]) << 8) + # G
            (int(pixel[2]) << 0))  # B


def _write_png(buf, width, height):
    # reverse the vertical line order and add null bytes at the start
    width_byte_4 = width * 4
    raw_data = b''.join(
        b'\x00' + buf[span:span + width_byte_4]
        for span in range((height - 1) * width_byte_4, -1, - width_byte_4)
    )

    def png_pack(png_tag, data):
        chunk_head = png_tag + data
        return (struct.pack("!I", len(data)) +
                chunk_head +
                struct.pack("!I", 0xFFFFFFFF & zlib.crc32(chunk_head)))

    return b''.join([
        b'\x89PNG\r\n\x1a\n',
        png_pack(b'IHDR', struct.pack("!2I5B", width, height, 8, 6, 0, 0, 0)),
        png_pack(b'IDAT', zlib.compress(raw_data, 9)),
        png_pack(b'IEND', b'')])


def _saveAsPNG(array, filename):
    if any([len(row) != len(array[0]) for row in array]):
        raise ValueError("Array should have elements of equal size")
    
    # First row becomes top row of image.
    flat = np.flip(array)
    flat = flat.flatten()
    #Big-endian, unsigned 32-byte integer.
    buf = b''.join([struct.pack('>I', ((0xffFFff & i32)<<8)| (i32>>24) )
                    for i32 in flat])   #Rotate from ARGB to RGBA.
    data = _write_png(buf, len(array[0]), len(array))
    f = open(filename, 'wb')
    f.write(data)
    f.close()
    return f.name
    
def convert(img_file, output="temp.png"):
    frames, exts, image_specs = gif2numpy(img_file, BGR2RGB=False)
    # Supports animation - only want 1st frame.
    pic = frames[0]
    exts = exts[0]
    shape = pic.shape[0:2]
    ashex = np.array([_RGBtoHEX(pix) for pix in pic.reshape((-1,3))]).reshape(shape)
    filename = _saveAsPNG(ashex, output)
    # error will be handled outside
    return filename # This is should be same as //output.
