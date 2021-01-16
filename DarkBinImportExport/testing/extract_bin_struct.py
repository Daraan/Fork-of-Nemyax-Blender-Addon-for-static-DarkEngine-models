"""
A simple parser for binary files

bin_struct is a an OrderedDict
constructed either by a single string argument that gets passed to struct.unpack
or multiple key=value pairs.

The keys can be anything you specify the values can be strings for struct.unpack
or another initialized bin_struct type.

Examples:

ushort = bin_struct("H")
bin_float = bin_struct("f") 
bin_string = bin_struct("6c") # see also string_struct

vector = bin_struct("3f")
or
vector = bin_struct(x=bin_float, y=bin_float, z=bin_float)

---

After initializing a bin_struct object can be called to parse the given byte argument
The necessary byte size can be retrieved with bin_struct.size 
and updated with bin_struct.calcsize() if a changed was made to the object.

Out of a bytes object the data then can be retrieved by:
xyz = vector(byte[:vector.size])
Even if more bytes are provided only the onces according to size are evaluated.

my_int = ushort(b'00')
This returns a parsed_struct object also an OrderedDict.

The values can then be retrieved with their keywords
xyz.x, xyz.y, xyz.z
For single valued objects with the keyword value
my_int.value

NOTE: Unlike with struct.unpack single values are not in tuples
for example:
ushort(b'00').value -> 12366 and not (12366,)

-----------------

For values that shall be converted to strings 'c', 's', 'p'

There are the tostring(key="value") or get_string(bytes) methods.
The default decode argument for 's' and 'p' are is set with the variable DECODER = 'ASCII'.
# Others are not tested. I doublt utf works.

parsed_string = bin_string(b'hello\x00')
parsed_string.tostring(s)
bin_struct.get_string(parsed_string.s)
-> "hello"

NOTE: Empty bytes \x00 are removed by default.
    tostring and get_string have a remove=iterable or False parameter which let's you keep these or remove others.
    The default is set as a list: _default_remove = [b'\x00', 0]

easier to use is the string_struct

"""

short = "<h"
ushort = "<H"
bfloat = "<f"
uchar  = "<B"
ulong  = "<L"
uint   = "<I"
bint   = "<i"
char   = "<i"

from struct import calcsize, unpack
from collections import OrderedDict 

_default_remove = [b'\x00', 0]
DECODER = 'ASCII'

class parsed_struct(OrderedDict):
    _printdepth=0
    _mainprinter=None
    def tostring(self, key="value", remove=_default_remove):
        val = self[key]
        if len(val) == 1 and type(val)==int:
            return chr(val)
        return self.get_string(val, remove_empty)
    
    @staticmethod
    def get_string(bs, remove=_default_remove):
        if remove:
            iterr = filter(lambda x: x not in remove, bs)
        else:
            iterr = bs
        result = ""
        for c in iterr:
            if type(c) == int:
                result += chr(c)
            else:
                result += c.decode(DECODER)
        return result    
        
    def __getattr__(self, key):
        return self[key]
    
    def __repr__(self):
        if not self.__class__._mainprinter:
            self.__class__._mainprinter = self
            extra = ""
        else:
            self.__class__._printdepth += 1
            extra = "  " * self.__class__._printdepth
        for k,v in self.items():            
            if isinstance(v, OrderedDict) or isinstance(v, tuple) or isinstance(v, list):
                print(extra, f"{k} : \n", extra, "{", sep="")
                print(extra, v, sep="")
                print(extra +"}")
                
            else:
                print(extra, f"{k:<20}"," : ", v, sep="")
        if self.__class__._mainprinter == self:
            self.__class__._mainprinter = None
            self.__class__._printdepth = 0
        else:
            self.__class__._printdepth -= 1
        return "\r"


class bin_struct(OrderedDict):
    def __init__(self, single=None, **kwargs):
        assert not single or (not kwargs and type(single)==str), "Takes only a single string argument OR key=value arguments"
        if single:
            self["value"] = single
        else:
            super().__init__(**kwargs)
        self.size = self.calcsize()
    
    @classmethod
    def _calcsize(cls, v):
        if type(v) == str:
            return calcsize(v)
        elif type(v) != tuple:
            return v.size
        else:
            key, v = v
            print(key, v)
            if type(key) == int:
                return key * cls._calcsize(v)
            else:
                print("not def", key, type(key))
                return "not defined"
            
    # This is constant
    def calcsize(self):
        size = 0
        for v in self.values():
            vsize = self._calcsize(v)
            print(vsize)
            if type(vsize) == int:
                size += vsize
            else:
                print("not def", vsize, type(vsize))
                return "not defined"
        return size
    
    def parse(self, bytes):
        pos = 0
        finished = parsed_struct()
        for k, v in self.items():
            cycles = 1
            dorepeats = False
            while cycles > 0: # for (repeat, struct) values
            
                if type(v) in (tuple, list):
                    # determine repeats
                    dorepeats = True
                    key, v = v
                    if key == "*":
                        cycles = -1
                    elif type(key) == int:
                        cycles = key
                    else:
                        cycles_from = finished[key]
                        if type(cycles_from) in [int, float]:
                            cycles = int(cycles_from)
                        else:
                            cycles = len(cycles_from)
                # Extract
                if type(v) == str:
                    size = calcsize(v)
                    #print("SIZE", size, len(bytes[pos:pos+size]))
                    data = unpack(v, bytes[pos:pos+size])                    
                    if len(data) == 1:
                        data=data[0]
                else: # isinstance(v, bin_struct):
                    size = v.size
                    data = v.parse(bytes[pos:pos+size])
                
                pos+=size
                cycles -= 1
                
                # Save data
                if dorepeats:
                    finished.setdefault(k, []).append(data)
                    if cycles < 0 and pos >= len(bytes):
                        break
                else:
                    finished[k] = data
                  
        finished._size = self.size
        return finished
        
    def __call__(self, bytes):
        return self.parse(bytes)

class string_struct(bin_struct):
    def __init__(self, single):
        super().__init__(single)

    def parse(self, bytes):
        data = unpack(self["value"], bytes[:self.size])
        if len(data) == 1:
            data=data[0]
        return parsed_struct.get_string(data)
    
    def __call__(self, bytes):
        return self.parse(bytes)
      