import sys
import struct
import numpy
import os

if len(sys.argv) < 2:
    print('Missing required argument')
    exit()

BWT_DELIMITER = 0
BWT_CHUNK_SIZE = 64 * 1024
MAX_DIC_LENGTH = 65536
READ_WIDTH = 2

FILE_TYPE = 0
class BufferedOutput():
    def __init__(self, file, passthru=True):
        self.file = file
        self.buffer = []
        self.passthru = passthru

    def write(self, out):
        if self.passthru:
            self.file.write(struct.pack('>H', out))
            return
        self.buffer += list(struct.pack('>H', out))

        while(len(self.buffer) > 0 and len(self.buffer) % 4 == 0):
            b0 = (self.buffer[0] << 4) & 0xff
            b1 = (self.buffer[1] << 4) & 0xff
            b1_hi = self.buffer[1] & 0xf0
            b1_lo = self.buffer[1] & 0x0f
            b1_hi = (b1_hi >> 4) & 0xff
            self.buffer[0] = b0 | b1_hi
            self.buffer[1] = (b1 | b1_lo) & 0xf0
            self.buffer[1] = self.buffer[1] | self.buffer[2]
            self.buffer[2] = self.buffer[3]

            self.file.write(bytes(self.buffer[0:3]))
            if len(self.buffer) > 4:
                self.buffer = self.buffer[3:]
            else:
                self.buffer = []

    def flush(self):
        if len(self.buffer) != 0 and len(self.buffer) < 4 and not self.passthru:
            self.write(0)

class BWT:
    def encode(self, s):
        result = []
        #Use a suffix array
        suffixes = sorted([(s[i:], i) for i in range(0, len(s) + 1)])
        suffixes = map(lambda x: x[1], suffixes)
        for i in suffixes:
            if i == 0:
                result.append(BWT_DELIMITER)
            else:
                result.append(s[i - 1])
        return result

class LZW:
    def __init__(self):
        self.reset()

    def encode(self, c, out_file):
        c = chr(c)
        if (self.m + c) in self.dic:
            self.m = self.m + c
        else:
            if c not in self.dic:
                print('Invalid symbol ' + str(c) + ' found in file. Maybe a character outside ASCII range was used in the original file. This shouldn\'t happen since the compressor runs byte-wise.')
                return False
            out_file.write(self.dic[self.m])
            if len(self.dic) < MAX_DIC_LENGTH:
                self.dic[self.m + c] = len(self.dic)
            self.m = c

    def get_final(self):
        return self.dic[self.m]
    
    def get_dic(self):
        return self.dic

    def reset(self):
        self.m = ''
        self.dic = {'': 0}
        for i in range(0, 256):
            self.dic[chr(i)] = len(self.dic)

if os.path.getsize(sys.argv[1]) > BWT_CHUNK_SIZE:
    FILE_TYPE = 1

if FILE_TYPE == 0:
    BWT_CHUNK_SIZE = 64 * 1024
    MAX_DIC_LENGTH = 4096
    READ_WIDTH = 3
elif FILE_TYPE == 1:
    BWT_CHUNK_SIZE = 64 * 1024
    MAX_DIC_LENGTH = 65536
    READ_WIDTH = 2

in_file = open(sys.argv[1], mode='rb')
outfile_name = ''.join(sys.argv[1].split('.')[0:-1]) + '.lz'
out_file = open(outfile_name, mode='wb')
passthru = False
if READ_WIDTH == 2:
    passthru = True
buffered_out = BufferedOutput(out_file, passthru)
buffer = []
bwt = BWT()
lzw = LZW()

out_file.write(bytes([FILE_TYPE]))
while True:
    buffer = in_file.read(BWT_CHUNK_SIZE)
    if not buffer:
        break
    transformed = bwt.encode(buffer)
    for t in transformed:
        result = lzw.encode(t, buffered_out)
        if result == False:
            print('Error detected. Compression failed.')
            in_file.close()
            out_file.close()
            exit()
    buffered_out.write(lzw.get_final())
    buffered_out.write(0)
    lzw.m = ''
        
buffered_out.flush()

in_file.close()
out_file.close()
