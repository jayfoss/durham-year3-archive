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
class BufferedInput():
    def __init__(self, file, passthru=True):
        self.file = file
        self.buffer = []
        self.passthru = passthru
        self.total = 0

    def read(self):
        inp = self.file.read(READ_WIDTH)
        self.total += len(inp)
        if not inp:
            return False
        vals = []
        if READ_WIDTH == 2:
            vals.append(struct.unpack('>H', inp)[0])
        elif READ_WIDTH == 3:
            #print(inp)
            """
            Unpack 3 bytes representing 2x12 bits into 4 bytes representing 2x16 bits
            """
            b3 = inp[2]
            b1_hi = inp[1] & 0xf0
            b1_lo = inp[1] & 0x0f
            b2 = b1_lo
            b1 = (b1_hi >> 4) & 0xff
            b1 = (((inp[0] & 0x0f) << 4) & 0xff) | b1
            b0 = (inp[0] >> 4) & 0xff
            vals.append(struct.unpack('>H', bytes([b0, b1]))[0])
            vals.append(struct.unpack('>H', bytes([b2, b3]))[0])
        return vals

    def read_block(self):
        while True:
            v = self.read()
            if not v:
                b = self.buffer
                self.buffer = []
                return b
            if READ_WIDTH == 2:
                if v[0] != 0:
                    self.buffer += v
                else:
                    b = self.buffer
                    self.buffer = []
                    return b
            else:
                #Make sure block delimiters are not added to the block
                if v[0] != 0:
                    self.buffer.append(v[0])
                else:
                    b = self.buffer
                    self.buffer = []
                    if v[1] != 0:
                        self.buffer.append(v[1])
                    return b
                if v[1] != 0:
                    self.buffer.append(v[1])
                else:
                    b = self.buffer
                    self.buffer = []
                    return b
                
class BWT:
    def decode(self, e):
        totals = {}
        ranks = []
        for c in e:
            if c not in totals:
                totals[c] = 0
            ranks.append(totals[c])
            totals[c] += 1
        first = {}
        total_count = 0
        for k, v in sorted(totals.items()):
            first[k] = total_count
            total_count += v
        i = 0
        t = [BWT_DELIMITER]
        while e[i] != BWT_DELIMITER:
            c = e[i]
            u = t
            t = [c]
            t += u
            i = first[c] + ranks[i]
        return t[0:-1]

class LZW:
    def __init__(self):
        self.reset()

    def decode(self, buffer):
        out_buffer = ''
        self.prev = buffer[0]
        out_buffer += self.dic[self.prev]
        for i in range(1, len(buffer)):
            self.curr = buffer[i]
            if self.curr == len(self.dic):
                self.dic.append(self.dic[self.prev] + self.dic[self.prev][0])
                out_buffer += self.dic[self.prev] + self.dic[self.prev][0]
            else:
                e = self.dic[self.curr]
                out_buffer += e
                s0 = e[0]
                self.dic.append(self.dic[self.prev] + s0)
            self.prev = self.curr
        out = []
        for i in out_buffer:
            out.append(ord(i))
        return out
        
    def get_final(self):
        return self.dic[self.m]
    
    def get_dic(self):
        return self.dic

    def reset(self):
        self.prev = -1
        self.curr = -1
        self.dic = ['']
        for i in range(0, 256):
            self.dic.append(chr(i))

in_file = open(sys.argv[1], mode='rb')
outfile_name = ''.join(sys.argv[1].split('.')[0:-1]) + '-decoded.tex'
out_file = open(outfile_name, mode='wb')
buffered_in = BufferedInput(in_file, True)
lzw = LZW()
bwt = BWT()
ftype = bytearray(b'\x00')
ftype.extend(in_file.read(1))
FILE_TYPE = struct.unpack('>H', bytes(ftype))[0]
if FILE_TYPE == 0:
    BWT_CHUNK_SIZE = 64 * 1024
    MAX_DIC_LENGTH = 4096
    READ_WIDTH = 3
elif FILE_TYPE == 1:
    BWT_CHUNK_SIZE = 64 * 1024
    MAX_DIC_LENGTH = 65536
    READ_WIDTH = 2

while True:
    block = buffered_in.read_block()
    if len(block) == 0:
        break
    block = lzw.decode(block)
    block = bwt.decode(block)
    out_file.write(bytes(block))

in_file.close()
out_file.close()
