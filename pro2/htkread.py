# -*- coding: utf-8 -*-
# @Author: richman
# @Date:   2017-10-23 13:13:42
# @Last Modified by:   richman
# @Last Modified time: 2017-10-23 13:15:53
import struct
import numpy as np


def readhtk(fname):
    # Read header
    with open(fname, 'rb') as f:
        nSamples, sampPeriod, sampSize, parmKind = struct.unpack(
            ">iihh", f.read(12))

        # Read data
        data = struct.unpack(
            ">%df" % (nSamples * sampSize // 4), f.read(nSamples * sampSize))
        return np.array(data).reshape(nSamples, sampSize // 4)


# x = readhtk("chen_0004092_A.mfcc")
# print(x.shape)
# print(x)
