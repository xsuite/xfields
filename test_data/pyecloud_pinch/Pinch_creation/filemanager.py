### Adapted from https://github.com/PyECLOUD/myfilemanager.py

import h5py
import numpy

class obj_from_dict:
    def __init__(self, mydict):
        for key in mydict.keys():
            setattr(self, key, mydict[key])

def h5_to_dict(filename, group=None):
    fid = h5py.File(filename, 'r')
    if group == None :
        grp = fid 
    else:
        grp = fid[group]
    mydict={}
    for key in grp.keys():
        mydict[key] = grp[key][()]
    return mydict

def h5_to_obj(filename, group=None):
    return obj_from_dict(h5_to_dict(filename, group=group))

def overwrite(dict_save, filename, group=None, verbose=False):
    with h5py.File(filename, 'a') as fid:
        if group == None :
            grp = fid 
        else:
            grp = fid[group]

        for kk in dict_save.keys():
            if verbose: print('Overwriting '+kk)
            grp[kk][...] = dict_save[kk]

def dict_to_h5(dict_save, filename, compression_opts=4, group=None, readwrite_opts='w', verbose=False):
    with h5py.File(filename, readwrite_opts) as fid:
        if group == None :
            grp = fid 
        else:
            grp = fid.create_group(group)

        for kk in dict_save.keys():
            if verbose: print('Writing '+kk)
            if isinstance(dict_save[kk], numpy.ndarray):
                dset = grp.create_dataset(kk, shape=dict_save[kk].shape, dtype=dict_save[kk].dtype, compression='gzip', compression_opts=compression_opts)
                dset[...] = dict_save[kk]
            else:
                grp[kk] = dict_save[kk]

def print_h5(filename):
    def print_this(x,y): print(x,y)
    with h5py.File(filename,'r') as fid:
        fid.visititems(print_this)
