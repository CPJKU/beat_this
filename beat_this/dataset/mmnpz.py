"""
Support for memory-mapping uncompressed .npz files.
"""
from collections.abc import Mapping
from zipfile import ZipFile
from threading import Lock
from contextlib import nullcontext
import struct

import numpy as np


class MemmappedNpzFile(Mapping):
    """
    A dictionary-like object with lazy-loading of numpy arrays in the given
    uncompressed .npz file. Upon construction, creates a memory map of the
    full .npz file, returning views for the arrays within on request.

    Attributes
    ----------
    files : list of str
        List of all uncompressed files in the archive with a ``.npy`` extension
        (listed without the extension). These are supported as dictionary keys.
    zip : ZipFile instance
        The ZipFile object initialized with the zipped archive.
    mmap : np.memmap
        The memory map of the full .npz file.
    arrays : dict
        Preloaded or cached arrays.

    Parameters
    ----------
    fn : str or Path
        The zipped archive to open.
    cache : bool, optional
        Whether to cache array objects in case they are requested again.
    preload : bool, optional
        Whether to precreate all array objects upon opening. Enforces caching.
    threadsafe : bool, optional
        Whether to guard multi-threaded access with a lock when not preloading.
    """
    def __init__(self, fn: str, cache: bool = True, preload: bool = False,
                 threadsafe: bool = False):
        self.zip = ZipFile(fn, mode='r')
        self.mmap = np.memmap(fn, mode='r')
        self.cache = cache or preload
        self.preload = preload
        self.files = [zinfo.filename[:-4] for zinfo in self.zip.infolist()
                      if zinfo.filename.endswith('.npy')
                      and zinfo.compress_type == 0]
        if self.preload:
            self.arrays = {name: self.load(name) for name in self.files}
        else:
            self.arrays = {}
        if threadsafe:
            self.lock = Lock()
        else:
            self.lock = nullcontext()

    def load(self, name):
        zinfo = self.zip.getinfo(name + '.npy')
        fp = self.zip.fp
        # parse lengths of local header file name and extra fields
        # (zinfo is based on the global directory, not local header)
        fp.seek(zinfo.header_offset + 26)
        fn_len, extra_len = struct.unpack('<2H', fp.read(4))
        # compute offset of start and end of data
        npy_start = zinfo.header_offset + 30 + fn_len + extra_len
        npy_end = npy_start + zinfo.file_size
        # read NPY header
        fp.seek(npy_start)
        version = np.lib.format.read_magic(fp)
        np.lib.format._check_version(version)
        shape, fortran, dtype = np.lib.format._read_array_header(fp, version)
        # produce slice of memmap
        data_start = fp.tell()
        return self.mmap[data_start:npy_end].view(dtype=dtype).reshape(shape, order='F' if fortran else 'C')

    def close(self):
        if hasattr(self, 'zip'):
            self.zip.close()
        if hasattr(self, 'mmap'):
            self.mmap.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __iter__(self):
        return iter(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, key):
        if self.cache:
            try:
                return self.arrays[key]
            except KeyError:
                pass
        with self.lock:
            array = self.load(key)
        if self.cache:
            self.arrays[key] = array
        return array

    def __contains__(self, key):
        # Mapping.__contains__ calls __getitem__, which could be expensive
        return (key in self.files)
