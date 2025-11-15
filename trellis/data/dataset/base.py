# -*- coding:utf-8 -*-
###
# File: base.py
# Created Date: Friday, November 14th 2025, 12:06:12 am
# Author: iYuqinL
# -----
# Last Modified: 
# Modified By: 
# -----
# Copyright © 2025 iYuqinL Holding Limited
# 
# All shall be well and all shall be well and all manner of things shall be well.
# Nope...we're doomed!
# -----
# HISTORY:
# Date      	By	Comments
# ----------	---	----------------------------------------------------------
###
from typing import Union, List, Tuple, Optional, Dict, Any
from abc import abstractmethod
import traceback
import os
import glob
from easydict import EasyDict
import numpy as np
from torch.utils.data import Dataset

import pyarrow.parquet as pq


class ParquetDatasetBase(Dataset):
    """
    Base class for parquet dataset.
    
    Args
    -----
    datacfg (EasyDict): Configuration dictionary for dataset.
        - rootdir (str): Root directory of dataset.
        - datadirs (list[str]): List of subdirectories in rootdir, each containing data files.
        - parquetdirs (list[str]): List of subdirectories in rootdir, each containing parquet files.
        - parquet_glob (str, optional): Glob pattern to match parquet files. Defaults to "*.parquet".
        
    logfunc (callable, optional): Logging function. Defaults to print.

    warnfunc (callable, optional): Warning logging function. Defaults to print.
    """
    def __init__(self, datacfg: EasyDict, logfunc=print, warnfunc=print):
        super(ParquetDatasetBase, self).__init__()
        self.logfunc, self.warnfunc = logfunc, warnfunc
        self.datacfg = EasyDict(datacfg.copy())

        self.rootdir = self.datacfg.get("rootdir", None)
        assert self.rootdir is not None, f"datacfg must have rootdir item"
        assert os.path.exists(self.rootdir), f"data rootdir {self.rootdir} is not exists"
        
        # datadirs & parquetdirs: must both be present and same length
        # and pair of datadir and parquetdir is a dataset.
        self.datadirs: list[str] = self.datacfg.get("datadirs", None)
        self.parquetdirs: list[str] = self.datacfg.get("parquetdirs", None)
        assert (
            self.datadirs is not None and self.parquetdirs is not None
            and isinstance(self.datadirs, (list, tuple))
            and isinstance(self.parquetdirs, (list, tuple))
            and len(self.datadirs) >= 1
            and len(self.parquetdirs) == len(self.datadirs)
        ), "datacfg must have matching 'datadirs' and 'parquetdirs' lists of same length"

        # options
        self._parquet_glob = self.datacfg.get("parquet_glob", "*.parquet")
        
        # Internal flattened structures:
        # absolute paths of discovered parquet files (global list, across all datasets)
        self._parquet_files: List[str] = []
        # number of rows per parquet file
        self._file_row_counts: List[int] = []
        # for each file in _parquet_files, 
        # which dataset index it belongs to (0..len(parquetdirs)-1)
        self._file_dataset_idx: List[int] = []
        # cumulative prefix sums of rows (length = len(_file_row_counts) + 1)
        self._cumulative_row_counts: List[int] = [0]

        # store resolved absolute datadir paths for each dataset index
        self._abs_datadirs: List[str] = [self._abs_dir(d) for d in self.datadirs]

        # store discovered files per-dataset if you want (not required, but handy)
        self._per_dataset_files: List[List[str]] = (
            [[] for _ in range(len(self.parquetdirs))])

    def _abs_dir(self, dir: str) -> str:
        """
        Return absolute path for dir: if already absolute, keep it; 
        else join with rootdir.
        """
        if os.path.isabs(dir):
            return dir
        return os.path.abspath(os.path.join(self.rootdir, dir))

    def _list_parquet_files_in_dir(self, dirpath: str) -> List[str]:
        """
        Return sorted list (non-recursive) of parquet files matching glob under dirpath.
        """
        if not os.path.isdir(dirpath):
            return []
        pattern = os.path.join(dirpath, self._parquet_glob)
        return sorted(glob.glob(pattern))

    def _count_rows_in_parquet(self, path: str) -> int:
        """
        Use pyarrow.parquet metadata to read number of rows (fast).
        """
        pf = pq.ParquetFile(path)
        return int(pf.metadata.num_rows)

    def _discover_all_parquets(self):
        """
        Scan each parquetdir (one per dataset), find parquet files, 
        count rows and populate flattened lists:
        - self._parquet_files
        - self._file_row_counts
        - self._file_dataset_idx
        - self._per_dataset_files
        - self._cumulative_row_counts
        """
        files = []
        counts = []
        file_dataset_idx = []
        per_ds_files = [[] for _ in range(len(self.parquetdirs))]

        for ds_idx, pdir in enumerate(self.parquetdirs):
            abs_pdir = self._abs_dir(pdir)
            if not os.path.exists(abs_pdir):
                self.warnfunc(f"Warning: parquet dir for dataset {ds_idx} "
                              f"does not exist: {abs_pdir}")
                continue
            pfiles = self._list_parquet_files_in_dir(abs_pdir)
            if len(pfiles) == 0:
                self.warnfunc(f"Warning: no parquet files in {abs_pdir} "
                              f"(pattern {self._parquet_glob})")
            for f in pfiles:
                try:
                    nrows = self._count_rows_in_parquet(f)
                except Exception as e:
                    print(f"Warning: failed to read metadata for parquet file {f}: {e}")
                    continue
                if nrows <= 0:
                    # skip empty files
                    continue
                files.append(os.path.abspath(f))
                counts.append(int(nrows))
                file_dataset_idx.append(ds_idx)
                per_ds_files[ds_idx].append(os.path.abspath(f))

        # assign to internals
        self._parquet_files = files
        self._file_row_counts = counts
        self._file_dataset_idx = file_dataset_idx
        self._per_dataset_files = per_ds_files

        # build cumulative
        cum = [0]
        s = 0
        for c in counts:
            s += c
            cum.append(s)
        self._cumulative_row_counts = cum
    
    # -------------------------
    # Public helpers: locating
    # -------------------------
    def _locate(self, index: int) -> Tuple[int, int, int]:
        """
        Map global index -> (file_idx, row_in_file, dataset_idx)
        - file_idx: index into self._parquet_files
        - row_in_file: row offset within that parquet file
        - dataset_idx: which dataset (index into datadirs/parquetdirs) this file belongs to
        Supports negative indexing.
        Raises IndexError if out of range.
        """
        total = len(self)
        if index < 0:
            index = total + index
        if index < 0 or index >= total:
            raise IndexError(f"index {index} out of range [0, {total})")

        import bisect
        file_idx = bisect.bisect_right(self._cumulative_row_counts, index) - 1
        row_in_file = index - self._cumulative_row_counts[file_idx]
        ds_idx = self._file_dataset_idx[file_idx]
        return int(file_idx), int(row_in_file), int(ds_idx)

    def get_datadir_for_index(self, index: int) -> str:
        """
        Return absolute datadir path for the dataset that contains the given global index.
        Useful so subclasses know where to read the actual data for that index.
        """
        _, _, ds_idx = self._locate(index)
        return self._abs_datadirs[ds_idx]

    def get_parquet_file_for_index(self, index: int) -> Tuple[str, int]:
        """
        Return (parquet_file_path, row_in_file) for a global index.
        """
        file_idx, row_in_file, _ = self._locate(index)
        return self._parquet_files[file_idx], row_in_file
    
    def get_datainfo(self, index, columns: list[str]=None):
        """
        Return dataset_idx, and datainfo in the parquet table
        
        Returns
        -------
        ds_idx (int): dataset index.
        
        datainfo (dict): data information in the parquet table of [index] data.
        """
        file_idx, row_in_file, ds_idx = self._locate(index)
        parquet_file = self._parquet_files[file_idx]
        datainfo = read_parquet_row(parquet_file, row_in_file, columns=columns)
        return ds_idx, datainfo

    @abstractmethod
    def get_item(self, index: Union[int, str]):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_item method")

    @abstractmethod
    def get_datapath(self, index: Union[int, str]):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_datapath method")

    # -------------------------
    # torch Dataset requirements
    # -------------------------
    def __len__(self) -> int:
        """Total number of rows across all parquet files across all datasets."""
        return int(self._cumulative_row_counts[-1])


    def __getitem__(self, index: int):
        try:
            rets = self.get_item(index)
            return rets
        except Exception as e:
            print(e)
            traceback.print_exc()
            datapath = self.get_datapath(index)
            print(f"data path: {datapath}")
            aindex = np.random.randint(0, len(self))
            rets = self.__getitem__(aindex)
            return rets
        
    # ------------------------------------------
    def list_parquet_files(self) -> List[str]:
        """Return discovered parquet files (absolute paths)."""
        return list(self._parquet_files)

    def per_dataset_parquet_files(self) -> List[List[str]]:
        """Return list (per dataset) of parquet files (absolute paths)."""
        return [list(x) for x in self._per_dataset_files]

    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f"  - Total data items: {len(self)}")
        lines.append(f"  - datasets: ")
        
        for i in range(len(self.datadirs)):
            lines.append(f"{i:03d}: datadir={self.datadirs[i]}, "
                         f"parquetdir={self.parquetdirs[i]}")

        return "\n".join(lines)

    def __repr__(self):
        repr_str = self.__str__()
        return repr_str


def read_parquet_row(
        parquet_path: str, row_in_file: int, 
        columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Read a single row from a parquet file using pyarrow,
    without loading the entire file.

    Args
    -----
    parquet_path (str): path to the parquet file.

    row_in_file (int): 0-based row index within this parquet file.

    columns (list): optional list of columns to read (None => read all columns).

    Returns
    --------
    A dict mapping column name -> value for that row.

    Raises
    ------
    IndexError if row_in_file is out of range.

    FileNotFoundError if parquet_path does not exist.

    RuntimeError for other read errors.
    """
    # open parquet file metadata
    pf = pq.ParquetFile(parquet_path)

    # total rows in file
    total_rows = int(pf.metadata.num_rows)
    if row_in_file < 0 or row_in_file >= total_rows:
        raise IndexError(
            f"row_in_file {row_in_file} out of range "
            f"[0, {total_rows}) for file {parquet_path}")

    # find which row_group contains this row
    rg_count = pf.num_row_groups
    rg_index = None
    rows_seen = 0
    for i in range(rg_count):
        rg_rows = pf.metadata.row_group(i).num_rows
        if row_in_file < rows_seen + rg_rows:
            rg_index = i
            offset_in_rg = row_in_file - rows_seen
            break
        rows_seen += rg_rows

    # defensive: should have found a row_group
    if rg_index is None:
        # should not happen, but handle gracefully
        raise RuntimeError(
            f"Failed to locate row {row_in_file} "
            f"in any row group (total rows {total_rows})")

    # read only the targeted row group and slice out the row
    try:
        table = pf.read_row_group(rg_index, columns=columns)  # pyarrow.Table
    except Exception as e:
        # fallback: try reading whole file (less efficient) if row_group read fails
        try:
            table = pq.read_table(parquet_path, columns=columns)
        except Exception as e2:
            raise RuntimeError(
                f"Failed to read parquet row group or full file: {e}; "
                f"fallback failed: {e2}") from e2

    # slice out the single-row Table
    single = table.slice(offset_in_rg, 1)

    # convert to python dict: column -> single value
    # use to_pydict() => column -> list_of_length_1, take [0]
    pyd = single.to_pydict()
    # transform to scalar per column
    row_dict = {}
    for k, v in pyd.items():
        if isinstance(v, (list, tuple)) and len(v) > 0:
            row_dict[k] = v[0]
        else:
            row_dict[k] = v
    return row_dict
