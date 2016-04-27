#imports
import pandas as pd
import numpy as np

from magellan.blocker.blocker import Blocker
from magellan.core.mtable import MTable
import magellan as mg
import math
import pyprind

import multiprocessing
from multiprocessing import Pool
#from multiprocessing.dummy import Pool
import time
import pathos.multiprocessing as mp
from itertools import chain
    
#def block_data_frames_skd(args_tuple):
#    l_df, r_df, l_block_attr, r_block_attr = args_tuple
#    candset = pd.merge(l_df, r_df, left_on=l_block_attr, right_on=r_block_attr,
#                       suffixes=('_ltable', '_rtable'))
#    return candset

class AttrEquivalenceBlocker(Blocker):

    def block_tables(self, ltable, rtable, l_block_attr, r_block_attr,
                     l_output_attrs=None, r_output_attrs=None):
        """
        Block tables based on l_block_attr, r_block_attr equivalence (similar to equi-join)

        Parameters
        ----------
        ltable, rtable : MTable
            Input MTables
        l_block_attr, r_block_attr : string,
            attribute names in ltable, rtable
        l_output_attrs, r_output_attrs : list (of strings), defaults to None
            attribute names to be included in the output table

        Returns
        -------
        blocked_table : MTable
            Containing tuple pairs whose l_block_attr and r_block_attr values are same

        Notes
        -----
        Output MTable contains the following three attributes
            * _id
            * id column from ltable
            * id column from rtable

        Also, the properties of blocked table is updated with following key-value pairs
            * ltable - ref to ltable
            * rtable - ref to rtable
            * key
            * foreign_key_ltable - string, ltable's  id attribute name
            * foreign_key_rtable - string, rtable's id attribute name
        """

        # do integrity checks
        l_output_attrs, r_output_attrs = self.check_attrs(ltable, rtable, l_block_attr, r_block_attr,
                                                     l_output_attrs, r_output_attrs)
        # remove nans
        l_df = self.rem_nan(ltable, l_block_attr)
        r_df = self.rem_nan(rtable, r_block_attr)
	
	t0 = time.time()
        candset = pd.merge(l_df, r_df, left_on=l_block_attr, right_on=r_block_attr,
                           suffixes=('_ltable', '_rtable'))
	t1 = time.time()
        # get output columns
        retain_cols, final_cols = self.output_columns(ltable.get_key(), rtable.get_key(), list(candset.columns),
                                                   l_output_attrs, r_output_attrs)
	t2 = time.time()
        candset = candset[retain_cols]
	t3 = time.time()
        candset.columns = final_cols
	t4 = time.time()
        candset = MTable(candset)
	t5 = time.time()

        # set metadata
        candset.set_property('ltable', ltable)
        candset.set_property('rtable', rtable)
        candset.set_property('foreign_key_ltable', 'ltable.'+ltable.get_key())
        candset.set_property('foreign_key_rtable', 'rtable.'+rtable.get_key())	
	t6 = time.time()
	print "Time taken to merge A and B:", (t1 - t0)
	print "Time taken to get output cols:", (t2 - t1)
	print "Time taken to project C cols:", (t3 - t2)
	print "Time taken to set C final cols:", (t4 - t3)
	print "Time taken to create table C:", (t5 - t4)
	print "Time taken to set props for C:", (t6 - t5)

        return candset


    # blocking over candidate set
    def block_candset(self, vtable, l_block_attr, r_block_attr):
        """
        Block candidate set (virtual MTable) based on l_block_attr, r_block_attr equivalence (similar to equi-join)

        Parameters
        ----------
        vtable : MTable
            Input candidate set
        l_block_attr, r_block_attr : string,
            attribute names in ltable, rtable

        Returns
        -------
        blocked_table : MTable
            Containing tuple pairs whose l_block_attr and r_block_attr values are same

        Notes
        -----
        Output MTable contains the following three attributes
            * _id
            * id column from ltable
            * id column from rtable

        Also, the properties of blocked table is updated with following key-value pairs
            * ltable - ref to ltable
            * rtable - ref to rtable
            * key
            * foreign_key_ltable - string, ltable's  id attribute name
            * foreign_key_rtable - string, rtable's id attribute name
        """
	start_time = time.time()
        # do integrity checks
        ltable = vtable.get_property('ltable')
        rtable = vtable.get_property('rtable')

        self.check_attrs(ltable, rtable, l_block_attr, r_block_attr, None, None)
        l_key = 'ltable.' + ltable.get_key()
        r_key = 'rtable.' + rtable.get_key()

	t000 = time.time()
	print "Time taken to do integrity checks:", (t000 - start_time)
        # convert to dataframes
        l_df = ltable.to_dataframe()
        r_df = rtable.to_dataframe()

	t001 = time.time()
	print "Time taken to convert tables A and B to data frames:", (t001 - t000)
        # set index for convenience
        l_df.set_index(ltable.get_key(), inplace=True)
        r_df.set_index(rtable.get_key(), inplace=True)

	t002 = time.time()
	print "Time taken to set indexes for tables A and B:", (t002 - t001)
        #if mg._verbose:
        #    count = 0
        #    per_count = math.ceil(mg._percent/100.0*len(vtable))
        #    print per_count

        #elif mg._progbar:
        #    bar = pyprind.ProgBar(len(vtable))


        # keep track of valid ids
        valid = []
        # iterate candidate set and process each row
        for idx, row in vtable.iterrows():

            #if mg._verbose:
            #    count += 1
            #    if count%per_count == 0:
            #        print str(mg._percent*count/per_count) + ' percentage done !!!'
            #elif mg._progbar:
            #    bar.update()

            # get the value of block attribute from ltuple
            l_val = l_df.ix[row[l_key], l_block_attr]
            r_val = r_df.ix[row[r_key], r_block_attr]
            if l_val != np.NaN and r_val != np.NaN:
                if l_val == r_val:
                    valid.append(True)
                else:
                    valid.append(False)
            else:
                valid.append(False)
        
	t6 = time.time()
	print "Time taken to get valid ids:", (t6 - t002)
        # should be modified
        if len(vtable) > 0:
            out_table = MTable(vtable[valid], key=vtable.get_key())
        else:
            out_table = MTable(columns=vtable.columns, key=vtable.get_key())
	t7 = time.time()
	print "Time taken to create mtable for candset:", (t7 - t6)
        out_table.set_property('ltable', ltable)
        out_table.set_property('rtable', rtable)
        out_table.set_property('foreign_key_ltable', 'ltable.'+ltable.get_key())
        out_table.set_property('foreign_key_rtable', 'rtable.'+rtable.get_key())
	end_time = time.time()
	print "Time taken to set properties of candset: ", (end_time - t7)
	print "Total time to block candset: ", (end_time - start_time)
        return out_table

    # blocking over tuples
    def block_tuples(self, ltuple, rtuple, l_block_attr, r_block_attr):
        """
        Block tuples based on l_block_attr and r_block_attr

        Parameters
        ----------
        ltuple, rtuple : pandas Series,
            Input tuples
        l_block_attr, r_block_attr : string,
            attribute names in ltuple and r_tuple

        Returns
        -------
        status : boolean,
         True is the tuple pair is blocked (i.e the values in l_block_attr and r_block_attr are different)
        """
        return ltuple[l_block_attr] != rtuple[r_block_attr]

    # check integrity of attrs in l_block_attr, r_block_attr
    def check_attrs(self, ltable, rtable, l_block_attr, r_block_attr, l_output_attrs, r_output_attrs):

        # check keys are set
        assert ltable.get_key() is not None, 'Key is not set for left table'
        assert rtable.get_key() is not None, 'Key is not set for right table'

        # check block attributes form a subset of left and right tables
        if not isinstance(l_block_attr, list):
            l_block_attr = [l_block_attr]
        assert set(l_block_attr).issubset(ltable.columns) is True, 'Left block attribute is not in left table'

        if not isinstance(r_block_attr, list):
            r_block_attr = [r_block_attr]
        assert set(r_block_attr).issubset(rtable.columns) is True, 'Right block attribute is not in right table'

        # check output columns form a part of left, right tables
        if l_output_attrs:
            if not isinstance(l_output_attrs, list):
                l_output_attrs = [l_output_attrs]
            assert set(l_output_attrs).issubset(ltable.columns) is True, 'Left output attributes ' \
                                                                         'are not in left table'
            l_output_attrs = [x for x in l_output_attrs if x not in [ltable.get_key()]]

        if r_output_attrs:
            if not isinstance(r_output_attrs, list):
                r_output_attrs = [r_output_attrs]
            assert set(r_output_attrs).issubset(rtable.columns) is True, 'Right output attributes ' \
                                                                         'are not in right table'
            r_output_attrs = [x for x in r_output_attrs if x not in [rtable.get_key()]]

        return l_output_attrs, r_output_attrs



    # get output columns
    def output_columns(self, l_key, r_key, col_names, l_output_attrs, r_output_attrs):

        ret_cols = []
        fin_cols = []

        # retain id columns from merge
        ret_l_id = [self.retain_names(x, col_names, '_ltable') for x in [l_key]]
        ret_r_id = [self.retain_names(x, col_names, '_rtable') for x in [r_key]]
        ret_cols.extend(ret_l_id)
        ret_cols.extend(ret_r_id)

        # retain output attrs from merge
        if l_output_attrs:
            ret_l_col = [self.retain_names(x, col_names, '_ltable') for x in l_output_attrs]
            ret_cols.extend(ret_l_col)
        if r_output_attrs:
            ret_r_col = [self.retain_names(x, col_names, '_rtable') for x in r_output_attrs]
            ret_cols.extend(ret_r_col)

        # final columns in the output
        fin_l_id = [self.final_names(x, 'ltable.') for x in [l_key]]
        fin_r_id = [self.final_names(x, 'rtable.') for x in [r_key]]
        fin_cols.extend(fin_l_id)
        fin_cols.extend(fin_r_id)

        # final output attrs from merge
        if l_output_attrs:
            fin_l_col = [self.final_names(x, 'ltable.') for x in l_output_attrs]
            fin_cols.extend(fin_l_col)
        if r_output_attrs:
            fin_r_col = [self.final_names(x, 'rtable.') for x in r_output_attrs]
            fin_cols.extend(fin_r_col)

        return ret_cols, fin_cols

    def get_retain_cols(self, l_key, r_key, col_names, l_output_attrs, r_output_attrs):
        ret_cols = []

        # retain id columns from merge
        ret_l_id = [self.retain_names(x, col_names, '_ltable') for x in [l_key]]
        ret_r_id = [self.retain_names(x, col_names, '_rtable') for x in [r_key]]
        ret_cols.extend(ret_l_id)
        ret_cols.extend(ret_r_id)

        # retain output attrs from merge
        if l_output_attrs:
            ret_l_col = [self.retain_names(x, col_names, '_ltable') for x in l_output_attrs]
            ret_cols.extend(ret_l_col)
        if r_output_attrs:
            ret_r_col = [self.retain_names(x, col_names, '_rtable') for x in r_output_attrs]
            ret_cols.extend(ret_r_col)
	return ret_cols

    def get_final_cols(self, l_key, r_key, l_output_attrs, r_output_attrs):
        fin_cols = []
        # final columns in the output
        fin_l_id = [self.final_names(x, 'ltable.') for x in [l_key]]
        fin_r_id = [self.final_names(x, 'rtable.') for x in [r_key]]
        fin_cols.extend(fin_l_id)
        fin_cols.extend(fin_r_id)

        # final output attrs from merge
        if l_output_attrs:
            fin_l_col = [self.final_names(x, 'ltable.') for x in l_output_attrs]
            fin_cols.extend(fin_l_col)
        if r_output_attrs:
            fin_r_col = [self.final_names(x, 'rtable.') for x in r_output_attrs]
            fin_cols.extend(fin_r_col)
	return fin_cols

    def retain_names(self, x, col_names, suffix):
        if x in col_names:
            return x
        else:
            return str(x) + suffix

    def final_names(self, x, prefix):
        return prefix + str(x)


    def block_data_frames_skd(self, args):
	t0 = time.time()
        l_df, r_df, l_block_attr, r_block_attr, l_key, r_key, l_output_attrs, r_output_attrs = args
	t1 = time.time()
        candset = pd.merge(l_df, r_df, left_on=l_block_attr, right_on=r_block_attr,
                           suffixes=('_ltable', '_rtable'))
	t2 = time.time()
        # get retain columns
        retain_cols = self.get_retain_cols(l_key, r_key, list(candset.columns),
                                                   l_output_attrs, r_output_attrs)
	t3 = time.time()
        candset = candset[retain_cols]
	t4 = time.time()
	print "Time taken to extract args:", (t1 - t0)
	print "Time taken to pd merge:", (t2 - t1)
	print "Time taken to get out cols:", (t3 - t2)
	print "Time taken to project cols:", (t4 - t3)
        return candset

    def block_tables_skd(self, ltable, rtable, l_block_attr, r_block_attr,
                     l_output_attrs=None, r_output_attrs=None):
        """
        Block tables based on l_block_attr, r_block_attr equivalence (similar to equi-join)

        Parameters
        ----------
        ltable, rtable : MTable
            Input MTables
        l_block_attr, r_block_attr : string,
            attribute names in ltable, rtable
        l_output_attrs, r_output_attrs : list (of strings), defaults to None
            attribute names to be included in the output table

        Returns
        -------
        blocked_table : MTable
            Containing tuple pairs whose l_block_attr and r_block_attr values are same

        Notes
        -----
        Output MTable contains the following three attributes
            * _id
            * id column from ltable
            * id column from rtable

        Also, the properties of blocked table is updated with following key-value pairs
            * ltable - ref to ltable
            * rtable - ref to rtable
            * key
            * foreign_key_ltable - string, ltable's  id attribute name
            * foreign_key_rtable - string, rtable's id attribute name
        """

        # do integrity checks
        l_output_attrs, r_output_attrs = self.check_attrs(ltable, rtable, l_block_attr, r_block_attr,
                                                     l_output_attrs, r_output_attrs)
        # remove nans
        l_df = self.rem_nan(ltable, l_block_attr)
        r_df = self.rem_nan(rtable, r_block_attr)
        
	#print 'cpu_count() = %d\n' % multiprocessing.cpu_count() 
	cpu_count = multiprocessing.cpu_count()
	m = int(math.sqrt(cpu_count)) # no. of splits of l_df
	n = cpu_count/m # no. of splits of r_df
        print "m: ", m, ", n: ", n
	t0 = time.time() 
        l_splits = np.array_split(l_df, m)
	t1 = time.time()
        r_splits = np.array_split(r_df, n)
	t2 = time.time()
        l_key = ltable.get_key()
        r_key = rtable.get_key()
        lr_splits = [(l, r, l_block_attr, r_block_attr, l_key, r_key, l_output_attrs, r_output_attrs) for l in l_splits for r in r_splits]
	t3 = time.time()
        #pool = Pool(4)
        pool = mp.ProcessingPool(processes=cpu_count, maxtasksperchild=1)
	t4 = time.time()
        c_splits = pool.map(self.block_data_frames_skd, lr_splits)
	t5 = time.time()
        pool.close()
	t6 = time.time()
        pool.join() 
	t7 = time.time()
        candset = pd.concat(c_splits, ignore_index=True)
	#candset = c_splits[0].append(c_splits[1], ignore_index=True)
	t8 = time.time()
	print "Time taken to split table A:", (t1 - t0)
	print "Time taken to split table B:", (t2 - t1)
	print "Time taken to get AB splits:", (t3 - t2)
	print "Time taken to start workers:", (t4 - t3)
	print "Time taken to get  C splits:", (t5 - t4)
	print "Time taken to close    pool:", (t6 - t5)
	print "Time taken to join     pool:", (t7 - t6)
	print "Time taken to combine splits:", (t8 - t7)
        final_cols = self.get_final_cols(l_key, r_key,
                                                   l_output_attrs, r_output_attrs)
        candset.columns = final_cols
        candset = MTable(candset)

        # set metadata
        candset.set_property('ltable', ltable)
        candset.set_property('rtable', rtable)
        candset.set_property('foreign_key_ltable', 'ltable.'+ltable.get_key())
        candset.set_property('foreign_key_rtable', 'rtable.'+rtable.get_key())
        return candset

    def get_valid_ids(self, args):
        c_df, l_df, r_df, l_key, r_key, l_block_attr, r_block_attr = args
	#print "l_key: ", l_key, "r_key: ", r_key, "l_block_attr: ", l_block_attr, "r_block_attr: ", r_block_attr
        #if mg._verbose:
        #    count = 0
        #    per_count = math.ceil(mg._percent/100.0*len(c_df))
        #    print per_count

        #elif mg._progbar:
        #    bar = pyprind.ProgBar(len(c_df))


        # keep track of valid ids
        valid = []
        # iterate candidate set and process each row
        for idx, row in c_df.iterrows():

            #if mg._verbose:
            #    count += 1
            #    if count%per_count == 0:
            #        print str(mg._percent*count/per_count) + ' percentage done !!!'
            #elif mg._progbar:
            #    bar.update()

            # get the value of block attribute from ltuple
            l_val = l_df.ix[row[l_key], l_block_attr]
            r_val = r_df.ix[row[r_key], r_block_attr]
            if l_val != np.NaN and r_val != np.NaN:
                if l_val == r_val:
                    valid.append(True)
                else:
                    valid.append(False)
            else:
                valid.append(False)
	return valid

    def block_candset_skd(self, vtable, l_block_attr, r_block_attr):
        """
        Block candidate set (virtual MTable) based on l_block_attr, r_block_attr equivalence (similar to equi-join)

        Parameters
        ----------
        vtable : MTable
            Input candidate set
        l_block_attr, r_block_attr : string,
            attribute names in ltable, rtable

        Returns
        -------
        blocked_table : MTable
            Containing tuple pairs whose l_block_attr and r_block_attr values are same

        Notes
        -----
        Output MTable contains the following three attributes
            * _id
            * id column from ltable
            * id column from rtable

        Also, the properties of blocked table is updated with following key-value pairs
            * ltable - ref to ltable
            * rtable - ref to rtable
            * key
            * foreign_key_ltable - string, ltable's  id attribute name
            * foreign_key_rtable - string, rtable's id attribute name
        """
	start_time = time.time()
        # do integrity checks
        ltable = vtable.get_property('ltable')
        rtable = vtable.get_property('rtable')

        self.check_attrs(ltable, rtable, l_block_attr, r_block_attr, None, None)
        l_key = 'ltable.' + ltable.get_key()
        r_key = 'rtable.' + rtable.get_key()
	
	t000 = time.time()
	print "Time taken to do integrity checks:", (t000 - start_time)

        # convert to dataframes
        l_df = ltable.to_dataframe()
        r_df = rtable.to_dataframe()

	t001 = time.time()
	print "Time taken to convert tables A and B to data frames:", (t001 - t000)

        # set index for convenience
        l_df.set_index(ltable.get_key(), inplace=True)
        r_df.set_index(rtable.get_key(), inplace=True)

	t002 = time.time()
	print "Time taken to set indexes for tables A and B:", (t002 - t001)

	cpu_count = multiprocessing.cpu_count() 
        #pool = Pool(4)
	t00 = time.time()
        pool = mp.ProcessingPool(processes=cpu_count, maxtasksperchild=1)
	t01 = time.time()
	print "Time taken to initialize the pool of workers:", (t01 - t00)
	c_df = vtable.to_dataframe()
	t0 = time.time()
	print "Time taken to convert mtable to data frame:", (t0 - t01)
        c_splits = np.array_split(c_df, cpu_count)
	t1 = time.time()
	print "Time taken to split table C:", (t1 - t0)
        args_splits = [(c, l_df, r_df, l_key, r_key, l_block_attr, r_block_attr) for c in c_splits]
	t2 = time.time()
	print "Time taken to get args splits:", (t2 - t1)
        valid_splits = pool.map(self.get_valid_ids, args_splits)
	t3 = time.time()
	print "Time taken to get valid splits:", (t3 - t2)
        pool.close()
	t4 = time.time()
	print "Time taken to close    pool:", (t4 - t3)
        pool.join() 
	t5 = time.time()
	print "Time taken to join     pool:", (t5 - t4)
        #valid = pd.concat(valid_splits, ignore_index=True)
	#valid = list(chain(valid_splits))
	valid = sum(valid_splits, [])
	t6 = time.time()
	print "Time taken to combine valid splits:", (t6 - t5)
 
        # should be modified
        if len(vtable) > 0:
            out_table = MTable(vtable[valid], key=vtable.get_key())
        else:
            out_table = MTable(columns=vtable.columns, key=vtable.get_key())
	t7 = time.time()
	print "Time taken to create mtable from data frame:", (t7 - t6)
        out_table.set_property('ltable', ltable)
        out_table.set_property('rtable', rtable)
        out_table.set_property('foreign_key_ltable', 'ltable.'+ltable.get_key())
        out_table.set_property('foreign_key_rtable', 'rtable.'+rtable.get_key())
	end_time = time.time()
	print "Time taken to set properties of mtable:", (end_time - t7)
	print "Time taken to block candset:", (end_time - start_time)
        return out_table
