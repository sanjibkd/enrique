from nose.tools import *
import numpy as np
import time

from magellan.tests import mg


def test_ab_block_tables():
    A = mg.load_dataset('table_A')
    B = mg.load_dataset('table_B')
    ab = mg.AttrEquivalenceBlocker()
    C = ab.block_tables(A, B, 'zipcode', 'zipcode', 'zipcode', 'zipcode')
    s1 = sorted(['_id', 'ltable.ID', 'rtable.ID', 'ltable.zipcode', 'rtable.zipcode'])
    assert_equal(s1, sorted(C.columns))
    assert_equal(C.get_key(), '_id')
    assert_equal(C.get_property('foreign_key_ltable'), 'ltable.ID')
    assert_equal(C.get_property('foreign_key_rtable'), 'rtable.ID')
    k1 = np.array(C[['ltable.zipcode']])
    k2 = np.array(C[['rtable.zipcode']])
    assert_equal(all(k1 == k2), True)

def test_ab_block_candset():
    A = mg.load_dataset('table_A')
    B = mg.load_dataset('table_B')
    ab = mg.AttrEquivalenceBlocker()
    C = ab.block_tables(A, B, 'zipcode', 'zipcode', ['zipcode', 'birth_year'], ['zipcode', 'birth_year'])
    D = ab.block_candset(C, 'birth_year', 'birth_year')
    s1 = sorted(['_id', 'ltable.ID', 'rtable.ID', 'ltable.zipcode', 'ltable.birth_year', 'rtable.zipcode',
                 'rtable.birth_year'])
    assert_equal(s1, sorted(D.columns))
    assert_equal(D.get_key(), '_id')
    assert_equal(D.get_property('foreign_key_ltable'), 'ltable.ID')
    assert_equal(D.get_property('foreign_key_rtable'), 'rtable.ID')
    k1 = np.array(D[['ltable.birth_year']])
    k2 = np.array(D[['rtable.birth_year']])
    assert_equal(all(k1 == k2), True)

def test_ab_block_tuples():
    A = mg.load_dataset('table_A')
    B = mg.load_dataset('table_B')
    ab = mg.AttrEquivalenceBlocker()
    assert_equal(ab.block_tuples(A.ix[1], B.ix[2], 'zipcode', 'zipcode'), False)
    assert_equal(ab.block_tuples(A.ix[2], B.ix[2], 'zipcode', 'zipcode'), True)


def test_ab_block_tables_wi_no_tuples():
    A = mg.load_dataset('table_A')
    B = mg.load_dataset('table_B')
    ab = mg.AttrEquivalenceBlocker()
    C = ab.block_tables(A, B, 'name', 'name')
    assert_equal(len(C),  0)
    assert_equal(sorted(C.columns), sorted(['_id', 'ltable.ID', 'rtable.ID']))
    assert_equal(C.get_key(), '_id')
    assert_equal(C.get_property('foreign_key_ltable'), 'ltable.ID')
    assert_equal(C.get_property('foreign_key_rtable'), 'rtable.ID')


def test_ab_block_candset_wi_no_tuples():
    A = mg.load_dataset('table_A')
    B = mg.load_dataset('table_B')
    ab = mg.AttrEquivalenceBlocker()
    C = ab.block_tables(A, B, 'name', 'name')
    D = ab.block_candset(C, 'birth_year', 'birth_year')
    assert_equal(len(D),  0)
    assert_equal(sorted(D.columns), sorted(['_id', 'ltable.ID', 'rtable.ID']))
    assert_equal(D.get_key(), '_id')
    assert_equal(D.get_property('foreign_key_ltable'), 'ltable.ID')
    assert_equal(D.get_property('foreign_key_rtable'), 'rtable.ID')

def test_ab_block_tables_skd():
    start_time = time.time()
    A = mg.load_dataset('bikedekho_clean', 'ID')
    #A = mg.load_dataset('bowker', 'ID')
    a_load_time = time.time()
    print("Loading table A --- %s seconds ---" % (a_load_time - start_time))
    B = mg.load_dataset('bikewale_clean', 'ID')
    #B = mg.load_dataset('walmart', 'ID')
    b_load_time = time.time()
    print("Loading table B --- %s seconds ---" % (b_load_time - a_load_time))
    ab = mg.AttrEquivalenceBlocker()
    ab_time = time.time()
    print("Created an AE blocker --- %s seconds ---" % (ab_time - b_load_time))
    C = ab.block_tables(A, B, 'city_posted', 'city_posted', 'city_posted', 'city_posted')
    #C = ab.block_tables(A, B, 'pubYear', 'pubYear', 'pubYear', 'pubYear')
    c_time = time.time()
    print("Block tables --- %s seconds ---" % (c_time - ab_time))

    s1 = sorted(['_id', 'ltable.ID', 'rtable.ID', 'ltable.city_posted', 'rtable.city_posted'])
    #s1 = sorted(['_id', 'ltable.ID', 'rtable.ID', 'ltable.pubYear', 'rtable.pubYear'])
    assert_equal(s1, sorted(C.columns))
    assert_equal(C.get_key(), '_id')
    assert_equal(C.get_property('foreign_key_ltable'), 'ltable.ID')
    assert_equal(C.get_property('foreign_key_rtable'), 'rtable.ID')
    k1 = np.array(C[['ltable.city_posted']])
    #k1 = np.array(C[['ltable.pubYear']])
    k2 = np.array(C[['rtable.city_posted']])
    #k2 = np.array(C[['rtable.pubYear']])
    assert_equal(all(k1 == k2), True)
