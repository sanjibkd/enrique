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
    #A = mg.load_dataset('bikedekho_clean', 'ID')
    A = mg.load_dataset('bowker', 'ID')
    a_load_time = time.time()
    print("Loading table A --- %s seconds ---" % (a_load_time - start_time))
    #B = mg.load_dataset('bikewale_clean', 'ID')
    B = mg.load_dataset('walmart', 'ID')
    b_load_time = time.time()
    print("Loading table B --- %s seconds ---" % (b_load_time - a_load_time))
    ab = mg.AttrEquivalenceBlocker()
    ab_time = time.time()
    print("Created an AE blocker --- %s seconds ---" % (ab_time - b_load_time))
    #C = ab.block_tables_skd(A, B, 'city_posted', 'city_posted', 'city_posted', 'city_posted')
    C = ab.block_tables_skd(A, B, 'pubYear', 'pubYear', 'pubYear', 'pubYear')
    #C = ab.block_tables_skd(A, B, 'isbn', 'isbn', 'isbn', 'isbn')
    print("Size of candset C: %d" % (len(C)))
    c_time = time.time()
    print("Block tables --- %s seconds ---" % (c_time - ab_time))

    #s1 = sorted(['_id', 'ltable.ID', 'rtable.ID', 'ltable.city_posted', 'rtable.city_posted'])
    s1 = sorted(['_id', 'ltable.ID', 'rtable.ID', 'ltable.pubYear', 'rtable.pubYear'])
    #s1 = sorted(['_id', 'ltable.ID', 'rtable.ID', 'ltable.isbn', 'rtable.isbn'])
    assert_equal(s1, sorted(C.columns))
    assert_equal(C.get_key(), '_id')
    assert_equal(C.get_property('foreign_key_ltable'), 'ltable.ID')
    assert_equal(C.get_property('foreign_key_rtable'), 'rtable.ID')
    #k1 = np.array(C[['ltable.city_posted']])
    k1 = np.array(C[['ltable.pubYear']])
    #k1 = np.array(C[['ltable.isbn']])
    #k2 = np.array(C[['rtable.city_posted']])
    k2 = np.array(C[['rtable.pubYear']])
    #k2 = np.array(C[['rtable.isbn']])
    assert_equal(all(k1 == k2), True)

def test_ab_block_candset_skd():
    #A = mg.load_dataset('table_A')
    A = mg.load_dataset('bikedekho_clean', 'ID')
    #B = mg.load_dataset('table_B')
    B = mg.load_dataset('bikewale_clean', 'ID')
    ab = mg.AttrEquivalenceBlocker()
    #C = ab.block_tables(A, B, 'zipcode', 'zipcode', ['zipcode', 'birth_year'], ['zipcode', 'birth_year'])
    C = ab.block_tables_skd(A, B, 'city_posted', 'city_posted',
	['bike_name', 'city_posted', 'km_driven', 'price', 'color', 'model_year'],
	['bike_name', 'city_posted', 'km_driven', 'price', 'color', 'model_year'])
    print "Size of C: ", len(C)
    #D = ab.block_candset_skd(C, 'birth_year', 'birth_year')
    D = ab.block_candset_skd(C, 'model_year', 'model_year')
    print "Size of D: ", len(D)
    #s1 = sorted(['_id', 'ltable.ID', 'rtable.ID', 'ltable.zipcode', 'ltable.birth_year', 'rtable.zipcode',
    #             'rtable.birth_year'])
    s1 = sorted(['_id', 'ltable.ID', 'rtable.ID', 'ltable.bike_name', 'ltable.city_posted',
	'ltable.km_driven', 'ltable.price', 'ltable.color', 'ltable.model_year',
	'rtable.bike_name', 'rtable.city_posted', 'rtable.km_driven', 'rtable.price',
	'rtable.color', 'rtable.model_year'])
    assert_equal(s1, sorted(D.columns))
    assert_equal(D.get_key(), '_id')
    assert_equal(D.get_property('foreign_key_ltable'), 'ltable.ID')
    assert_equal(D.get_property('foreign_key_rtable'), 'rtable.ID')
    #k1 = np.array(D[['ltable.birth_year']])
    k1 = np.array(D[['ltable.model_year']])
    #k2 = np.array(D[['rtable.birth_year']])
    k2 = np.array(D[['rtable.model_year']])
    assert_equal(all(k1 == k2), True)
