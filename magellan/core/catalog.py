import logging
import logging.config

import numpy as np
import pandas as pd

import magellan.utils.hfuncs as helper

# import magellan.utils.metadata

logger = logging.getLogger(__name__)



class Singleton:
    """
    A non-thread-safe helper class to ease implementing singletons.
    This should be used as a decorator -- not a metaclass -- to the
    class that should be a singleton.
    The decorated class can define one `__init__` function that
    takes only the `self` argument. Other than that, there are
    no restrictions that apply to the decorated class.
    To get the singleton instance, use the `Instance` method. Trying
    to use `__call__` will result in a `TypeError` being raised.
    Limitations: The decorated class cannot be inherited from.
    """

    def __init__(self, decorated):
        self._decorated = decorated

    def Instance(self):
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.
        """
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated()
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `Instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)


@Singleton
class Catalog(object):
    """
    Class to store and retrieve catalog information
    """

    def __init__(self):
        self.properties_catalog = {}

    def init_properties_for_id(self, obj_id):
        self.properties_catalog[obj_id] = {}
        return True

    def init_properties(self, df):
        df_id = id(df)
        self.init_properties_for_id(df_id)

    def get_property_for_id(self, obj_id, name):
        d = self.properties_catalog[obj_id]
        return d[name]

    def get_property(self, df, name):
        df_id = id(df)
        return self.get_property_for_id(df_id, name)

    def set_property_for_id(self, obj_id, name, value):
        d = self.properties_catalog[obj_id]
        d[name] = value
        self.properties_catalog[obj_id] = d
        return True

    def set_property(self, df, name, value):
        df_id = id(df)
        return self.set_property_for_id(df_id, name, value)

    def get_all_properties_for_id(self, obj_id):
        d = self.properties_catalog[obj_id]
        return d

    def get_all_properties(self, df):
        df_id = id(df)
        return self.get_all_properties_for_id(df_id)

    def del_property_for_id(self, obj_id, name):
        d = self.properties_catalog[obj_id]
        del d[name]
        self.properties_catalog[obj_id] = d
        return True

    def del_property(self, df, name):
        df_id = id(df)
        return self.del_property_for_id(df_id, name)

    def del_all_properties_for_id(self, obj_id):
        del self.properties_catalog[obj_id]
        return True

    def del_all_properties(self, df):
        df_id = id(df)
        return self.del_all_properties_for_id(df_id)

    def get_catalog(self):
        return self.properties_catalog

    def del_catalog(self):
        self.properties_catalog = {}
        return True

    def get_catalog_len(self):
        return len(self.properties_catalog)

    def is_catalog_empty(self):
        return self.properties_catalog > 0

    def is_dfinfo_present(self, df):
        return id(df) in self.properties_catalog

    def is_property_present_for_id(self, obj_id, name):
        d = self.properties_catalog[obj_id]
        return name in d

    def is_property_present_for_df(self, df, name):
        df_id = id(df)
        return self.is_property_present_for_id(df_id, name)


def get_property(df, name):
    """
    Get property for a dataframe

    Args:
        df (pandas dataframe): Dataframe for which the property should be retrieved
        name (str): Name of the property that should be retrieved

    Returns:
        Property value (pandas object) for the given property name

    Raises:
        AttributeError: If the input dataframe in null
        KeyError: If the dataframe is not present in the catalog, or the requested property is not
            present in the catalog

    """

    catalog = Catalog.Instance()

    if df is None:
        raise AttributeError('Input dataframe cannot be null')

    if not catalog.is_dfinfo_present(df):
        raise KeyError('Dataframe information is not present in the catalog')

    if not catalog.is_property_present_for_df(df, name):
        raise KeyError('Requested metadata (' + name + ') for the given dataframe is not present in the catalog')

    return catalog.get_property(df, name)


def set_property(df, name, value):
    """
    Set property for a dataframe

    Args:
        df (pandas dataframe): Dataframe for which the property has to be set
        name (str): Property name
        value (pandas object): Property value

    Returns:
        status (bool). Returns True if the property was set successfully

    Raises:
        AttributeError: If the input dataframe is null

    """
    catalog = Catalog.Instance()

    if df is None:
        raise AttributeError('Input dataframe cannot be null')

    if not catalog.is_dfinfo_present(df):
        catalog.init_properties(df)

    catalog.set_property(df, name, value)


def get_all_properties_for_id(obj_id):
    catalog = Catalog.Instance()
    if obj_id not in catalog.get_catalog():
        raise KeyError('Object not in the catalog')
    return catalog.get_all_properties_for_id(obj_id)


def get_all_properties(df):
    """
    Get all the properties for a dataframe

    Args:
        df (pandas dataframe): Dataframe for which the properties must be retrieved

    Returns:
        Property dictionary (dict). The keys are property names (str) and the values are property values (pandas object)

    Raises:
        AttributeError: If the input dataframe is null
        KeyError: If the information about the input dataframe is not present in the catalog

    """
    catalog = Catalog.Instance()

    if df is None:
        raise AttributeError('Input dataframe cannot be null')

    if not catalog.is_dfinfo_present(df):
        raise KeyError('Dataframe information is not present in the catalog')

    return catalog.get_all_properties(df)


def del_property(df, name):
    """
    Delete a property from the catalog

    Args:
        df (pandas dataframe): Input dataframe for which a property must be deleted
        name (str): Property name

    Returns:
        status (bool). Returns True if the deletion was successful

    Raises:
        AttributeError: If the input dataframe is null
        KeyError: If the Dataframe info. is not present or the given property is not present for that dataframe in the
            catalog
    """
    catalog = Catalog.Instance()
    if df is None:
        raise AttributeError('Input Dataframe cannot be null')

    if not catalog.is_dfinfo_present(df):
        raise KeyError('Dataframe information is not present in the catalog')

    if not catalog.is_property_present_for_df(df, name):
        raise KeyError('Requested metadata (' + name + ') for the given dataframe is not present in the catalog')

    return catalog.del_property(df, name)


def del_all_properties(df):
    """
    Delete all properties for a dataframe

    Args:
        df (pandas dataframe): Input dataframe for which all the properties must be deleted.

    Returns:
        status (bool). Returns True if the deletion was successful

    Raises:
        AttributeError: If the input dataframe is null
        KeyError: If the dataframe information is not present in the catalog
    """
    catalog = Catalog.Instance()
    if df is None:
        raise AttributeError('Input Dataframe cannot be null')

    if not catalog.is_dfinfo_present(df):
        raise KeyError('Dataframe information is not present in the catalog')

    return catalog.del_all_properties(df)


def get_catalog():
    """
    Get Catalog information.


    Returns:
        Catalog information in a dictionary format.

    """
    catalog = Catalog.Instance()
    return catalog.get_catalog()


def del_catalog():
    """
    Delete catalog information

    Returns:
        status (bool). Returns True if the deletion was successful.
    """
    catalog = Catalog.Instance()
    return catalog.del_catalog()


def is_catalog_empty():
    """
    Check if the catalog is empty

    Returns:
        result (bool). Returns True if the catalog is empty, else returns False.

    """
    catalog = Catalog.Instance()
    return catalog.is_catalog_empty()


def is_dfinfo_present(df):
    """
    Check if the dataframe information is present in the catalog

    Args:
        df (pandas dataframe): Input dataframe

    Returns:
        result (bool). Returns True if the dataframe information is present in the catalog, else returns False

    Raises:
        AttributeError: If the input dataframe is null

    """
    catalog = Catalog.Instance()
    if df is None:
        raise AttributeError('Input Dataframe cannot be null')

    return catalog.is_dfinfo_present(df)


def is_property_present_for_df(df, name):
    """
    Check if the property is present for the dataframe

    Args:
        df (pandas dataframe): Input dataframe
        name (str): Property name

    Returns:
        result (bool). Returns True if the property is present for the input dataframe

    Raises:
        AttributeError: If the input dataframe is null
        KeyError: If the dataframe is not present in the catalog

    """
    catalog = Catalog.Instance()
    if df is None:
        raise AttributeError('Input Dataframe cannot be null')

    if catalog.is_dfinfo_present(df) is False:
        raise KeyError('Dataframe information is not present in the catalog')

    return catalog.is_property_present_for_df(df, name)


def get_catalog_len():
    """
    Get the number of entries in the catalog

    Returns:
        length (int) of the catalog

    """
    catalog = Catalog.Instance()
    return catalog.get_catalog_len()


def set_properties(df, prop_dict, replace=True):
    """
    Set properties for a dataframe in the catalog
    Args:
        df (pandas dataframe): Input dataframe
        prop_dict (dict): Property dictionary with keys as property names and values as python objects
        replace (bool): Flag to indicate whether the input properties can replace the properties in the catalog

    Returns:
        status (bool). Returns True if the setting of properties was successful

    Notes:
        The function is intended to set all the properties in the catalog with the given property dictionary.
          The replace flag is just a check where the properties will be not be disturbed if they exist already in the
          catalog

    """
    catalog = Catalog.Instance()
    if catalog.is_dfinfo_present(df):
        if replace is False:
            logger.warning('Properties already exists for df (' + id(df) + ' ). Not replacing it')
            return False

    catalog.del_all_properties(df)
    for k, v in prop_dict.iteritems():
        catalog.set_property(df, k, v)
    return True


def copy_properties(src, tar, replace=True):
    """
    Copy properties from one dataframe to another
    Args:
        src (pandas dataframe): Dataframe from which the properties to be copied from
        tar (pandas dataframe): Dataframe to which the properties to be copied
        replace (bool): Flag to indicate whether the source properties can replace the tart properties

    Returns:
        status (bool). Returns True if the copying was successful

    Notes:
        This function internally calls set_properties and get_all_properties


    """
    # copy catalog information from src to tar
    catalog = Catalog.Instance()
    metadata = catalog.get_all_properties(src)
    return set_properties(tar, metadata, replace)


# key related methods
def get_key(df):
    """
    Get the key attribute for a dataframe

    Args:
        df (pandas dataframe): Dataframe for which the key must be retrieved

    Returns:
        key (str)

    """
    return get_property(df, 'key')


def set_key(df, key):
    """
    Set the key attribute for a dataframe

    Args:
        df (pandas dataframe): Dataframe for which the key must be set
        key (str): Key attribute in the dataframe

    Returns:
        status (bool). Returns True if the key attribute was set successfully, else returns False

    """
    if is_key_attribute(df, key) is False:
        logger.warning('Attribute (' + key + ') does not qualify to be a key')
        return False
    else:
        return set_property(df, 'key', key)


def get_fk_ltable(df):
    return get_property(df, 'fk_ltable')


def get_fk_rtable(df):
    return get_property(df, 'fk_rtable')


def set_fk_ltable(df, fk_ltable):
    """
    Set foreign key attribute to the left table
    Args:
        df (pandas dataframe): Dataframe for which the foreign key must be set
        fk_ltable (str): Foreign key attribute in the dataframe

    Returns:
        status (bool). Returns True if the ltable foreign key attribute was set successfully, else returns False
    """
    return set_property(df, 'fk_ltable', fk_ltable)


def set_fk_rtable(df, fk_rtable):
    """
    Set foreign key attribute to the right table
    Args:
        df (pandas dataframe): Dataframe for which the foreign key must be set
        fk_rtable (str): Foreign key attribute in the dataframe

    Returns:
        status (bool). Returns True if the rtable foreign key attribute was set successfully, else returns False
    """
    return set_property(df, 'fk_rtable', fk_rtable)


def get_reqd_metadata_from_catalog(df, reqd_metadata):
    """
    Get a list of properties from the catalog

    Args:
        df (pandas dataframe): Dataframe for which the properties must be retrieved
        reqd_metadata (list): List of properties to be retrieved

    Returns:
        properties (dict)

    Notes:
        This is an internal helper function.


    """
    if not isinstance(reqd_metadata, list):
        reqd_metadata = [reqd_metadata]

    metadata = {}
    d = get_all_properties(df)
    for m in reqd_metadata:
        if m in d:
            metadata[m] = d[m]
    return metadata


def update_reqd_metadata_with_kwargs(metadata, kwargs_dict, reqd_metadata):
    """
    Update the metadata with input args

    Args:
        metadata (dict): Properties dictonary
        kwargs_dict (dict): Input key-value args
        reqd_metadata (list): List of properties to be updated.

    Returns:
        updated properties (dict)

    Notes:
        This is an internal helper function.


    """
    if not isinstance(reqd_metadata, list):
        reqd_metadata = [reqd_metadata]

    for m in reqd_metadata:
        if m in kwargs_dict:
            metadata[m] = kwargs_dict[m]
    return metadata


def get_diff_with_reqd_metadata(metadata, reqd_metadata):
    """
    Find what metadata is missing from the required list

    Args:
        metadata (dict): Property dictionary
        reqd_metadata (list): List of properties

    Returns:
        diff list (list) of properties between the property dictionary and the properties list

    Notes:
        This is an internal helper function
    """
    k = metadata.keys()
    if not isinstance(reqd_metadata, list):
        reqd_metadata = [reqd_metadata]
    d = set(reqd_metadata).difference(k)
    return d


def is_all_reqd_metadata_present(metadata, reqd_metadata):
    """
    Check if all the required metadata are present

    Args:
        metadata (dict): Property dictionary
        reqd_metadata (list): List of properties

    Returns:
        result (bool). Returns True if all the required metadata is present, else returns False

    Notes:
        This is an internal helper function

    """
    d = get_diff_with_reqd_metadata(metadata, reqd_metadata)
    if len(d) == 0:
        return True
    else:
        return False


def check_fk_constraint(df_foreign, attr_foreign, df_base, attr_base):
    """
    Check if the foreign key is a primary key
    Args:
        df_foreign (pandas dataframe): Foreign dataframe
        attr_foreign (str): Attribute in the foreign dataframe
        df_base (pandas dataframe): Base dataframe
        attr_base (str): Attribute in the base dataframe
    Returns:
        result (bool). Returns True if the foreign key contraint is satisfied, else returns False
    Notes:
        This is an internal helper function
    """
    if isinstance(attr_base, basestring) is False:
        return False
    if helper.check_attrs_present(df_base, attr_base) is False:
        return False
    t = df_base[df_base[attr_base].isin(pd.unique(df_foreign[attr_foreign]))]
    return is_key_attribute(t, attr_base)


def does_contain_rows(df):
    """
    Check if the dataframe is non-empty

    Args:
        df (pandas dataframe): Input dataframe

    Returns:
        result (bool). Returns True, if the length of the dataframe is greater than 0, else returns False

    Notes:
        This is an internal helper function

    """
    return len(df) > 0


def is_attr_unique(df, attr):
    """
    Check if the attribute is unique in a dataframe

    Args:
        df (pandas dataframe): Input dataframe
        attr (str): Attribute in the pandas dataframe

    Returns:
        result (bool). Returns True, if the attribute contains unique values, else returns False

    Notes:
        This is an internal helper function

    """
    uniq_flag = len(np.unique(df[attr])) == len(df)
    if not uniq_flag:
        return False
    else:
        return True


def does_contain_missing_vals(df, attr):
    """
    Check if the attribute contains missing values in the dataframe

    Args:
        df (pandas dataframe): Input dataframe
        attr (str): Attribute in the pandas dataframe

    Returns:
        result (bool). Returns True, if the attribute contains missing values, else returns False

    Notes:
        This is an internal helper function

    """
    nan_flag = sum(df[attr].isnull()) == 0
    if not nan_flag:
        return False
    else:
        return True


def is_key_attribute(df, attr, verbose=False):
    """
    Check if an attribute is a key attribute
    Args:
        df (pandas dataframe): Input dataframe
        attr (str): Attribute in the pandas dataframe
        verbose (bool):  Flag to indicate whether warnings should be printed out

    Returns:
        result (bool). Returns True, if the attribute is a key attribute (unique and without missing values), else
         returns False

    """
    # check if the length is > 0
    if len(df) > 0:
        # check for uniqueness
        uniq_flag = len(np.unique(df[attr])) == len(df)
        if not uniq_flag:
            if verbose:
                logger.warning('Attribute ' + attr + ' does not contain unique values')
            return False

        # check if there are missing or null values
        nan_flag = sum(df[attr].isnull()) == 0
        if not nan_flag:
            if verbose:
                logger.warning('Attribute ' + attr + ' contains missing values')
            return False
        return uniq_flag and nan_flag
    else:
        return True


def show_properties(df):
    if is_dfinfo_present(df) == False:
        logger.warning('Dataframe information is not present in the catalog')
        return
    metadata = get_all_properties(df)
    print 'id: ' + str(id(df))
    for prop in metadata.iterkeys():
        value = metadata[prop]
        if isinstance(value, basestring):
            print prop + ": " + value
        else:
            print prop + "(obj.id): " + str(id(value))


def show_properties_for_id(obj_id):
    metadata = get_all_properties_for_id(obj_id)
    print 'id: ' + str(obj_id)
    for prop in metadata.iterkeys():
        value = metadata[prop]
        if isinstance(value, basestring):
            print prop + ": " + value
        else:
            print prop + "(obj.id): " + str(id(value))


def set_candset_properties(candset, key, fk_ltable, fk_rtable, ltable, rtable):
    set_property(candset, 'key', key)
    set_fk_ltable(candset, fk_ltable)
    set_fk_rtable(candset, fk_rtable)
    set_property(candset, 'ltable', ltable)
    set_property(candset, 'rtable', rtable)


def validate_metadata_for_table(table, key, out_str, lgr, verbose):
    helper.log_info(lgr, 'Validating ' + out_str + ' key: ' + str(key), verbose)
    assert isinstance(key, basestring) is True, 'Key attribute must be a string.'
    assert helper.check_attrs_present(table, key) is True, 'Key attribute is not present in the ' + out_str + ' table'
    assert is_key_attribute(table, key, verbose) == True, 'Attribute ' + str(key) + ' in the ' + out_str + ' table ' \
                                                                                                           'does not qualify to be the key'
    helper.log_info(lgr, '..... Done', verbose)


def validate_metadata_for_candset(candset, key, fk_ltable, fk_rtable, ltable, rtable, l_key, r_key,
                                  lgr, verbose):
    validate_metadata_for_table(candset, key, 'cand.set', lgr, verbose)

    helper.log_info(lgr, 'Validating foreign key constraint for left table', verbose)
    assert check_fk_constraint(candset, fk_ltable, ltable, l_key) == True, 'Cand.set does not satisfy foreign key ' \
                                                                           'constraint with the left table'
    helper.log_info(lgr, '..... Done', verbose)
    helper.log_info(lgr, 'Validating foreign key constraint for right table', verbose)
    assert check_fk_constraint(candset, fk_rtable, rtable, r_key) == True, 'Cand.set does not satisfy foreign key ' \
                                                                           'constraint with the right table'
    helper.log_info(lgr, '..... Done', verbose)

    return True

def get_keys_for_ltable_rtable(ltable, rtable, lgr, verbose):
    helper.log_info(lgr, 'Required metadata: ltable key, rtable key', verbose)
    helper.log_info(lgr, 'Getting metadata from the catalog', verbose)
    l_key = get_key(ltable)
    r_key = get_key(rtable)
    helper.log_info(lgr, '..... Done', verbose)
    return l_key, r_key

def get_metadata_for_candset(candset, lgr, verbose):
    helper.log_info(lgr, 'Getting metadata from the catalog', verbose)
    key = get_key(candset)
    fk_ltable = get_fk_ltable(candset)
    fk_rtable = get_fk_rtable(candset)
    ltable = get_property(candset, 'ltable')
    rtable = get_property(candset, 'rtable')
    l_key = get_key(ltable)
    r_key = get_key(rtable)
    helper.log_info(lgr, '..... Done', verbose)
    return key, fk_ltable, fk_rtable, ltable, rtable, l_key, r_key
