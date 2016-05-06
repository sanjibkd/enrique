import logging
import os



logger = logging.getLogger(__name__)


def are_all_attributes_present(df, col_names, verbose=False):
    df_columns_names = list(df.columns)
    for c in col_names:
        if c not in df_columns_names:
            if verbose:
                logger.warning('Column name (' +c+ ') is not present in dataframe')
            return False
    return True


def get_install_path():
    plist = install_path.split(os.sep)
    return os.sep.join(plist[0:len(plist)-1])


def add_key_column(table, key):
    table.insert(0, key, range(0, len(table)))
    return table


def get_name_for_key(columns):
    k = '_id'
    i = 0
    # try attribute name of the form "_id", "_id0", "_id1", ... and
    # return the first available name
    while True:
        if k not in columns:
            break
        else:
            k = '_id' + str(i)
        i += 1
    return k

def log_info(lgr, s, verbose):
    if verbose:
        lgr.info(s)

def get_prop_from_dict(d, name):
    return d[name]



def check_attrs_present(table, attrs):
    if isinstance(attrs, list) is False:
        attrs = [attrs]
    status = are_all_attributes_present(table, attrs, verbose=True)
    return status


# remove non-ascii characters from string
def remove_non_ascii(s):
    s = ''.join(i for i in s if ord(i) < 128)
    s = str(s)
    return str.strip(s)

# find the list difference
def diff(a, b):
  b = set(b)
  return [aa for aa in a if aa not in b]

