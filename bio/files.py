from functools import reduce
import pandas as pd
import requests
import xport
from bio.columns import input_col_map, output_col_map, all_cols

NHANES_BASE_URL = 'https://wwwn.cdc.gov/Nchs/Nhanes/'
NHANES_SUFFIX = '.XPT'

input_files = {
    '1999-2000': [
	'LAB25',
	'DEMO',
	'LAB13',
	'LAB18',
	'LAB06',
	'LAB13AM',
    ],
    '2001-2002': [
        'L25_B',
        'L25_2_B',
        'DEMO_B',
        'L40_B',
        'L13AM_B',
        'L13_B',
        'L13_2_B',
        'L40_2_B',
        'L06_2_B',
        'L40FE_B',
    ],
    '2003-2004': [
        'L25_C',
        'DEMO_C',
        'L40_C',
        'L13_C',
        'L06COT_C',
        'L06BMT_C',
        'L06TFR_C',
        'L13AM_C',
        'L40FE_C',
    ],
    '2005-2006': [
        'CBC_D',
        'DEMO_D',
        'HDL_D',
        'BIOPRO_D',
        'FERTIN_D',
        'FETIB_D',
        'TCHOL_D',
    ],
    '2007-2008': [
        'CBC_E',
        'DEMO_E',
        'BIOPRO_E',
        'HDL_E',
        'FERTIN_E',
        'TCHOL_E',
    ],
    '2009-2010': [
        'CBC_F',
        'BIOPRO_F',
        'DEMO_F',
        'HDL_F',
        'FERTIN_F',
        'TCHOL_F',
    ],
    '2011-2012': [
        'CBC_G',
        'DEMO_G',
        'HDL_G',
        'BIOPRO_G',
        'TCHOL_G',
    ],
    '2013-2014': [
        'CBC_H',
        'DEMO_H',
        'HDL_H',
        'BIOPRO_H',
        'TCHOL_H',
        'TRIGLY_H',
    ],
    '2015-2016': [
        'CBC_I',
        'DEMO_I',
        'HDL_I',
        'BIOPRO_I',
        'TCHOL_I',
    ]
}

def get_fname(fname):
    return f'{fname}{NHANES_SUFFIX}'

def download(datadir, input_files=input_files):
    for (year, files) in input_files.items():
        for fname in files:
            fname = get_fname(fname)
            ofname = datadir / fname
            if ofname.exists():
                print(f'Skipping {ofname} (file exists)')
            else:
                url = f'{NHANES_BASE_URL}{year}/{fname}'
                print(f'Downloading {url} to {ofname}')
                r = requests.get(url, stream=True)
                if r.status_code == 200:
                    with open(ofname, 'wb') as f:
                        f.write(r.content)
                else:
                    raise FileNotFoundError(f'{url}')
    print('Done')

def get_df(datadir, fname, key):
    fname = get_fname(fname)
    df = xport.to_dataframe(open(datadir / fname, 'rb'))
    df.set_index(key, inplace=True)
    df.drop(df.columns.difference(all_cols), axis=1, inplace=True)
    df.rename(columns={**input_col_map, **output_col_map}, inplace=True)
    return df

def join_input(datadir, year, key='SEQN'):
    dfs = [get_df(datadir, fname, key) for fname in input_files[year]]
    df = pd.concat(dfs, ignore_index=True, sort=False)
    df = reduce(lambda x, y: x.merge(y,
                                     left_index=True,
                                     right_index=True,
                                     how='outer',
                                     suffixes=('', '_y')),
				dfs)

    year = int(year.split('-')[0])
    df['YEAR'] = [year] * len(df)
    return df

def join_all(datadir):
    dfs = []
    included_markers = input_col_map.values()
    for year in sorted(input_files.keys(), reverse=True):
        df = join_input(datadir, year)
        dfs.append(df)
        print(f'Checking for missing columns in {year}... ', end='')
        diff = df.columns.symmetric_difference(included_markers).drop('YEAR')
        if len(diff) == 0:
            print('OK')
            continue
        print('\nMissing columns: {}'.format(', '.join([f'"{d}"' for d in diff])))
    print('Done')
    return pd.concat(dfs, ignore_index=True, sort=False)
