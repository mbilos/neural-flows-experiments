import pandas as pd

from sklearn.model_selection import train_test_split
from pathlib import Path

from nfe.experiments.gru_ode_bayes.lib.data_utils import ITSDataset


DATA_DIR = Path('/opt/ml/input/data/training')
if DATA_DIR.exists():
    OU_FILE = DATA_DIR / '2dou.csv'
    MIMIC3_FILE = MIMIC4_FILE = DATA_DIR / 'full_dataset.csv'
else:
    OU_FILE = Path(__file__).parents[2] / 'data/2dou/2dou.csv'
    MIMIC3_FILE = Path(__file__).parents[2] / 'data/mimic3/mimic3_full_dataset.csv'
    MIMIC4_FILE = Path(__file__).parents[2] / 'data/mimic4/mimic4_full_dataset.csv'
    MIMIC3_FILE_LONG = Path(__file__).parents[2] / 'data/mimic3/mimic3_full_dataset_long.csv'
    MIMIC4_FILE_LONG = Path(__file__).parents[2] / 'data/mimic4/mimic4_full_dataset_long.csv'


def get_OU_data(t_val=4, max_val_samples=1):
    full_data = pd.read_csv(OU_FILE, index_col=0)

    val_options = {'T_val': t_val, 'max_val_samples': max_val_samples}
    train_idx, val_idx = train_test_split(full_data.index.unique(), test_size=0.2, random_state=432)

    train = ITSDataset(in_df=full_data.loc[train_idx].reset_index())
    val = ITSDataset(in_df=full_data.loc[val_idx].reset_index(), validation=True, val_options=val_options)
    return train, val, val


def get_MIMIC_data(name, t_val=2.160, max_val_samples=3, return_vc=False):
    if name == "mimic3":
        full_data = pd.read_csv(MIMIC3_FILE, index_col=0)
        full_data = full_data.reset_index()
        full_data = full_data.rename(columns={'HADM_ID':'ID', 'TIME_STAMP':'Time'})
    elif name == "mimic4":
        full_data = pd.read_csv(MIMIC4_FILE, index_col=0)
        full_data = full_data.reset_index()
        full_data = full_data.rename(columns={'hadm_id':'ID', 'time_stamp':'Time'})
    else:
        raise NotImplementedError()
    full_data = full_data.set_index('ID')
    full_data.loc[:, 'Time'] = full_data['Time'] / 1000

    value_cols = [c.startswith('Value') for c in full_data.columns]
    value_cols = full_data.iloc[:, value_cols]
    mask_cols = [('Mask' + x[5:]) for x in value_cols]

    for item in zip(value_cols, mask_cols):
        temp = full_data.loc[full_data[item[1]].astype('bool'), item[0]]
        full_data.loc[full_data[item[1]].astype('bool'), item[0]] = (temp - temp.mean()) / temp.std()
    full_data.dropna(inplace=True)

    # Remove outliers in mimic4
    sdevs = 5
    if name == "mimic4":
        for item, m in zip(value_cols, mask_cols):
            outlier_mask = ((full_data[item] < sdevs) & (full_data[item] > -sdevs))
            full_data[item].loc[~outlier_mask] = 0
            full_data[m].loc[~outlier_mask] = 0

        full_data = full_data.loc[full_data[mask_cols].sum(axis=1) > 0]

    val_options = {'T_val': t_val, 'max_val_samples': max_val_samples}
    train_idx, eval_idx = train_test_split(full_data.index.unique(), test_size=0.3, random_state=0)
    val_idx, test_idx = train_test_split(full_data.loc[eval_idx].index.unique(), test_size=0.5, random_state=0)

    train = ITSDataset(in_df=full_data.loc[train_idx].reset_index())
    val = ITSDataset(in_df=full_data.loc[val_idx].reset_index(), validation=True, val_options=val_options)
    test = ITSDataset(in_df=full_data.loc[test_idx].reset_index(), validation=True, val_options=val_options)

    if return_vc:
        return train, val, test, value_cols
    else:
        return train, val, test


def get_MIMIC_data_long(idx, value_cols, name, t_val=2.160, t_stop=3.600, max_val_samples=5):
    if name == "mimic3":
        full_data = pd.read_csv(MIMIC3_FILE_LONG, index_col=0)
        full_data = full_data.reset_index()
        full_data = full_data.rename(columns={'HADM_ID':'ID', 'TIME_STAMP':'Time'})
        full_data_norm = pd.read_csv(MIMIC3_FILE, index_col=0)
        full_data_norm = full_data_norm.reset_index()
        full_data_norm = full_data_norm.rename(columns={'hadm_id':'ID', 'time_stamp':'Time'})
    elif name == "mimic4":
        full_data = pd.read_csv(MIMIC4_FILE_LONG, index_col=0)
        full_data = full_data.reset_index()
        full_data = full_data.rename(columns={'hadm_id':'ID', 'time_stamp':'Time'})
        full_data_norm = pd.read_csv(MIMIC4_FILE, index_col=0)
        full_data_norm = full_data_norm.reset_index()
        full_data_norm = full_data_norm.rename(columns={'hadm_id':'ID', 'time_stamp':'Time'})
    else:
        raise NotImplementedError()

    mask_cols = [('Mask' + x[5:]) for x in value_cols]
    full_data = full_data[['ID', 'Time'] + value_cols.tolist() + mask_cols]
    full_data = full_data.set_index('ID')
    full_data.loc[:, 'Time'] = full_data['Time'] / 1000

    value_cols = [c.startswith('Value') for c in full_data.columns]
    value_cols = full_data.iloc[:, value_cols]

    for item in zip(value_cols, mask_cols):
        temp = full_data.loc[full_data[item[1]].astype('bool'), item[0]]
        temp_norm = full_data_norm.loc[full_data_norm[item[1]].astype('bool'), item[0]]
        full_data.loc[full_data[item[1]].astype('bool'), item[0]] = (temp - temp_norm.mean()) / temp_norm.std()
    full_data.dropna(inplace=True)

    sdevs = 5
    if name == "mimic4":
        for item, m in zip(value_cols, mask_cols):
            outlier_mask = ((full_data[item] < sdevs) & (full_data[item] > -sdevs))
            full_data[item].loc[~outlier_mask] = 0
            full_data[m].loc[~outlier_mask] = 0

        full_data = full_data.loc[full_data[mask_cols].sum(axis=1) > 0]

    val_options = {'T_val': t_val, 'max_val_samples': max_val_samples, 'T_stop': t_stop}
    test = ITSDataset(in_df=full_data.loc[idx].reset_index(), validation=True, val_options=val_options)
    return test
