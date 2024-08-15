import pandas as pd
import numpy as np


ENERGY_SPAIN = '/Users/ishanshastri/Desktop/nus/CPMF/energy_datasets/spain_energy_weather/energy_dataset.csv'

def get_raw_dataframe(filepath=ENERGY_SPAIN, print_desc=True, dropna=True, drop_cols=list()):

    df_energy = pd.read_csv(filepath)#.drop(columns=drop_cols)#, axis=1)#.dropna()

    # Remove all-0 columns
    df_energy = df_energy.replace(0, np.nan).dropna(axis=1,how="all").dropna() \
        if dropna else \
        df_energy.replace(0, np.nan).dropna(axis=1,how="all")
    
    df_energy['time'] = df_energy['time'].apply(pd.to_datetime)
    df_energy.set_index('time')

    
    if print_desc:
        df_energy.describe()

    return df_energy

def get_gluonts_dataset(filepath=ENERGY_SPAIN, time_col='time', tgt_col='price actual', fut_length=14, print_desc=True):
    # from gluonts.dataset.pandas import PandasDataset, infer_freq
    # import gluonts.dataset as wtf
    from gluonts.dataset.pandas import PandasDataset

    raw_df = get_raw_dataframe(filepath=filepath, print_desc=print_desc)
    # raw_df.set_index('time', drop=True, inplace=True)
    raw_df.set_index('time', inplace=True)
    # raw_df.reset_index(drop=True, inplace=True)
    # raw_df[time_col] = pd.to_datetime(raw_df[time_col], utc=True)
    #raw_df.index = pd.to_datetime(raw_df.index)
    # raw_df.index = pd.to_datetime(raw_df[time_col], utc=False)
    # raw_df.index = pd.to_datetime(raw_df[time_col]).to_timestamp()
    # f = pd.infer_freq(raw_df.index)#, utc=Tr)
    # # f = 
    # s = infer_freq(raw_df.index)
    raw_df.rename(columns=
                    {
                        time_col: "timestamp", 
                        tgt_col:"target",
                    }, 
                    inplace=True)
    ds = PandasDataset(raw_df, 
                       timestamp='timestamp', 
                       freq='H', 
                       target='target',
                       past_feat_dynamic_real=list(str(raw_df.drop('target', axis=1).columns)),
                       future_length=fut_length)#, freq='H')
    #TODO make this more dynamic. so you dont have to hardocde the damn frequence. 
    return ds

# ds = get_gluonts_dataset()

def proc_dataframe_dates(df, set_time_features=False, drop_cols=list()):#, start='2015-01-01 00:00:00+01:00', end='2018-12-29 00:00:00+01:00'):
    # df = df[start: end] if isinstance(start, str) else df.iloc[start:end]
    
    df['year'] = df.time.apply(lambda x : x.year / pd.Timestamp.now().year)
    df['week'] = df.time.apply(lambda x : x.weekofyear / 52)
    df['hour'] = df.time.apply(lambda x : x.hour)
    
    if set_time_features:
        df['time_features'] = df.apply(lambda x : [x.year, x.week, x.hour], axis=1)#df[['year', 'week', 'hour']].apply()
#    df['time_features'] = df[['year', 'week', 'hour']].apply(lambda x : [x.year, x.week, x.hour], axis=1)#df[['year', 'week', 'hour']].apply()

    df = df.drop(['time'], axis=1)

    df = df.drop(columns=drop_cols)

    return df




# def get_train_test_split(df, train_pct=0.8, test_pct=)
    
# def get_tensor_rep(df)