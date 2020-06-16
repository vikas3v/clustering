import gzip
import shutil
import pandas as pd
# import srtm

class utils:
    def save_csv(self, dataframe, name):
        dataframe.to_csv('results/' + name + '.csv')

    def extract_csv(self, file_name, target_name):
        with gzip.open(file_name, 'rb') as f_in:
            with open('data/' + target_name, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    def read_data(self, dataframe):
        """
        Method to do initial cleanup and setting datatypes
        """
        dataframe['ts_local'] = pd.to_datetime(dataframe['ts_local'], format='%Y-%m-%d %H:%M:%S')
        dataframe = dataframe[['ts_local','lat6','lon6','tide5','month','hour_of_week','dev_count','rq_count','province']]
        return dataframe

    def get_elevation(self, latitude, longitude):
        # elevation_data = srtm.get_data(local_cache_dir="tmp_cache")
        elevation = elevation_data.get_elevation(latitude, longitude)
        print('CGN Airport elevation (meters):', elevation)

        return elevation

if __name__ == '__main__':
    # File name is filename with .csv.gz as extension
    util = utils()
    util.extract_csv('../data/p1024_tide_wjava_ciesin_201912.csv.gz', 'p1024_tide_wjava_ciesin_201912.csv')