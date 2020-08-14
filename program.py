from analyses import initialAnalysis, clustering
from utils import common
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
# import seaborn as sns
# import srtm

class program:

    def __init__(self, dataset = None):
        self.dataset = dataset
        self.util = common.utils()
        self.analysis = initialAnalysis.commonAnalysis()

    def timeseries_all_days(self):
        df_ts_all = self.analysis.group_by_day_count(self.dataset)
        self.util.save_csv(df_ts_all, 'df_ts_all_test2')

    def aggregate_dataset(self):
        df_hour = self.analysis.aggregate_by_num_hours(self.dataset, 4)
        self.util.save_csv(df_hour, 'df_4hourbins')

    def filter_for_jakarta(self):
        # Filtering provinces near Jakarta
        provinces = ['JK', 'BT', 'JB']
        df_provinces = self.analysis.filter_by_provinces(self.dataset, provinces)
        print(df_provinces.describe())

    def calculate_avg_std(self):
        df = self.dataset[['tide5', 'dev_count', 'lat6', 'lon6', 'rq_count', 'ts_local']]
        df_before = df[df['ts_local'] < '2020-1-1']
        df_after = df[df['ts_local'] >= '2020-1-3']
        df_before_calc = self.analysis.cal_basic_parameters_regionwise(df_before)
        df_after_calc = self.analysis.cal_basic_parameters_regionwise(df_after)

        df_combined = pd.merge(df_before_calc, df_after_calc, on = (['tide5', 'lat6', 'lon6']), how = 'inner', suffixes = ('_before', '_after'))
        # print df_combined.columns
        df_combined.columns = ['tide5', 'lat6', 'lon6', 'dev_mean_before', 'dev_max_before', 'dev_min_before', 'dev_std_before', 'dev_count_before',\
            'rq_mean_before', 'rq_max_before', 'rq_min_before', 'rq_std_before', 'rq_count_before',\
            'dev_mean_after', 'dev_max_after', 'dev_min_after', 'dev_std_after', 'dev_count_after',\
            'rq_mean_after', 'rq_max_after', 'rq_min_after', 'rq_std_after', 'rq_count_after']
        # print df_combined.columns
        print(df_combined.head())
        self.util.save_csv(df_combined, 'df_bef_after_tide5_std_avg_06_22')

        df_baseline = df_combined
        # print (df_baseline.shape)
        df_bl_filter = df_baseline[df_baseline['dev_count_before'] > 5]
        df_bl_filter = df_bl_filter[df_bl_filter['dev_count_after'] > 5]
        self.util.save_csv(df_combined, 'df_bef_after_tide5_std_avg_count_5_filter_06_22')

        return df_combined

    def baseline_comparison(self, baselinePath = 'results/df_bef_after_tide5_std_avg_min_max.csv'):
        df_baseline = pd.read_csv(baselinePath)
        print (df_baseline.shape)
        df_bl_filter = df_baseline[df_baseline['dev_count_before'] > 5]
        df_bl_filter = df_bl_filter[df_bl_filter['dev_count_after'] > 5]

        print (df_bl_filter.shape)
        # df_avg = df_baseline[['dev_mean_before','dev_mean_after']]
        x1 = df_baseline.dev_mean_before
        y1 =  df_baseline.dev_mean_after
        self.comparison_plot(x1, y1, 'dev_mean_bef_ater_filter02')

        x2 = df_baseline.dev_std_before
        y2 =  df_baseline.dev_std_after
        self.comparison_plot(x2, y2, 'dev_std_bef_ater_filter02')

    def baseline_comparison_excl_outlier(self, baselinePath = 'results/df_bef_after_tide5_std_avg_06_22.csv'):
        df_baseline = pd.read_csv(baselinePath)
        print (df_baseline.shape)
        df_bl_filter = df_baseline[df_baseline['dev_count_before'] > 5]
        df_bl_filter = df_bl_filter[df_bl_filter['dev_count_after'] > 5]
        print (df_bl_filter.shape)

        df_bl_filter = df_bl_filter[df_bl_filter['dev_mean_before'] < 100]

        print (df_bl_filter.shape)
        # df_avg = df_baseline[['dev_mean_before','dev_mean_after']]
        x1 = df_baseline.dev_mean_before
        y1 =  df_baseline.dev_mean_after
        self.comparison_plot(x1, y1, 'dev_mean_bef_ater_excl_outlier')

        x2 = df_baseline.dev_std_before
        y2 =  df_baseline.dev_std_after
        self.comparison_plot(x2, y2, 'dev_std_bef_ater_excl_outlier')

    def histogram_bef_aft_dev_counts(self, baselinePath = 'results/df_bef_after_tide5_std_avg_06_22.csv'):
        df_baseline = pd.read_csv(baselinePath)
        df_bl_filter = df_baseline[df_baseline['dev_count_before'] >50]
        df_bl_filter = df_bl_filter[df_bl_filter['dev_count_after'] >50]
        plt.figure(figsize=(8,6))
        plt.hist(df_bl_filter.dev_count_before, bins=100, alpha=0.5, label="Device count before the floods")
        plt.hist(df_bl_filter.dev_count_after, bins=100, alpha=0.5, label="Device count after the floods")

        plt.xlabel("Number of times device count of a tile is recorded in the dataset", size=14)
        plt.ylabel("Number of tiles", size=14)
        plt.title("Distribution of the number of entries for each tile")
        plt.legend(loc='upper right')
        plt.savefig("figures/histogram_dev_ount_bef_aft_great50.png")

    def histogram_change_dev_counts(self, baselinePath = 'results/df_bef_after_tide5_std_avg_06_22.csv'):
        df_baseline = pd.read_csv(baselinePath)
        # df_bl_filter = df_baseline[df_baseline['dev_count_before'] >50]
        # df_bl_filter = df_bl_filter[df_bl_filter['dev_count_after'] >50]
        plt.figure(figsize=(8,6))
        without_zero = df_baseline.dev_mean_before - df_baseline.dev_mean_after
        # without_zero = without_zero[without_zero!=0]
        plt.hist(without_zero, bins=100, alpha=1, label="Change in Device mean before and after the floods")

        plt.xlabel("Number of times device count of a tile is recorded in the dataset", size=14)
        plt.ylabel("Number of tiles", size=14)
        plt.title("Distribution of the number of entries for each tile")
        plt.legend(loc='upper right')
        plt.savefig("figures/histogram_dev_mean_change.png")

    def comparison_plot(self, x, y, figName):
        lineStart = x.min() 
        lineEnd = x.max()  

        plt.figure(figsize=(7, 7))
        plt.scatter(x, y, color = 'b', alpha=0.2)
        plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color = 'r')
        plt.xlim(lineStart - 5, x.max() + 5)
        plt.ylim(y.min() - 5, lineEnd + 5)
        plt.xlabel(x.name)
        plt.ylabel(y.name)

        plt.savefig('figures/'+ figName + '.png')

    def bl_quantitative(self, baselinePath = 'results/df_bef_after_tide5_std_avg_min_max.csv'):
        df_baseline = pd.read_csv(baselinePath)
        df_bl_filter = df_baseline[df_baseline['dev_count_before'] > 5]
        df_bl_filter = df_bl_filter[df_bl_filter['dev_count_after'] > 5]

        df_bl_before = df_bl_filter[(df_bl_filter['dev_mean_before'] - df_bl_filter['dev_mean_after']) > 5]
        df_bl_after = df_bl_filter[(df_bl_filter['dev_mean_before'] - df_bl_filter['dev_mean_after']) < -5]
        print (df_bl_before.shape)
        print (df_bl_after.shape)

        self.util.save_csv(df_bl_before, 'df_bl_before_mean5')
        self.util.save_csv(df_bl_after, 'df_bl_after_mean5')
        
    def bl_elevation(self, baselinePath = 'results/df_bef_after_tide5_std_avg_min_max.csv'):
        # df_bl = pd.read_csv(baselinePath)
        # print ("Full df shape: ", df_bl.shape, df_bl.describe)
        print('Full df shape: ', self.dataset.shape)
        df_bl_tide5_unique = self.dataset[['tide5', 'lat6', 'lon6']]
        df_bl_tide5_unique = df_bl_tide5_unique.drop_duplicates(subset=['tide5', 'lat6', 'lon6'])
        tide5_uniques = df_bl_tide5_unique.drop_duplicates(subset=['tide5'])
        lat_lon_uniques = df_bl_tide5_unique.drop_duplicates(subset=['lat6', 'lon6'])
        print ("Unique shape: ", df_bl_tide5_unique.shape)
        print("Tide5 shape: ", tide5_uniques.shape)
        print('Lat-lon shape', lat_lon_uniques.shape)
        # print(df_bl_tide5_unique)
        utils = common.utils()
        print(utils.get_elevation(-6.216140, 106.864826))

    def read_test_data(self, filepath = 'data/test_tide_jakarta_jan2020.csv'):
        df = self.util.read_data(pd.read_csv(filepath))
        self.dataset = df
        return df

    def read_full_data(self):
        util = common.utils()
        df1 = util.read_data(pd.read_csv('data/p1024_tide_wjava_ciesin_201912.csv'))
        df2 = util.read_data(pd.read_csv('data/p1024_tide_wjava_ciesin_202001.csv'))
        df = pd.concat([df1, df2])
        self.dataset = df
        print(df.shape)
        return df       

if __name__ == '__main__':
    pgm = program()
    # pgm.read_test_data()
    # pgm.read_full_data()
    # pgm.timeseries_all_days()
    # pgm.filter_for_jakarta()
    # pgm.aggregate_dataset()
    # pgm.calculate_avg_std()
    # pgm.baseline_comparison()
    # pgm.bl_quantitative()
    # pgm.bl_elevation()
    # pgm.baseline_comparison_excl_outlier()
    pgm.histogram_change_dev_counts()

