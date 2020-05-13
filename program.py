from analyses import initialAnalysis
from utils import common
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

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
        print df_provinces.describe()

    def calculate_avg_std(self):
        df = self.dataset[['tide5', 'dev_count', 'rq_count', 'ts_local']]
        df_before = df[df['ts_local'] < '2020-1-1']
        df_after = df[df['ts_local'] >= '2020-1-3']
        df_before_calc = self.analysis.cal_basic_parameters_regionwise(df_before)
        df_after_calc = self.analysis.cal_basic_parameters_regionwise(df_after)

        df_combined = pd.merge(df_before_calc, df_after_calc, on = 'tide5', how = 'inner', suffixes = ('_before', '_after'))
        # print df_combined.columns
        df_combined.columns = ['tide5', 'dev_mean_before', 'dev_max_before', 'dev_min_before', 'dev_std_before', 'dev_count_before',\
            'rq_mean_before', 'rq_max_before', 'rq_min_before', 'rq_std_before', 'rq_count_before',\
            'dev_mean_after', 'dev_max_after', 'dev_min_after', 'dev_std_after', 'dev_count_after',\
            'rq_mean_after', 'rq_max_after', 'rq_min_after', 'rq_std_after', 'rq_count_after']
        # print df_combined.columns
        print df_combined.head()
        self.util.save_csv(df_combined, 'df_bef_after_tide5_std_avg_min_max')

        return df_combined

    def baseline_comparison(self, baselinePath = 'results/df_bef_after_tide5_std_avg_min_max.csv'):
        df_baseline = pd.read_csv(baselinePath)
        print df_baseline.shape
        df_bl_filter = df_baseline[df_baseline['dev_count_before'] > 5]
        df_bl_filter = df_bl_filter[df_bl_filter['dev_count_after'] > 5]

        print df_bl_filter.shape
        # df_avg = df_baseline[['dev_mean_before','dev_mean_after']]
        x1 = df_baseline.dev_mean_before
        y1 =  df_baseline.dev_mean_after
        self.comparison_plot(x1, y1, 'dev_mean_bef_ater_filter02')

        x2 = df_baseline.dev_std_before
        y2 =  df_baseline.dev_std_after
        self.comparison_plot(x2, y2, 'dev_std_bef_ater_filter02')

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
        print df_bl_before.shape
        print df_bl_after.shape

        self.util.save_csv(df_bl_before, 'df_bl_before_mean5')
        self.util.save_csv(df_bl_after, 'df_bl_after_mean5')
        

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
    pgm.bl_quantitative()