import pandas as pd
import matplotlib.pyplot as plt

class commonAnalysis:

    def group_by_day_count(self, dataframe):
        dataframe = dataframe[['ts_local', 'dev_count', 'rq_count']]
        df_group_by_day_count = dataframe.set_index('ts_local').groupby(pd.Grouper(freq='D')).sum()
        df_group_by_day_count['ts_local'] = pd.to_datetime(df_group_by_day_count.index)
        df_group_by_day_count.to_csv('day_count_all.csv')

        # print df_group_by_day_count.dtypes
        df_group_by_day_count[['ts_local', 'dev_count']].plot()
        plt.savefig('figures/dev_count_ts_test2.png')
        # print df_group_by_day_count.head(60)
        return df_group_by_day_count

    def group_by_day_region(self, dataframe):
        """
        Takes too long and number of regions quite high
        """
        dataframe = dataframe[['ts_local', 'tide5', 'dev_count', 'rq_count']]
        dataframe['day'] = dataframe['ts_local'].dt.date
        dataframe.drop('ts_local', axis = 1, inplace = True)

        df_group_day_reg = dataframe.groupby(['day', 'tide5']).sum().reset_index()
        # df_group_day_reg['day'] = df_group_day_reg.index['day']
        # df_group_day_reg['tide5'] = df_group_day_reg.index['tide5']
        
        print (df_group_day_reg.dtypes,'\n', df_group_day_reg.head(10))
        df_day_reg_devCount = df_group_day_reg[['day', 'tide5', 'dev_count']]
        df_day_reg = df_day_reg_devCount.pivot(index = 'day', columns = 'tide5', values = 'dev_count').reset_index()
        df_day_reg.plot()
        plt.savefig('dev_count_ts_region.png')

    def aggregate_by_num_hours(self, dataframe, numOfHours):
        print (dataframe.describe())
        df_hours = dataframe.groupby([dataframe['ts_local'].dt.hour // numOfHours,\
            dataframe['ts_local'].dt.date,\
                'lat6', 'lon6']).sum()
        df_hours.index.names = [str(numOfHours) + '_hourBin', 'ts_date', 'lat6', 'lon6']
        df_hours = df_hours.reset_index()
        print (df_hours.head(30))
        print (df_hours.describe())
        return df_hours

    def filter_by_provinces(self, dataframe, provinces):
        # if type(provinces) == list:
        #     for province in provinces:
        df_provinces = dataframe[dataframe['province'].isin(provinces)]
        # print df_provinces.head()
        return df_provinces

    def cal_basic_parameters_regionwise(self, dataframe):
        dataframe = dataframe[['tide5', 'lat6', 'lon6','dev_count','rq_count']]
        df_params = dataframe.groupby(['tide5', 'lat6', 'lon6'])\
            .agg({'dev_count':['mean', 'max', 'min', 'std', 'count'],\
            'rq_count':['mean', 'max', 'min', 'std', 'count']})
        
        print (df_params.head())

        return df_params.reset_index()


class figures:
    def scatter_plot(self, df_scatter, figName):
        fig = plt.scatter(df_scatter[0], df_scatter[1], marker='ro', alpha=0.3)
        fig.savefig('figures/' + figName + '.png')

if __name__ == '__main__':
    analyses = commonAnalysis()
    """
    df = pd.read_csv('test_tide_jakarta_jan2020.csv')
    df_clean = analyses.read_data(df)
    df_day_count = analyses.group_by_day_count(df_clean)
    print df_day_count.head(31)
    """

    # df_all = analyses.read_data(pd.read_csv('test_tide_jakarta_jan2020.csv'))

    # Number of regions quite high for making a multiple time series graph
    # df_day_region = analyses.group_by_day_region(df_all)
    #print df_all['tide5'].nunique()



