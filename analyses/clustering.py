import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import time, datetime
# from sklearn.cluster import dbScan
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import silhouette_score, silhouette_samples

class kmeans:
    def __init__(self, dataset = None, n_cluster = [2, 3], startTime = time.time()):
        self.dataset = dataset
        self.n_cluster = n_cluster
        self.startTime = startTime
        # self.max_iter = max_iter
        # self.random_state = random_state

    def kmean_bef_after(self):
        principalDf = self.dataset[['elev_norm', 'dev_count_norm']]
        # print(principalDf.head())
        # principalDf.dropna(inplace=True)
        print('IsNan: ', np.any(np.isnan(principalDf)))
        print('IsFinite: ', np.all(np.isfinite(principalDf)))
        range_n_clusters = self.n_cluster
        now = datetime.datetime.now()

        for n_clusters in range_n_clusters:
            start = time.time()
            now = datetime.datetime.now()
            print('\nStarting n_cluster = ', n_clusters, ' at ', datetime.datetime.now())
            suffix = '_' + str(n_clusters) + '_' + str(now.month).zfill(2) + str(now.day).zfill(2) +\
                     '_' + str(now.hour).zfill(2)
            fig, (ax1, ax2) = plt.subplots(1,2)
            fig.set_size_inches(12, 5)

            ax1.set_xlim([-0.1, 1])
            ax1.set_ylim([0, len(principalDf) + (n_clusters + 1) * 10])

            clusterer = KMeans(n_clusters = n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(principalDf)
            # print(cluster_labels)
            silhouette_avg = silhouette_score(principalDf, cluster_labels)
            clusterDf = principalDf
            clusterDf['cluster'] = cluster_labels
            clusterDf['lat6'] = self.dataset['lat6']
            clusterDf['lon6'] = self.dataset['lon6']
            # print(clusterDf.head())
            clusterDf.to_csv('results/cluster_df' + suffix + '.csv')

            print("For n_clusters =", n_clusters,
                "The average silhouette_score is :", silhouette_avg)
            print("Clustering completed at: ", datetime.datetime.now())
            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(principalDf, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                0, ith_cluster_silhouette_values,
                                facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.scatter(principalDf.loc[:, 'elev_norm'], principalDf.loc[:, 'dev_count_norm'], marker='.', s=30, lw=0, alpha=0.7,
                        c=colors, edgecolor='k')

            # Labeling the clusters
            centers = clusterer.cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                        c="white", alpha=1, s=200, edgecolor='k')

            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                            s=50, edgecolor='k')

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")

            plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                        "with n_clusters = %d" % n_clusters),
                        fontsize=14, fontweight='bold')
        #     plt.tight_layout()
            # plt.show()
            # now = datetime.datetime.now()
            plotname = 'silhoutte_n' + suffix
            plt.savefig('figures/' + plotname + '.png')

            end = time.time()
            hours, rem = divmod(end-start, 3600)
            minutes, seconds = divmod(rem, 60)
            print('Time taken for cluster ', n_clusters, ": {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

    def read_before_after_elevation_data(self, filepath = 'data/before_after_wElevation.csv'): #results/before_after_wElevation_sample.csv'): #
        df = pd.read_csv(filepath)

        # print(df.columns)
        df.rename(columns={'s10e105_de': 'elevation'}, inplace = True)

        # Looking at percentage change in device count as it is more representative of the data
        df = df.loc[df['dev_mean_b'] != 0]
        df['incr_in_device_count'] = (df['dev_mean_a'] - df['dev_mean_b'])/df['dev_mean_b']
        df = df[['OBJECTID', 'lat6', 'lon6', 'elevation', 'incr_in_device_count']]
        
        # Normalizing each column and adding latitude and longitude back
        minMaxScaler = preprocessing.MinMaxScaler()
        df_normalize = df[['elevation', 'incr_in_device_count']]
        df_normal = pd.DataFrame(minMaxScaler.fit_transform(df_normalize))
        df_normal.rename(columns={0 : 'elev_norm', 1 : 'dev_count_norm'}, inplace = True)
        df_normal['lat6'] = df['lat6']
        df_normal['lon6'] = df['lon6']
        df_normal['OBJECTID'] = df['OBJECTID']
        # print(df_normalize.head())
        # print(df_normal.head())
        df_normal.dropna(inplace=True)
        df_normal.to_csv('results/bef_aft_norm_elev_devCount.csv')

        self.dataset = df_normal
        return df
    
    def save_sample_data(self, filepath = 'data/cluster_df_3_0624_18.csv'):
        df = pd.read_csv(filepath)
        df_sample = df.head(10000)
        df_sample.to_csv('results/before_after_wElevation_sample.csv')

    def hist_cludters(self, filepath = 'results/cluster_df_dev_3_0624_18.csv'):
        df_cluster = pd.read_csv(filepath)
        print(df_cluster.head())
        clstr0_df = df_cluster[df_cluster['cluster'] == 0]
        clstr1_df = df_cluster[df_cluster['cluster'] == 1]
        clstr2_df = df_cluster[df_cluster['cluster'] == 2]
        
        clstr0_chng = clstr0_df.dev_count_norm
        clstr1_chng = clstr1_df.dev_count_norm
        clstr2_chng = clstr2_df.dev_count_norm
        # plt.hist(clstr0_chng, bins=100, alpha=0.5, label="Change in Device mean for cluster 0", color='red')
        # plt.hist(clstr1_chng, bins=100, alpha=0.5, label="Change in Device mean for cluster 1", color='blue')
        plt.hist(clstr2_chng, bins=100, alpha=0.5, label="Change in Device mean for cluster 2",color='green')

        plt.xlabel("Change in device count normalized", size=14)
        plt.ylabel("Count", size=14)
        plt.title("Distribution of the change in device count for different clusters")
        plt.legend(loc='upper right')
        plt.savefig("figures/hist_cludters_dev_count_3_2.png")

    def hist_cludters_dev(self, filepath = 'results/clstr_df_dev_3_0701_17.csv'):
        df_cluster = pd.read_csv(filepath)
        print(df_cluster.head())
        clstr0_df = df_cluster[df_cluster['cluster'] == 0]
        clstr1_df = df_cluster[df_cluster['cluster'] == 1]
        clstr2_df = df_cluster[df_cluster['cluster'] == 2]
        
        clstr0_chng = clstr0_df.dev_mean_a_norm - clstr0_df.dev_mean_b_norm
        clstr1_chng = clstr1_df.dev_mean_a_norm - clstr1_df.dev_mean_b_norm
        clstr2_chng = clstr2_df.dev_mean_a_norm - clstr2_df.dev_mean_b_norm
        plt.hist(clstr0_chng, bins=100, alpha=0.5, label="Change in Device mean for cluster 0", color='red')
        # plt.hist(clstr1_chng, bins=100, alpha=0.5, label="Change in Device mean for cluster 1", color='blue')
        # plt.hist(clstr2_chng, bins=100, alpha=1, label="Change in Device mean for cluster 2",color='green')

        plt.xlabel("Change in device count normalized", size=14)
        plt.ylabel("Count", size=14)
        plt.title("Distribution of the change in device count for different clusters")
        plt.legend(loc='upper right')
        plt.savefig("figures/hist_cludters_dev_count_3_0.png")

if __name__ == '__main__':
    clstr = kmeans()
    # clstr.save_sample_data()
    # clstr.read_before_after_elevation_data()
    # clstr.kmean_bef_after()
    # start = clstr.startTime
    # end = time.time()
    # hours, rem = divmod(end-start, 3600)
    # minutes, seconds = divmod(rem, 60)
    # print("Total time taken: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    # clstr.hist_cludters()
    clstr.hist_cludters_dev()
