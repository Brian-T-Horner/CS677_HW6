"""
Brian Horner
CS 677 - Summer 2
Date: 8/17/2021
Week 6 Homework Question 3
This program applies k-means classifier with k values of 1-8. It plots the
errors and k values in order for the user to determine the best k value.
It then plot the clusters and 2 randomly selected feature points. It assigns
a label to each cluster based on the majority of True class labels and uses
this to assign a cluster label based on the lowest euclidean distance from each
cluster for the two randomly selected features. Finally we compare this
accuracy for all classes and our two classes from the remainder of our bu id.
"""

# Imports
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from random import seed, randint
from math import sqrt
from sklearn.metrics import accuracy_score, confusion_matrix
from bthorner_hw_6_2 import table_printer, print_list

# Reading data and assigning columns (features and class)
seeds_df = pd.read_csv('seeds_dataset.txt', delim_whitespace=True,
                       names=['Area', 'Perimeter', 'Compactness',
                              'Length of kernel', 'Width of kernel',
                              'Asymmetry coefficient',
                              'Length of kernel groove', 'Class'])

# Splitting Y and X data
Y_seeds = seeds_df[['Class']].values
X_seeds = seeds_df[['Area', 'Perimeter', 'Compactness',
                    'Length of kernel', 'Width of kernel',
                    'Asymmetry coefficient', 'Length of kernel groove']].copy(
                     deep=True)

# Scaling X data
scaler = MinMaxScaler()
X_seeds = scaler.fit_transform(X_seeds)

"""Question 3.1"""

# Finding best k-means clustering k value by inertia
SSE = []
# K value of 1-8
for k in range(1, 9):
    km = KMeans(n_clusters=k)
    km.fit(X_seeds)
    SSE.append(km.inertia_)

# Plotting k value and SSE to determine best k value by knee method
elbow_plot = plt.figure('K-Means Knee Plot')
plt.xlabel('K')
plt.ylabel('SSE')
plt.plot(range(1, 9), SSE)
plt.title('K-Means Knee Plot')
# Saving plot as a pdf to avoid any program interruption
elbow_plot.savefig('KMeans_Elbow_Plot.pdf')
print("The best k value for the K-Means model and the data is is 3.\n")


"""Question 3.2"""

# Assigning KMeans the best n_cluster value and fitting model
km = KMeans(n_clusters=3)
Y_predicted = km.fit_predict(X_seeds)


# Getting random feature values with randint and a seed value of 33
i = 0
randint_list = []
seed(33)
while i < 2:
    randint_list.append(randint(1, 7))
    i += 1

# Copying X_seeds
X_seeds = pd.DataFrame(X_seeds).copy(deep=True)
# Assigning randomly chosen features to new dataframe
plotting_df = X_seeds.iloc[:, [randint_list[0], randint_list[1]]]
# Assigning appropriate column values of the chosen features to new dataframe
plotting_df.columns = [seeds_df.columns[randint_list[0]], seeds_df.columns[
    randint_list[1]]]

# Adding a column to new dataframe for the predicted clusters
plotting_df = plotting_df.assign(Clusters=Y_predicted)
# Grabbing the centroids of the clusters [:,0] = x and [:, 1] = y
centroids = np.array(km.cluster_centers_)

# Grabbing columns for columns names used in sns.scatterplot
plotting_columns = plotting_df.columns

"""At this point I'm not sure if I miss understood which centroids we were 
suppose to use. Originally I did it by the 2 dimensional (x and y) centroids. 
I ended up choosing to go with the features dimension centroids."""


# Plotting clusters and two features as X and Y of scatter plot
scatter_plot = plt.figure('Feature and K-Means Scatter Plot')
sns.scatterplot(data=plotting_df, x=plotting_columns[0], y=plotting_columns[1],
                hue=plotting_columns[2]).set(title=f"Scatter plot of "
                                                   f"Two Random Features with "
                                                   f"K-Means Predicted as Hue")
# Plotting centroids of the clusters
sns.scatterplot(data=centroids, x=centroids[:, randint_list[0]], y=centroids[:,
                                                                 randint_list[1]],
                marker='X',
                color='Blue', label='Centroid', s=50)
# Saving plot as a pdf to avoid program interruption
scatter_plot.savefig('Feature_KMeans_Scatter.pdf')

print("Observing the scatter plot is seems that Asymmetry coefficient and "
      "compactness are not easily sorted into clusters. \nUltimately with all "
      "the features the dimensions would be greater and the sorting would be "
      "cleaner as well.\n")

"""Question 3.3"""

# Grabbing copy of Class column from seeds dataframe
plotting_df['Class'] = seeds_df['Class'].copy(deep=True)
# Splitting dataframe into three by class value
cluster1_df = plotting_df.loc[plotting_df['Clusters'] == 0]
cluster2_df = plotting_df.loc[plotting_df['Clusters'] == 1]
cluster3_df = plotting_df.loc[plotting_df['Clusters'] == 2]


def class_counter(cluster_df):
    """Computes the predicted class value of a dataset row by majority
    voting."""
    max_count = 0
    majority_class = 0
    # Range of class values 1-3
    for j in range(1, 4):
        # If current class count is greater than previous set to max_count
        current_count = (cluster_df.Class == j).sum()
        if current_count > max_count:
            max_count = current_count
            # Setting majority class to highest class count
            majority_class = j
    # Assigning majority voted class to data row
    cluster_df = cluster_df.assign(predicted_class=majority_class)

    return cluster_df


# Getting class label for cluster 1
cluster1_df = class_counter(cluster1_df)
cluster1_label = cluster1_df.predicted_class.mode()[0]
# Getting centroids for cluster 1
cluster1_centroid = (float(centroids[0:1, randint_list[0]]), float(centroids[0:1,
                                                                   randint_list[
                                                                    1]]))


print(f"Cluster 1 centroid is {cluster1_centroid} and its label is "
      f"{cluster1_label}.\n")

# Getting class label for cluster 2
cluster2_df = class_counter(cluster2_df)
cluster2_label = cluster2_df.predicted_class.mode()[0]
# Getting centroids for cluster 2
cluster2_centroid = (float(centroids[1:2, randint_list[0]]), float(centroids[
                                                                 1:2,
                                                                   randint_list[1]]))

print(f"Cluster 2 centroid is {cluster2_centroid} and its la"
      f"bel is {cluster2_label}.\n")

# Getting class label for cluster 3
cluster3_df = class_counter(cluster3_df)
cluster3_label = cluster3_df.predicted_class.mode()[0]
# Getting centroids for cluster 3
cluster3_centroid = (float(centroids[2:3, randint_list[0]]), float(centroids[
                                                                 2:3,
                                                                   randint_list[1]]))

print(f"Cluster 3 centroid is {cluster3_centroid} and its label is"
      f" {cluster3_label}.\n")

"""Question 3.4 (Multi-label Classifier)"""

# Getting X and Y coordinates from randomly selected features
x_cords = plotting_df.iloc[:, 0:1].values
y_cords = plotting_df.iloc[:, 1:2].values

# Calculating label of each dataframe row through euclidean distance from
# each clusters centroids
euclidean_labels = []
# Iterating through each row of dataframe
for i in range(0, len(x_cords)):
    # Calculating distance of dataframe rows X and Y from each cluster centroids
    cluster3_dist = sqrt((cluster3_centroid[0] - x_cords[i]) ** 2 +
                         (cluster3_centroid[1] - y_cords[i]) ** 2)
    cluster2_dist = sqrt((cluster2_centroid[0] - x_cords[i]) ** 2 +
                         (cluster2_centroid[1] - y_cords[i]) ** 2)
    cluster1_dist = sqrt((cluster1_centroid[0] - x_cords[i]) ** 2 +
                         (cluster1_centroid[1] - y_cords[i]) ** 2)
    # Comparing euclidean distance for each row of data to determine class label
    if cluster1_dist < cluster2_dist and cluster1_dist < cluster3_dist:
        euclidean_labels.append(cluster1_label)
    elif cluster2_dist < cluster3_dist:
        euclidean_labels.append(cluster2_label)
    else:
        euclidean_labels.append(cluster3_label)


# Adding euclidean predicted labels to dataframe
plotting_df['Euclidean_Labels'] = euclidean_labels
# Calculating accuracy of the euclidean multi-label classifier
euclidean_acc = round(accuracy_score(plotting_df['Class'].values,
                                     plotting_df['Euclidean_Labels'].values), 2)

print(f"\nThe accuracy for the euclidean distance multi-label classifier on "
      f"all features is "
      f"{euclidean_acc}.")
print("This accuracy is more than likely lower than other models as we are "
      "only taking two dimensions centroids into consideration. \nThe data is "
      "not easily split into distinct clusters given the selected features "
      "centroids.")


"""Question 3.5"""


# Grabbing true class labels for our two selected features
true_labels = plotting_df[plotting_df.Class != 3]
true_labels = true_labels.Class
predicted_labels = plotting_df[plotting_df.Euclidean_Labels != 3]
predicted_labels = predicted_labels.Euclidean_Labels


# Slicing label arrays if one if len is not equal
if len(true_labels) > len(predicted_labels):
    true_labels = true_labels[:len(predicted_labels)]
elif len(predicted_labels) > len(true_labels):
    predicted_labels = predicted_labels[:len(true_labels)]

# Calculating accuracy of euclidean distance model
euclidean_acc_features = round(accuracy_score(true_labels,
                               predicted_labels), 2)

# Computing confusion matrix for euclidean distance model
euclidean_conf = confusion_matrix(true_labels.values, predicted_labels.values)

# # Computing TP, TN, FP, FN, TPR and TNR for euclidean distance model
euclidean_tp = euclidean_conf[0][0]; euclidean_fp = euclidean_conf[1][0]
euclidean_tn = euclidean_conf[1][1]; euclidean_fn = euclidean_conf[0][1]
euclidean_tpr = round(euclidean_tp/(euclidean_tp+euclidean_fn), 2)
euclidean_tnr = round(euclidean_tn/(euclidean_tn+euclidean_fp), 2)
#
# # Adding euclidean distance model stats to print list for table printing
euclidean_list = ['Euclidean Model', euclidean_tp, euclidean_fp,
                  euclidean_tn, euclidean_fn, euclidean_acc_features,
                  euclidean_tpr, euclidean_tnr]

print_list.append(euclidean_list)

print("\nThe accuracy of the Euclidean Model on the randomly selected features "
      f"is {euclidean_acc_features} and its confusion matrix is...")
print(euclidean_conf)
print('\n')
table_printer(print_list)


print("\nCompared to other models the multi-label classifier using Euclidean "
      "distance is lacking.\nIts accuracy is lower as well as its TPR, TNR, "
      "TP and TN. Its FP and FN are higher. \nI believe this is due to the "
      "fact that I only did the euclidean with two features centroids and "
      "values as opposed to all.\nIf done with all centroid values and all "
      "feature values this should in the same range as the other models.")
