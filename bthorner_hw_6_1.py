"""
Brian Horner
CS 677 - Summer 2
Date: 8/17/2021
Week 6 Homework Question 1
This program reads in the seeds data, grabs the data corresponding to class
L=1 and L =3 and applies linear SVM, Gaussian kernel SVM and polynomial SVM.
It prints the accuracy and the confusion matrix as well as creating lists of
stats for each model for table printing
"""

# Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import svm


# Reading in data and assigning column names (features & class)
seeds_df = pd.read_csv('seeds_dataset.txt', delim_whitespace=True,
                       names=['Area', 'Perimeter', 'Compactness',
                               'Length of kernel', 'Width of kernel',
                               'Asymmetry coefficient',
                               'Length of kernel groove', 'Class'])

# Isolating just the classes L = 1 and L =2
seeds_df = seeds_df[seeds_df.Class != 3]

"""Linear SVM"""
# Splitting X and Y data
Y_seeds = seeds_df[['Class']].values
X_seeds = seeds_df[['Area', 'Perimeter', 'Compactness',
                 'Length of kernel', 'Width of kernel',
                 'Asymmetry coefficient', 'Length of kernel groove']].copy(
    deep=True)
# Scaling X data
scaler = StandardScaler()
scaler.fit(X_seeds)
X_seeds = scaler.transform(X_seeds)

# Splitting X and Y data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X_seeds, Y_seeds,
                                                    test_size=.5,
                                                    random_state=33)

# Applying linear SVM
linear_svm = svm.SVC(kernel='linear')
linear_svm.fit(X_train, Y_train.ravel())
linear_predict = linear_svm.predict(X_test)

# Computing accuracy and confusion matrix
linear_acc = round(accuracy_score(Y_test, linear_predict), 2)
linear_conf = confusion_matrix(Y_test, linear_predict)

# Computing TP, TN, FP, FN, TPR and TNR
linear_tp = linear_conf[0][0]; linear_fp = linear_conf[1][0]
linear_tn = linear_conf[1][1]; linear_fn = linear_conf[0][1]

linear_tpr = round(linear_tp/(linear_tp+linear_fn), 2)
linear_tnr = round(linear_tn/(linear_tn+linear_fp), 2)

# Adding statistics to list for table printing
linear_svm_list = ['Linear SVM', linear_tp, linear_fp, linear_tn, linear_fn,
                   linear_acc, linear_tpr, linear_tnr]



"""Gaussian SVM"""

# Splitting X and Y Data
Y_seeds = seeds_df[['Class']].values
X_seeds = seeds_df[['Area', 'Perimeter', 'Compactness',
                 'Length of kernel', 'Width of kernel',
                 'Asymmetry coefficient', 'Length of kernel groove']].copy(
    deep=True)

# Scaling X data
scaler = StandardScaler()
scaler.fit(X_seeds)
X_seeds = scaler.transform(X_seeds)

# Splitting X and Y into train and test data
X_train, X_test, Y_train, Y_test = train_test_split(X_seeds, Y_seeds,
                                                    test_size=.5,
                                                    random_state=33)
# Applying Gaussian kernel SVM
gaussian_svm = svm.SVC(kernel='rbf')
gaussian_svm.fit(X_train, Y_train.ravel())
gaussian_predict = gaussian_svm.predict(X_test)

# Computing accuracy and confusion matrix for model
gaussian_acc = round(accuracy_score(Y_test, gaussian_predict), 2)
gaussian_conf = confusion_matrix(Y_test, gaussian_predict)

# Computing TP, TN, FP, FN, TPR and TNR
gaussian_tp = gaussian_conf[0][0]; gaussian_fp = gaussian_conf[1][0]
gaussian_tn = gaussian_conf[1][1]; gaussian_fn = gaussian_conf[0][1]
gaussian_tpr = round(gaussian_tp/(gaussian_tp+gaussian_fn), 2)
gaussian_tnr = round(gaussian_tn/(gaussian_tn+gaussian_fp), 2)

# Adding statistics to list for table printing
gaussian_svm_list = ['Gaussian SVM', gaussian_tp, gaussian_fp, gaussian_tn,
                    gaussian_fn, gaussian_acc, gaussian_tpr, gaussian_tnr]



"""Polynomial"""

# Splitting X and Y data
Y_seeds = seeds_df[['Class']].values
X_seeds = seeds_df[['Area', 'Perimeter', 'Compactness',
                 'Length of kernel', 'Width of kernel',
                 'Asymmetry coefficient', 'Length of kernel groove']].copy(
    deep=True)

# Scaling X data
scaler = StandardScaler()
scaler.fit(X_seeds)
X_seeds = scaler.transform(X_seeds)

# Splitting X and Y data into train and test splits
X_train, X_test, Y_train, Y_test = train_test_split(X_seeds, Y_seeds,
                                                    test_size=.5,
                                                    random_state=33)

# Applying polynomial SVM
poly_svm = svm.SVC(kernel='poly', degree=3)
poly_svm.fit(X_train, Y_train.ravel())
poly_predict = poly_svm.predict(X_test)

# Computing accuracy and confusion matrix
poly_acc = round(accuracy_score(Y_test, poly_predict), 2)
poly_conf = confusion_matrix(Y_test, poly_predict)

# Computing TP, TN, FP, FN, TPR and TNR
poly_tp = poly_conf[0][0]; poly_fp = poly_conf[1][0]
poly_tn = poly_conf[1][1]; poly_fn = poly_conf[0][1]
poly_tpr = round(poly_tp/(poly_tp+poly_fn), 2)
poly_tnr = round(poly_tn/(poly_tn+poly_fp), 2)

poly_svm_list = ['Polynomial SVM', poly_tp, poly_fp, poly_tn,
                    poly_fn, poly_acc, poly_tpr, poly_tnr]

# Appending model statistics to print list for table printing
print_list = [linear_svm_list, gaussian_svm_list, poly_svm_list]

"""-------------------------------------------------------------------"""

if __name__ == "__main__":

    # Printing accuracy and confusion matrix for each model
    print(f"Accuracy of a linear kernel SVM is {linear_acc}. Its confusion matrix "
          f"is...")
    print(linear_conf)

    print(f"\nAccuracy of a Gaussian kernel SVM is {gaussian_acc}. Its confusion "
          f"matrix is... ")
    print(gaussian_conf)

    print(f"\nAccuracy of a polynomial kernel SVM of degree 3 is {poly_acc}. Its "
          f"confusion matrix is...")
    print(poly_conf)

