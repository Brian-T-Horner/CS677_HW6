"""
Brian Horner
CS 677 - Summer 2
Date: 8/17/2021
Week 6 Homework Question 2
This program applies Gaussian Naive Bayesian to our data set and returns the
accuracy and confusion matrix. Program then prints a table of the statistics
for linear SVM, Gaussian SVM, polynomial SVM and Gaussian Naive Bayesian.
"""

# imports
from bthorner_hw_6_1 import seeds_df, print_list
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB


def table_printer(print_list):
    """Formats the computations for table printing."""
    # Header list for the top of the table
    header_list = ['Model', 'TP', 'FP', 'TN', 'FN', 'Accuracy', 'TPR',
                   'TNR']
    # Adding each models list to the print_list
    print("--- Summary of Models and Confusion Matrix Stats. ---\n")
    print_list.insert(0, header_list)
    for index, stuff in enumerate(print_list):
        # Adding a | in front of each value of the lists in print list
        row = '|'.join(str(value).ljust(15) for value in stuff)
        # Printing the row for the list in print list
        print(row)
        # Adding a line between the header and the data rows
        if index == 0:
            print('-' * len(row))


"""Naive Bayesian"""

# Splitting X and Y data
Y_seeds = seeds_df[['Class']].values
X_seeds = seeds_df[['Area', 'Perimeter', 'Compactness',
                 'Length of kernel', 'Width of kernel',
                 'Asymmetry coefficient', 'Length of kernel groove']].copy(
    deep=True)
# Splitting X and Y data into train and test splits
X_train, X_test, Y_train, Y_test = train_test_split(X_seeds, Y_seeds,
                                                    test_size=.5,
                                                    random_state=33)
# Applying Gaussian Naive Bayesian
g_naive_bayes = GaussianNB()

# Fitting model. Used ravel instead of reshape on NP Array to avoid warning
gnb = g_naive_bayes.fit(X_train, Y_train.ravel())
y_predict = gnb.predict(X_test)

# Computing accuracy and Confusion Matrix
gnb_acc = round(accuracy_score(Y_test, y_predict), 2)
gnb_conf = confusion_matrix(Y_test, y_predict)

# Computing TP, TN, FP, FN, TPR and TNR
gnb_tp = gnb_conf[0][0]; gnb_fp = gnb_conf[1][0]
gnb_tn = gnb_conf[1][1]; gnb_fn = gnb_conf[0][1]
gnb_tpr = round(gnb_tp/(gnb_tp+gnb_fn), 2)
gnb_tnr = round(gnb_tn/(gnb_tn+gnb_fp), 2)

# Adding computations to list for table printing
gnb_list = ['Naive Bayesian', gnb_tp, gnb_fp,
            gnb_tn, gnb_fn, gnb_acc, gnb_tpr, gnb_tnr
            ]
print_list.append(gnb_list)

"""-------------------------------------------------------------------"""

if __name__ == "__main__":
    print(f"\nAccuracy of a Gaussian Naive Bayesian model is {gnb_acc}. Its "
          f"confusion matrix is...")
    print(gnb_conf)
    print("\n")
    table_printer(print_list)
