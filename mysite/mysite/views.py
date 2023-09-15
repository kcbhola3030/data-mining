# import csv
# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt

# @csrf_exempt  # Exempt CSRF protection for demonstration purposes, consider adding proper protection
# def upload_csv(request):
#     if request.method == 'POST':
#         csv_file = request.FILES.get('file')  # Make sure the field name matches what's sent from the client

#         if csv_file is None:
#             return JsonResponse({'error': 'No file uploaded'}, status=400)

#         csv_data = csv_file.read().decode('utf-8')
#         rows = csv.reader(csv_data.splitlines())

#         data = []
#         for row in rows:
#             data.append(row)  # Add each row as a list to the data list

#         response_data = {'data': data}
#         return JsonResponse(response_data)

#     return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import math
import csv
from statistics import mean, median, mode, variance, stdev
from django.http import JsonResponse
from math import ceil
from django.views.decorators.csrf import csrf_exempt
# views.py
from django.shortcuts import render
from scipy.stats import chi2_contingency
from scipy.stats import pearsonr
import numpy as np
import json


@csrf_exempt  # Exempt CSRF protection for demonstration purposes, consider adding proper protection
def upload_csv(request):
    if request.method == 'POST':
        # Make sure the field name matches what's sent from the client
        csv_file = request.FILES.get('file')

        if csv_file is None:
            return JsonResponse({'error': 'No file uploaded'}, status=400)

        csv_data = csv_file.read().decode('utf-8')
        rows = csv.reader(csv_data.splitlines())

        data = list(rows)  # Convert rows to a list

        if len(data) < 2:
            return JsonResponse({'error': 'CSV file must contain at least two rows'}, status=400)

        header = data[0]  # Extract header row
        data = data[1:]  # Remove header row from data

        # Initialize a dictionary to hold column data
        columns = {column: [] for column in header}

        for row in data:
            for col_idx, value in enumerate(row):
                columns[header[col_idx]].append(value)

        calculated_stats = {}

        for column, values in columns.items():
            # Convert values to float or int if possible
            converted_values = []
            for value in values:
                try:
                    converted_value = float(value)
                    converted_values.append(converted_value)
                except ValueError:
                    pass

            if converted_values:
                calculated_stats[column] = {
                    'mean': mean(converted_values),
                    'mode': mode(converted_values),
                    'median': median(converted_values),
                    'variance': variance(converted_values),
                    'std_deviation': stdev(converted_values),
                    'mid_range': calculate_midrange(converted_values),
                    'range': max(converted_values) - min(converted_values),
                    'quartiles': [percentile(converted_values, 25),
                                  percentile(converted_values, 50),
                                  percentile(converted_values, 75)],
                    'interquartile_range': percentile(converted_values, 75) - percentile(converted_values, 25),
                    'five_number_summary': [min(converted_values),
                                            percentile(converted_values, 25),
                                            percentile(converted_values, 50),
                                            percentile(converted_values, 75),
                                            max(converted_values)],
                    "mean_cal": calculate_mean(converted_values),
                    "mode_cal": calculate_mode(converted_values),
                    "median_cal": calculate_median(converted_values),
                    "variance_cal": calculate_variance(converted_values),
                    "std_deviation_cal": calculate_standard_deviation(converted_values),
                    "midrange_cal": calculate_midrange(converted_values),
                    'converted_values': converted_values
                }
            else:
                calculated_stats[column] = {'error': 'No valid numeric data'}

        response_data = {'statistics': calculated_stats}
        return JsonResponse(response_data)

    return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)


def percentile(data, percent):
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * percent / 100
    f = int(k)
    c = ceil(k)
    if f == c:
        return sorted_data[int(k)]
    else:
        return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


def calculate_midrange(data):
    if not data:
        return None

    max_value = max(data)
    min_value = min(data)
    midrange = (max_value + min_value) / 2

    return midrange


def calculate_mean(data):
    if len(data) == 0:
        return None
    return sum(data) / len(data)


def calculate_mode(data):
    if len(data) == 0:
        return None

    counts = {}
    for value in data:
        if value in counts:
            counts[value] += 1
        else:
            counts[value] = 1

    mode = max(counts, key=counts.get)
    return mode


def calculate_median(data):
    if len(data) == 0:
        return None

    sorted_data = sorted(data)
    n = len(sorted_data)
    if n % 2 == 1:
        return sorted_data[n // 2]
    else:
        middle1 = sorted_data[n // 2 - 1]
        middle2 = sorted_data[n // 2]
        return (middle1 + middle2) / 2


def calculate_variance(data):
    if len(data) == 0:
        return None

    mean = calculate_mean(data)
    squared_diffs = [(x - mean) ** 2 for x in data]
    variance = sum(squared_diffs) / len(data)
    return variance


def calculate_standard_deviation(data):
    variance = calculate_variance(data)
    if variance is None:
        return None
    return math.sqrt(variance)


@csrf_exempt  # Exempt CSRF protection for demonstration purposes, consider adding proper protection
def calculate_pearson(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        array1 = data.get('array1', [])
        array2 = data.get('array2', [])

        # Convert the string values in the arrays to floats
        array1 = [float(value) if value is not None else 0 for value in array1]
        array2 = [float(value) if value is not None else 0 for value in array2]

        # Calculate the Pearson correlation coefficient and p-value
        # pearson_coefficient, p_value = pearsonr(array1, array2)
        stdx = np.std(array1)
        stdy = np.std(array2)
        meanx = np.mean(array1)
        meany = np.mean(array2)

        sum_of_product = sum(x * y for x, y in zip(array1, array2))
        cov = (sum_of_product//(len(array1)))-(meanx*meany)
        pearson_coefficient = cov//(stdx*stdy)

        response_data = {
            'pearson_coefficient': (pearson_coefficient),
        }

        return JsonResponse(response_data)

    return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)


# assign 3



from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score



# classifier/views.py

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score



# classifier/views.py

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn import tree
import matplotlib.pyplot as plt
import os

# @csrf_exempt
# def classify(request, method):
#     if request.method == 'POST':
#         try:
#             # Load the uploaded dataset from the POST request
#             uploaded_file = request.FILES['file']
#             data = pd.read_csv(uploaded_file)
            
#             # Get the target column specified in the POST request
#             target_column = request.POST.get('target_column', None)
#             if not target_column:
#                 return JsonResponse({"error": "Target column not specified."})

#             # Split the data into features (X) and target labels (y)
#             X = data.drop(target_column, axis=1)
#             y = data[target_column]

#             # Split the data into training and testing sets
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#             # Train a decision tree classifier using the specified method
#             if method == 'info_gain':
#                 clf = DecisionTreeClassifier(criterion='entropy')
#                 plt.figure(figsize=(12, 8)) 
#                 tree.plot_tree(clf, feature_names=X.columns, class_names=y.unique(), filled=True)
#                 plt.title("Decision Tree with Information Gain")
#                 img_path = 'decision_tree.png'
#                 plt.savefig(os.path.join('media', img_path))
#             elif method == 'gain_ratio':
#                 clf = DecisionTreeClassifier(criterion='entropy')
#             elif method == 'gini':
#                 clf = DecisionTreeClassifier(criterion='gini')
#             else:
#                 return JsonResponse({"error": "Invalid method specified."})

#             clf.fit(X_train, y_train)
            
#             # Make predictions on the testing dataset
#             predictions = clf.predict(X_test)



#             # Calculate evaluation metrics for multi-class classification
#             accuracy = accuracy_score(y_test, predictions)
#             misclassification_rate = 1 - accuracy
#             precision_micro = precision_score(y_test, predictions, average='micro')
#             recall_micro = recall_score(y_test, predictions, average='micro')
#             f1_micro = f1_score(y_test, predictions, average='micro')

#             precision_macro = precision_score(y_test, predictions, average='macro')
#             recall_macro = recall_score(y_test, predictions, average='macro')
#             f1_macro = f1_score(y_test, predictions, average='macro')

            
#             confusion = confusion_matrix(y_test, predictions)

#             # Access True Negatives, False Positives, False Negatives, and True Positives
#             tn = confusion[0, 0]
#             fp = confusion[0, 1]
#             fn = confusion[1, 0]
#             tp = confusion[1, 1]

#             # Calculate Sensitivity (True Positive Rate)
#             sensitivity = tp / (tp + fn)

#             # Calculate Specificity (True Negative Rate)
#             specificity = tn / (tn + fp)

#             # Calculate Recognition Rate (Overall Accuracy)
#             recognition_rate = (tp + tn) / (tp + tn + fp + fn)
           

#             # Include these metrics in the response_data dictionary
#             response_data = {
#                 "image_url": os.path.join('media', img_path),
#                 "accuracy": accuracy,
#                 "misclassification_rate": misclassification_rate,
#                 "precision_micro": precision_micro,
#                 "recall_micro": recall_micro,
#                 "f1_micro": f1_micro,
#                 "precision_macro": precision_macro,
#                 "recall_macro": recall_macro,
#                 "f1_macro": f1_macro,
#                 "sensitivity": sensitivity,
#                 "specificity": specificity,
#                 "recognition_rate": recognition_rate,
#             }

#             return JsonResponse(response_data)
#         except Exception as e:
#             return JsonResponse({"error": str(e)})
#     else:
#         return JsonResponse({"error": "Only POST requests are supported."})

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from django.http import JsonResponse
from django.conf import settings
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text

@csrf_exempt
def classify(request, method):
    if request.method == 'POST':
        try:
            # Load the uploaded dataset from the POST request
            uploaded_file = request.FILES['file']
            data = pd.read_csv(uploaded_file)

            # Get the target column specified in the POST request
            target_column = request.POST.get('target_column', None)
            if not target_column:
                return JsonResponse({"error": "Target column not specified."})

            # Split the data into features (X) and target labels (y)
            X = data.drop(target_column, axis=1)
            y = data[target_column]

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train a decision tree classifier using the specified method
            if method == 'info_gain':
                clf = DecisionTreeClassifier(criterion='entropy')
            elif method == 'gain_ratio':
                clf = DecisionTreeClassifier(criterion='entropy')
            elif method == 'gini':
                clf = DecisionTreeClassifier(criterion='gini')
            else:
                return JsonResponse({"error": "Invalid method specified."})

            clf.fit(X_train, y_train)

            # Create a figure for the decision tree visualization
            plt.figure(figsize=(12, 8))
            plot_tree(clf, feature_names=X.columns.tolist(), class_names=y.unique().tolist(), filled=True)
            plt.title("Decision Tree with Information Gain")
            img_path = os.path.join('client/src', 'decision_tree.png')
            # plt.savefig(img_path)
            plt.savefig(img_path, bbox_inches='tight', pad_inches=0.1)

            # Make predictions on the testing dataset
            predictions = clf.predict(X_test)

            # Calculate evaluation metrics for multi-class classification
            accuracy = accuracy_score(y_test, predictions)
            misclassification_rate = 1 - accuracy
            precision_micro = precision_score(y_test, predictions, average='micro')
            recall_micro = recall_score(y_test, predictions, average='micro')
            f1_micro = f1_score(y_test, predictions, average='micro')

            precision_macro = precision_score(y_test, predictions, average='macro')
            recall_macro = recall_score(y_test, predictions, average='macro')
            f1_macro = f1_score(y_test, predictions, average='macro')

            confusion = confusion_matrix(y_test, predictions)

            # Access True Negatives, False Positives, False Negatives, and True Positives
            tn = confusion[0, 0]
            fp = confusion[0, 1]
            fn = confusion[1, 0]
            tp = confusion[1, 1]

            # Calculate Sensitivity (True Positive Rate)
            sensitivity = tp / (tp + fn)

            # Calculate Specificity (True Negative Rate)
            specificity = tn / (tn + fp)

            # Calculate Recognition Rate (Overall Accuracy)
            recognition_rate = (tp + tn) / (tp + tn + fp + fn)

            rules = export_text(clf, feature_names=X.columns.tolist())
            
            y_pred = clf.predict(X_test)

            # Calculate Coverage
            coverage = sum(y_pred == y_test) / len(y_test)

            # Calculate Tree Depth (Toughness)
            tree_depth = clf.get_depth()
            
            # Include the image path and metrics in the response_data dictionary
            response_data = {
                "image_url": img_path,
                "rules": rules,
                "accuracy": accuracy,
                "misclassification_rate": misclassification_rate,
                "precision_micro": precision_micro,
                "recall_micro": recall_micro,
                "f1_micro": f1_micro,
                "precision_macro": precision_macro,
                "f1_macro": f1_macro,
                "recall_macro": recall_macro,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "recognition_rate": recognition_rate,
                "coverage": coverage,
                "toughness": tree_depth,

            }

            return JsonResponse(response_data)
        except Exception as e:
            return JsonResponse({"error": str(e)})
    else:
        return JsonResponse({"error": "Only POST requests are supported."})

from django.http import JsonResponse
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the datasets
iris_data = load_iris()
breast_cancer_data = load_breast_cancer()

# Create view functions for each classifier
@csrf_exempt
def regression_classifier(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        dataset_name = data.get('dataset')
        
        if dataset_name == 'IRIS':
            data = iris_data
        elif dataset_name == 'BreastCancer':
            data = breast_cancer_data
        else:
            return JsonResponse({'error': 'Invalid dataset name'})

        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict and calculate metrics
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        response_data = {
            "mae": mae,
            "mse": mse,
            "r2": r2,
        }

        return JsonResponse(response_data)
    
@csrf_exempt
def naive_bayesian_classifier(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        dataset_name = data.get('dataset')
        
        if dataset_name == 'IRIS':
            data = iris_data
        elif dataset_name == 'BreastCancer':
            data = breast_cancer_data
        else:
            return JsonResponse({'error': 'Invalid dataset name'})

        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a Naïve Bayesian Classifier
        model = GaussianNB()
        model.fit(X_train, y_train)

        # Predict and calculate metrics
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        if len(np.unique(y_test)) == 2:
            tn, fp, fn, tp = cm.ravel()
            if tp + fn == 0:
                sensitivity = 0.0
            else:
                sensitivity = tp / (tp + fn)
            
            if tn + fp == 0:
                specificity = 0.0
            else:
                specificity = tn / (tn + fp)
        else:
            sensitivity = specificity = None

        # Predict and calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        # f1 = f1_score(y_test, y_pred, average='weighted')
        misclassification_rate = 1 - accuracy

        response_data = {
            "confusionMatrix":cm.tolist(),
            "accuracy": accuracy,
            "misclassificationRate": misclassification_rate,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "recall": recall,
        }
        
        return JsonResponse(response_data)

@csrf_exempt
def knn_classifier(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        dataset_name = data.get('dataset')
        # k_value = int(request.POST.get('k_value', 1))  # You can pass the selected k-value
        k_value = 1
        if dataset_name == 'IRIS':
            data = iris_data
        elif dataset_name == 'BreastCancer':
            data = breast_cancer_data
        else:
            return JsonResponse({'error': 'Invalid dataset name'})

        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a k-NN classifier
        model = KNeighborsClassifier(n_neighbors=k_value)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        if len(np.unique(y_test)) == 2:
            tn, fp, fn, tp = cm.ravel()
            if tp + fn == 0:
                sensitivity = 0.0
            else:
                sensitivity = tp / (tp + fn)
            
            if tn + fp == 0:
                specificity = 0.0
            else:
                specificity = tn / (tn + fp)
        else:
            sensitivity = specificity = None

        # Predict and calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        # f1 = f1_score(y_test, y_pred, average='weighted')
        misclassification_rate = 1 - accuracy

        response_data = {
            "confusionMatrix":cm.tolist(),
            "accuracy": accuracy,
            "misclassificationRate": misclassification_rate,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "recall": recall,
        }
        
        return JsonResponse(response_data)
 

# You can implement the ANN classifier similarly
from sklearn.neural_network import MLPClassifier

# @csrf_exempt
# def ann_classifier(request):
#     if request.method == 'POST':
#         data = json.loads(request.body.decode('utf-8'))
#         dataset_name = data.get('dataset')
        
#         if dataset_name == 'IRIS':
#             data = iris_data
#         elif dataset_name == 'BreastCancer':
#             data = breast_cancer_data
#         else:
#             return JsonResponse({'error': 'Invalid dataset name'})
        
#         X, y = data.data, data.target
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

#         # Initialize an ANN classifier
#         hidden_layer_sizes = (5,)  # Adjust the number of neurons in the hidden layer(s) as needed
#         model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=1000)

#         # Train the ANN and record error values
#         errors = []
#         for i in range(1, 1001):  # Train for 1000 iterations (adjust as needed)
#             model.fit(X_train, y_train)
#             errors.append(model.loss_)
        
#         # Create and save the error plot
#         plt.figure()
#         plt.plot(range(1, 1001), errors)
#         plt.xlabel('Iteration')
#         plt.ylabel('Error')
#         plt.title('Error vs. Iteration')
#         plt.savefig('error_plot.png')  # Save the plot as a file
#         plt.close()

#         return JsonResponse({'error_plot': 'error_plot.png'})

import json
import numpy as np
from django.http import JsonResponse
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

@csrf_exempt
def ann_classifier(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        dataset_name = data.get('dataset')
        
        if dataset_name == 'IRIS':
            data = iris_data
        elif dataset_name == 'BreastCancer':
            data = breast_cancer_data
        else:
            return JsonResponse({'error': 'Invalid dataset name'})

        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a three-layer ANN classifier
        model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        # Predict and calculate metrics
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        if len(np.unique(y_test)) == 2:
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
        else:
            sensitivity = specificity = None

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        response_data = {
            "confusionMatrix": cm.tolist(),
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "recall": recall,
        }
        
        return JsonResponse(response_data)
