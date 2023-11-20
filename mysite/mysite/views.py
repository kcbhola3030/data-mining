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

import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn import datasets

def generate_dendrogram(dataset_name, method):
    if dataset_name == 'IRIS':
        data = datasets.load_iris().data
        title = "IRIS Dendrogram"
    elif dataset_name == 'BreastCancer':
        data = datasets.load_breast_cancer().data
        title = "Breast Cancer Dendrogram"
    else:
        return None  # Invalid dataset choice

    if method == 'agnes':
        linkage_method = 'ward'  # Use 'ward' method for AGNES
    elif method == 'diana':
        linkage_method = 'single'  # Use 'single' method for DIANA
    else:
        return None  # Invalid clustering method

    linkage_matrix = linkage(data, method=linkage_method)

    plt.figure(figsize=(10, 6))
    dendrogram(linkage_matrix)
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Distance")

    # Save the dendrogram to a file and close the plot
    img_path = os.path.join('client/src', 'dendrogram.png')
            # plt.savefig(img_path)
    plt.savefig(img_path)

    # Print the dendrogram
    plt.show()

    # Read the saved image and return it
    with open('dendrogram.png', 'rb') as image_file:
        dendrogram_image = image_file.read()

    return dendrogram_image



from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST


@csrf_exempt
@require_POST
def dendrogram_view(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        dataset_name = data.get('dataset')
        clustering_method = data.get('clustering_method')
        
        if dataset_name == 'IRIS':
            data = iris_data
        elif dataset_name == 'BreastCancer':
            data = breast_cancer_data
        else:
            return JsonResponse({'error': 'Invalid dataset name'})
        
    
        
    if dataset_name and clustering_method:
        dendrogram_image = generate_dendrogram(dataset_name, clustering_method)

        if dendrogram_image:
            response = HttpResponse(dendrogram_image, content_type='image/png')
            response['Content-Disposition'] = 'attachment; filename="dendrogram.png"'
            return response
        else:
            return JsonResponse({'error': 'Invalid dataset or clustering method'}, status=400)
    else:
        return JsonResponse({'error': 'Missing dataset_name or clustering_method in POST data'}, status=400)




# views.py in your Django app

from django.http import JsonResponse
import networkx as nx
import pandas as pd

# Function to calculate PageRank and return the top 10 pages with their ranks
def calculate_pagerank(file_path):
    G = nx.read_edgelist(file_path, create_using=nx.DiGraph(), nodetype=int)
    pagerank = nx.pagerank(G)
    top_pages = sorted(pagerank, key=pagerank.get, reverse=True)[:10]
    results = pd.DataFrame(columns=['Page', 'PageRank'])
    results['Page'] = top_pages
    results['PageRank'] = [pagerank[page] for page in top_pages]
    return results.to_dict(orient='records')

# Django view function
def get_page_rank(request):
    # Define the file path (replace this with your file path)
    file_path = '/Users/krishnacharan/Desktop/kc/acads/sem7/dm/data-mining/mysite/mysite/web-Stanford.txt'

    # Calculate PageRank and get the top 10 pages with their ranks
    top_pages_with_ranks = calculate_pagerank(file_path)

    # Return the top pages with their ranks in JSON format
    return JsonResponse({'top_pages': top_pages_with_ranks})


# views.py in your Django app

from django.http import JsonResponse
import networkx as nx
import pandas as pd

# Function to calculate HITS scores and return the top 10 authoritative and hub pages
def calculate_hits(file_path):
    G = nx.read_edgelist(file_path, create_using=nx.DiGraph(), nodetype=int)
    hits_scores = nx.hits(G)
    authority_scores = hits_scores[1]
    hub_scores = hits_scores[0]
    top_authority_pages = sorted(authority_scores, key=authority_scores.get, reverse=True)[:10]
    top_hub_pages = sorted(hub_scores, key=hub_scores.get, reverse=True)[:10]
    authority_results = pd.DataFrame(columns=['Page', 'AuthorityScore'])
    authority_results['Page'] = top_authority_pages
    authority_results['AuthorityScore'] = [authority_scores[page] for page in top_authority_pages]
    hub_results = pd.DataFrame(columns=['Page', 'HubScore'])
    hub_results['Page'] = top_hub_pages
    hub_results['HubScore'] = [hub_scores[page] for page in top_hub_pages]
    return authority_results.to_dict(orient='records'), hub_results.to_dict(orient='records')

# Django view function
def get_hits_scores(request):
    file_path = '/Users/krishnacharan/Desktop/kc/acads/sem7/dm/data-mining/mysite/mysite/web-Stanford.txt'
    authority_pages, hub_pages = calculate_hits(file_path)
    return JsonResponse({'top_authority_pages': authority_pages, 'top_hub_pages': hub_pages})

import requests
from bs4 import BeautifulSoup
from collections import deque

def dfs_crawler(seed_url, max_pages=10):
    visited = set()
    stack = [(seed_url, 0)]
    result = []

    while stack and len(visited) < max_pages:
        url, depth = stack.pop()

        if url in visited:
            continue

        visited.add(url)
        result.append(url)

        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')

            for link in soup.find_all('a', href=True):
                next_url = link['href']
                if next_url.startswith('http') and next_url not in visited:
                    stack.append((next_url, depth + 1))
        except Exception as e:
            print(f"Error: {e}")
    
    return result


def bfs_crawler(seed_url, max_pages=10):
    visited = set()
    queue = deque([(seed_url, 0)])
    result = []
    

    while queue and len(visited) < max_pages:
        url, depth = queue.popleft()
        
        if url in visited:
            continue
        
        visited.add(url)
        result.append(url)
        
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for link in soup.find_all('a', href=True):
                next_url = link['href']
                if next_url.startswith('http') and next_url not in visited:
                    queue.append((next_url, depth + 1))
        except Exception as e:
            print(f"Error: {e}")
    
    return result


from django.http import JsonResponse

@csrf_exempt
def crawl_urls(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        url = data.get('url')  # Get the URL from the form input
        # Call BFS and DFS crawlers
        bfs_result = bfs_crawler(url)
        dfs_result = dfs_crawler(url)

        # Prepare the response data
        response_data = {
            'bfs_result': bfs_result,
            'dfs_result': dfs_result
        }

        return JsonResponse(response_data)


#assignment 6
# ==============kmeans==================
from io import BytesIO  # Add this import statement

import os

def generate_kmeans(dataset_name, k_value):
    if dataset_name == 'IRIS':
        data = datasets.load_iris()
        title = "IRIS K-Means Clustering"
    elif dataset_name == 'BreastCancer':
        data = datasets.load_breast_cancer()
        title = "Breast Cancer K-Means Clustering"
    else:
        return None  # Invalid dataset choice

    if k_value:
        kmeans = KMeans(n_clusters=k_value, random_state=42)
        kmeans.fit(data.data)
        y_pred = kmeans.predict(data.data)
        plt.figure(figsize=(10, 6))
        plt.scatter(data.data[:, 0], data.data[:, 1], c=y_pred)
        plt.title(title)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")

        # Specify the directory where you want to save the image
        save_directory = 'client/src'

        # Ensure the directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Construct the full path for the saved image with k_value in the filename
        img_filename = f'kmeansOf_{dataset_name}.png'
        img_path = os.path.join(save_directory, img_filename)

        # Save the image to the specified directory
        plt.savefig(img_path)
        plt.close()

        # Read the saved image and return it
        with open(img_path, 'rb') as image_file:
            kmeans_image = image_file.read()

        return kmeans_image
    else:
        return None  # Invalid k-value


@csrf_exempt
@require_POST
def kmeans_view(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        print("Received data:", data)  # Print the received data for debugging
        dataset_name = data.get('dataset')
        k_value_str = data.get('k_value')
        
        print("dataset_name:", dataset_name)
        print("k_value_str:", k_value_str)
        if not k_value_str:
            return JsonResponse({'error': 'k_value is missing or empty'}, status=400)

        try:
            k_value = int(k_value_str)
        except ValueError:
            return JsonResponse({'error': 'Invalid k_value. Must be an integer.'}, status=400)

        if dataset_name == 'IRIS':
            data = iris_data
        elif dataset_name == 'BreastCancer':
            data = breast_cancer_data
        else:
            return JsonResponse({'error': 'Invalid dataset name'})
    
        if dataset_name and k_value:
            kmeans_image = generate_kmeans(dataset_name, k_value)

            if kmeans_image:
                response = HttpResponse(kmeans_image, content_type='image/png')
                response['Content-Disposition'] = 'attachment; filename="kmeans.png"'
                return response
            else:
                return JsonResponse({'error': 'Invalid dataset or k-value'}, status=400)
        else:
            return JsonResponse({'error': 'Missing dataset_name or k_value in POST data'}, status=400)

# ================k-medoids================
import numpy as np
import matplotlib.pyplot as plt
import os
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import json
from sklearn_extra.cluster import KMedoids
import numpy as np
import matplotlib.pyplot as plt
import os

# Sample data (Replace this with your actual dataset)
iris_data = np.random.rand(150, 4)  # Example for Iris dataset
breast_cancer_data = np.random.rand(569, 30)  # Example for Breast Cancer dataset

# Directory to save the generated images
save_directory = 'client/src'

# Ensure the directory exists
os.makedirs(save_directory, exist_ok=True)

# def generate_kmedoids(dataset_name, k_value):
#     if dataset_name == 'IRIS':
#         data = iris_data
#         title = "IRIS k-Medoids Clustering"
#     elif dataset_name == 'BreastCancer':
#         data = breast_cancer_data
#         title = "BREAST CANCER k-Medoids Clustering"
#     else:
#         return None  # Invalid dataset choice

#     if k_value:
#         # initial_medoids = np.random.choice(len(data), k_value, replace=False)
#         kmedoids_instance = KMedoids(n_clusters=k_value, random_state=0, init='k-medoids++')
#         kmedoids_instance.fit(data)
#         labels = kmedoids_instance.labels_
#         medoids = kmedoids_instance.medoid_indices_

#     # Plotting the clusters and medoids
#     plt.figure(figsize=(8, 6))
#     for cluster_index in range(3):
#         cluster_points = data[labels == cluster_index]
#         plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_index + 1}')
#         plt.scatter(data[medoids, 0], data[medoids, 1], c='red', marker='x', s=200, label='Medoids')
#         plt.title(title)
#         plt.xlabel("Feature 1")
#         plt.ylabel("Feature 2")
#         plt.legend()

#         # Construct the full path for the saved image with k_value in the filename
#         img_filename = f'kmedoidsOf_{dataset_name}.png'
#         img_path = os.path.join(save_directory, img_filename)

#         # Save the image to the specified directory
#         plt.savefig(img_path)
#         plt.close()

#         # Read the saved image and return it
#         with open(img_path, 'rb') as image_file:
#             kmedoids_image = image_file.read()

#         return kmedoids_image
#     else:
#         return None  # Invalid k-value


import numpy as np
import os
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_breast_cancer

def generate_kmedoids(dataset_name, k_value):
    if dataset_name == 'IRIS':
        data = load_iris().data[:, :2]  # Considering only the first two features for visualization purposes
        title = "IRIS k-Medoids Clustering"
    elif dataset_name == 'BreastCancer':
        data = load_breast_cancer().data[:, :2]  # Considering only the first two features for visualization purposes
        title = "BREAST CANCER k-Medoids Clustering"
    else:
        return None  # Invalid dataset choice

    if k_value:
        kmedoids_instance = KMedoids(n_clusters=k_value, random_state=0, init='k-medoids++')
        kmedoids_instance.fit(data)
        labels = kmedoids_instance.labels_
        medoids = kmedoids_instance.medoid_indices_

        # Plotting the clusters and medoids
        plt.figure(figsize=(8, 6))
        for cluster_index in range(k_value):
            cluster_points = data[labels == cluster_index]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_index + 1}')
        plt.scatter(data[medoids, 0], data[medoids, 1], c='red', marker='x', s=200, label='Medoids')
        plt.title(title)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()

        # Construct the full path for the saved image with k_value in the filename
        save_directory = 'client/src'
        img_filename = f'kmedoidsOf_{dataset_name}.png'
        img_path = os.path.join(save_directory, img_filename)

        # Save the image to the specified directory
        plt.savefig(img_path)
        plt.close()

        # Read the saved image and return it
        with open(img_path, 'rb') as image_file:
            kmedoids_image = image_file.read()

        return kmedoids_image
    else:
        return None  # Invalid k-value

@csrf_exempt
@require_POST
def kmedoids_view(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        print("Received data:", data)  # Print the received data for debugging
        dataset_name = data.get('dataset')
        k_value_str = data.get('k_value')

        print("dataset_name:", dataset_name)
        print("k_value_str:", k_value_str)
        if not k_value_str:
            return JsonResponse({'error': 'k_value is missing or empty'}, status=400)

        try:
            k_value = int(k_value_str)
        except ValueError:
            return JsonResponse({'error': 'Invalid k_value. Must be an integer.'}, status=400)

        if dataset_name == 'IRIS':
            data = iris_data
        elif dataset_name == 'BreastCancer':
            data = breast_cancer_data
        else:
            return JsonResponse({'error': 'Invalid dataset name'})

        if dataset_name and k_value:
            kmedoids_image = generate_kmedoids(dataset_name, k_value)

            if kmedoids_image:
                response = HttpResponse(kmedoids_image, content_type='image/png')
                response['Content-Disposition'] = 'attachment; filename="kmedoids.png"'
                return response
            else:
                return JsonResponse({'error': 'Invalid dataset or k-value'}, status=400)
        else:
            return JsonResponse({'error': 'Missing dataset_name or k_value in POST data'}, status=400)


# ==============
# Import necessary libraries
from django.http import JsonResponse, HttpResponse
from sklearn.cluster import Birch, DBSCAN
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import json
from io import BytesIO

# BIRCH Clustering
def generate_birch(dataset_name):
    if dataset_name == 'IRIS':
        data = load_iris()
        title = "IRIS BIRCH Clustering"
    elif dataset_name == 'BreastCancer':
        data = load_breast_cancer()
        title = "Breast Cancer BIRCH Clustering"
    else:
        return None  # Invalid dataset choice

    birch = Birch(threshold=0.5, branching_factor=50)
    birch.fit(data.data)
    labels = birch.predict(data.data)

    plt.figure(figsize=(10, 6))
    plt.scatter(data.data[:, 0], data.data[:, 1], c=labels)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    save_directory = 'client/src'
    os.makedirs(save_directory, exist_ok=True)
    img_filename = f'birchOf_{dataset_name}.png'
    img_path = os.path.join(save_directory, img_filename)

    plt.savefig(img_path)
    plt.close()

    with open(img_path, 'rb') as image_file:
        birch_image = image_file.read()

    return birch_image

# DBSCAN Clustering
def generate_dbscan(dataset_name):
    if dataset_name == 'IRIS':
        data = load_iris()
        title = "IRIS DBSCAN Clustering"
    elif dataset_name == 'BreastCancer':
        data = load_breast_cancer()
        title = "Breast Cancer DBSCAN Clustering"
    else:
        return None  # Invalid dataset choice

    # Normalize the data
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data.data)

    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(data_normalized)

    plt.figure(figsize=(10, 6))
    plt.scatter(data.data[:, 0], data.data[:, 1], c=labels)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    save_directory = 'client/src'
    os.makedirs(save_directory, exist_ok=True)
    img_filename = f'dbscanOf_{dataset_name}.png'
    img_path = os.path.join(save_directory, img_filename)

    plt.savefig(img_path)
    plt.close()

    with open(img_path, 'rb') as image_file:
        dbscan_image = image_file.read()

    return dbscan_image

# View for BIRCH clustering
@csrf_exempt
@require_POST
def birch_view(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        dataset_name = data.get('dataset')

        if dataset_name == 'IRIS':
            data = iris_data
        elif dataset_name == 'BreastCancer':
            data = breast_cancer_data
        else:
            return JsonResponse({'error': 'Invalid dataset name'})

        birch_image = generate_birch(dataset_name)

        if birch_image:
            response = HttpResponse(birch_image, content_type='image/png')
            response['Content-Disposition'] = f'attachment; filename="birchOf_{dataset_name}.png"'
            return response
        else:
            return JsonResponse({'error': 'Invalid dataset for BIRCH clustering'}, status=400)

# View for DBSCAN clustering
@csrf_exempt
@require_POST
def dbscan_view(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        dataset_name = data.get('dataset')

        if dataset_name == 'IRIS':
            data = iris_data
        elif dataset_name == 'BreastCancer':
            data = breast_cancer_data
        else:
            return JsonResponse({'error': 'Invalid dataset name'})

        dbscan_image = generate_dbscan(dataset_name)

        if dbscan_image:
            response = HttpResponse(dbscan_image, content_type='image/png')
            response['Content-Disposition'] = f'attachment; filename="dbscanOf_{dataset_name}.png"'
            return response
        else:
            return JsonResponse({'error': 'Invalid dataset for DBSCAN clustering'}, status=400)


# ========
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering, KMeans, Birch
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
from pyclustering.cluster.kmedoids import kmedoids
import matplotlib.pyplot as plt
import io
import base64
import json
import numpy as np
import pandas as pd

results_df = pd.DataFrame(columns=['Algorithm', 'ARI Score', 'Silhouette Score'])

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import load_iris, load_breast_cancer

# Directory to save the generated images
save_directory = 'client/src'
os.makedirs(save_directory, exist_ok=True)

def hierarchical_clustering(dataset_name, algorithm_name):
    if dataset_name == 'iris':
        dataset = datasets.load_iris()
        title = "IRIS "
    elif dataset_name == 'breast_cancer':
        dataset = datasets.load_breast_cancer()
        title = "Breast Cancer "
    else:
        return None  # Invalid dataset choice
    if algorithm_name == 'agnes':
        linkage_matrix = linkage(dataset.data, method='single')  # 'single' linkage method for AGNES
        title += "AGNES Hierarchical Clustering"
    elif algorithm_name == 'diana':
        linkage_matrix = linkage(dataset.data, method='complete')  # 'complete' linkage method for DIANA
        title += "DIANA Hierarchical Clustering"
    else:
        return None  # Invalid algorithm choice

    plt.figure(figsize=(10, 6))
    dendrogram(linkage_matrix)
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    
    # Save the dendrogram plot as an image
    img_filename = f'clustering_result.png'
    img_path = os.path.join(save_directory, img_filename)
    plt.savefig(img_path)
    plt.close()

    return img_path  # Return the path to the generated image


@csrf_exempt
@require_POST
def clustering_view(request):
    global results_df
    data = json.loads(request.body.decode('utf-8'))
    dataset_name = data.get('dataset')
    algorithm_name = data.get('algorithm')
    k_value = data.get('k_value')

    if dataset_name == 'iris':
        dataset = datasets.load_iris()
    elif dataset_name == 'breast_cancer':
        dataset = datasets.load_breast_cancer()
    else:
        return JsonResponse({'error': 'Invalid dataset name'})

    data_to_cluster = dataset.data
    true_labels = dataset.target

    print(dataset_name,algorithm_name)
    if algorithm_name == 'agnes' or algorithm_name == 'diana':
        img_path = hierarchical_clustering(dataset_name,algorithm_name)
        

    # Convert the plot to base64 for sending to the frontend
        image_base64 = plot_to_base64(img_path)
        return JsonResponse({'image': image_base64})

    elif algorithm_name == 'kmeans':
        # k-Means Clustering
        kmeans = KMeans(n_clusters=int(k_value), random_state=42)
        clusters = kmeans.fit_predict(data_to_cluster)

        ari_score, sil_score = evaluate_clusters(data_to_cluster, true_labels, clusters)
        results_df = update_results(results_df, algorithm_name, ari_score, sil_score)
        return JsonResponse({'ARI Score': ari_score, 'Silhouette Score': sil_score})

    elif algorithm_name == 'kmedoids':
        # k-Medoids Clustering (PAM)
        if not k_value:
            return JsonResponse({'error': 'k_value is missing or empty'}, status=400)

        try:
            k_value = int(k_value)
        except ValueError:
            return JsonResponse({'error': 'Invalid k_value. Must be an integer.'}, status=400)

        # Use pyclustering's kmedoids for K-Medoids
        kmedoids_instance = kmedoids(data_to_cluster, initial_index_medoids=np.random.randint(0, len(data_to_cluster), int(k_value)))
        clusters = kmedoids_instance.process().get_clusters()

        ari_score, sil_score = evaluate_clusters(data_to_cluster, true_labels, clusters)
        results_df = update_results(results_df, algorithm_name, ari_score, sil_score)
        return JsonResponse({'ARI Score': ari_score, 'Silhouette Score': sil_score})
    
    elif algorithm_name == 'dbscan':
        # DBSCAN Clustering
        if not k_value:
            return JsonResponse({'error': 'k_value is missing or empty'}, status=400)

        try:
            eps = float(k_value)
        except ValueError:
            return JsonResponse({'error': 'Invalid k_value. Must be a float.'}, status=400)

        dbscan = DBSCAN(eps=eps)
        clusters = dbscan.fit_predict(data_to_cluster)

        ari_score, sil_score = evaluate_clusters(data_to_cluster, true_labels, clusters)
        results_df = update_results(results_df, algorithm_name, ari_score, sil_score)
        return JsonResponse({'ARI Score': ari_score, 'Silhouette Score': sil_score})



    elif algorithm_name == 'birch':
        # BIRCH Clustering
        if not k_value:
            return JsonResponse({'error': 'k_value is missing or empty'}, status=400)

        try:
            k_value = int(k_value)
        except ValueError:
            return JsonResponse({'error': 'Invalid k_value. Must be an integer.'}, status=400)

        birch = Birch(n_clusters=k_value)
        clusters = birch.fit_predict(data_to_cluster)

        ari_score, sil_score = evaluate_clusters(data_to_cluster, true_labels, clusters)
        results_df = update_results(results_df, algorithm_name, ari_score, sil_score)
        return JsonResponse({'ARI Score': ari_score, 'Silhouette Score': sil_score})

    # Add other clustering algorithms as needed

    return JsonResponse({'error': 'Invalid algorithm name'})

def evaluate_clusters(data_to_cluster, true_labels, predicted_labels):
    unique_labels = np.unique(predicted_labels)

    # Check if there is only one label (e.g., all points assigned to the same cluster)
    if len(unique_labels) < 2:
        return 0.0, 0.0  # Return default scores for this case

    ari_score = adjusted_rand_score(true_labels, predicted_labels)
    sil_score = silhouette_score(data_to_cluster, predicted_labels)
    return ari_score, sil_score


def plot_to_base64(plot_path):
    with open(plot_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_image

def update_results(results_df, algorithm_name, ari_score, sil_score):
    new_row = pd.DataFrame({
        'Algorithm': [algorithm_name],
        'ARI Score': [ari_score],
        'Silhouette Score': [sil_score]
    })
    results_df = pd.concat([results_df, new_row], ignore_index=True)
    return results_df

#assign7
# ================ass7===============
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from ucimlrepo import fetch_ucirepo
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
import json


@csrf_exempt
def run_association_rules(request):
    if request.method == 'POST':
        try:
            print('Received POST request')
            
            # Load dataset
            dataset_id = 105  # Use the appropriate dataset ID
            dataset = fetch_ucirepo(id=dataset_id)
            X = dataset.data.features
            y = dataset.data.targets
            data = pd.concat([X, pd.DataFrame(y, columns=['Class'])], axis=1)

            # Convert categorical columns to boolean using one-hot encoding
            data = pd.get_dummies(data, drop_first=True)

            # Get parameters from the request
            data_json = json.loads(request.body.decode('utf-8'))
            support_values = data_json.get('support_values', [])
            confidence_values = data_json.get('confidence_values', [])
            
            results = []

            for support in support_values:
                for confidence in confidence_values:
                    support = float(support)
                    confidence = float(confidence)

                    # Find frequent itemsets
                    frequent_itemsets = apriori(data, min_support=support, use_colnames=True)
                    frequent_itemsets_list = frequent_itemsets['itemsets'].apply(list).tolist()
                    # print("frequent_itemsets_list:", frequent_itemsets_list)

                    # Generate association rules
                    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence)
                    rules_list = rules.to_dict(orient='records')
                    # print("rules_list:", rules.shape[0])

                    # Prepare results
                    

                    # for rule in rules_list:
                    #     lift_val = calculate_lift(data, rule)
                    #     chi_square_val = calculate_chi_square(data, rule)
                    #     all_confidence_val = calculate_all_confidence(data, rule)
                    #     max_confidence_val = calculate_max_confidence(data, rule)
                    #     kulczynski_val = calculate_kulczynski(data, rule)
                    #     cosine_val = calculate_cosine(data, rule)

                    # rule.update({
                    #     'lift': lift_val,
                    #     'chi_square': chi_square_val,
                    #     'all_confidence': all_confidence_val,
                    #     'max_confidence': max_confidence_val,
                    #     'kulczynski': kulczynski_val,
                    #     'cosine': cosine_val
                    # })
                    
                    
                    result = {
                        'support': support,
                        'confidence': confidence,
                        'frequent_itemsets': frequent_itemsets_list,
                        'total_rules': len(rules),
                        'rules': rules_list[0],
                    #     'data': {
                    #     'lift': lift_val,
                    #     'chi_square': chi_square_val,
                    #     'all_confidence': all_confidence_val,
                    #     'max_confidence': max_confidence_val,
                    #     'kulczynski': kulczynski_val,
                    #     'cosine': cosine_val
                    # }
                    }
                    results.append(result)

            if results:
                # Convert frozensets to lists before sending the response
                results_serializable = json.loads(json.dumps(results, default=list))
                return JsonResponse(results_serializable, safe=False)
            else:
                return JsonResponse({'error': 'No results found'})

        except Exception as e:
            return JsonResponse({'error': str(e)})

    return JsonResponse({'error': 'Invalid request method'})


    import numpy as np
from mlxtend.frequent_patterns import apriori


def calculate_lift(rules, data):
    # Calculate Lift measure
    # Lift = P(A and B) / (P(A) * P(B))
    lift_values = []
    for rule in rules:
        antecedent = rule[0]
        consequent = rule[1]
        support_A_B = np.sum((data.loc[:, antecedent] == 1) & (data.loc[:, consequent] == 1)) / len(data)
        support_A = np.sum(data.loc[:, antecedent] == 1) / len(data)
        support_B = np.sum(data.loc[:, consequent] == 1) / len(data)
        lift = support_A_B / (support_A * support_B)
        lift_values.append(lift)
    return lift_values

def calculate_chi_square(rules, data):
    # Calculate Chi-Square measure
    # Chi-Square = N * (ad - bc)^2 / [(a + b)(c + d)(a + c)(b + d)]
    chi_square_values = []
    for rule in rules:
        antecedent = rule[0]
        consequent = rule[1]
        a = np.sum((data.loc[:, antecedent] == 1) & (data.loc[:, consequent] == 1))
        b = np.sum((data.loc[:, antecedent] == 0) & (data.loc[:, consequent] == 1))
        c = np.sum((data.loc[:, antecedent] == 1) & (data.loc[:, consequent] == 0))
        d = np.sum((data.loc[:, antecedent] == 0) & (data.loc[:, consequent] == 0))
        chi_square = len(data) * ((a * d - b * c) ** 2) / ((a + b) * (c + d) * (a + c) * (b + d))
        chi_square_values.append(chi_square)
    return chi_square_values

def calculate_all_confidence(rules, data):
    # Calculate All Confidence measure
    # All Confidence = P(A and B) / max(P(A), P(B))
    all_confidence_values = []
    for rule in rules:
        antecedent = rule[0]
        consequent = rule[1]
        support_A_B = np.sum((data.loc[:, antecedent] == 1) & (data.loc[:, consequent] == 1)) / len(data)
        support_A = np.sum(data.loc[:, antecedent] == 1) / len(data)
        support_B = np.sum(data.loc[:, consequent] == 1) / len(data)
        all_confidence = support_A_B / max(support_A, support_B)
        all_confidence_values.append(all_confidence)
    return all_confidence_values

def calculate_max_confidence(rules, data):
    # Calculate Max Confidence measure
    # Max Confidence = P(A and B) / P(A)
    max_confidence_values = []
    for rule in rules:
        antecedent = rule[0]
        consequent = rule[1]
        support_A_B = np.sum((data.loc[:, antecedent] == 1) & (data.loc[:, consequent] == 1)) / len(data)
        support_A = np.sum(data.loc[:, antecedent] == 1) / len(data)
        max_confidence = support_A_B / support_A
        max_confidence_values.append(max_confidence)
    return max_confidence_values

def calculate_kulczynski(rules, data):
    # Calculate Kulczynski measure
    # Kulczynski = (P(A and B) / P(A)) + (P(A and B) / P(B)) / 2
    kulczynski_values = []
    for rule in rules:
        antecedent = rule[0]
        consequent = rule[1]
        support_A_B = np.sum((data.loc[:, antecedent] == 1) & (data.loc[:, consequent] == 1)) / len(data)
        support_A = np.sum(data.loc[:, antecedent] == 1) / len(data)
        support_B = np.sum(data.loc[:, consequent] == 1) / len(data)
        kulczynski = (support_A_B / support_A + support_A_B / support_B) / 2
        kulczynski_values.append(kulczynski)
    return kulczynski_values

def calculate_cosine(rules, data):
    # Calculate Cosine measure
    # Cosine = P(A and B) / sqrt(P(A) * P(B))
    cosine_values = []
    for rule in rules:
        antecedent = rule[0]
        consequent = rule[1]
        support_A_B = np.sum((data.loc[:, antecedent] == 1) & (data.loc[:, consequent] == 1)) / len(data)
        support_A = np.sum(data.loc[:, antecedent] == 1) / len(data)
        support_B = np.sum(data.loc[:, consequent] == 1) / len(data)
        cosine = support_A_B / np.sqrt(support_A * support_B)
        cosine_values.append(cosine)
    return cosine_values







