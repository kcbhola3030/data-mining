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
#             elif method == 'gain_ratio':
#                 clf = DecisionTreeClassifier(criterion='entropy')
#             elif method == 'gini':
#                 clf = DecisionTreeClassifier(criterion='gini')
#             else:
#                 return JsonResponse({"error": "Invalid method specified."})

#             clf.fit(X_train, y_train)
            
#             # Make predictions on the testing dataset
#             predictions = clf.predict(X_test)
#             confusion = confusion_matrix(y_test, predictions)

#             # Calculate evaluation metrics
#             accuracy = accuracy_score(y_test, predictions)
#             precision = precision_score(y_test, predictions, average='weighted')  # Specify 'weighted' for multi-class
#             recall = recall_score(y_test, predictions, average='weighted')  # Specify 'weighted' for multi-class
#             f1 = f1_score(y_test, predictions, average='weighted')  # Specify 'weighted' for multi-class


#             # accuracy = accuracy_score(y_test, predictions)
#             misclassification_rate = 1 - accuracy
#             sensitivity = recall_score(y_test, predictions)
#             specificity = (confusion[0, 0]) / (confusion[0, 0] + confusion[0, 1])
#             # precision = precision_score(y_test, predictions)
#             # recall = sensitivity


#             # Return the classification results as JSON response
#             response_data = {
#                 "accuracy": accuracy,
#                 "precision": precision,
#                 "recall": recall,
#                 "f1_score": f1,
#                 "misclassification_rate": misclassification_rate,
#                 "specificity":specificity,
#                 "sensitivity":sensitivity
#             }

#             return JsonResponse(response_data)
#         except Exception as e:
#             return JsonResponse({"error": str(e)})
#     else:
#         return JsonResponse({"error": "Only POST requests are supported."})


# classifier/views.py

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

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

#             # Determine the positive label dynamically from the dataset
#             unique_labels = y.unique()
#             if len(unique_labels) != 2:
#                 return JsonResponse({"error": "Dataset must have exactly two unique labels for binary classification."})
            
#             positive_label = unique_labels[1]  # Assume the second unique label is the positive class

#             # Split the data into training and testing sets
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#             # Train a decision tree classifier using the specified method
#             if method == 'info_gain':
#                 clf = DecisionTreeClassifier(criterion='entropy')
#             elif method == 'gain_ratio':
#                 clf = DecisionTreeClassifier(criterion='entropy')
#             elif method == 'gini':
#                 clf = DecisionTreeClassifier(criterion='gini')
#             else:
#                 return JsonResponse({"error": "Invalid method specified."})

#             clf.fit(X_train, y_train)
            
#             # Make predictions on the testing dataset
#             predictions = clf.predict(X_test)

#             # Calculate the confusion matrix
#             confusion = confusion_matrix(y_test, predictions)

#             # Calculate evaluation metrics
#             accuracy = accuracy_score(y_test, predictions)
#             misclassification_rate = 1 - accuracy
#             sensitivity = recall_score(y_test, predictions, pos_label=positive_label)
#             specificity = recall_score(y_test, predictions, pos_label=unique_labels[0])
#             precision = precision_score(y_test, predictions, pos_label=positive_label)
#             recall = sensitivity

#             # Return all the metrics as JSON response
#             response_data = {
#                 "accuracy": accuracy,
#                 "misclassification_rate": misclassification_rate,
#                 "sensitivity": sensitivity,
#                 "specificity": specificity,
#                 "precision": precision,
#                 "recall": recall,
#             }
#             return JsonResponse(response_data)
#         except Exception as e:
#             return JsonResponse({"error": str(e)})
#     else:
#         return JsonResponse({"error": "Only POST requests are supported."})


# classifier/views.py

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

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
            
            # Make predictions on the testing dataset
            predictions = clf.predict(X_test)

            # Calculate the confusion matrix
            confusion = confusion_matrix(y_test, predictions)

            # Calculate evaluation metrics for multi-class classification
            accuracy = accuracy_score(y_test, predictions)
            misclassification_rate = 1 - accuracy
            precision_micro = precision_score(y_test, predictions, average='micro')
            recall_micro = recall_score(y_test, predictions, average='micro')
            f1_micro = f1_score(y_test, predictions, average='micro')

            precision_macro = precision_score(y_test, predictions, average='macro')
            recall_macro = recall_score(y_test, predictions, average='macro')
            f1_macro = f1_score(y_test, predictions, average='macro')

            # Return all the metrics as JSON response
            response_data = {
                "accuracy": accuracy,
                "misclassification_rate": misclassification_rate,
                "precision_micro": precision_micro,
                "recall_micro": recall_micro,
                "f1_micro": f1_micro,
                "precision_macro": precision_macro,
                "recall_macro": recall_macro,
                "f1_macro": f1_macro,
            }
            return JsonResponse(response_data)
        except Exception as e:
            return JsonResponse({"error": str(e)})
    else:
        return JsonResponse({"error": "Only POST requests are supported."})
