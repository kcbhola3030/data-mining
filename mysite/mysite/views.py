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


def chi_square_analysis(attribute1_data, attribute2_data):
    # Perform Chi-Square Test
    contingency_table = np.histogram2d(attribute1_data, attribute2_data, bins=2)[0]
    chi2, p, _, _ = chi2_contingency(contingency_table)

    # Determine correlation conclusion based on p-value
    is_correlated = p < 0.05

    response_data = {
        'chi2': chi2,
        'is_correlated': is_correlated
    }

    return JsonResponse(response_data)

@csrf_exempt  # Exempt CSRF protection for demonstration purposes, consider adding proper protection
def chi_square_analysis2(request):
    if request.method == 'POST':
        data = request.POST  # Assuming you're sending data in the POST request

        attribute1_data = [value for value in data.getlist('attribute1_data[]')]
        attribute2_data = [int(value) for value in data.getlist('attribute2_data[]')]  # Convert to integers

        # Create a mapping from unique categorical values to numerical identifiers
        unique_values = set(attribute1_data + attribute2_data)
        value_to_id = {value: id for id, value in enumerate(unique_values)}

        # Map categorical values to numerical identifiers
        attribute1_data = [value_to_id[value] for value in attribute1_data]
        attribute2_data = [value_to_id[value] for value in attribute2_data]

        # Create a contingency table
        contingency_table = np.histogram2d(attribute1_data, attribute2_data, bins=len(unique_values))[0]

        # Calculate the chi-square statistic, p-value, and expected frequencies
        chi2, p, dof, expected = chi2_contingency(contingency_table)

        response_data = {
            'chi2': chi2,
            'p_value': p,
            'degrees_of_freedom': dof,
            'expected_frequencies': expected.tolist()
        }

        return JsonResponse(response_data)
    else:
        return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)
