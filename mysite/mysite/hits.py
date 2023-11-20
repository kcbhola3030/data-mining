import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import networkx as nx

# Function to read Google web graph and calculate HITS
def calculate_hits(file_path):
    G = nx.read_edgelist(file_path, create_using=nx.DiGraph(), nodetype=int)
    hits_result = nx.hits(G)
    return hits_result

# Function to create a Dash table from the results
def create_table(data, max_rows=10):
    hub_df = pd.DataFrame(list(data[0].items()), columns=['Page', 'Hub'])
    authority_df = pd.DataFrame(list(data[1].items()), columns=['Page', 'Authority'])
    
    hub_df = hub_df.sort_values(by='Hub', ascending=False).head(max_rows)
    authority_df = authority_df.sort_values(by='Authority', ascending=False).head(max_rows)

    return html.Div(children=[
        html.H2(children='Hub Scores'),
        html.Table(
            # Header
            [html.Tr([html.Th(col) for col in hub_df.columns])] +
            # Body
            [html.Tr([html.Td(hub_df.iloc[i][col]) for col in hub_df.columns]) for i in range(min(len(hub_df), max_rows))]
        ),

        html.H2(children='Authority Scores'),
        html.Table(
            # Header
            [html.Tr([html.Th(col) for col in authority_df.columns])] +
            # Body
            [html.Tr([html.Td(authority_df.iloc[i][col]) for col in authority_df.columns]) for i in range(min(len(authority_df), max_rows))]
        )
    ])

# Define the Dash app
app = dash.Dash(__name__)

# Specify the path to the downloaded file
file_path = r'C:\Users\nisha\Desktop\New folder\web-Google.txt'

# Calculate HITS
hits_data = calculate_hits(file_path)

# Define the layout of the app
app.layout = html.Div(children=[
    html.H1(children='HITS Algorithm Dashboard'),

    # Display the adjacency matrix
    html.Div(children=[
        html.H2(children='Adjacency Matrix'),
        dcc.Markdown(children='''
            The adjacency matrix is not displayed here due to its large size.
            However, it is used internally for HITS calculations.
        ''')
    ]),

    # Display the HITS results in tables
    html.Div(children=[
        html.H2(children='HITS Results'),
        create_table(hits_data)
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True)
