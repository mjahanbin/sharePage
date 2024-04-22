# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 15:26:30 2024

@author: w10
"""

import dash
from dash import html, dash_table
import pandas as pd

# Create a sample DataFrame
data = {
    'ID': [1, 2, 3, 4],
    'Sales': [1234, 56789, 9101112, 131415]
}
df = pd.DataFrame(data)

# Initialize the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    dash_table.DataTable(
        id='data-table',
        columns=[
            {"name": "ID", "id": "ID"},
            {"name": "Sales", "id": "Sales", 'type': 'numeric', 
             'format': {'specifier': ',.0f'}}
        ],
        data=df.to_dict('records'),
        filter_action="native",
        sort_action="native",
        style_table={'height': '300px', 'overflowY': 'auto'}
    )
])

if __name__ == '__main__':
    app.run_server(debug=False)
