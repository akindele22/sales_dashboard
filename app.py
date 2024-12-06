from flask import Flask
from dash import Dash, dcc, html, Input, Output
import pandas as pd
import plotly.graph_objs as go
import joblib
import numpy as np
from datetime import timedelta

# Load data and model
df = pd.read_csv('m_train.csv')  # Replace with your CSV file path
df['date'] = pd.to_datetime(df['date'])

df1 = pd.read_csv('final_data11.csv')  # Replace with your CSV file path
df1['date'] = pd.to_datetime(df1['date'])

# Load the pre-trained model
model = joblib.load('model.pkl')  # Load your pre-trained model (ensure it's a regression model)

# Initialize Flask and Dash
server = Flask(__name__)
app = Dash(__name__, server=server, url_base_pathname='/dashboard/')
app.title = "Modern Sales Dashboard"

# Dashboard Layout
app.layout = html.Div(
    style={'backgroundColor': '#1e2132', 'padding': '20px', 'fontFamily': 'Arial, sans-serif'},
    children=[
        html.H1("Sales Analytics Dashboard", style={
            'text-align': 'center', 
            'padding': '10px', 
            'color': 'white',
            'font-family': 'Roboto, Arial, sans-serif'
        }),
        
        # Filters Section
        html.Div([
            html.Div([
                html.Label("Store Number", style={'color': 'white', 'font-family': 'Open Sans, Arial, sans-serif'}),
                dcc.Dropdown(
                    id="store_filter",
                    options=[{"label": f"Store {s}", "value": s} for s in df1['store_nbr'].unique()],
                    placeholder="Select Store",
                    style={'width': '100%', 'font-family': 'Open Sans'}
                ),
            ], style={'width': '30%', 'padding': '10px'}),
            html.Div([
                html.Label("Store Type", style={'color': 'white', 'font-family': 'Open Sans, Arial, sans-serif'}),
                dcc.Dropdown(
                    id="type_filter",
                    options=[{"label": t, "value": t} for t in df1['type'].unique()],
                    placeholder="Select Store Type",
                    style={'width': '100%', 'font-family': 'Open Sans'}
                ),
            ], style={'width': '30%', 'padding': '10px'}),
            html.Div([
                html.Label("Locale Name", style={'color': 'white', 'font-family': 'Open Sans, Arial, sans-serif'}),
                dcc.Dropdown(
                    id="locale_filter",
                    options=[{"label": l, "value": l} for l in df1['locale_name'].unique()],
                    placeholder="Select Locale",
                    style={'width': '100%', 'font-family': 'Open Sans'}
                ),
            ], style={'width': '30%', 'padding': '10px'}),
        ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-bottom': '20px'}),
        
        # KPIs Section
        html.Div([
            html.Div(id="total_revenue", style={
                'backgroundColor': '#2d3038', 'color': '#00cc96', 'padding': '15px', 
                'borderRadius': '10px', 'text-align': 'center', 'width': '22%', 'font-size': '1.2em'
            }),
            html.Div(id="total_transactions", style={
                'backgroundColor': '#2d3038', 'color': '#00cc96', 'padding': '15px', 
                'borderRadius': '10px', 'text-align': 'center', 'width': '22%', 'font-size': '1.2em'
            }),
            html.Div(id="avg_discounted_sales", style={
                'backgroundColor': '#2d3038', 'color': '#00cc96', 'padding': '15px', 
                'borderRadius': '10px', 'text-align': 'center', 'width': '22%', 'font-size': '1.2em'
            }),
            html.Div(id="avg_oil_price", style={
                'backgroundColor': '#2d3038', 'color': '#00cc96', 'padding': '15px', 
                'borderRadius': '10px', 'text-align': 'center', 'width': '22%', 'font-size': '1.2em'
            }),
        ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-bottom': '20px'}),
        
        # Graphs Section
        html.Div([
            html.Div([dcc.Graph(id="sales_trend")], style={'width': '48%'}),
            html.Div([dcc.Graph(id="product_analysis")], style={'width': '48%'}),
        ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-bottom': '20px'}),
        
        html.Div([
            html.Div([dcc.Graph(id="sales_by_type")], style={'width': '48%'}),
            html.Div([dcc.Graph(id="promotion_analysis")], style={'width': '48%'}),
        ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-bottom': '20px'}),
        

        
        # Forecasting Section
        html.Div([
            html.H3("Forecasted Sales Analysis", style={
                'text-align': 'center', 
                'margin-top': '30px', 
                'color': 'white', 
                'font-family': 'Roboto'
            }),
            dcc.Slider(
                id="forecast_period",
                min=7, max=365, step=7,
                value=30,
                marks={i: f"{i} days" for i in range(7, 366, 30)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            dcc.Graph(id="forecast_sales_graph"),
        ], style={'margin-top': '20px'}),
    ]
)

# Callback for dynamic updates
@app.callback(
    [
        Output("total_revenue", "children"),
        Output("total_transactions", "children"),
        Output("avg_discounted_sales", "children"),
        Output("avg_oil_price", "children"),
        Output("sales_trend", "figure"),
        Output("product_analysis", "figure"),
        Output("sales_by_type", "figure"),
        Output("promotion_analysis", "figure"),
        Output("forecast_sales_graph", "figure"),
    ],
    [
        Input("store_filter", "value"),
        Input("type_filter", "value"),
        Input("locale_filter", "value"),
        Input("forecast_period", "value"),
    ]
)
def update_dashboard(store, store_type, locale, forecast_period):
    # Filter data
    filtered_df = df1.copy()
    if store:
        filtered_df = filtered_df[filtered_df['store_nbr'] == store]
    if store_type:
        filtered_df = filtered_df[filtered_df['type'] == store_type]
    if locale:
        filtered_df = filtered_df[filtered_df['locale_name'] == locale]

    # KPIs
    total_revenue = f"Total Revenue: ${filtered_df['sales'].sum():,.2f}"
    total_transactions = f"Total Transactions: {filtered_df['transactions'].sum():,.0f}"
    avg_discounted_sales = f"Avg Discounted Sales: ${filtered_df[filtered_df['onpromotion'] > 0]['sales'].mean():,.2f}"
    avg_oil_price = f"Avg Oil Price: ${filtered_df['dcoilwtico'].mean():,.2f}"

    # Sales Trend (Bar Chart)
    sales_trend_fig = go.Figure()
    sales_trend_fig.add_trace(go.Bar(x=filtered_df['date'], y=filtered_df['sales'], name='Sales'))
    sales_trend_fig.update_layout(
        title="Sales Trend", 
        xaxis_title="Date", 
        yaxis_title="Sales",
        plot_bgcolor="#1e2132",  # Background color of the chart
        paper_bgcolor="#1e2132", # Background color of the chart's paper
        font=dict(color='white') # Font color for the axis titles and ticks
    )

    # Promotion Sales by Family (Bar Chart)
    promotion_sales_by_family_fig = go.Figure()

    # Group data by 'family' and 'onpromotion', summing sales
    promotion_sales_by_family = filtered_df.groupby(['family', 'onpromotion'])['sales'].sum().reset_index()

    # Pivot the table to create a matrix of 'family' vs. 'onpromotion'
    promotion_sales_by_family_pivot = promotion_sales_by_family.pivot(
        index='family',
        columns='onpromotion',
        values='sales'
    ).fillna(0)  # Fill NaN values with 0

    # Plot the bar chart for Promotion Sales by Family
    promotion_sales_by_family_fig = go.Figure()

    # Add bars for each promotion type (0 = not on promotion, >0 = on promotion)
    for promotion_value in promotion_sales_by_family_pivot.columns:
        promotion_sales_by_family_fig.add_trace(go.Bar(
            x=promotion_sales_by_family_pivot.index,
            y=promotion_sales_by_family_pivot[promotion_value],
            name=f'Promotion {promotion_value}'
        ))

    # Customize layout
    promotion_sales_by_family_fig.update_layout(
        title="Promotion Sales by Family",
        xaxis_title="Family",
        yaxis_title="Sales",
        plot_bgcolor="#1e2132", 
        paper_bgcolor="#1e2132", 
        font=dict(color='white')
    )

    # Forecasting
    # Get the last known data point
    last_date = filtered_df['date'].max()
    last_sales = filtered_df[filtered_df['date'] == last_date]['sales'].values[0]

    # Generate future sales predictions using the pre-trained model
    forecast_dates = pd.date_range(last_date + timedelta(days=1), periods=forecast_period).strftime('%Y-%m-%d').tolist()
    forecast_values = []

    for i in range(forecast_period):
        # Use previous sales and other features for the prediction
        forecast_input = np.array([last_sales] + [0] * (len(filtered_df.columns) - 1)).reshape(1, -1)  # Adjust based on features
        forecast_value = model.predict(forecast_input)[0]
        forecast_values.append(max(forecast_value, 0))  # Ensure no negative sales
        last_sales = forecast_values[-1]  # Update last sales with the forecasted value for the next iteration

    # Create the forecast plot
    forecast_sales_fig = go.Figure()
    forecast_sales_fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_values, mode='lines', name='Forecasted Sales'))
    forecast_sales_fig.update_layout(
        title="Sales Forecast", 
        xaxis_title="Date", 
        yaxis_title="Forecasted Sales",
        plot_bgcolor="#1e2132",
        paper_bgcolor="#1e2132",
        font=dict(color='white')
    )

    return (
        total_revenue,
        total_transactions,
        avg_discounted_sales,
        avg_oil_price,
        sales_trend_fig,
        promotion_sales_by_family_fig,
        promotion_sales_by_family_fig,
        promotion_sales_by_family_fig,
        forecast_sales_fig
    )

if __name__ == "__main__":
    app.run_server(debug=True, port= 8700)
