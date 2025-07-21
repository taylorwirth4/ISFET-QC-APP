import dash
from dash import dcc, html, Input, Output, State, ctx
import pandas as pd
import plotly.express as px
import io
import base64

# Use Bootstrap styling
external_stylesheets = ["https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
app.title = "Modern Dash App"

app.layout = html.Div([
    html.Div([
        html.H1("📊 Multi-Tab Dash App", className="text-center my-4"),
    ], className="container"),

    dcc.Tabs(id="tabs", value='tab-upload', children=[
        dcc.Tab(label='Upload & Plot', value='tab-upload', className="custom-tab", selected_className="custom-tab-selected"),
        dcc.Tab(label='About This App', value='tab-info', className="custom-tab", selected_className="custom-tab-selected"),
    ], className="mb-3"),

    html.Div(id='tabs-content', className="container"),

    dcc.Store(id='stored-data')
])

# Tab content logic
@app.callback(Output('tabs-content', 'children'), Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-upload':
        return html.Div([
            html.H4("Step 1: Upload CSV", className="mb-3"),
            dcc.Upload(
                id='upload-data',
                children=html.Button('Upload CSV File', className="btn btn-primary"),
                accept='.csv',
                multiple=False
            ),
            html.Div(id='upload-status', className="mt-2"),
            html.Hr(),
            html.H4("Step 2: View Plot", className="mb-3"),
            dcc.Graph(id='data-plot'),
        ])

    elif tab == 'tab-info':
        return html.Div([
            html.H3("About This App", className="mb-3"),
            html.P("""
                This is a simple multi-tab Dash app built with Python, Plotly, and Dash.
                The first tab lets you upload a CSV file and plots 'DTUTC' vs. 'VINT' if those columns are present.
                The second tab (this one) is a placeholder to demonstrate how you can easily add more pages!
            """, className="lead"),
            html.P("You can add tables, graphs, maps, or anything else to future tabs.")
        ])

    return html.Div("Unknown tab")

# File upload and data parsing
@app.callback(
    Output('upload-status', 'children'),
    Output('stored-data', 'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def parse_upload(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        if 'DTUTC' not in df.columns or 'VINT' not in df.columns:
            return html.Div(f"❌ Uploaded file '{filename}' missing required columns.", className="text-danger"), None
        return html.Div(f"✅ Uploaded '{filename}' with {len(df)} rows.", className="text-success"), df.to_json(date_format='iso', orient='split')
    except Exception as e:
        return html.Div(f"❌ Error processing file: {str(e)}", className="text-danger"), None

# Plot figure from DataFrame
@app.callback(
    Output('data-plot', 'figure'),
    Input('stored-data', 'data'),
    prevent_initial_call=True
)
def update_plot(data_json):
    if not data_json:
        return dash.no_update
    df = pd.read_json(data_json, orient='split')
    fig = px.line(df, x='DTUTC', y='VINT', title="DTUTC vs VINT")
    fig.update_layout(margin=dict(l=40, r=20, t=40, b=40))
    return fig

if __name__ == '__main__':
    app.run(debug=True)
