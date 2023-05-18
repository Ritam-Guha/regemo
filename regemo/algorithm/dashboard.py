import regemo.config as config
from regemo.problems.get_problem import problems
from regemo.configs.create_config import create_config
from regemo.algorithm.regularity_search import main as reg_search_main

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

import pickle
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import numpy as np
from pdf2image import convert_from_path
import cv2
import base64
import io
from flask import send_file

app = dash.Dash(__name__)


def get_initial_setting(problem_name):
    current_config = pickle.load(open(f"{config.BASE_PATH}/configs/algorithm_configs/{problem_name}.pickle", "rb"))
    return current_config


def auto_crop_box(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path)
    # Apply Canny edge detection
    edges = cv2.Canny(image, 50, 150)
    # Find contours in the edge image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find the contour with the largest area
    contour = max(contours, key=cv2.contourArea)
    # Get the bounding rectangle of the contour
    x, y, width, height = cv2.boundingRect(contour)
    # Crop the region of interest
    cropped_image = image[y:y+height, x:x+width]
    # Display the cropped image
    cv2.imwrite(image_path, cropped_image)


def convert_rules_to_image(problem_name,
                           format="png"):
    pdf_file = f"{config.BASE_PATH}/results/{problem_name}/regularity_pf_long.pdf"
    output_path = f"{config.BASE_PATH}/results/{problem_name}/regularity_pf_long"
    # Convert PDF pages to PIL images
    images = convert_from_path(pdf_file, dpi=300)

    # Save each image to the output path
    for i, image in enumerate(images):
        image_path = f'{output_path}.{format}'
        image.save(image_path, format=format)
        auto_crop_box(image_path)


def get_go_image_from_rules(problem_name):
    # Specify the path to the PNG image file
    convert_rules_to_image(problem_name)
    image_path = f"{config.BASE_PATH}/results/{problem_name}/regularity_pf_long.png"

    # Load the PNG image using PIL
    image = Image.open(image_path)
    # width, height = 1000, 800
    # image = image.resize((width, height))

    # Convert the image to a numpy array
    image_array = np.array(image)

    # Create a Plotly Express go.Figure object
    fig = go.Figure()

    # Add the image as a trace
    fig.add_trace(go.Image(z=image_array))

    # Display the figure
    return fig


@app.callback(
    Output('original_scatter_plot', 'figure'),
    [Input('dropdown_problem', 'value')]
)
def create_original_scatter_plot(problem_name):
    data_orig = pickle.load(
        open(
            f"{config.BASE_PATH}/results/hierarchical_search/{problem_name}/initial_population_{problem_name}.pickle",
            "rb"))
    df_orig_F = pd.DataFrame(data_orig["F"], columns=[f"F{i}" for i in range(data_orig["F"].shape[1])])

    # Original Front
    # Efficient Front
    if df_orig_F.shape[1] == 2:
        fig_scatter = go.Figure(data=go.Scatter(x=df_orig_F["F0"], y=df_orig_F["F1"], mode='markers',
                                                marker=dict(color='red', line=dict(color='white', width=1))))
        fig_scatter.update_layout(xaxis_title="F0", yaxis_title="F1")
    elif df_orig_F.shape[1] == 3:
        fig_scatter = go.Figure(
            data=go.Scatter3d(x=df_orig_F["F0"], y=df_orig_F["F1"], z=df_orig_F["F2"], mode='markers',
                              marker=dict(color='red', line=dict(color='white', width=1))))
        fig_scatter.update_layout(scene=dict(xaxis_title='F0',
                                             yaxis_title='F1',
                                             zaxis_title='F2'))
    fig_scatter.update_layout(title="Original Efficient Front", template="plotly_dark")
    return fig_scatter


@app.callback(
    Output('original_pcp', 'figure'),
    [Input('dropdown_problem', 'value')]
)
def create_original_pcp(problem_name):
    data_orig = pickle.load(
        open(
            f"{config.BASE_PATH}/results/hierarchical_search/{problem_name}/initial_population_{problem_name}.pickle",
            "rb"))
    df_orig_X = pd.DataFrame(data_orig["X"], columns=[f"X{i}" for i in range(data_orig["X"].shape[1])])

    # PCP
    fig_pcp = go.Figure(data=go.Parcoords(
        line=dict(color="red"),
        dimensions=list([dict(label=f'X{i}',
                              values=df_orig_X[f'X{i}']) for i in range(df_orig_X.shape[1])])))
    fig_pcp.update_layout(title="Parallel Coordinate Plot for the Pareto Front", template="plotly_dark")

    return fig_pcp


# @app.callback(
#     Output('regular_scatter_plot', 'figure'),
#     [Input('dropdown_problem', 'value')]
# )
def create_reg_scatter_plot(problem_name):
    data_reg = pickle.load(
        open(f"{config.BASE_PATH}/results/{problem_name}/final_regular_population.pickle", "rb"))
    df_reg_F = pd.DataFrame(data_reg["F"], columns=[f"F{i}" for i in range(data_reg["F"].shape[1])])

    # Original Front
    # Efficient Front
    if df_reg_F.shape[1] == 2:
        fig_scatter = go.Figure(data=go.Scatter(x=df_reg_F["F0"], y=df_reg_F["F1"], mode='markers',
                                                marker=dict(color='blue', line=dict(color='white', width=1))))
        fig_scatter.update_layout(xaxis_title="F0", yaxis_title="F1")
    elif df_reg_F.shape[1] == 3:
        fig_scatter = go.Figure(data=go.Scatter3d(x=df_reg_F["F0"], y=df_reg_F["F1"], z=df_reg_F["F2"], mode='markers',
                                                  marker=dict(color='blue', line=dict(color='white', width=1))))
        fig_scatter.update_layout(scene=dict(xaxis_title='F0',
                                             yaxis_title='F1',
                                             zaxis_title='F2'))
    fig_scatter.update_layout(title="Regular Efficient Front", template="plotly_dark")

    return fig_scatter


# @app.callback(
#     Output('regular_pcp', 'figure'),
#     [Input('dropdown_problem', 'value')]
# )
def create_reg_pcp(problem_name):
    data_reg = pickle.load(
        open(f"{config.BASE_PATH}/results/{problem_name}/final_regular_population.pickle", "rb"))
    df_reg_X = pd.DataFrame(data_reg["X"], columns=[f"X{i}" for i in range(data_reg["X"].shape[1])])

    # PCP
    fig_pcp = go.Figure(data=go.Parcoords(
        line=dict(color="blue"),
        dimensions=list([dict(label=f'X{i}',
                              values=df_reg_X[f'X{i}']) for i in range(df_reg_X.shape[1])])))
    fig_pcp.update_layout(title="Parallel Coordinate Plot for the Regular Front", template="plotly_dark")

    return fig_pcp


def create_regularity_principle(problem_name):
    # Load the JPG image file
    image_path = f"{config.BASE_PATH}/results/{problem_name}/regularity_pf_long.jpg"
    image = Image.open(image_path)
    image = image.crop()

    # Convert the image to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Get the image data as a numpy array
    image_data = np.array(image)

    # Create the go.Image trace
    image_trace = go.Image(
        z=image_data,
        hoverinfo='skip'  # Skip displaying hover information
    )

    # Create the go.Figure
    figure = go.Figure(data=image_trace)

    # Set layout properties if needed
    figure.update_layout(
        width=800,
        height=600,
        template='plotly_dark'  # Set a dark theme template
    )

    # Display the figure
    return figure


@app.callback(
    [Output('output', 'children'),
     Output('regular_scatter_plot', 'figure'),
     Output('regular_pcp', 'figure')],
     # Output('regularity_principle', 'figure')],
    [Input('dropdown_problem', 'value'),
     Input('start-button', 'n_clicks'),
     Input('coef', 'value'),
     Input('dep_percent', 'value'),
     Input('delta', 'value'),
     Input('rand_bins', 'value'),
     Input('degree', 'value')]
)
def start_process(problem_name,
                  n_clicks,
                  coef,
                  dep_percent,
                  delta,
                  rand_bins,
                  degree):
    if n_clicks > 0:
        # Perform your process here 
        create_config(problem_name=problem_name,
                      non_fixed_regularity_coef_factor=float(coef),
                      non_fixed_dependency_percent=float(dep_percent),
                      delta=float(delta),
                      n_rand_bins=int(rand_bins),
                      non_fixed_regularity_degree=int(degree))

        reg_search_main(problem_name=problem_name)
        return f"Done for parameter setting: ({coef}, {dep_percent}, {delta}, {rand_bins}, {degree})", \
               create_reg_scatter_plot(problem_name), create_reg_pcp(problem_name)

    else:
        current_config = get_initial_setting(problem_name)
        coef = current_config["non_fixed_regularity_coef_factor"]
        dep_percent = current_config["non_fixed_dependency_percent"]
        delta = current_config["delta"]
        rand_bins = current_config["n_rand_bins"]
        degree = current_config["non_fixed_regularity_degree"]
        return f"Initial parameter setting: ({coef}, {dep_percent}, {delta}, {rand_bins}, {degree})", \
               create_reg_scatter_plot(problem_name), create_reg_pcp(problem_name)


# Callback to update the download link
@app.callback(
    Output('download-link', 'href'),
    Input('download-link', 'n_clicks'),
    Input('dropdown_problem', 'value')
)
def download_image(n_clicks, problem_name):
    if n_clicks:
        # Generate the image file
        image_data = Image.open("/home/ritz/projects/coin_lab/Research_Directions/EC/regemo/mod_zdt_3d_illustration_f.png")

        # Save the image to a BytesIO object
        img_io = io.BytesIO()
        image_data.save(img_io, format='png')
        img_io.seek(0)

        # Return the BytesIO object as a downloadable file
        return send_file(img_io, mimetype='image/png')

        # If the button is not clicked, don't show the download link
    return ''


def main():
    # Create the dashboard layout
    # Apply custom CSS styles to the dropdown_problem menu
    dropdown_style = {
        'backgroundColor': 'gray',  # Background color
        'color': 'black',  # Text color
        'border': 'none',  # Border style
        'fontFamily': 'Arial',  # Font family
        'fontWeight': 'bold',  # Font weight
        'width': '1000px'  # Width of the dropdown_problem
    }
    app.layout = html.Div(style={'backgroundColor': 'black', 'color': 'white', 'height': '100vh'},
                          children=[
                              html.Header(html.H1('RegEMO Dashboard', className='header-title'),
                                          className='app-header'),
                              dcc.Dropdown(
                                  id='dropdown_problem',
                                  options=[{'label': prob, 'value': prob} for prob in problems],
                                  style=dropdown_style,
                                  value='bnh'
                              ),
                              # Add the scatter plot
                              # html.Div(children=[
                              html.Div(children=[
                                  dcc.Graph(id='original_scatter_plot', style={'display': 'inline-block'}),
                                  dcc.Graph(id='original_pcp', style={'display': 'inline-block'})
                              ]),
                              html.Header(html.H2('Regularity Corner', className='header-title'),
                                          className='app-header'),
                              html.Div(children=[
                                  dcc.Input(id="coef",
                                            type="text",
                                            placeholder="Coefficient Multiplier"),
                                  dcc.Input(id="dep_percent",
                                            type="text",
                                            placeholder="Dependency Percentage"),
                                  dcc.Input(id="delta",
                                            type="text",
                                            placeholder="Delta"),
                                  dcc.Input(id="rand_bins",
                                            type="text",
                                            placeholder="Number of random bins"),
                                  dcc.Input(id="degree",
                                            type="text",
                                            placeholder="Degree"),
                                  html.Button('Start Regularity Search', id='start-button', n_clicks=0),
                                  html.Div(id='output'),
                                  html.A(
                                      'Download Image',
                                      id='download-link',
                                      download='image.png',
                                      href='',
                                      target='_blank'
                                  )
                              ]),
                              html.Div(children=[
                                  dcc.Graph(id='regular_scatter_plot', style={'display': 'inline-block'}),
                                  dcc.Graph(id='regular_pcp', style={'display': 'inline-block'}),
                                  # dcc.Graph(id='regularity_principle', style={'display': 'inline-block'})
                              ])
                              # ]),
                          ])


if __name__ == '__main__':
    main()
    app.run_server(debug=True)