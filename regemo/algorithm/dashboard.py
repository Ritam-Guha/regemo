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

app = dash.Dash(__name__)


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


@app.callback(
    Output('regular_scatter_plot', 'figure'),
    [Input('dropdown_problem', 'value')]
)
def create_reg_scatter_plot(problem_name):
    data_reg = pickle.load(
        open(f"{config.BASE_PATH}/results/{problem_name}/final_regular_population.pickle", "rb"))
    df_reg_F = pd.DataFrame(data_reg["F"], columns=[f"F{i}" for i in range(data_reg["F"].shape[1])])

    # Original Front
    # Efficient Front
    if df_reg_F.shape[1] == 2:
        fig_scatter = go.Figure(data=go.Scatter(x=df_reg_F["F0"], y=df_reg_F["F1"], mode='markers',
                                                marker=dict(color='blue', line=dict(color='white', width=1))))
        fig_scatter.update_layout(xaxis_title="$F0$", yaxis_title="$F1$")
    elif df_reg_F.shape[1] == 3:
        fig_scatter = go.Figure(data=go.Scatter3d(x=df_reg_F["F0"], y=df_reg_F["F1"], z=df_reg_F["F2"], mode='markers',
                                                  marker=dict(color='blue', line=dict(color='white', width=1))))
        fig_scatter.update_layout(scene=dict(xaxis_title='F0',
                                             yaxis_title='F1',
                                             zaxis_title='F2'))
    fig_scatter.update_layout(title="Regular Efficient Front", template="plotly_dark")

    return fig_scatter


@app.callback(
    Output('regular_pcp', 'figure'),
    [Input('dropdown_problem', 'value')]
)
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


@app.callback(
    # [dash.dependencies.Output('regular_scatter_plot', 'figure'),
    #  dash.dependencies.Output('regular_pcp', 'figure')],
    dash.dependencies.Output('output', 'value'),
    [Input('dropdown_problem', 'value'),
     dash.dependencies.Input('start-button', 'n_clicks'),
     dash.dependencies.Input('coef', 'value'),
     dash.dependencies.Input('dep_percent', 'value'),
     dash.dependencies.Input('delta', 'value'),
     dash.dependencies.Input('rand_bins', 'value'),
     dash.dependencies.Input('degree', 'value')]
)
def start_process(problem_name, n_clicks, coef, dep_percent, delta, rand_bins, degree):
    if n_clicks > 0:
        # Perform your process here 
        create_config(problem_name=problem_name,
                      non_fixed_regularity_coef_factor=coef,
                      non_fixed_dependency_percent=dep_percent,
                      delta=delta,
                      n_rand_bins=rand_bins,
                      non_fixed_regularity_degree=degree)

        reg_search_main(problem_name=problem_name)

    else:
        return None


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
                              ]),
                              html.Div(children=[
                                  dcc.Graph(id='regular_scatter_plot', style={'display': 'inline-block'}),
                                  dcc.Graph(id='regular_pcp', style={'display': 'inline-block'})
                              ])
                              # ]),
                          ])


if __name__ == '__main__':
    # convert_pdf_to_image("/home/ritz/projects/coin_lab/Research_Directions/EC/regemo/regemo/results/crashworthiness/regularity_pf_long.pdf")
    # convert_image_to_go_obj("/home/ritz/projects/coin_lab/Research_Directions/EC/regemo/regemo/results/crashworthiness/regularity_pf_long.jpg")
    main()
    app.run_server(debug=True)
