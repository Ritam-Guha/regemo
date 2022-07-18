import argparse
import os
import sys
import warnings
from signal import signal, SIGINT

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly.graph_objs as go
import h5py

from innovization.legacy.graph_matrix_innovization import GraphMatrixInnovization
import innovization.legacy.graph_matrix as gm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from file_io import open_file_selection_dialog


sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
sup = str.maketrans("0123456789.-", "⁰¹²³⁴⁵⁶⁷⁸⁹⋅⁻")
sub_ij = str.maketrans("ij", "ᵢⱼ")

# Parse input arguments
parser = argparse.ArgumentParser("Dash-based visualization for innovized rules")
parser.add_argument('--result-path', type=str, help="path to the results to be visualized")
parser.add_argument("--port", type=int, default=8050, help="Port to host the visualization portal")
parser.add_argument("--special-flag", type=str, default=None, help="Any special flag to be passed to the viz portal")
args = parser.parse_args()

if args.result_path is None:
    args.result_path = open_file_selection_dialog(multi_file=False, title="Select optimization history file",
                                                  initialdir=".")

hf = h5py.File(args.result_path, 'r')

gen_arr = []
for key in hf.keys():
    gen_no = int(key[3:])
    gen_arr.append(gen_no)
gen_arr.sort()
gen_arr = np.array(gen_arr)

latest_innov_gen_key = f"gen{hf.attrs['innov_info_latest_gen']}"
if hf.attrs['current_gen'] != gen_arr[-1]:
    print("Mismatch in gen numbering")
plaw_err_max = np.array(hf[latest_innov_gen_key]['power_law_error_max'])
innov = GraphMatrixInnovization(min_ineq_rule_significance=0.8, corr_min_ineq=0.,
                                max_power_law_error_ratio=0.05,
                                corr_min_power=0.,
                                # power_law_error_max=hf.attrs['power_law_error_max'])
                                power_law_error_max=plaw_err_max)

# if 'power_law_error_max' in hf.attrs and hf.attrs['power_law_error_max'] is not None:
#     innov.power_law_error_max = hf.attrs['power_law_error_max']
#     print(innov.power_law_error_max)
# else:
#     for g in gen_arr:
#         print(f"Generation {g}")
#         x = np.array(hf[f"gen{g}"]['X'])
#         rank = np.array(hf[f"gen{g}"]['rank'])
#         x_nd = x[rank == 0, :]
#         if len(x_nd) > 1:
#             innov.update_max_power_law_error(x_nd)

xl = hf.attrs['xl']
xu = hf.attrs['xu']
ignore_vars = hf.attrs['ignore_vars']


def handler(signal_received, frame):
    # Handle any cleanup here
    print('SIGINT or CTRL-C detected. Closing HDF file. Exiting.')
    hf.close()
    sys.exit(0)


signal(SIGINT, handler)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', 'static/reset.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

config = {"toImageButtonOptions": {"width": None, "height": None, "format": 'svg'}}
config_heatmap = {"toImageButtonOptions": {"width": None, "height": None, "format": 'svg'}, 'editable': True}
html_layout = [
    html.H1(children='Optimization dashboard v0.1',
            style={'padding': '20px 20px 20px 20px', 'background-color': 'white',
                   'margin': '0px 0px 20px 0px', 'border-bottom': '1px #EBEDEF solid'}),

    html.Div([
        html.Div([
            html.Div([html.H2(children='Scatter plot', id='scatter-heading')],
                     style={'padding': '20px 0px 0px 20px', 'color': '#3C4B64'}),
            html.Div([dcc.Graph(id='objective-space-scatter',
                                hoverData={'points': [{'customdata': ''}]}, config=config)],
                     style={'padding': '20px 20px 20px 20px',  # 'background-color': 'white',
                            'margin': '0px 0px 20px 0px', 'border-bottom': '1px #EBEDEF solid',
                            'background-color': 'white'},
                     ),

            html.Div([html.H6(children='Generation', id='generation-no')], style={'padding': '0px 20px 0px 20px'}),

            html.Div([
                dcc.Slider(
                    id='cross-filter-gen-slider',
                    min=min(gen_arr),
                    max=max(gen_arr),
                    value=max(gen_arr),
                    step=None,
                    tooltip={"placement": "bottom", "always_visible": True},
                    marks={str(int(gen)): '' for gen in gen_arr}
                    # marks={str(int(gen)): str(int(gen)) for gen in gen_arr}
                )], style={'padding': '0px 20px 20px 20px'}),
        ], style={'width': '25%', 'display': 'inline-block', 'padding': '0px 20px 20px 10px',
                  'vertical-align': 'top', 'border': '1px solid #969696', 'border-radius': '5px',
                  'margin': '0px 20px 20px 20px', 'background-color': 'white'}),
        # Padding: top right bottom left
        html.Div([
            html.Div([html.H2(children='Parallel coordinate plot (PCP)', id='pcp-heading')],
                     style={'padding': '20px 20px 0px 20px', 'color': '#3C4B64', }),
            html.Div([
                html.Div([dcc.Graph(id='pcp-interactive',
                                    hoverData={'points': [{'customdata': ''}]}, config=config)],
                         style={'width': '1000%'}
                         )
            ], style={'overflow': 'scroll', 'border': '1px solid #969696',
                      'border-radius': '5px',
                      'background-color': 'white', 'padding': '0px 20px 20px 10px', 'margin': '0px 20px 20px 20px'}),
            html.Div([html.H2(children='Power law graph', id='power-law-graph-heading')],
                     style={'padding': '20px 20px 0px 20px', 'color': '#3C4B64', }),
            html.Div([
                html.Div([dcc.Graph(id='power-law-graph',
                                    hoverData={'points': [{'customdata': ''}]}, config=config)]),
                    ],
                style={'border': '1px solid #969696', 'border-radius': '5px', 'background-color': 'white'})
        ], style={'width': '30%', 'display': 'inline-block', 'float': 'center', 'padding': '20px 20px 20px 20px',
                  'vertical-align': 'top', 'background-color': 'white',
                  'border': '1px solid #969696', 'border-radius': '5px', 'margin': '0px 20px 20px 0px',
                  'overflow': 'scroll', 'height': '90%'}),
        html.Div([
            html.H4(children='Inequality rules', id='rule-list'),
            html.Div([
                html.H5(children='Min. ineq. score', id='minscore_ineq_text'),
                dcc.Input(id="minscore_ineq", type="number", placeholder="Min. ineq. score", debounce=True,
                          inputMode='numeric', value=innov.min_ineq_rule_significance),
            ], style={'display': 'inline-block'}),
            html.Div([
                html.H5(children='Min. ineq. corr.', id='mincorr_ineq_text'),
                dcc.Input(id="mincorr_ineq", type="number", placeholder="Min. ineq. corr", debounce=True,
                          inputMode='numeric', value=innov.corr_min_ineq),
            ]),
            html.Div([
                html.H5(children='Vars per rule', id='varsperrule_ineq_text'),
                dcc.Input(id="varsperrule_ineq", type="number", placeholder="Vars per rule", debounce=True,
                          inputMode='numeric', value=2),
            ]),
            html.Div([html.Button('Reset', id='ineq-reset', n_clicks=0)],
                     style={'padding': '10px 0px 10px 0px'}),

            html.Div([
                dcc.Checklist(
                    id='inequality-rule-checklist',
                    options=[
                        # {'label': 'New York City', 'value': 'NYC'},
                        # {'label': 'Montréal', 'value': 'MTL'},
                        # {'label': 'San Francisco', 'value': 'SF'}
                    ],
                    value=[]  # ['NYC', 'MTL']
                )],
                style={'height': '38%', 'overflow': 'scroll'})
        ], style={'width': '15%', 'height': '90%',
                  'display': 'inline-block', 'float': 'center', 'padding': '20px 0px 20px 20px',
                  'vertical-align': 'top', 'background-color': 'white',
                  'border': '1px solid #969696', 'border-radius': '5px',
                  'margin': '0px 20px 40px 0px',
                  # 'overflow': 'scroll'
                  }),
        html.Div([
            html.H4(children='Power laws (normalized)', id='power-law-rule-list'),
            html.Div([
                html.H5(children='Max. power law error ratio', id='maxerror_power_text'),
                dcc.Input(id="maxerror_power", type="number", placeholder="Max. power law error ratio", debounce=True,
                          inputMode='numeric', value=innov.max_power_law_error_ratio),
            ]),
            html.Div([
                html.H5(children='Min. power law score', id='maxscore_power_text'),
                dcc.Input(id="minscore_power", type="number", placeholder="Max. power law score", debounce=True,
                          inputMode='numeric', value=1),
            ]),
            html.Div([
                html.H5(children='Min. power law corr.', id='mincorr_power_text'),
                dcc.Input(id="mincorr_power", type="number", placeholder="Min. power corr", debounce=True,
                          inputMode='numeric', value=innov.corr_min_power),
            ]),
            html.Div([
                html.H5(children='Vars per rule', id='varsperrule_power_text'),
                dcc.Input(id="varsperrule_power", type="number", placeholder="Vars per rule", debounce=True,
                          inputMode='numeric', value=2),
            ]),
            html.Div([
                dcc.Checklist(
                    id='power-law-select-all',
                    options=[
                        {'label': 'Select all', 'value': 'select_all'},
                    ],
                    value=[]  # ['NYC', 'MTL']
                ),
                # html.Button('Reset', id='power-reset', n_clicks=0)
            ],
                style={'display': 'inline-block', 'padding': '10px 0px 10px 0px'}),
            html.Div([
                dcc.Checklist(
                    id='power-law-rule-checklist',
                    options=[
                        # {'label': 'New York City', 'value': 'NYC'},
                        # {'label': 'Montréal', 'value': 'MTL'},
                        # {'label': 'San Francisco', 'value': 'SF'}
                    ],
                    value=[]  # ['NYC', 'MTL']
                )
            ],
                style={'height': '30%', 'overflow': 'scroll'})
        ], style={'width': '15%', 'height': '90%',
                  'display': 'inline-block', 'float': 'center', 'padding': '20px 20px 20px 20px',
                  'vertical-align': 'top', 'background-color': 'white',
                  'border': '1px solid #969696', 'border-radius': '5px',
                  'margin': '0px 20px 20px 0px'}),
    ], style={'width': '100%', 'overflow': 'scroll', 'height': '700px'}),
    html.Div([
        html.Div([
            html.Div([dcc.Graph(id='design-fig',
                                hoverData={'points': [{'customdata': ''}]}, config=config)],
                     style={'padding': '20px 20px 20px 20px',  # 'background-color': 'white',
                            'margin': '0px 0px 20px 0px', 'border-bottom': '1px #EBEDEF solid',
                            'background-color': 'white'},
                     ),
        ], style={'width': '75%', 'display': 'inline-block', 'padding': '0px 20px 20px 10px',
                  'vertical-align': 'top', 'border': '1px solid #969696', 'border-radius': '5px',
                  'margin': '0px 20px 20px 20px', 'background-color': 'white'}),
    ], # style={'height': '2000px'}
    )
]
app.layout = html.Div(html_layout)


# @app.callback(
#     dash.dependencies.Output("power-law-rule-checklist", "value"),
#     dash.dependencies.Input("power-law-select-all", "value"),
#     dash.dependencies.State("power-law-rule-checklist", "options"),
# )
# def select_all_none(all_selected, options):
#     all_or_none = []
#     all_or_none = [option["value"] for option in options if all_selected]
#
#     return all_or_none


@app.callback(
    dash.dependencies.Output(component_id='generation-no', component_property='children'),
    [dash.dependencies.Input(component_id='cross-filter-gen-slider', component_property='value')]
)
def update_generation_no(selected_gen):
    return f'Generation {selected_gen}'


@app.callback(
    dash.dependencies.Output(component_id='inequality-rule-checklist', component_property='value'),
    dash.dependencies.Input(component_id='ineq-reset', component_property='n_clicks')
)
def reset_ineq_checklist(btn):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'ineq-reset' in changed_id:
        print("Ineq. reset button pressed")
        return []


# @app.callback(
#     dash.dependencies.Output(component_id='power-law-rule-checklist', component_property='value'),
#     dash.dependencies.Input(component_id='power-reset', component_property='n_clicks')
# )
def reset_power_checklist(btn):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'power-reset' in changed_id:
        print("Power law reset button pressed")
        return []


# @app.callback(
#     [dash.dependencies.Output(component_id='inequality-rule-checklist', component_property='options'),
#      dash.dependencies.Output(component_id='inequality-rule-checklist', component_property='value')],
#     [dash.dependencies.Input('cross-filter-gen-slider', 'value'),
#      dash.dependencies.Input("minscore_ineq", "value"),
#      dash.dependencies.Input("mincorr_ineq", "value"),
#      dash.dependencies.Input("varsperrule_ineq", "value"),
#      dash.dependencies.Input(component_id='objective-space-scatter', component_property='selectedData')]
# )
def ineq_rule_checklist(selected_gen, minscore_ineq, mincorr_ineq, vars_per_rule, selected_data):
    ctx = dash.callback_context

    if ctx.triggered:
        print("Inequality law triggers")
        # print(ctx.triggered)
        id_which_triggered = ctx.triggered[0]['prop_id'].split('.')[0]
        print(id_which_triggered)
        if id_which_triggered == 'objective-space-scatter':
            if selected_data is not None:
                print("Inequality law checklist triggered by selected data")
                # raise dash.exceptions.PreventUpdate
    # print("minscore_ineq = ", minscore_ineq)
    # print("mincorr_power = ", mincorr_ineq)

    if minscore_ineq is not None:
        innov.min_ineq_rule_significance = float(minscore_ineq)
    if mincorr_ineq is not None:
        innov.corr_min_ineq = float(mincorr_ineq)

    # print("innov.min_ineq_rule_significance = ", innov.min_ineq_rule_significance)
    # print("innov.corr_min_ineq = ", innov.corr_min_ineq)

    all_gen_val = gen_arr
    nearest_gen_value = int(all_gen_val[np.abs(all_gen_val - selected_gen).argmin()])
    gen_key = f'gen{nearest_gen_value}'
    current_gen_data = hf[gen_key]

    obj = np.array(current_gen_data['F'])
    x = np.array(current_gen_data['X'])
    rank = np.array(current_gen_data['rank'])
    f_nd = obj[rank == 0, :]
    x_nd = x[rank == 0, :]
    n_var = x_nd.shape[1]
    xl, xu = hf.attrs['xl'], hf.attrs['xu']

    solution_id = np.array([f"{nearest_gen_value}_{indx}_r{rank[indx]}" for indx in range(obj.shape[0])])
    solution_id_nd = solution_id[rank == 0]

    data_arr = []
    # print(selected_data)
    id_indx_arr = []
    if selected_data is not None:
        for data in selected_data['points']:
            # print(data)
            data_id = data['customdata']
            if data_id[-2:] != 'r0':
                continue
            id_indx = np.where(solution_id_nd == data_id)[0][0]
            data_arr.append(x_nd[id_indx, :].tolist())
            id_indx_arr.append(id_indx)

        data_arr = np.array(data_arr)
    else:
        data_arr = x_nd

    # Inequality rules
    var_groups, var_group_score, rel_type, corr = innov.learn_inequality_rules(data_arr)
    var_grp_score_significant = var_group_score[var_group_score >= innov.min_ineq_rule_significance]
    # print(var_grp_score_significant)
    var_grp_significant = var_groups[var_group_score >= innov.min_ineq_rule_significance, :]
    rel_type_significant = rel_type[var_group_score >= innov.min_ineq_rule_significance]

    # print(f"Min score = {innov.min_ineq_rule_significance}")
    # print(var_grp_score_significant)

    ineq_data = []
    for i in range(len(var_grp_significant)):
        indx_min = np.min(var_grp_significant[i, :])
        indx_max = np.max(var_grp_significant[i, :])
        if indx_min == indx_max:
            continue
        if xl[indx_min] != xl[indx_max] and xu[indx_min] != xu[indx_max]:
            continue
        if np.abs(corr[indx_max, indx_min]) < innov.corr_min_ineq:
            continue
        # print(var_grp_score_significant[i])
        istr = f"x{var_grp_significant[i, 0]} <= x{var_grp_significant[i, 1]} ".translate(sub) + \
               f"(score={np.round(var_grp_score_significant[i], decimals=2)}, " \
               f"corr={np.round(corr[var_grp_significant[i, 0], var_grp_significant[i, 1]], decimals=2)}, " \
               f"rtype={rel_type_significant[i]})"
        ineq_data.append({'label': istr, 'value': var_grp_significant[i]})

    return ineq_data, []


def normalize_x(x_nd):
    x_nd_normalized = (innov.normalize_to_range[0]
                       + ((x_nd - xl) / (xu - xl)
                          * (innov.normalize_to_range[1] - innov.normalize_to_range[0])))
    return x_nd_normalized


@app.callback(
    [dash.dependencies.Output(component_id='power-law-rule-checklist', component_property='options'),
     dash.dependencies.Output(component_id='power-law-rule-checklist', component_property='value')],
    [dash.dependencies.Input('cross-filter-gen-slider', 'value'),
     dash.dependencies.Input("maxerror_power", "value"),
     dash.dependencies.Input("minscore_power", "value"),
     dash.dependencies.Input("mincorr_power", "value"),
     dash.dependencies.Input("varsperrule_power", "value"),
     dash.dependencies.Input(component_id='objective-space-scatter', component_property='selectedData')]
)
def power_law_rule_checklist(selected_gen, maxerror_power, minscore_power, mincorr_power,
                             vars_per_rule, selected_data):
    # print("Selected data ", selected_data)
    # print("maxerror_power = ", maxerror_power)
    # print("mincorr_power = ", mincorr_power)
    ctx = dash.callback_context

    if ctx.triggered:
        print("Power law triggers")
        # print(ctx.triggered)
        id_which_triggered = ctx.triggered[0]['prop_id'].split('.')[0]
        print(id_which_triggered)
        if id_which_triggered == 'objective-space-scatter':
            if selected_data is not None:
                print("Power law checklist triggered by selected data")
                # raise dash.exceptions.PreventUpdate

    if maxerror_power is not None:
        innov.max_power_law_error_ratio = float(maxerror_power)
    if mincorr_power is not None:
        innov.corr_min_power = float(mincorr_power)

    # print("innov.max_power_law_error_ratio = ", innov.max_power_law_error_ratio)
    # print("innov.corr_min_power = ", innov.corr_min_power)

    all_gen_val = gen_arr
    nearest_gen_value = int(all_gen_val[np.abs(all_gen_val - selected_gen).argmin()])
    gen_key = f'gen{nearest_gen_value}'
    current_gen_data = hf[gen_key]

    obj = np.array(current_gen_data['F'])
    x = np.array(current_gen_data['X'])
    rank = np.array(current_gen_data['rank'])
    f_nd = obj[rank == 0, :]
    x_nd = x[rank == 0, :]
    n_var = x_nd.shape[1]

    solution_id = np.array([f"{nearest_gen_value}_{indx}_r{rank[indx]}" for indx in range(obj.shape[0])])
    solution_id_nd = solution_id[rank == 0]

    data_arr = []
    # print(selected_data)
    id_indx_arr = []
    if selected_data is not None:
        for data in selected_data['points']:
            # print(data)
            data_id = data['customdata']
            if data_id[-2:] != 'r0':
                continue
            id_indx = np.where(solution_id_nd == data_id)[0][0]
            data_arr.append(x_nd[id_indx, :].tolist())
            id_indx_arr.append(id_indx)

        data_arr = np.array(data_arr)
    else:
        data_arr = x_nd
    # print("data", data_arr)

    if args.special_flag is not None:
        n_var = x.shape[1]
        if n_var == 279 or n_var == 86:
            n_shape_var = 19
        elif n_var == 579 or n_var == 176:
            n_shape_var = 39
        elif n_var == 879 or n_var == 266:
            n_shape_var = 59
        else:
            return {'data': [], 'layout': None}
        for i in range(data_arr.shape[0]):
            if n_var == 279 or n_var == 579 or n_var == 879:
                symmetry = ()
                shape_var = data_arr[i, -n_shape_var:]
                shape_var[:n_shape_var // 2 + 1] = np.sort(shape_var[:n_shape_var // 2 + 1])
                # shape_var[n_shape_var // 2 + 1:] = np.flip(np.sort(shape_var[n_shape_var // 2 + 1:]))
                shape_var[n_shape_var//2 + 1:] = np.flip(shape_var[:n_shape_var//2 + 1] + 0.001 + np.random.random()*0.001)[1:]
                shape_var[n_shape_var // 2] = (shape_var[n_shape_var // 2] + shape_var[n_shape_var // 2 - 1]) / 2
            else:
                symmetry = ('xz', 'yz')
                shape_var = data_arr[i, -(n_shape_var // 2 + 1):]
                shape_var = np.sort(shape_var)
            data_arr[i, -n_shape_var:] = shape_var

    # TODO: Handle exception where only single data point is selected (corr is not an array)
    # Learn power laws and constant vars from selected data.
    b_arr, c_arr, power_law_error_arr, _, corr, const_var_list, mean_x\
        = innov.learn_power_laws(data_arr, normalize=True,
                                 normalize_min=xl,
                                 normalize_max=xu,
                                 ignore_vars=ignore_vars,
                                 xl=xl, xu=xu)
    mean_x_normalized = normalize_x(mean_x)

    # if innov.power_law_error_min is not None or len(innov.power_law_error_min) > 0:
    #     np.savetxt("error_min", innov.power_law_error_min, delimiter=',')
    # if innov.power_law_error_max is not None or len(innov.power_law_error_max) > 0:
    #     np.savetxt("error_max", innov.power_law_error_max, delimiter=',')

    const_var_already_printed = []
    power_law_data = []
    for i in range(n_var - 1):
        if i in ignore_vars:
            continue
        for j in range(i + 1, n_var):
            if j in ignore_vars:
                continue
            # Show rules for normal power laws.
            if i not in const_var_list and j not in const_var_list:
                power_law_error_normalized = power_law_error_arr[i, j] / innov.power_law_error_max[i, j]
                if power_law_error_normalized > innov.max_power_law_error_ratio:
                    continue
                if np.abs(corr[j, i]) < innov.corr_min_power:
                    continue
                # Show the rule pair (i, j) or (j, i) that gives |b| <= 1
                if np.abs(b_arr[i, j]) > 1:
                    indx1, indx2 = j, i
                else:
                    indx1, indx2 = i, j

                x_nd_normalized = normalize_x(data_arr)
                power_law = [indx1, indx2, b_arr[indx1, indx2], c_arr[indx1, indx2]]
                rule_compliance_id_power, non_compliance_mat, error_mat, error_ratio_mat = \
                    gm.get_power_law_compliance(x_orig=x_nd_normalized, power_law=[power_law],
                                                power_law_error_max=plaw_err_max,
                                                max_power_law_error_ratio=innov.max_power_law_error_ratio)
                rule_compliance_id = rule_compliance_id_power
                score = len(rule_compliance_id) / data_arr.shape[0]

                if score < minscore_power:
                    continue

                pstr = f"x\u0302{indx1} * x\u0302{indx2}".translate(sub) \
                       + f"{np.round(b_arr[indx1, indx2], decimals=2)}".translate(sup) \
                       + f" = {np.round(c_arr[indx1, indx2], decimals=2)}" \
                       + f" (score = {np.round(score, decimals=2)}," \
                       + " err ratio = {:.1e},".format(power_law_error_normalized) \
                       + f" err = {np.round(power_law_error_arr[indx1, indx2], decimals=2)}," \
                       + f" corr = {np.round(corr[j, i], decimals=2)}," \
                       + f" errmax = {np.round(innov.power_law_error_max[i, j], decimals=2)})"
                power_law_data.append({'label': pstr,
                                       'value': power_law})

            # Show rules of type x = constant.
            else:
                # TODO: Get a score metric for x = c type rules.
                # TODO: Convert pstr creation to a function for re-usability.
                std_x = np.std(data_arr, axis=0)
                if i in const_var_list and i not in const_var_already_printed:
                    diff_i = np.abs(data_arr[:, i] - std_x[i])
                    score_const = len(np.where(diff_i > std_x[i])[0]) / data_arr.shape[0]
                    pstr = f"x\u0302{i} = ".translate(sub) \
                           + f"{np.round(mean_x_normalized[i], decimals=2)} " \
                             f"(score = {np.round(score_const, decimals=2)})"
                    eq_law = [i, mean_x_normalized[i]]
                    if score_const >= minscore_power:
                        power_law_data.append({'label': pstr,
                                               'value': eq_law})
                        const_var_already_printed.append(i)
                if j in const_var_list and j not in const_var_already_printed:
                    diff_j = np.abs(data_arr[:, j] - std_x[j])
                    score_const = len(np.where(diff_j > std_x[j])[0]) / data_arr.shape[0]
                    pstr = f"x\u0302{j} = ".translate(sub) \
                           + f"{np.round(mean_x_normalized[j], decimals=2)} " \
                             f"(score = {np.round(score_const, decimals=2)})"
                    eq_law = [j, mean_x_normalized[j]]
                    if score_const >= minscore_power:
                        power_law_data.append({'label': pstr,
                                               'value': eq_law})
                        const_var_already_printed.append(j)

    # if selected_data is not None:
    #     raise dash.exceptions.PreventUpdate
    # print(power_law_data)
    for i in range(len(power_law_data)):
        power_law_data[i]['value'] = str(power_law_data[i]['value'])
    return power_law_data, []
    # return [{'label': "x̂₀ = 1.49 (score = 1.0)", 'value': str([0, 1.4930975618820586])}], []


@app.callback(
    dash.dependencies.Output('objective-space-scatter', 'figure'),
    [dash.dependencies.Input('cross-filter-gen-slider', 'value'),
     dash.dependencies.Input('inequality-rule-checklist', 'value'),
     dash.dependencies.Input('power-law-rule-checklist', 'value'),
     dash.dependencies.Input(component_id='objective-space-scatter', component_property='selectedData')]
)
def update_objective_space_scatter_graph(selected_gen, var_grp, power_law, selected_data):
    ctx = dash.callback_context

    layout_update = True
    if ctx.triggered:
        id_which_triggered = ctx.triggered[0]['prop_id'].split('.')[0]
        print(id_which_triggered)
        if id_which_triggered == 'objective-space-scatter':
            if selected_data is not None:
                print("Exception raised")
                raise dash.exceptions.PreventUpdate
        elif id_which_triggered == 'power-law-rule-checklist':
            print(power_law)
            print(selected_data)
            layout_update = False
            if selected_data is not None and len(power_law) == 0:
                print("Power law trigger obj graph")
                raise dash.exceptions.PreventUpdate
        elif id_which_triggered == 'inequality-rule-checklist':
            layout_update = False
            if selected_data is not None and len(var_grp) == 0:
                print("Inequality trigger obj graph")
                raise dash.exceptions.PreventUpdate
    # print(var_grp)
    all_gen_val = gen_arr
    nearest_gen_value = int(all_gen_val[np.abs(all_gen_val - selected_gen).argmin()])
    gen_key = f'gen{nearest_gen_value}'
    current_gen_data = hf[gen_key]
    obj_label = hf.attrs['obj_label']

    obj = np.array(current_gen_data['F'])
    x = np.array(current_gen_data['X'])
    rank = np.array(current_gen_data['rank'])
    f_nd = obj[rank == 0, :]
    x_nd = x[rank == 0, :]
    x_dominated = x[rank > 0, :]
    f_dominated = obj[rank > 0, :]
    n_obj = obj.shape[1]

    solution_id = np.array([f"{nearest_gen_value}_{indx}_r{rank[indx]}" for indx in range(obj.shape[0])])
    solution_id_nd = solution_id[rank == 0]

    rule_compliance_id_ineq = np.array([])
    if var_grp is not None:
        for grp in var_grp:
            id_list = np.where(x_nd[:, grp[0]] <= x_nd[:, grp[1]])[0]
            if len(rule_compliance_id_ineq) == 0:
                rule_compliance_id_ineq = id_list
            else:
                rule_compliance_id_ineq = np.intersect1d(rule_compliance_id_ineq, id_list)

    # if power_law.ndim == 1:
    #     power_law = power_law.reshape([1, -1])

    x_nd_normalized = normalize_x(x_nd)
    rule_compliance_id_power, non_compliance_mat, error_mat, error_ratio_mat \
        = gm.get_power_law_compliance(x_orig=x_nd_normalized, power_law=power_law,
                                      power_law_error_max=innov.power_law_error_max,
                                      max_power_law_error_ratio=innov.max_power_law_error_ratio)
    if len(rule_compliance_id_ineq) == 0:
        rule_compliance_id = rule_compliance_id_power
    elif len(rule_compliance_id_power) == 0:
        rule_compliance_id = rule_compliance_id_ineq
    else:
        rule_compliance_id = np.intersect1d(rule_compliance_id_ineq, rule_compliance_id_power)
    print(f"No. of pop members following selected rule(s) = {len(rule_compliance_id)}")

    f_nd_unselected = np.copy(f_nd)
    solution_id_nd_unselected = np.copy(solution_id_nd)
    if len(rule_compliance_id) > 0:
        del_row_indx = []
        f_rule = obj[rule_compliance_id, :]
        solution_id_rule = solution_id[rule_compliance_id]
        for id_indx, id in enumerate(solution_id_nd_unselected):
            if id in solution_id_rule:
                del_row_indx.append(id_indx)

        f_nd_unselected = np.delete(f_nd_unselected, del_row_indx, axis=0)
        solution_id_nd_unselected = np.delete(solution_id_nd_unselected, del_row_indx, axis=0)

    # TODO: Make 3d graph plot possible
    return_data = {'data': []}

    if n_obj == 2:
        return_data['data'] += [
                go.Scatter(
                    x=f_dominated[:, 0],
                    y=f_dominated[:, 1],
                    customdata=solution_id,
                    mode='markers',
                    name='Population',
                    marker={
                        'size': 10,
                        'opacity': 0.5,
                        'line': {'width': 0.5, 'color': 'white'}  # TODO: Why is this set to white?
                    },
                ),
                go.Scatter(
                    x=f_nd_unselected[:, 0],
                    y=f_nd_unselected[:, 1],
                    customdata=solution_id_nd_unselected,
                    mode='markers',
                    name='Non-dominated',
                    marker={
                        'size': 10,
                        'opacity': 0.8,
                        'color': 'OrangeRed',
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                ),
            ]
        # TODO: Selecting a xi = c type rule does not highlight points on the scatter plot
        if len(rule_compliance_id) > 0:
            f_rule = obj[rule_compliance_id, :]
            solution_id_rule = solution_id[rule_compliance_id]
            return_data['data'].append(
                go.Scatter(
                    x=f_rule[:, 0],
                    y=f_rule[:, 1],
                    customdata=solution_id_rule,
                    mode='markers',
                    name='Rule compliance',
                    marker={
                        'size': 10,
                        'opacity': 0.6,
                        'color': 'Green',
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                    # legendrank=2000,
                ))

    xaxis = {
        'title': obj_label[0],
        'titlefont': {'size': 18},
        'tickfont': {'size': 16},
        # 'showline': True,
        'linecolor': 'black',
        'zeroline': False,
        'mirror': True,
        'type': 'linear',
        'autorange': True,
        'automargin': True,
        # 'rangemode': 'tozero'
    }
    yaxis = {
        'title': obj_label[1],
        'titlefont': {'size': 18},
        'tickfont': {'size': 16},
        'tickprefix': "   ",
        # 'showline': True,
        'linecolor': 'black',
        'zeroline': False,
        'mirror': True,
        'type': 'linear',
        'autorange': True,
        'automargin': True,
        # 'rangemode': 'tozero'
    }
    # if layout_update:
    #     xaxis['autorange'] = True
    #     yaxis['autorange'] = True

    return_data['layout'] = go.Layout(
        xaxis=xaxis,
        yaxis=yaxis,
        margin={'l': 50, 'b': 30, 't': 10, 'r': 0},
        height=400,
        legend=dict(orientation="v",
                    x=0.6, y=0.95, xanchor='left', font={'size': 14}, bordercolor="Black", borderwidth=1),
        hovermode='closest',
    )

    return return_data


def convert_power_law_str_to_list(power_law):
    return power_law.strip('][').split(', ')


@app.callback(
    dash.dependencies.Output('power-law-graph', 'figure'),
    [dash.dependencies.Input('power-law-rule-checklist', 'value')]
)
def update_power_law_plot(power_law):
    return_data = {'data': []}
    # print(power_law)
    # ll, ul = 0, 5
    xl, xu = hf.attrs['xl'], hf.attrs['xu']
    for indx, law_str in enumerate(power_law):
        law = convert_power_law_str_to_list(law_str)
        if len(law) == 4:
            i, j, b, c = law
            i = int(i)
            j = int(j)
            b = float(b)
            c = float(c)
            print(i, j, b, c)
            # ll_i, ul_i = xl[i], xu[i]
            # ll_j, ul_j = xl[j], xu[j]
            ll_i, ul_i = innov.normalize_to_range[0], innov.normalize_to_range[1]
            ll_j, ul_j = innov.normalize_to_range[0], innov.normalize_to_range[1]
            # print(ll, ul)
            xj = np.linspace(ll_j, ul_j, int(100 * (ul_j - ll_j)))
            xi = c / (xj ** b)
            # print(xi, xj)
            xj = xj[(xi >= ll_i) & (xi <= ul_i)]
            xi = xi[(xi >= ll_i) & (xi <= ul_i)]

            # print(xi, xj)

            return_data['data'] += \
                [
                    go.Scatter(
                        x=xi,
                        y=xj,
                        mode='lines',
                        name=f"x\u0302{i} * x\u0302{j}".translate(sub) + f"{np.round(b, decimals=2)}".translate(sup) +
                             f" = {np.round(c, decimals=2)}",
                        marker={
                            'size': 10,
                            'opacity': 0.5,
                            'line': {'width': 0.5, 'color': 'white'}
                        },
                        showlegend=True
                    ),
                ]
        elif len(law) == 2:
            j, mean_xj = law
            j = int(j)
            mean_xj = float(mean_xj)
            ll_i, ul_i = innov.normalize_to_range[0], innov.normalize_to_range[1]
            xi = np.linspace(ll_i, ul_i, int(100 * (ul_i - ll_i)))
            # xj = ((innov.normalize_to_range[0]
            #       + (mean_xj - xl[j])/(xu[j] - xl[j])*(innov.normalize_to_range[1] - innov.normalize_to_range[0]))
            #       * np.ones_like(xi))
            xj = mean_xj * np.ones_like(xi)
            return_data['data'] += \
                [
                    go.Scatter(
                        x=xi,
                        y=xj,
                        mode='lines',
                        name=f"x\u0302{j}".translate(sub) + f"= {np.round(mean_xj, decimals=2)}",
                        marker={
                            'size': 10,
                            'opacity': 0.5,
                            'line': {'width': 0.5, 'color': 'white'}
                        },
                        showlegend=True
                    ),
                ]
        else:
            warnings.warn("Incorrect length of power law list.")

        wl, wu = 0.95, 1.05
        return_data['layout'] = go.Layout(
                xaxis={
                    'title': 'x\u0302i'.translate(sub_ij),
                    'titlefont': {'size': 18},
                    'tickfont': {'size': 16},
                    # 'showline': True,
                    'linecolor': 'black',
                    'zeroline': False,
                    'mirror': True,
                    'type': 'linear',
                    # 'autorange': True,
                    'automargin': True,
                    # 'rangemode': 'tozero',
                    'range': [innov.normalize_to_range[0] * wl, innov.normalize_to_range[1] * wu],
                },
                yaxis={
                    'title': 'x\u0302j'.translate(sub_ij),
                    'titlefont': {'size': 18},
                    'tickfont': {'size': 16},
                    'tickprefix': "   ",
                    # 'showline': True,
                    'linecolor': 'black',
                    'zeroline': False,
                    'mirror': True,
                    'type': 'linear',
                    # 'autorange': True,
                    'automargin': True,
                    # 'rangemode': 'tozero',
                    'range': [innov.normalize_to_range[0] * wl, innov.normalize_to_range[1] * wu],
                },
                margin={'l': 50, 'b': 30, 't': 10, 'r': 0},
                height=400,
                legend=dict(orientation="v",
                            x=0.6, y=0.95, xanchor='left', font={'size': 14}, bordercolor="Black", borderwidth=1),
                hovermode='closest',
            )

    # print(return_data)

    return return_data


@app.callback(
    dash.dependencies.Output('pcp-interactive', 'figure'),
    [dash.dependencies.Input('cross-filter-gen-slider', 'value'),
     dash.dependencies.Input(component_id='objective-space-scatter', component_property='selectedData')]
)
def update_pcp(selected_gen, selected_data):
    all_gen_val = gen_arr
    nearest_gen_value = int(all_gen_val[np.abs(all_gen_val - selected_gen).argmin()])
    gen_key = f'gen{nearest_gen_value}'
    current_gen_data = hf[gen_key]
    # obj_label = hf.attrs['obj_label']

    # final_obj = np.array(current_gen_data['F'])
    obj = np.array(current_gen_data['F'])
    constr = np.array(current_gen_data['G'])
    x = np.array(current_gen_data['X'])
    rank = np.array(current_gen_data['rank'])
    f_nd = obj[rank == 0, :]
    g_nd = constr[rank == 0, :]
    x_nd = x[rank == 0, :]
    x_dominated = x[rank > 0, :]
    f_dominated = obj[rank > 0, :]
    n_obj = obj.shape[1]
    n_constr = constr.shape[1]

    solution_id = np.array([f"{nearest_gen_value}_{indx}_r{rank[indx]}" for indx in range(obj.shape[0])])
    solution_id_nd = solution_id[rank == 0]

    data_arr = []
    data_arr_f = []
    data_arr_g = []
    # print(selected_data)
    id_indx_arr = []
    if selected_data is not None:
        for data in selected_data['points']:
            # print(data)
            data_id = data['customdata']
            if data_id[-2:] != 'r0':
                continue
            id_indx = np.where(solution_id_nd == data_id)[0][0]
            data_arr.append(x_nd[id_indx, :].tolist())
            data_arr_f.append(f_nd[id_indx, :].tolist())
            data_arr_g.append(g_nd[id_indx, :].tolist())
            id_indx_arr.append(id_indx)

        data_arr = np.array(data_arr)
        data_arr_f = np.array(data_arr_f)
        data_arr_g = np.array(data_arr_g)
    else:
        data_arr = x_nd
        data_arr_f = f_nd
        data_arr_g = g_nd
    data_list = []
    # obj_label = hf.attrs['obj_label']
    # Add objectives
    # FIXME: Plotly only shows max. 60 dimension PCP
    w = 0.05
    for i in range(n_obj):
        # w_l, w_u = 0.95, 1.05
        # if i > 0:
        #     w_l, w_u = 1.2, 0.8

        # label = obj_label[i].translate(sub)
        label = f"f{i + 1}".translate(sub)
        min_f, max_f = np.min(data_arr_f, axis=0), np.max(data_arr_f, axis=0)
        data_list.append(dict(range=[np.round(min_f[i] - w*np.abs(min_f[i]), decimals=2),
                                     np.round(max_f[i] + w*np.abs(max_f[i]), decimals=2)],
                              label=label, values=data_arr_f[:, i]),
                         )

    # Add variables
    for i in range(data_arr.shape[1]):
        data_list.append(dict(range=[xl[i], xu[i]],
                              # constraintrange=[0.6, 4.5],
                              label=f"x{i}".translate(sub), values=data_arr[:, i]
                              )
                         )

    # Add constraints
    for i in range(n_constr):
        # w_l, w_u = 0.95, 1.05
        # w_l, w_u = 1.2, 0.8
        label = f"g{i}".translate(sub)
        min_g, max_g = np.min(data_arr_g, axis=0), np.max(data_arr_g, axis=0)
        data_list.append(dict(range=[np.round(min_g[i] - w*np.abs(min_g[i]), decimals=2),
                                     np.round(max_g[i] + w*np.abs(max_g[i]), decimals=2)],
                              label=label, values=data_arr_g[:, i]
                              ),
                         )
    return_data = {'data': []}

    return_data['data'] += [
        go.Parcoords(
            line=dict(#color=iris_df['FlowerType'],
                color='red',
                colorscale='Electric',
                showscale=True,
                # cmin=-4000,
                # cmax=-100
            ),
            dimensions=list(data_list),
            tickfont=dict(size=18),
            labelfont=dict(size=24),
            rangefont=dict(size=12),
        )]

    xaxis = {
        # 'title': obj_label[0],
        # 'title': 'Mass',
        # 'titlefont': {'size': 18},
        # 'tickfont': {'size': 16},
        # 'showline': True,
        'linecolor': 'black',
        # 'zeroline': False,
        'mirror': True,
        # 'type': 'linear',
        # 'autorange': True,
        # 'automargin': True,
        # 'rangemode': 'tozero'
    }
    yaxis = {
        # 'title': obj_label[1],
        'title': 'y',
        'titlefont': {'size': 18},
        'tickfont': {'size': 16},
        'tickprefix': "   ",
        # 'showline': True,
        'linecolor': 'black',
        'zeroline': False,
        'mirror': True,
        'type': 'linear',
        'autorange': True,
        'automargin': True,
        # 'rangemode': 'tozero'
    }

    return_data['layout'] = go.Layout(
        # title=dict(text="My Title", pad=dict(b=90, l=130, r=50)),
        # xaxis=xaxis,
        # yaxis=yaxis,
        # margin={'l': 50, 'b': 30, 't': 10, 'r': 0},
        # height=400,
        # legend=dict(orientation="v",
        #             x=0.6, y=0.95, xanchor='left', font={'size': 14}, bordercolor="Black", borderwidth=1),
        # hovermode='closest',
    )

    return return_data


@app.callback(
    dash.dependencies.Output(component_id='design-fig', component_property='figure'),
    [dash.dependencies.Input(component_id='objective-space-scatter', component_property='hoverData'),
     dash.dependencies.Input('power-law-rule-checklist', 'value')]
)
def plot_design(hover_data, power_law):
    if args.special_flag is None:
        return
    solution_id = hover_data['points'][0]['customdata']
    # print("solution_id=", solution_id)
    if solution_id == "":
        return {'data': [], 'layout': None}
    print("Selected data for design plot ", solution_id)

    current_gen, pop_indx, rank = solution_id.split("_")
    current_gen_data = hf[f'gen{current_gen}']
    x = np.array(current_gen_data['X'])
    x_hover_selected = x[int(pop_indx), :]

    # KLUGE: VERY BIG KLUGE!!
    n_var = x.shape[1]
    if n_var == 279 or n_var == 86:
        n_shape_var = 19
    elif n_var == 579 or n_var == 176:
        n_shape_var = 39
    elif n_var == 879 or n_var == 266:
        n_shape_var = 59
    else:
        return {'data': [], 'layout': None}
    if n_var == 279 or n_var == 579 or n_var == 879:
        symmetry = ()
        shape_var = x_hover_selected[-n_shape_var:]
        shape_var[:n_shape_var//2 + 1] = np.sort(shape_var[:n_shape_var//2 + 1])
        # shape_var[n_shape_var//2 + 1:] = np.flip(np.sort(shape_var[n_shape_var//2 + 1:]))
        shape_var[n_shape_var//2 + 1:] = np.flip(shape_var[:n_shape_var//2 + 1] + 0.001 + np.random.random()*0.001)[1:]
        shape_var[n_shape_var // 2] = (shape_var[n_shape_var // 2] + shape_var[n_shape_var // 2 - 1]) / 2
    else:
        symmetry = ('xz', 'yz')
        shape_var = x_hover_selected[-(n_shape_var//2 + 1):]
        shape_var = np.sort(shape_var)

    from scalable_truss.truss.generate_truss import gen_truss
    from scalable_truss.truss.truss_problem_general import TrussProblemGeneral
    coordinates, connectivity, fixed_nodes, load_nodes, member_groups = gen_truss(n_shape_nodes=n_shape_var)
    coordinates = TrussProblemGeneral.set_coordinate_matrix(
        coordinates=coordinates, shape_var=shape_var,
        n_shape_var=n_shape_var, shape_var_mode='l', symmetry=symmetry
    )

    return_data = {'data': []}

    return_data['data'] += [
        go.Scatter3d(
            x=coordinates[:, 0],
            y=coordinates[:, 1],
            z=coordinates[:, 2],
            mode='markers',
            # name='Nodes',
            marker={
                'size': 2,
                'opacity': 0.5,
                'color': 'blue',
                # 'line': {'width': 0.5, 'color': 'blue'}
            },
        )
    ]

    for i, nodes in enumerate(connectivity):
        return_data['data'] += [
            go.Scatter3d(
                x=[coordinates[int(nodes[0] - 1), 0], coordinates[int(nodes[1] - 1), 0]],
                y=[coordinates[int(nodes[0] - 1), 1], coordinates[int(nodes[1] - 1), 1]],
                z=[coordinates[int(nodes[0] - 1), 2], coordinates[int(nodes[1] - 1), 2]],
                mode='lines',
                line=dict(
                    color='black',
                    width=1),
                # name='Population',
                # marker={
                #     'size': 2,
                #     'opacity': 0.5,
                #     'color': 'blue',
                #     # 'line': {'width': 0.5, 'color': 'blue'}
                # },
            )
        ]

    for indx, law_str in enumerate(power_law):
        law = convert_power_law_str_to_list(law_str)
        member_color = 'red'
        if len(law) == 4:
            i, j, b, c = law
            i = int(i)
            j = int(j)
            b = float(b)
            c = float(c)
            member_indx = [i, j]
        elif len(law) == 2:
            i, mean_i_normalized = law
            i = int(i)
            mean_i_normalized = float(mean_i_normalized)
            member_indx = [i]
            member_color = 'blue'
        else:
            member_indx = []
        for k in member_indx:
            if k >= connectivity.shape[0]:
                continue
            nodes = connectivity[k, :]
            return_data['data'] += [
                go.Scatter3d(
                    x=[coordinates[int(nodes[0] - 1), 0], coordinates[int(nodes[1] - 1), 0]],
                    y=[coordinates[int(nodes[0] - 1), 1], coordinates[int(nodes[1] - 1), 1]],
                    z=[coordinates[int(nodes[0] - 1), 2], coordinates[int(nodes[1] - 1), 2]],
                    mode='lines',
                    line=dict(
                        color=member_color,
                        width=8),
                )
            ]

    return_data['layout'] = go.Layout(
        # width=700,
        # aspectratio=dict(x=1, y=0.5, z=0.5),
        scene=go.layout.Scene(aspectratio=dict(x=3, y=1, z=1)),
        showlegend=False,
        xaxis={
            # 'title': 'x\u0302i'.translate(sub_ij),
            # 'titlefont': {'size': 18},
            # 'tickfont': {'size': 16},
            # # 'showline': True,
            # 'linecolor': 'black',
            # 'zeroline': False,
            # 'mirror': True,
            # 'type': 'linear',
            # 'autorange': True,
            'automargin': True,
            # 'rangemode': 'tozero',
            # 'automargin': True,
            'range': [0, np.max(coordinates[:, 0] + 1)],
        },
        yaxis={
            'automargin': True,
            'range': [-5, np.max(coordinates[:, 1] + 10)],
        },
        # zaxis={
        # 'automargin': True,
        #     'range': [np.min(coordinates[:, 2]) - 5, np.max(coordinates[:, 2] + 5)],
        # }
    )

    return return_data


if __name__ == '__main__':
    if args.port is not None:
        app.run_server(debug=True, port=int(args.port))
    else:
        app.run_server(debug=True)
