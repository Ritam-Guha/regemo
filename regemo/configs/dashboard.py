from regemo.problems.get_problem import problems
import regemo.config as config

from dash import Dash, dcc, html, Input, Output, State

import pickle

configs = {}
total_keys = []

for problem in problems:
    configs[problem] = {}

    problem_config = pickle.load(open(f"{config.BASE_PATH}/configs/problem_configs/{problem}.pickle", "rb"))
    algorithm_config = pickle.load(open(f"{config.BASE_PATH}/configs/algorithm_configs/{problem}.pickle", "rb"))
    for key in problem_config.keys():
        configs[problem][key] = problem_config[key]
    for key in algorithm_config.keys():
        configs[problem][key] = algorithm_config[key]
    total_keys = list(set(total_keys).union(set(configs[problem].keys())))


app = Dash(__name__)
app.layout = html.Div([
    dcc.Dropdown(problems, 'bnh', id='problem-dropdown'),
    dcc.Dropdown(total_keys, 'lb', id='config-dropdown'),
    html.Div(id='dd-output-container'),
])


@app.callback(
    Output('dd-output-container', 'children'),
    Input('problem-dropdown', 'value'),
    Input('config-dropdown', 'value')
)
def update_output(value_1,
                  value_2):
    return_str = ""

    if value_2 in configs[value_1].keys():
        if type(configs[value_1][value_2])==type([]):
            for i in list(configs[value_1][value_2]):
                return_str += f"{i}, "
        else:
            return_str += f"{configs[value_1][value_2]}\n"
        if type(configs[value_1][value_2]) is dict and "ref_dirs" in configs[value_1][value_2].keys():
            return_str += f"ref_dir shape: {configs[value_1][value_2]['ref_dirs'].shape}"
    else:
        return_str += "none"

    return return_str


if __name__ == '__main__':
    app.run_server(debug=True)
