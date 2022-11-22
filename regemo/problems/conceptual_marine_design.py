import numpy as np


def evaluate(X,
             problem_args=None,
             constr=False):
    # evaluation function for conceptual marine design problem
    """
    :param X: population of solutions
    :param problem_args: the arguments needed to define the problem
    :param constr: whether constraints are needed
    :return: evaluation metrics
    """
    x_L = X[:, 0]
    x_B = X[:, 1]
    x_D = X[:, 2]
    x_T = X[:, 3]
    x_Vk = X[:, 4]
    x_CB = X[:, 5]

    displacement = 1.025 * x_L * x_B * x_T * x_CB
    V = 0.5144 * x_Vk
    g = 9.8065
    Fn = V / np.power(g * x_L, 0.5)
    a = (4977.06 * x_CB * x_CB) - (8105.61 * x_CB) + 4456.51
    b = (-10847.2 * x_CB * x_CB) + (12817.0 * x_CB) - 6960.32

    power = (np.power(displacement, 2.0 / 3.0) * np.power(x_Vk, 3.0)) / (a + (b * Fn))
    outfit_weight = 1.0 * np.power(x_L, 0.8) * np.power(x_B, 0.6) * np.power(x_D, 0.3) * np.power(x_CB, 0.1)
    steel_weight = 0.034 * np.power(x_L, 1.7) * np.power(x_B, 0.7) * np.power(x_D, 0.4) * np.power(x_CB, 0.5)
    machinery_weight = 0.17 * np.power(power, 0.9)
    light_ship_weight = steel_weight + outfit_weight + machinery_weight

    ship_cost = 1.3 * (
                (2000.0 * np.power(steel_weight, 0.85)) + (3500.0 * outfit_weight) + (2400.0 * np.power(power, 0.8)))
    capital_costs = 0.2 * ship_cost

    DWT = displacement - light_ship_weight

    running_costs = 40000.0 * np.power(DWT, 0.3)

    round_trip_miles = 5000.0
    sea_days = (round_trip_miles / 24.0) * x_Vk
    handling_rate = 8000.0

    daily_consumption = ((0.19 * power * 24.0) / 1000.0) + 0.2
    fuel_price = 100.0
    fuel_cost = 1.05 * daily_consumption * sea_days * fuel_price
    port_cost = 6.3 * np.power(DWT, 0.8)

    fuel_carried = daily_consumption * (sea_days + 5.0)
    miscellaneous_DWT = 2.0 * np.power(DWT, 0.5)

    cargo_DWT = DWT - fuel_carried - miscellaneous_DWT
    port_days = 2.0 * ((cargo_DWT / handling_rate) + 0.5)
    RTPA = 350.0 / (sea_days + port_days)

    voyage_costs = (fuel_cost + port_cost) * RTPA
    annual_costs = capital_costs + running_costs + voyage_costs
    annual_cargo = cargo_DWT * RTPA

    f_0 = annual_costs / annual_cargo
    f_1 = light_ship_weight
    # f_2 is dealt as a minimization problem
    f_2 = -annual_cargo

    # Reformulated objective functions
    g_0 = (x_L / x_B) - 6.0
    g_1 = -(x_L / x_D) + 15.0
    g_2 = -(x_L / x_T) + 19.0
    g_3 = 0.45 * np.power(DWT, 0.31) - x_T
    g_4 = 0.7 * x_D + 0.7 - x_T
    g_5 = 500000.0 - DWT
    g_6 = DWT - 3000.0
    g_7 = 0.32 - Fn

    KB = 0.53 * x_T
    BMT = ((0.085 * x_CB - 0.002) * x_B * x_B) / (x_T * x_CB)
    KG = 1.0 + 0.52 * x_D
    g_8 = (KB + BMT - KG) - (0.07 * x_B)
    G = np.column_stack((g_0, g_1, g_2, g_3, g_4, g_5, g_6, g_7, g_8))
    G = np.where(G < 0, -G, 0)

    F = np.column_stack((f_0, f_1, f_2))

    if constr:
        return F, G
    else:
        return F
