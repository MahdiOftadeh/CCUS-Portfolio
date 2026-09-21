from src.kinetics.ldf import LinearDrivingForce

def test_ldf_kinetics():
    k_ldf_params = {'CO2': 0.05, 'N2': 0.20}
    ldf_model = LinearDrivingForce(k_ldf=k_ldf_params)

    q_eq = {'CO2': 3.5, 'N2': 0.5}
    q_act = {'CO2': 1.0, 'N2': 0.5}

    rates = ldf_model.calculate_rate(q_eq, q_act)

    assert abs(rates['CO2'] - 0.125) < 1e-6
    assert abs(rates['N2'] - 0.0) < 1e-6
    print('? All Kinetics tests passed successfully!')

if __name__ == '__main__':
    test_ldf_kinetics()
