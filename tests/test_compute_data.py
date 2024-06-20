import os

import math
import numpy as np
import numpy.testing as npt
from pathlib import Path

import pytest

def test_analyse_data():
    from inflammation.compute_data import analyse_data
    path = Path.cwd() / "data"

    expected_result = [0.        , 0.22510286, 0.18157299, 0.1264423 , 0.9495481 ,
       0.27118211, 0.25104719, 0.22330897, 0.89680503, 0.21573875,
       1.24235548, 0.63042094, 1.57511696, 2.18850242, 0.3729574 ,
       0.69395538, 2.52365162, 0.3179312 , 1.22850657, 1.63149639,
       2.45861227, 1.55556052, 2.8214853 , 0.92117578, 0.76176979,
       2.18346188, 0.55368435, 1.78441632, 0.26549221, 1.43938417,
       0.78959769, 0.64913879, 1.16078544, 0.42417995, 0.36019114,
       0.80801707, 0.50323031, 0.47574665, 0.45197398, 0.22070227]

    result = analyse_data(path)

    npt.assert_array_almost_equal(result['standard deviation by day'], expected_result)
    # npt.assert_allclose(result, expected_result)

@pytest.mark.parametrize(
    "test, expected",
    [
        ([[[0, 1, 0], [0, 2, 0]]], [0, 0, 0]),
        ([[[0, 2, 0]], [[0, 1, 0]]], [0, math.sqrt(0.25), 0]),
        ([[[0, 1, 0], [0, 2, 0]], [[0, 1, 0], [0, 2, 0]]], [0, 0, 0])
    ])

def test_compute_daily_std(test, expected):
    """Test mean function works for array of zeroes and positive integers."""
    from inflammation.compute_data import compute_daily_standard_deviation
    npt.assert_array_almost_equal(compute_daily_standard_deviation(test), expected)

def test_purer_analyse_data():
    from inflammation.compute_data import load_data_from_path
    from inflammation.compute_data import compute_daily_standard_deviation
    path = Path.cwd() / "data"

    expected_result = [0.        , 0.22510286, 0.18157299, 0.1264423 , 0.9495481 ,
       0.27118211, 0.25104719, 0.22330897, 0.89680503, 0.21573875,
       1.24235548, 0.63042094, 1.57511696, 2.18850242, 0.3729574 ,
       0.69395538, 2.52365162, 0.3179312 , 1.22850657, 1.63149639,
       2.45861227, 1.55556052, 2.8214853 , 0.92117578, 0.76176979,
       2.18346188, 0.55368435, 1.78441632, 0.26549221, 1.43938417,
       0.78959769, 0.64913879, 1.16078544, 0.42417995, 0.36019114,
       0.80801707, 0.50323031, 0.47574665, 0.45197398, 0.22070227]
    
    data = load_data_from_path(path)
    result = compute_daily_standard_deviation(data)

    npt.assert_array_almost_equal(result, expected_result)
    # npt.assert_allclose(result, expected_result)