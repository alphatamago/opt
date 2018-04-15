""" Some examples of using Twiddle and with Scipy Optimize module to minimize
some functions.
"""

import logging

import numpy as np
from scipy.optimize import minimize

from twiddle import twiddle

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    tol = 0.001

    for run_func, start_params, expected_params in [
            (lambda x: x[0]**2, [-300], [0]),
            (lambda x: x[0]**2 + x[1]**2 + x[2]**2, [-300, 500, -3500],
             [0, 0, 0])
    ]:
        logging.info("start_params: %s", start_params)
        
        initial_value = run_func(start_params)

        # Optimize using Scipy Optimize
        logging.info("Optimizing with Nelder-Mead from scipy.optimize.")
        result = minimize(fun=run_func,
                          x0=start_params,
                          method='Nelder-Mead')
        logging.info("OptimizeResult: %s", result)
        
        for i in range(len(result.x)):
            assert abs(expected_params[i] - result.x[i]) < tol

        final_value = run_func(result.x)
            
        logging.info("initial_value=%s final_value=%s",
                     initial_value, final_value)
        logging.info("num iterations scipy.optimize: %s", result.nit)

        # Optimize the same using Twiddle
        logging.info("Optimizing with Twiddler.")
        final_params, best_val, num_iter = twiddle(fun=run_func,
                                                   x0=start_params,
                                                   tol=tol)
        for i in range(len(final_params)):
            assert abs(expected_params[i] - final_params[i]) < tol
        
        logging.info("Final parameters: %s", final_params)

        # Double-checking, we should get best_val
        final_value = run_func(final_params)
        assert abs(final_value - best_val) < tol

        logging.info("initial_value=%s final_value=%s", initial_value,
                     final_value)
        logging.info("num iterations twiddle: %s", num_iter)
