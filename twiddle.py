""" Twiddle algorithm for paramter optimization.
As described in Sebastian Thurn's Artificial Intelligence for Robotics online
course from Udacity: https://classroom.udacity.com/courses/cs373 in lecture 5
("PID Control").
Short video with the algorithm description:
https://www.youtube.com/watch?v=2uQ2BSzDvXs
"""

import logging

import numpy as np

def twiddle(fun, x0, start_dparams=None, tol=0.001,
            dp_percent_delta=0.1, min_value=-1e+32):
    """
    Find params that minimize fun.
    x0: array, initial choice of parameters, can be all zeros or random.
    start_dparams: array, initial choice of the step sizes for each parameter
    tol: float, tolerance which dictates to stop the algorithm when the sum of all step sizes is below this value
    dp_percent_delta: float in (0, 1), the multiplier that dictates how step sizes change up/down
    """

    if start_dparams is None:
        dparams = [1.0] * len(x0)
    else:
        dparams = start_dparams[:]
        for dp in dparams:
            assert dp > 0

    assert tol > 0
    assert sum([abs(dp) for dp in dparams]) >= tol
    assert dp_percent_delta > 0 and dp_percent_delta < 1

    params = x0[:]
    x0 = None
    best_so_far = fun(params)
    logging.debug("current best: %s", best_so_far)
    num_iter = 0
    while sum([abs(dp) for dp in dparams]) >= tol:
        num_iter += 1
        if best_so_far <=  min_value:
            logging.warn("Hard stop: best_so_far exceeded max_val: %s",
                         best_so_far)
            break
        logging.debug("......")
        logging.debug("p: %s dp: %s best_so_far: %s",
                     params, dparams, best_so_far)
        for i in range(len(params)):
            logging.debug("Trying to update param %s p=%s dp=%s",
                         i, params[i], dparams[i])

            # Saving the current value of the i-th param since we may need to
            # restore it.
            original_param = params[i]

            # Try to move forward.
            params[i] = original_param + dparams[i]
            value = fun(params)
            logging.debug("If moving param %s fowrard to %s we get value=%s",
                         i, params[i], value)
            if value < best_so_far:
                best_so_far = value
                params = params[:]
                # Increase step size
                dparams[i] *= (1.0 + dp_percent_delta)
                logging.debug("Moved fowrard, improved value to %s, increased "
                             " step size for param %s to %s",
                             value, i, params[i])
            else:
                # Try to move in the opposite direction.
                params[i] = original_param - dparams[i]
                value = fun(params)
                logging.debug("If moving param %s backward to %s we get value=%s",
                             i, params[i], value)
                if value < best_so_far:
                    best_so_far = value
                    params = params[:]
                    dparams[i] *= (1.0 + dp_percent_delta)
                    logging.debug("Improved value to: %s increased step size for "
                                 "param %s to %s",
                                 best_so_far, i, dparams[i])
                else:
                    # We tried to move in both directions and failed
                    # Undo the param changes since they didn't help.
                    params[i] = original_param
                    # Maybe the step size is too large - decrease it.
                    dparams[i] *= (1.0 - dp_percent_delta)
                    logging.debug("Moving fowrard/backward both failed; reset the"
                                 " value for param %s to %s and decreased step "
                                 "size to %s",
                                 i, params[i], dparams[i])

    return params, best_so_far, num_iter
