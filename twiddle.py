""" Twiddle algorithm for paramter optimization.
As described in Sebastian Thurn's Artificial Intelligence for Robotics online
course from Udacity: https://classroom.udacity.com/courses/cs373 in lecture 5
("PID Control").
Short video with the algorithm description:
https://www.youtube.com/watch?v=2uQ2BSzDvXs
"""

import logging

import numpy as np

def twiddle(run_func, start_params, start_dparams=None, tol=0.001,
            dp_percent_delta=0.1, max_val=1000000000):
    """
    Find params that maximize run_func.
    """

    if start_dparams is None:
        dparams = [1.0] * len(start_params)
    else:
        dparams = start_dparams[:]
        for dp in dparams:
            assert dp > 0

    assert tol > 0
    assert sum([abs(dp) for dp in dparams]) >= tol
    assert dp_percent_delta > 0 and dp_percent_delta < 1

    params = start_params[:]
    start_params = None
    best_so_far = run_func(*params)
    logging.info("current best: %s", best_so_far)
    num_iter = 0
    while sum([abs(dp) for dp in dparams]) >= tol:
        num_iter += 1
        if best_so_far >= max_val:
            logging.warn("Hard stop: best_so_far exceeded max_val: %s",
                         best_so_far)
            break
        logging.info("......")
        logging.info("p: %s dp: %s best_so_far: %s",
                     params, dparams, best_so_far)
        for i in range(len(params)):
            logging.info("Trying to update param %s p=%s dp=%s",
                         i, params[i], dparams[i])

            # Saving the current value of the i-th param since we may need to
            # restore it.
            original_param = params[i]

            # Try to move forward.
            params[i] = original_param + dparams[i]
            value = run_func(*params)
            logging.info("If moving param %s fowrard to %s we get value=%s",
                         i, params[i], value)
            if value > best_so_far:
                best_so_far = value
                params = params[:]
                # Increase step size
                dparams[i] *= (1.0 + dp_percent_delta)
                logging.info("Moved fowrard, improved value to %s, increased "
                             " step size for param %s to %s",
                             value, i, params[i])
            else:
                # Try to move in the opposite direction.
                params[i] = original_param - dparams[i]
                value = run_func(*params)
                logging.info("If moving param %s backward to %s we get value=%s",
                             i, params[i], value)
                if value > best_so_far:
                    best_so_far = value
                    params = params[:]
                    dparams[i] *= (1.0 + dp_percent_delta)
                    logging.info("Improved value to: %s increased step size for "
                                 "param %s to %s",
                                 best_so_far, i, dparams[i])
                else:
                    # We tried to move in both directions and failed
                    # Undo the param changes since they didn't help.
                    params[i] = original_param
                    # Maybe the step size is too large - decrease it.
                    dparams[i] *= (1.0 - dp_percent_delta)
                    logging.info("Moving fowrard/backward both failed; reset the"
                                 " value for param %s to %s and decreased step "
                                 "size to %s",
                                 i, params[i], dparams[i])

    return params, best_so_far, num_iter


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    tol = 0.001
    
    for run_func, start_params, expected_params in [
            ((lambda x: - x**2), [-300], [0]),
            ((lambda x, y, z: - (x**2 + y**2 + z**2)), [-300, 500, -3500],
             [0, 0, 0])]:
    
        initial_value = run_func(*start_params)
        final_params, best_val, num_iter = twiddle(run_func, start_params,
                                                   tol=tol)
        for i in range(len(final_params)):
            assert abs(expected_params[i] - final_params[i]) < tol
        
        print("Final parameters: %s" % final_params)

        # Double-checking, we should get best_val
        final_value = run_func(*final_params)
        assert abs(final_value - best_val) < tol

        print("initial_value=%s final_value=%s" % (initial_value, final_value))
        print("num iterations: %s" % num_iter)
