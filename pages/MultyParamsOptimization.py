import streamlit as st
import math_module as mm
import pandas as pd
import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.termination import get_termination 
from pymoo.optimize import minimize

from pymoo.core.problem import Problem

class MyProblem(Problem):
    def __init__(self):
        super().__init__(n_var=2, n_obj=2, n_constr=0, xl=np.array([0, 0]), xu=np.array([10, 10]))

    def _evaluate(self, X, out, *args, **kwargs):
        f1 = []
        f2 = []
        samples = [
            [3.0, 5.0, 3.424],
            [5.0, 5.0, 2.972],
            [3.0, 2.0, 2.043],
            [5.0, 2.0, 1.831]
        ]

        for L, P in X:
            errors = []
            mass_deltas = []
            for s in samples:
                resData = pd.DataFrame(mm.run_calculation(s[0], s[1], 23.7, None, P=P, L=L)['stats'])
                minimum = resData['mean'].min()
                mass_delta = resData['mean'].std()
                mass_deltas.append(mass_delta)
                errors.append((minimum - s[2]) ** 2)
            mse_value = sum(errors) / len(samples)
            mass_value = sum(mass_deltas) / len(samples)
            f1.append(mse_value)
            f2.append(mass_value)
            st.write(f"L = `{L}`, P = `{P}`, MSE = `{mse_value}`, Mass = `{mass_value}`")
        out["F"] = np.column_stack([f1, f2])

def FindP():
    problem = MyProblem()
    algorithm = GA(pop_size=100)

    termination = get_termination("n_gen", 40)

    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   verbose=False)

    st.json(res.X)
    st.json(res.F)

go = st.button("Начать подбор!")

if go:
    FindP()