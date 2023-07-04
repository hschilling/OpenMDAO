import openmdao.api as om

prob = om.Problem()
prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

