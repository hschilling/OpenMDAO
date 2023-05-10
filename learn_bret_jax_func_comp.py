def f(a):
    return a + 2


from openmdao.components.func_comp_common import jac_forward
d_f = jac_forward(f)

d = d_f(3.2)

print(d)