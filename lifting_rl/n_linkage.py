from sympy import symbols
from sympy.physics.mechanics import *
from sympy import Dummy, lambdify


def kane(n=5, mode="rigid_body", hands_load=False):
    q = dynamicsymbols("q:" + str(n))
    u = dynamicsymbols("u:" + str(n))
    f = dynamicsymbols("f:" + str(n))
    m = symbols("m:" + str(n))
    l = symbols("l:" + str(n))
    g, t = symbols("g t")

    N = ReferenceFrame("N")
    frames = [N]
    points = []
    phys_objs = []
    loads = []
    kindiffs = []
    for i in range(n):
        RFi = N.orientnew("RF" + str(i), "Axis", [q[i], N.z])
        RFi.set_ang_vel(N, sum(u[: i + 1]) * N.z)

        if not points:
            Pi = Point("P" + str(i))
            Pi.set_vel(N, 0)
        else:
            Pi = points[-1].locatenew("P" + str(i), l[i - 1] * frames[-1].x)
        points.append(Pi)
        frames.append(RFi)
        Pcmi = Pi.locatenew("Pcm" + str(i), l[i] / 2 * RFi.x)
        Pcmi.v2pt_theory(Pi, N, RFi)
        if mode == "particle":
            obj = Particle("Pa" + str(i), Pcmi, m[i])

        if mode == "rigid_body":
            Ii = inertia(
                RFi,
                m[i] * l[i] * l[i] / 3,
                m[i] * l[i] * l[i] / 3,
                m[i] * l[i] * l[i] / 3,
            )
            obj = RigidBody("RB" + str(i), Pcmi, RFi, m[i], (Ii, Pi))
        phys_objs.append(obj)
        loads.append((Pcmi, -m[i] * g * N.y))
        loads.append((RFi, 0.5 * f[i] * l[i] * RFi.z))
        kindiffs.append(q[i].diff(t) - u[i])

    km = KanesMethod(N, q_ind=q, u_ind=u, kd_eqs=kindiffs)
    fr, frstar = km.kanes_equations(phys_objs, loads=loads)
    fr.simplify()
    frstar.simplify()
    dynamic = q + u + f
    dummy_symbols = [Dummy() for i in dynamic]
    dummy_dict = dict(zip(dynamic, dummy_symbols))
    kindiff_dict = km.kindiffdict()
    params = [g]
    for i in range(n):
        params += [l[i], m[i]]
    M = km.mass_matrix_full.subs(kindiff_dict).subs(dummy_dict)
    F = km.forcing_full.subs(kindiff_dict).subs(dummy_dict)
    M.simplify()
    F.simplify()
    return M, F, dummy_symbols + params
