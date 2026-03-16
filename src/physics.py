import torch


def navier_stokes_loss(model, coords, nu=1e-5):
    """
    Computes PDE residual loss for 2D steady incompressible Navier-Stokes:
        continuity:   u_x + v_y = 0
        momentum_x:   u*u_x + v*u_y + p_x - nu*(u_xx + u_yy) = 0
        momentum_y:   u*v_x + v*v_y + p_y - nu*(v_xx + v_yy) = 0

    Args:
        model:  PINN model, output shape (N, 3) -> [u, v, p]
        coords: (N, 2) tensor of [x, y], must NOT have requires_grad before call
        nu:     kinematic viscosity (AirFRANS Re ~ 1e6, chord=1 -> nu ~ 1e-5 to 1e-6)

    Returns:
        scalar physics loss
    """
    coords = coords.detach().requires_grad_(True)

    pred = model(coords)
    u = pred[:, 0:1]
    v = pred[:, 1:2]
    p = pred[:, 2:3]

    def grad(f):
        return torch.autograd.grad(
            f, coords,
            grad_outputs=torch.ones_like(f),
            create_graph=True
        )[0]

    du = grad(u)
    dv = grad(v)
    dp = grad(p)

    u_x, u_y = du[:, 0:1], du[:, 1:2]
    v_x, v_y = dv[:, 0:1], dv[:, 1:2]
    p_x, p_y = dp[:, 0:1], dp[:, 1:2]

    u_xx = grad(u_x)[:, 0:1]
    u_yy = grad(u_y)[:, 1:2]
    v_xx = grad(v_x)[:, 0:1]
    v_yy = grad(v_y)[:, 1:2]

    continuity  = u_x + v_y
    momentum_x  = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    momentum_y  = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

    return (
        (continuity ** 2).mean() +
        (momentum_x ** 2).mean() +
        (momentum_y ** 2).mean()
    )
