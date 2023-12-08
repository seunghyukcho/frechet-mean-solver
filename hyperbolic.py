import wandb
import torch
import geoopt
import argparse
import numpy as np
import plotly.express as px
from plotly import graph_objects as go


def disk2lorentz(x):
    x_norm = x.pow(2).sum(dim=-1, keepdim=True)
    x = torch.concat([
        (1 + x_norm) / (1 - x_norm),
        2 * x / (1 - x_norm)
    ], dim=-1)
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--dim', type=int)
    parser.add_argument('--n_points', type=int)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=7777)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--init_mu', type=float, default=0.0)
    parser.add_argument('--init_sigma', type=float, default=1.0)
    parser.add_argument('--model', type=str,
                        choices=['poincare_disk', 'lorentz', 'tangent_space']
                        )
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'])
    parser.add_argument('--exp_name', type=str, default='UNTITLED')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.set_default_tensor_type(torch.DoubleTensor)

    manifold = geoopt.PoincareBall()
    x = args.init_mu + args.init_sigma * torch.randn([args.n_points, args.dim])
    frechet_mean = torch.zeros([args.dim])
    x = manifold.expmap0(x)
    frechet_mean = manifold.expmap0(frechet_mean)

    if args.model == 'lorentz':
        manifold = geoopt.Lorentz()
        x = disk2lorentz(x)
        frechet_mean = disk2lorentz(frechet_mean)

    if args.model == 'tangent_space':
        frechet_mean = manifold.logmap0(frechet_mean)
        frechet_mean = torch.nn.parameter.Parameter(
            frechet_mean,
            requires_grad=True
        )
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD([frechet_mean], lr=args.lr)
        else:
            optimizer = torch.optim.Adam([frechet_mean], lr=args.lr)
    else:
        frechet_mean = geoopt.ManifoldParameter(
            data=frechet_mean,
            manifold=manifold
        )
        if args.optimizer == 'sgd':
            optimizer = geoopt.optim.RiemannianSGD([frechet_mean], lr=args.lr)
        else:
            optimizer = geoopt.optim.RiemannianAdam([frechet_mean], lr=args.lr)

    trajectory = []
    wandb.init(project='hyperbolic-frechet-mean')
    wandb.run.name = args.exp_name
    for e in range(args.n_epochs):
        optimizer.zero_grad()

        if args.model == 'tangent_space':
            dists = manifold.dist(x, manifold.expmap0(frechet_mean[None]))
        else:
            dists = manifold.dist(x, frechet_mean[None])
        loss = dists.pow(2).mean()
        loss.backward()
        optimizer.step()
        print(loss.item())
        wandb.log({
            'epoch': e,
            'loss': loss.item()
        })

        if args.dim == 2:
            if args.model == 'tangent_space':
                trajectory.append(manifold.expmap0(
                    frechet_mean).clone().detach().numpy())
            else:
                trajectory.append(frechet_mean.clone().detach().numpy())

    if args.dim == 2:
        trajectory = np.stack(trajectory, axis=0)
        fig = px.scatter(x=trajectory[:, 0], y=trajectory[:, 1])
        fig.add_shape(
            type='circle',
            xref='x',
            yref='y',
            x0=-1,
            y0=-1,
            x1=1,
            y1=1
        )
        fig.update_yaxes(
            scaleanchor='x',
            scaleratio=1
        )
        wandb.log({
            'trajectory': fig
        })
