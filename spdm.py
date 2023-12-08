import wandb
import torch
import geoopt
import argparse
import numpy as np


def vec2mat(v, dim):
    M = torch.zeros([dim, dim])
    i, j = torch.triu_indices(dim, dim)
    M[i, j] = v
    M.T[i, j] = v
    return M


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
                        choices=['spdm', 'tangent_space']
                        )
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'])
    parser.add_argument('--exp_name', type=str, default='UNTITLED')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.set_default_tensor_type(torch.DoubleTensor)

    N = args.dim * (args.dim + 1) // 2
    manifold = geoopt.SymmetricPositiveDefinite()
    x = args.init_mu + args.init_sigma * \
        torch.randn([args.n_points, args.dim, args.dim])
    x = torch.bmm(x, x.transpose(-1, -2))

    if args.model == 'spdm':
        frechet_mean = torch.eye(args.dim)
        frechet_mean = geoopt.ManifoldParameter(
            data=frechet_mean,
            manifold=manifold
        )
        if args.optimizer == 'sgd':
            optimizer = geoopt.optim.RiemannianSGD([frechet_mean], lr=args.lr)
        else:
            optimizer = geoopt.optim.RiemannianAdam([frechet_mean], lr=args.lr)
    else:
        frechet_mean = torch.zeros([N])
        frechet_mean = torch.nn.parameter.Parameter(
            frechet_mean,
            requires_grad=True
        )
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD([frechet_mean], lr=args.lr)
        else:
            optimizer = torch.optim.Adam([frechet_mean], lr=args.lr)

    wandb.init(project='hyperbolic-frechet-mean')
    wandb.run.name = args.exp_name
    wandb.config.update(args)
    for e in range(args.n_epochs):
        optimizer.zero_grad()

        if args.model == 'tangent_space':
            dists = manifold.dist(x, manifold.expmap(
                torch.eye(args.dim)[None], vec2mat(frechet_mean, args.dim)[None]))
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
