import torch
import matplotlib.pyplot as plt

dataset = 'nyc-mix'
name = 'sensitivity-base1'
gcn_depth = 3
if __name__ == '__main__':
    model = torch.load(open(f"./saves/{dataset}/MSTFCN/{name}/best-model.pt", "rb"), map_location='cpu')
    model.graph_constructor.device = torch.device('cpu')
    graphs = model.graph_constructor().detach().cpu()

    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(2, 2)
    axs = gs.subplots(sharey=True)

    for i in range(2):
        for j in range(2):
            g = graphs[i, j]
            ax = axs[i, j]

            adj = g  # + torch.eye(g.size(0))
            d = adj.sum(1)
            a = adj / d.view(-1, 1)

            # ap = [torch.eye(len(a))]
            # for _ in range(gcn_depth):
            #     ap.append(torch.matmul(ap[-1], a))
            # g = torch.stack(ap, 0).sum(0) / (gcn_depth + 1)
            g = a

            ax.set_title(f"{i}-{j}")
            im = ax.imshow(g)

            cbar = fig.colorbar(im, ax=ax, orientation='horizontal', location='top')

    fig.tight_layout()
    plt.savefig(f"./saves/plot/ExpGraph_{dataset}_{name}.png", dpi=300)
    plt.show()
