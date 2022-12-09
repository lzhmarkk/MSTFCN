import os
import numpy as np
import matplotlib.pyplot as plt

dataset = "beijing-mix"
model = 'CRGNN'
expid = 'mixer-sematicGraph-noSkip-notime0'
selected_variables = [2, 30, 53, 70, 103]

if __name__ == '__main__':
    file = os.path.join("./saves/", dataset, model, expid, "test-results.npz")
    test_res = np.load(file)

    B, T, N, C = test_res['predictions'].shape
    pred = test_res['predictions'][0::12].reshape(-1, N, C)
    truth = test_res['targets'][0::12].reshape(-1, N, C)
    print(pred.shape)

    for i in selected_variables:
        for m in range(C):
            p = pred[:, i, m]
            t = truth[:, i, m]
            x = np.arange(p.shape[0])

            fig, ax = plt.subplots(figsize=(40, 10))
            plt.title(f"{dataset}-{model}-{expid}")
            plt.xticks([2 * 24 * day + 24 for day in range(7 * 4)], [f"day {day}" for day in range(7 * 4)])
            plt.plot(x, p, label=f'pred-{m}')
            plt.plot(x, t, label=f'truth-{m}')
            plt.legend(loc=0)

            plt.savefig(f"./saves/plot/output/{dataset}-{model}-{i}-{m}.png")
            plt.show()
