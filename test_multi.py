from util import *
from trainer import Trainer
from evaluate import get_scores
from data.load_dataset import load_dataset

args = get_config()
torch.set_num_threads(3)


def evaluate(args, engine, dataloader):
    """
    output: loss(float), scores(dict), pred(ndarray), real(ndarray)
    """
    pred, real, loss = [], [], []
    for _, (x, y) in enumerate(dataloader):
        x = torch.Tensor(x).to(args.device)  # [B, T, N, C + time], transformed
        y = torch.Tensor(y).to(args.device)  # [B, T, N, C + time], transformed
        with torch.no_grad():
            _loss, _pred, _real = engine.eval(x, real=y[..., :args.output_dim], pred_time=y[..., args.output_dim:])
        pred.append(_pred)  # [B, T, N, C], inverse_transformed
        real.append(_real)  # [B, T, N, C], inverse_transformed
        loss.append(_loss)

    pred = torch.cat(pred, dim=0).cpu().numpy()
    real = torch.cat(real, dim=0).cpu().numpy()
    scores = get_scores(pred, real, mask=args.mask0, out_catagory='multi', detail=True)

    return np.mean(loss).item(), scores, pred, real


def main(runid):
    # load data
    save_folder = os.path.join('./saves', args.data, args.model_name, args.expid + str(runid))
    model_path = os.path.join(save_folder, 'best-model.pt')
    dataloader = load_dataset(args.data, args.batch_size, args.window, args.horizon, args.input_dim,
                              args.output_dim, add_time=args.add_time)
    setattr(args, 'device', torch.device(args.device))
    setattr(args, 'scaler', dataloader['scaler'])

    auxiliary = get_auxiliary(args, dataloader)
    for k, v in auxiliary.items():
        setattr(args, k, v)

    with open(model_path, 'rb') as f:
        model = torch.load(f)

    engine = Trainer(model, args.learning_rate, args.weight_decay, args.clip, args.step_size,
                     args.horizon, args.scaler, args.device, args.early_stop_steps, args.cl, args.mask0)

    # valid model
    _, valid_scores, _, _ = evaluate(args, engine, dataloader['val_loader'])
    vrmse = valid_scores['RMSE']['all']
    vmae = valid_scores['MAE']['all']
    vcorr = valid_scores['CORR']['all']

    # test model
    _, test_scores, pred, tgt = evaluate(args, engine, dataloader['test_loader'])
    rmse, mae, corr = [], [], []
    for i in range(args.horizon):
        rmse.append(test_scores['RMSE'][f'horizon-{i}'])
        mae.append(test_scores['MAE'][f'horizon-{i}'])
        corr.append(test_scores['CORR'][f'horizon-{i}'])
    armse = test_scores['RMSE']['all']
    amae = test_scores['MAE']['all']
    acorr = test_scores['CORR']['all']

    print('test results:')
    print(json.dumps(test_scores, cls=MyEncoder, indent=4))
    with open(os.path.join(save_folder, 'test-scores.json'), 'w+') as f:
        json.dump(test_scores, f, cls=MyEncoder, indent=4)
    np.savez(os.path.join(save_folder, 'test-results.npz'), predictions=pred, targets=tgt)
    return vrmse, vmae, vcorr, rmse, mae, corr, armse, amae, acorr


if __name__ == "__main__":
    main(0)
