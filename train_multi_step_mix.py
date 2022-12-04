import time
from util import *
from tqdm import tqdm
from trainer import Trainer
from evaluate import get_scores
from tensorboardX import SummaryWriter
from data.load_dataset import load_dataset_mix

args = get_config()


def evaluate(args, engine, dataloader):
    """
    output: loss(float), scores(list[dict]), pred(ndarray), real(ndarray)
    """
    pred, real, loss = [], [], []
    for _, (x, y) in enumerate(dataloader):
        x = torch.Tensor(x).to(args.device)  # [B, T, N, C + time], transformed
        y = torch.Tensor(y).to(args.device)  # [B, T, N, C + time], transformed
        with torch.no_grad():
            _loss, _pred, _real = engine.eval(x, real=y[..., :sum(args.output_dim)],
                                              pred_time=y[..., sum(args.output_dim):])
        pred.append(_pred)  # [B, T, N, C], inverse_transformed
        real.append(_real)  # [B, T, N, C], inverse_transformed
        loss.append(_loss)

    pred = torch.cat(pred, dim=0).cpu().numpy()
    real = torch.cat(real, dim=0).cpu().numpy()
    scores = []
    idx = 0
    for dim in args.output_dim:
        scores.append(get_scores(pred[..., idx: idx + dim], real[..., idx: idx + dim],
                                 mask=args.mask0, out_catagory='multi', detail=True))
        idx += dim

    return np.mean(loss).item(), scores, pred, real


def main(runid):
    set_random_seed(args.seed + runid)
    # load data
    save_folder = os.path.join('./saves', args.data, args.model_name, args.expid + str(runid))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    model_path = os.path.join(save_folder, 'best-model.pt')
    sys.stdout = Logger(os.path.join(save_folder, 'log.txt'))
    dataloader = load_dataset_mix(args.data, args.batch_size, args.window, args.horizon, args.input_dim,
                                  args.output_dim, add_time=True)
    setattr(args, 'device', torch.device(args.device))
    setattr(args, 'scaler', dataloader['scaler'])

    auxiliary = get_auxiliary(args, dataloader)
    for k, v in auxiliary.items():
        setattr(args, k, v)
    model = get_model(args)

    print(args)
    print(model)
    print('Number of model parameters is', sum([p.nelement() for p in model.parameters()]))
    run_folder = os.path.join(save_folder, 'run')
    if not os.path.exists(run_folder):
        os.makedirs(run_folder)
    writer = SummaryWriter(run_folder)

    engine = Trainer(model, args.learning_rate, args.weight_decay, args.clip, args.step_size,
                     args.horizon, args.scaler, args.device, args.early_stop_steps, args.cl, args.mask0)

    # train model
    print("start training...", flush=True)
    his_loss = []
    train_time, valid_time = [], []
    min_valid_loss = best_epoch = 1e5
    for i in range(args.epochs):
        train_loss = []
        t = time.time()
        dataloader['train_loader'].shuffle()
        tqdm_loader = tqdm(dataloader['train_loader'], ncols=150)
        for iter, (x, y) in enumerate(tqdm_loader):
            x = torch.Tensor(x).to(args.device)  # [B, T, N, C + time], inverse_transformed
            y = torch.Tensor(y).to(args.device)  # [B, T, N, C + time], inverse_transformed
            loss = engine.train(x, real=y[..., :sum(args.output_dim)], pred_time=y[..., sum(args.output_dim):])
            train_loss.append(loss)

            tqdm_loader.set_description('Iter: {:03d}, Train Loss: {:.4f}'.format(iter, train_loss[-1]))

        engine.scheduler.step()
        train_time.append(time.time() - t)
        train_loss = np.mean(train_loss).item()

        # validation
        t = time.time()
        valid_loss, _, _, _ = evaluate(args, engine, dataloader['val_loader'])

        print('Epoch: {:03d}, Inference Time: {:.4f} secs'.format(i, (time.time() - t)))
        valid_time.append(time.time() - t)
        his_loss.append(valid_loss)

        # print('Epoch: {:03d}, Train Loss: {:.4f}, Valid Loss: {:.4f}, '
        #       'Training Time: {:.2f}s/epoch, Inference Time: {:.2f}s/epoch'.
        #       format(i, train_loss, valid_loss, train_time[-1], valid_time[-1]), flush=True)
        print('Epoch: {:03d}, lr: {:.8f}, Train Loss: {:.4f}, Valid Loss: {:.4f}, '
              'Training Time: {:.2f}s/epoch, Inference Time: {:.2f}s/epoch'.
              format(i, engine.scheduler.get_last_lr()[0], train_loss, valid_loss, train_time[-1], valid_time[-1]),
              flush=True)
        writer.add_scalars('loss', {'train': train_loss}, global_step=i)
        writer.add_scalars('loss', {'valid': valid_loss}, global_step=i)

        if valid_loss < min_valid_loss:
            with open(model_path, 'wb') as f:
                torch.save(engine.model, f)
            min_valid_loss = valid_loss
            best_epoch = i
            print(f'save best epoch {best_epoch} *****************')
        elif args.early_stop and i - best_epoch > args.early_stop_steps:
            print('best epoch:', best_epoch)
            break

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(valid_time)))

    # Load the best saved model
    with open(model_path, 'rb') as f:
        engine.model = torch.load(f)
    best_epoch = np.argmin(his_loss)
    print("Training finished")
    print('Best epoch:', best_epoch)
    print("The valid loss on best model is", str(his_loss[best_epoch]))

    # valid model
    _, valid_scores, _, _ = evaluate(args, engine, dataloader['val_loader'])
    vrmse = [valid_score['RMSE']['all'] for valid_score in valid_scores]
    vmae = [valid_score['MAE']['all'] for valid_score in valid_scores]
    vcorr = [valid_score['CORR']['all'] for valid_score in valid_scores]

    # test model
    _, test_scores, pred, tgt = evaluate(args, engine, dataloader['test_loader'])
    rmse, mae, corr = [], [], []
    for test_score in test_scores:
        _rmse, _mae, _corr = [], [], []
        for i in range(args.horizon):
            _rmse.append(test_score['RMSE'][f'horizon-{i}'])
            _mae.append(test_score['MAE'][f'horizon-{i}'])
            _corr.append(test_score['CORR'][f'horizon-{i}'])
        rmse.append(_rmse)
        mae.append(_mae)
        corr.append(_corr)
    armse = [test_score['RMSE']['all'] for test_score in test_scores]
    amae = [test_score['MAE']['all'] for test_score in test_scores]
    acorr = [test_score['CORR']['all'] for test_score in test_scores]

    print('test results:')
    print(json.dumps(test_scores, cls=MyEncoder, indent=4))
    with open(os.path.join(save_folder, 'test-scores.json'), 'w+') as f:
        json.dump(test_scores, f, cls=MyEncoder, indent=4)
    np.savez(os.path.join(save_folder, 'test-results.npz'), predictions=pred, targets=tgt)
    return vrmse, vmae, vcorr, rmse, mae, corr, armse, amae, acorr


if __name__ == "__main__":
    vrmse = []
    vmae = []
    vcorr = []
    rmse = []
    mae = []
    corr = []
    armse = []
    amae = []
    acorr = []
    for i in range(args.runs):
        v1, v2, v3, t1, t2, t3, a1, a2, a3 = main(i)
        vrmse.append(v1)
        vmae.append(v2)
        vcorr.append(v3)
        rmse.append(t1)
        mae.append(t2)
        corr.append(t3)
        armse.append(a1)
        amae.append(a2)
        acorr.append(a3)

    vrmse = np.array(vrmse)
    vmae = np.array(vmae)
    vcorr = np.array(vcorr)
    rmse = np.array(rmse)
    mae = np.array(mae)
    corr = np.array(corr)
    armse = np.array(armse)
    amae = np.array(amae)
    acorr = np.array(acorr)

    mrmse = np.mean(rmse, 0)
    mmae = np.mean(mae, 0)
    mcorr = np.mean(corr, 0)
    srmse = np.std(rmse, 0)
    smae = np.std(mae, 0)
    scorr = np.std(corr, 0)

    print(f'\n\nResults for {args.runs} runs\n\n')
    # valid data
    for j in range(len(args.input_dim)):
        print('valid\tMAE\tRMSE\tCORR')
        print('mean:\t{:.4f}\t{:.4f}\t{:.4f}'.format(np.mean(vmae[:, j]), np.mean(vrmse[:, j]), np.mean(vcorr[:, j])))
        print('std:\t{:.4f}\t{:.4f}\t{:.4f}'.format(np.std(vmae[:, j]), np.std(vrmse[:, j]), np.std(vcorr[:, j])))
        print('\n\n')
        # test data
        print('test|horizon\tMAE-mean\tRMSE-mean\tCORR-mean\tMAE-std\tRMSE-std\tcorr-std')
        for i in [2, 5, 11]:
            print('{:d}\t\t\t\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t\t{:.4f}\t{:.4f}'
                  .format(i + 1, mmae[j, i], mrmse[j, i], mcorr[j, i], smae[j, i], srmse[j, i], scorr[j, i]))
        print('test|All\t\tMAE-mean\tRMSE-mean\tCORR-mean\tMAE-std\tRMSE-std\tcorr-std')
        print('{:d}\t\t\t\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
              .format(0, np.mean(amae[:, j], 0), np.mean(armse[:, j], 0), np.mean(acorr[:, j], 0),
                      np.std(amae[:, j], 0), np.std(armse[:, j], 0), np.std(acorr[:, j], 0)))
        print("-" * 30)
