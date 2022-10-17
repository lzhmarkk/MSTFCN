import time
from tensorboardX import SummaryWriter
from util import *
from trainer import Trainer
import sys,os
from evaluate import get_scores
from data.dataloader import load_dataset


args = get_config()
torch.set_num_threads(3)

def main(runid):
    #load data
    save_folder = os.path.join('./saves',args.data,args.model_name, args.expid, str(runid))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    model_path = os.path.join(save_folder,'best-model.pt')
    sys.stdout = Logger(os.path.join(save_folder,'log.txt'))
    dataloader = load_dataset(args.data, args.batch_size, args.window, args.horizon, args.input_dim, args.output_dim, add_time=args.add_time)
    setattr(args, 'device', torch.device(args.device))
    setattr(args, 'scaler', dataloader['scaler'])

    auxiliary = get_auxiliary(args, dataloader)
    for k, v in auxiliary.items():
        setattr(args, k, v)
    model = get_model(args)
  
    print(args)
    print(model)
    print('Number of model parameters is', sum([p.nelement() for p in model.parameters()]))
    run_folder = os.path.join(save_folder,'run')
    if not os.path.exists(run_folder):
        os.makedirs(run_folder)
    writer = SummaryWriter(run_folder)

    engine = Trainer(model, args.learning_rate, args.weight_decay, args.clip, args.step_size, args.horizon, args.scaler, args.device, args.cl, args.mask0)

    print("start training...",flush=True)
    his_loss =[]
    val_time = []
    train_time = []
    minl = 1e5
    best_epoch = 1e5
    for i in range(args.epochs):
        train_loss = []
        t1 = time.time()
        # dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            tx = torch.Tensor(x).to(args.device)  # [B,T,N,C]
            ty = torch.Tensor(y).to(args.device)  # [B,T,N,C]

            loss = engine.train(tx, ty[..., :args.output_dim], ty[..., args.output_dim:])
            train_loss.append(loss)

            if iter % args.print_interval == 0 :
                log = 'Iter: {:03d}, Train Loss: {:.4f}'
                print(log.format(iter, train_loss[-1]),flush=True)
        t2 = time.time()
        train_time.append(t2-t1)
        #validation
        valid_loss = []


        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(args.device)
            testy = torch.Tensor(y).to(args.device)

            vloss = engine.eval(testx, testy[..., :args.output_dim], testy[..., args.output_dim:])
            valid_loss.append(vloss)

        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)

        mvalid_loss = np.mean(valid_loss)
        his_loss.append(mvalid_loss)


        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Valid Loss: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mvalid_loss, (t2 - t1)),flush=True)
        writer.add_scalars('loss', {'train':mtrain_loss}, global_step=i )
        writer.add_scalars('loss', {'valid':mvalid_loss}, global_step=i )


        if mvalid_loss<minl:
            with open(model_path, 'wb') as f:
                torch.save(engine.model, f)
            #torch.save(engine.model.state_dict(), os.path.join(save_folder,"exp.pth"))
            minl = mvalid_loss
            best_epoch = i
            print(f'save best epoch {best_epoch} *****************')
        elif args.early_stop and  i - best_epoch > args.early_stop_steps:
            print('best epoch:', best_epoch)
            break

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))


    bestid = np.argmin(his_loss)

    # engine.model.load_state_dict(torch.load(os.path.join(save_folder,"exp.pth")))

    # Load the best saved model.
    with open(model_path, 'rb') as f:
        engine.model = torch.load(f)

    print("Training finished")
    print('Best epoch:', bestid)
    print("The valid loss on best model is", str(round(his_loss[bestid],4)))

    #valid data
    outputs = []
    realy = torch.Tensor(dataloader['y_val']).to(args.device)
    realy = realy[:,:,:,:args.output_dim] #[B, T, N,C]

    for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
        testx = torch.Tensor(x).to(args.device)
        testy = torch.Tensor(y).to(args.device)
        with torch.no_grad():
            preds = engine.model(testx, pred_time=testy[..., args.output_dim:])  # [B,T,N,C]
        outputs.append(preds) # [B,T,N,C]

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]



    # mask = 1
    # if args.data == 'nyc-bike' or args.data == 'nyc-taxi':
    #     mask = 0
    preds = args.scaler.inverse_transform(yhat).cpu().numpy()
    targets = args.scaler.inverse_transform(realy).cpu().numpy()
    valid_scores = get_scores(preds, targets, mask = args.mask0, out_catagory='multi')
    vrmse = valid_scores['RMSE']['all']
    vmae = valid_scores['MAE']['all']
    vcorr = valid_scores['CORR']['all']


    #test data
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(args.device) #[B, T, N,C]
    realy = realy[:,:,:,:args.output_dim] #[B, T, N,C]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(args.device)
        testy = torch.Tensor(y).to(args.device)
        with torch.no_grad():
            preds = engine.model(testx, pred_time=testy[..., args.output_dim:])  #[B,T,N,C]
        outputs.append(preds)#[B,T,N,C]

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]
    save_predicitions = args.scaler.inverse_transform(yhat).cpu().numpy()
    save_targets = args.scaler.inverse_transform(realy).cpu().numpy()
    test_scores = get_scores(save_predicitions, save_targets, mask = args.mask0, out_catagory='multi', detail = True)
    rmse = []
    mae = []
    corr = []    
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
    np.savez(os.path.join(save_folder, 'test-results.npz'), predictions=save_predicitions, targets=save_targets)
    return vrmse, vmae, vcorr, rmse, mae, corr, armse, amae, acorr


if __name__ == "__main__":
    vrmse= []
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

    rmse = np.array(rmse)
    mae = np.array(mae)
    corr = np.array(corr)
    
    mrmse = np.mean(rmse,0)
    mmae = np.mean(mae,0)
    mcorr = np.mean(corr,0)  

    srmse = np.std(rmse,0)
    smae = np.std(mae,0)
    scorr = np.std(corr,0)
    

    print(f'\n\nResults for {args.runs} runs\n\n')
    #valid data
    print('valid\tRMSE\tMAE\tCORR')
    log = 'mean:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.mean(vrmse),np.mean(vmae),np.mean(vcorr)))
    log = 'std:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.std(vrmse),np.std(vmae),np.std(vcorr)))
    print('\n\n')
    #test data
    print('test|horizon\tRMSE-mean\tMAE-mean\tCORR-mean\tRMSE-std\tMAE-std\tcorr-std')
    for i in [2,5,11]:
        log = '{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
        print(log.format(i+1, mrmse[i], mmae[i], mcorr[i], srmse[i], smae[i], scorr[i]))
    print('test|All\tRMSE-mean\tMAE-mean\tCORR-mean\tRMSE-std\tMAE-std\tcorr-std')
    print(log.format(0, np.mean(armse,0), np.mean(amae,0), np.mean(acorr,0), np.std(armse,0), np.std(amae,0), np.std(acorr,0)))





