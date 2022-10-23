from util import *
from evaluate import get_scores
from data.dataloader import load_dataset

args = get_config()
torch.set_num_threads(3)
set_random_seed(args.seed)


def main():
    #load data
    save_folder = os.path.join('./saves', args.data, args.model_name, args.expid)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    model_path = os.path.join(save_folder, 'best-model.pt')
    sys.stdout = Logger(os.path.join(save_folder,'log.txt'))
    dataloader = load_dataset(args.data, args.batch_size, args.window, args.horizon, args.input_dim, args.output_dim, add_time=args.add_time)
    setattr(args, 'device', torch.device(args.device))
    setattr(args, 'scaler', dataloader['scaler'])

    auxiliary = get_auxiliary(args, dataloader)
    for k, v in auxiliary.items():
        setattr(args, k, v)
    # Load the best saved model.
    with open(model_path, 'rb') as f:
        model = torch.load(f, map_location=args.device)

    outputs = []
    realy = torch.Tensor(dataloader['y_val']).to(args.device)
    realy = realy[:,:,:,:args.output_dim] #[B, T, N,C]

    for iter, (x, y) in enumerate(dataloader['val_loader']):
        testx = torch.Tensor(x).to(args.device)
        testy = torch.Tensor(y).to(args.device)
        with torch.no_grad():
            preds = model(testx, pred_time=testy[..., args.output_dim:])  # [B,T,N,C]
        outputs.append(preds) # [B,T,N,C]

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]

    preds = args.scaler.inverse_transform(yhat).cpu().numpy()
    targets = args.scaler.inverse_transform(realy).cpu().numpy()
    valid_scores = get_scores(preds, targets, mask = args.mask0, out_catagory='multi', detail = True)

    print('valid results:')
    print(json.dumps(valid_scores, cls=MyEncoder, indent=4))

    #test data
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(args.device) #[B, T, N,C]
    realy = realy[:,:,:,:args.output_dim] #[B, T, N,C]

    for iter, (x, y) in enumerate(dataloader['test_loader']):
        testx = torch.Tensor(x).to(args.device)
        testy = torch.Tensor(y).to(args.device)
        with torch.no_grad():
            preds = model(testx, pred_time=testy[..., args.output_dim:])  #[B,T,N,C]
        outputs.append(preds)#[B,T,N,C]

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]
    save_predicitions = args.scaler.inverse_transform(yhat).cpu().numpy()
    save_targets = args.scaler.inverse_transform(realy).cpu().numpy()
    test_scores = get_scores(save_predicitions, save_targets, mask = args.mask0, out_catagory='multi', detail = True)
   
    print('test results:')
    print(json.dumps(test_scores, cls=MyEncoder, indent=4))
    with open(os.path.join(save_folder, 'test-scores.json'), 'w+') as f:
        json.dump(test_scores, f, cls=MyEncoder, indent=4)
    np.savez(os.path.join(save_folder, 'test-results.npz'), predictions=save_predicitions, targets=save_targets)


if __name__ == "__main__":
    main()
