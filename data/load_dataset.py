from .dataloader import DataLoaderM
from data.util import StandardScaler, generate_train_val_test


def load_dataset(dataset, batch_size, window, horizon, input_dim, output_dim, valid_batch_size=None, test_batch_size=None, add_time=False):
    [x_train, y_train], [x_val, y_val], [x_test, y_test] = generate_train_val_test(dataset, window, horizon, add_time, add_time)

    scaler = StandardScaler(x_train[..., :input_dim], dim=input_dim)

    x_train[..., : input_dim] = scaler.transform(x_train[..., : input_dim])
    y_train[..., : output_dim] = scaler.transform(y_train[..., : output_dim])
    x_val[..., : input_dim] = scaler.transform(x_val[..., : input_dim])
    y_val[..., : output_dim] = scaler.transform(y_val[..., : output_dim])
    x_test[..., : input_dim] = scaler.transform(x_test[..., : input_dim])
    y_test[..., : output_dim] = scaler.transform(y_test[..., : output_dim])

    valid_batch_size = valid_batch_size if valid_batch_size is not None else batch_size
    test_batch_size = test_batch_size if test_batch_size is not None else batch_size

    data = {}
    data['train_loader'] = DataLoaderM(x_train, y_train, batch_size)
    data['val_loader'] = DataLoaderM(x_val, y_val, valid_batch_size)
    data['test_loader'] = DataLoaderM(x_test, y_test, test_batch_size)
    data['x_train'] = x_train
    data['y_train'] = y_train
    data['x_val'] = x_val
    data['y_val'] = y_val
    data['x_test'] = x_test
    data['y_test'] = y_test
    data['scaler'] = scaler
    return data


def load_dataset_mix(dataset, batch_size, window, horizon, input_dim, output_dim, valid_batch_size=None, test_batch_size=None, add_time=False):
    assert isinstance(input_dim, list) and isinstance(output_dim, list) and input_dim == output_dim

    [x_train, y_train], [x_val, y_val], [x_test, y_test] = generate_train_val_test(dataset, window, horizon, add_time, add_time)

    sum_input_dim, sum_output_dim = sum(input_dim), sum(output_dim)
    scaler = StandardScaler(x_train[..., :sum_input_dim], dim=input_dim)

    x_train[..., : sum_input_dim] = scaler.transform(x_train[..., : sum_input_dim])
    y_train[..., : sum_output_dim] = scaler.transform(y_train[..., : sum_output_dim])
    x_val[..., : sum_input_dim] = scaler.transform(x_val[..., : sum_input_dim])
    y_val[..., : sum_output_dim] = scaler.transform(y_val[..., : sum_output_dim])
    x_test[..., : sum_input_dim] = scaler.transform(x_test[..., : sum_input_dim])
    y_test[..., : sum_output_dim] = scaler.transform(y_test[..., : sum_output_dim])

    valid_batch_size = valid_batch_size if valid_batch_size is not None else batch_size
    test_batch_size = test_batch_size if test_batch_size is not None else batch_size

    data = {}
    data['train_loader'] = DataLoaderM(x_train, y_train, batch_size)
    data['val_loader'] = DataLoaderM(x_val, y_val, valid_batch_size)
    data['test_loader'] = DataLoaderM(x_test, y_test, test_batch_size)
    data['x_train'] = x_train
    data['y_train'] = y_train
    data['x_val'] = x_val
    data['y_val'] = y_val
    data['x_test'] = x_test
    data['y_test'] = y_test
    data['scaler'] = scaler
    return data
