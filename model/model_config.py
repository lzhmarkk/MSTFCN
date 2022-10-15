config = {
    'dy_embedding_dim': 20,  # the dimension of evolving node representation
    'dy_interval': [1, 1, 1],  # time intervals for each layer
    'layers': 3,  # number of layers
    'conv_channels': 32,  # convolution channels
    'residual_channels': 32,  # residual channels
    'skip_channels': 64,  # skip channels
    'end_channels': 128,  # end channels
    'kernel_set': [2, 6],  # the kernel set in TCN
    'dilation_exponential': 1,  # dilation exponential
    'gcn_depth': 2,  # graph convolution depth
    'st_embedding_dim': 40,  # the dimension of static node representation
    'dropout': 0.3,  # dropout rate
    'propalpha': 0.05,  # prop alpha

    "add_time": False
}
