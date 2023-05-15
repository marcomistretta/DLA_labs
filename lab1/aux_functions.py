import torch.utils.tensorboard as tb


def convert_to_string(l_sizes):
    string = "layers["
    n_layers = len(l_sizes)
    if n_layers == 0:
        raise Exception
    else:
        for i in range(0, n_layers-1):
            string += str(l_sizes[i]) + "-"
        if n_layers > 1:
            string += str(l_sizes[n_layers-1])+"]"
    return string


def create_summary_writer(lr, batch_size, epochs, folder, mode="mlp", sizes=None, depth=None, residual=None, cam=False):
    if mode == "mlp":
        w_path = "./model/"+folder+"/mlp-" + convert_to_string(sizes) + "-ep" + str(epochs) + "-lr" + str(lr) + "-bs" + str(batch_size)
    elif mode == "cnn":
        sub_string = "-not-residual" if not residual else "-residual"
        w_path = "./model/"+folder+"/cnn-ep"+str(epochs)+"-lr" + str(lr) + "-bs" + str(batch_size) + "-depth" + str(depth) + sub_string
        if cam:
            w_path += "-cam"
    else:
        raise Exception
    writer = tb.SummaryWriter(w_path)
    print("Summary-writer " + mode + ":", w_path)
    return writer
