### TODO: add your import
import numpy
import matplotlib.pyplot as plt

def visualize_loss(train_loss_list, train_interval, val_loss_list, val_interval, dataset, out_dir):
    ### TODO: visualize loss of training & validation and save to [out_dir]/loss.png
    # train_interval = log_interval = 10
    # val_interval = eval_interval = 200
    # dataset = dataset
    # out_dir = out_dir
    x_train = list(range(len(train_loss_list)))
    x_train  = [i*train_interval for i in x_train]
    x_loss = list(range(len(val_loss_list)))
    x_loss = [i*val_interval for i in x_loss]
    plt.plot(x_train, train_loss_list, label='train_loss')
    plt.plot(x_loss, val_loss_list, label='val_loss')
    plt.legend()
    plt.title('Loss of training & validation')
    plt.savefig(f'{out_dir}/loss.png')
    ###