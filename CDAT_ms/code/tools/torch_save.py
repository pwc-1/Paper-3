import mindspore
import x2ms_adapter

def save_model(net,acc,epoch,optimizer,loss,global_step,lr,save_path):
    state = {
            'net': x2ms_adapter.nn_cell.state_dict(net) ,
            'acc': acc,
            'epoch': epoch,
            'optimizer_state_dict': x2ms_adapter.nn_cell.state_dict(optimizer),
            'loss': loss,
            'global_step':global_step,
            'lr':lr,
            }
    x2ms_adapter.save(state, save_path)

def load_model(net,load_path,optimizer=None):
    checkpoint = x2ms_adapter.load(load_path)
    x2ms_adapter.load_state_dict(net, checkpoint['net'])
    save_epoch=checkpoint['epoch']
    save_acc=checkpoint['acc']
    save_global_step=checkpoint['global_step']
    save_lr=checkpoint['lr']

    if optimizer==None:
        return net,save_epoch,save_acc,save_global_step,save_lr
    else:
        x2ms_adapter.load_state_dict(optimizer, checkpoint['optimizer_state_dict'])

        return net,save_epoch,save_acc,save_global_step,save_lr,optimizer