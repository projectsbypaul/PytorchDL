from torch.optim.lr_scheduler import LambdaLR

def get_linear_scheduler(optimizer, initial_lr, final_lr, n_epochs):
    def lr_lambda(epoch):
        return ((final_lr / initial_lr) - 1) * (epoch / n_epochs) + 1
    scheduler = LambdaLR(optimizer, lr_lambda)
    scheduler.final_lr = final_lr  # ← attach custom attribute
    return scheduler