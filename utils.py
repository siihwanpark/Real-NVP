import torch

def save_checkpoint(model, path):
    model_state = {
        'state_dict' : model.state_dict()
    }
    
    torch.save(model_state, path)
    print('A check point has been generated : ' + path)