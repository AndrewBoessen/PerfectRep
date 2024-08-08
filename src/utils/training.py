import collections
import warnings

class AverageMeter(object):
    '''
    Calcualte and store the current and average value of a given metric
    
    Methods:
        reset - Set current value and average to zero
        update - Update current value and averge with new value
    '''
    def __init__(self):
        self.reset() # Set all values to zero
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.count = 0
        self.sum = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def load_checkpoint_weights(model, checkpoint):
    '''
    Load previous learned weights from checkpoint to model
    
    Incompatible layers (unmatched in name or size) will be ignored
    Args:
        model (nn.Module): network model
        checkpoint (dict): checkpoint weights
    '''
    assert 'state_dict' in checkpoint, "checkpoint does not contain state dict"

    state_dict = checkpoint['state_dict'] # dictionary of weights from checkpoint
    model_dict = model.state_dict() # current model weights

    if set(state_dict.keys()) != set(model_dict.keys()):
        warnings.warn("model and checkpoint states do not align")
    
    new_state_dict = collections.OrderedDict()
    aligned_layers = []
    for k, v in state_dict.items():
        if k in model_dict and model_dict[k].size() == v.size(): # Layers align with model and checkpoint
            new_state_dict[k] = v
            aligned_layers.append(k)
    model_dict.update(new_state_dict) # Update model state with checkpoint values
    model.load_state_dict(model_dict, strict=True) # Load new state in model

    print('Loaded Checkpoint Weights', len(aligned_layers))
    return model

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
