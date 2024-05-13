from models.Update import LocalUpdate
from models.Fed import BadFedAvg
import copy
import time

class BadClient:
    def __init__(self, args, dataset, id):
        self.args = args
        self.dataset = dataset
        self.user = id

    def update(self, network, user):
        local = LocalUpdate(args=self.args, dataset=self.dataset, idxs=user)
        w, loss = local.train(net=copy.deepcopy(network).to(self.args.device))
        return w, loss
    
    def aggregate(self, w_locals):
        time.sleep(0.1)
        return BadFedAvg(w_locals)
    
    def get_id(self):
        return self.user