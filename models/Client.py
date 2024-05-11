from models.Update import LocalUpdate
from models.Fed import FedAvg
import copy

class SwarmClient:
    def __init__(self, args, dataset, id):
        self.args = args
        self.dataset = dataset
        self.user = id

    def update(self, network, user):
        local = LocalUpdate(args=self.args, dataset=self.dataset, idxs=user)
        w, loss = local.train(net=copy.deepcopy(network).to(self.args.device))
        return w, loss
    
    def aggregate(self, w_locals):
        return FedAvg(w_locals)
    
    def get_id(self):
        return self.user