import copy

class Reputation:
    def __init__(self, clients) -> None:
        self.client_dict = dict()
        for client in clients:
            id_client = client.get_id()
            self.client_dict[id_client] = 0
        
        self.rotation = list()
        self.ROTATION_SIZE = 5
        if len(self.client_dict) <= self.ROTATION_SIZE:
            self.ROTATION_SIZE = len(self.client_dict) - 3

    def recalculate_client(self, metrics, client):
        new_reputation = metrics["quality"]
        if self.client_dict[client] != 0:
            self.client_dict[client] = (0.6 * self.client_dict[client]) + (0.4 * new_reputation)
        else:
             self.client_dict[client] = new_reputation

    def highest_reputation(self, clients):
        return max(clients.items(), key=lambda x: x[1])[0]
    
    def non_computed_reputation(self, clients):
        return min(clients.items(), key=lambda x: x[1])

    def choose(self):
        clients = copy.deepcopy(self.client_dict)
        #favour the clients that have never aggregated before
        best_client = self.non_computed_reputation(clients)
        if best_client[1] == 0:
            return best_client[0]
        #select the best reputation that has not been in rotation for last 5 rounds
        best_client = self.highest_reputation(clients)
        while best_client in self.rotation:
            clients.pop(best_client)
            best_client = self.highest_reputation(clients)
        #update rotation list
        if len(self.rotation) >= self.ROTATION_SIZE and len(self.rotation) != 0:
            self.rotation.pop(0)
        self.rotation.append(best_client)
        return best_client

    def choose_new_aggregator(self, metrics, id_client):
        self.recalculate_client(metrics, id_client)
        return self.choose()
    
    def print_reputation(self):
        for client, rep in self.client_dict.items():
            print(f'client {client} : reputation {rep}')

