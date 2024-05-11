class Reputation:
    def __init__(self, clients) -> None:
        self.client_dict = dict()
        for client in clients:
            id_client = client.get_id()
            self.client_dict[id_client] = 0
        
        self.rotation = list()

    def recalculate_client(self, metrics, client):
        QUALITY = 0.33
        TIME = 0.33
        SUCCES_FAIL = 0.5
        new_reputation = (metrics["quality"] * QUALITY) + (metrics["time"] * TIME) + (metrics["success_fail"] * SUCCES_FAIL)
        self.client_dict[client] = new_reputation

    def highest_reputation(self, clients):
        return max(clients.items(), key=lambda x: x[1])[0]

    def choose(self):
        ROTATION_SIZE = 5
        clients = self.client_dict
        #select the best reputation that has not been in rotation for last 5 rounds
        best_client = self.highest_reputation(clients)
        while best_client in self.rotation:
            best_client = self.highest_reputation(clients.pop(best_client))
        #update rotation list
        if len(self.rotation) >= ROTATION_SIZE:
            self.rotation.pop(0)
        self.rotation.append(best_client)
        return best_client

    def choose_new_aggregator(self, metrics, id_client):
        self.recalculate_client(metrics, id_client)
        return self.choose()