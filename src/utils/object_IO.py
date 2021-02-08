import pickle

PATH = './data/trained_networks/'


def save_network(network, name):
    pickle.dump(network, open(PATH + name, 'wb'))


def load_network(name):
    return pickle.load(open(PATH + name, 'rb'))
