from reader import read_net
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import warnings
from modularity import modularity

warnings.simplefilter('ignore')
file_name = "net.in"
net = read_net(file_name)
communities = [1,1,1,2,2, 1]

A = np.matrix(net["mat"])
G=nx.from_numpy_matrix(A)
pos = nx.spring_layout(G)  # compute graph layout
plt.figure(figsize=(4, 4))  # image is 8 x 8 inches
nx.draw_networkx_nodes(G, pos, node_size = 600, cmap = plt.cm.RdYlBu, node_color = communities)
nx.draw_networkx_edges(G, pos, alpha = 0.3)
plt.show()

print(modularity(communities, net))

no_nodes = net["no_nodes"]
