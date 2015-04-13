import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# generating G matrix
def generate_G(graph, source_from, source_to):
	nodes = nx.nodes(graph)
	edges = nx.edges(graph)

	N_nodes = len(nodes)
	N_edges = len(edges)

	G = np.zeros((N_nodes - 1, N_nodes - 1))

	# only elements in the diagonal
	for x in nodes:
		if x == 0: 			# do not consider GND
			continue
		
		connected_nodes = nx.all_neighbors(graph, x)
		conductances = 0
		for i in connected_nodes:
			if (x == source_from and i == source_to) or (x == source_to and i == source_from):
				continue
			conductances += 1.0/(graph.get_edge_data(x, i)['weight'])
		G[x - 1][x - 1] = conductances

	# other, off diagonal elements
	for p in edges:
		if abs(p[0] - p[1]) == (p[0] + p[1]):		# if one node is GND 
			continue
		if (p[0] == source_from and p[1] == source_to) or (p[0] == source_to and p[1] == source_from):
			continue

		conductance = 1.0/graph.get_edge_data(p[0], p[1])['weight']
		G[p[0] - 1][p[1] - 1] = conductance
		G[p[1] - 1][p[0] - 1] = conductance

	return G

# generating B matrix
def generate_B(graph, source_from, source_to):
	N_nodes = nx.number_of_nodes(graph)
	B = np.zeros((N_nodes - 1, 1))

	B[source_from][0] = 1
	B[source_to][0] = -1

	return B

# making C matrix(in our case only transposed B)
def generate_C(B):
	C = B.T

	return C

# generating D after grave computing
def generate_D():
	D = np.zeros(1)
	return D

# combining G, B, C and D matrices into A
def generate_A(graph, source_from, source_to):
	G = generate_G(graph, source_from, source_to)
	B = generate_B(graph, source_from, source_to)
	C = generate_C(B)
	D = generate_D()

	A = np.zeros((map(sum, zip(G.shape, (1, 1)))))

	A[:-1, :-1] = G
	A[:, -1][:-1] = C
	A[-1][:-1] = C

	return A

# generating z matrix
def generate_z(graph):
	N_nodes = nx.number_of_nodes(graph)
	z = np.zeros((N_nodes, 1))

	z[-1] = emf

	return z

def create_graph_from_file(file):
	f = open(file, 'r')
	N = int(f.readline())

	graph = nx.Graph()
	
	for _ in range(N):
		fr, to, res = list(map(int, f.readline().split()))
		graph.add_edge(fr, to, weight=res)

	source_from, source_to, emf = list(map(int, f.readline().split()))

	f.close()

	return graph, source_from, source_to, emf

def print_output(graph, x):
	nodes_out = []
	edges_out = []

	edges = nx.edges(graph)
	for edge in edges:
		R = graph.get_edge_data(edge[0], edge[1])['weight']     # resistance between 2 nodes
		
		U = abs(x[edge[0]] - x[edge[1]]) 	# calculate voltage as potential difference
		
		if R == 0:
			I = 0 		# let's assume so
		else:
			I = float(U) / R 	# from Ohm's law
			
		edges_out.append({'from': edge[0], 'to': edge[1], 'value': I, 'label': R})

	nodes = nx.nodes(graph)
	for node in nodes:
		nodes_out.append({'id': node, 'label': str(node)})

	f = open('data.jsonp', 'w')

	f.write("nodes = " + str(nodes_out) + ";\n")

	f.write("edges = " + str(edges_out) + ";")

	f.close()

graph, source_from, source_to, emf = create_graph_from_file('input.txt')


A = generate_A(graph, source_from, source_to)

z = generate_z(graph)


x = np.linalg.solve(A, z) 

print_output(graph, x)

nx.draw_random(graph)
# plt.show()




