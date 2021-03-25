def modularity(communities, param):
    noNodes = param['no_nodes']
    mat = param['mat']
    degrees = param['degrees']
    noEdges = param['no_edges']
    M = 2 * noEdges
    Q = 0.0
    for i in range(0, noNodes):
        for j in range(0, noNodes):
            if communities[i] == communities[j]:
                Q += (mat[i][j] - degrees[i] * degrees[j] / M)
    return Q * 1 / M
