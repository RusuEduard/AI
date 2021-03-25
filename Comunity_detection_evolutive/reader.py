def read_net(file_name):
    f = open(file_name, "r")
    net = {}
    n = int(f.readline())
    net["no_nodes"] = n
    mat = []
    for i in range(n):
        mat.append([])
        line = f.readline()
        elems = line.split(" ")
        for j in range(n):
            mat[-1].append(int(elems[j]))
    net["mat"] = mat
    degrees = []
    no_edges = 0
    for i in range(n):
        d = 0
        for j in range(n):
            if mat[i][j] == 1:
                d += 1
            if j > i:
                no_edges += mat[i][j]
        degrees.append(d)
    net["no_edges"] = no_edges
    net["degrees"] = degrees
    f.close()
    return net
