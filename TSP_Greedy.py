def readfile(file_name):
    rez = []
    reader = open(file_name, 'r')
    nr = int(reader.readline())
    rez.append(nr)
    arr = []
    for i in range(nr):
        line = reader.readline()
        line = line.strip()
        arr.append([int(num) for num in line.split(',')])
    rez.append(arr)
    start = int(reader.readline())
    finish = int(reader.readline())
    rez.append(start)
    rez.append(finish)
    reader.close()
    return rez


def paths(data, start):
    path = [start+1]
    curent = start
    poz = 0
    dist = 0
    min_val = 0
    visited = {start: 1}
    for j in range(len(data)-1):
        for i in range(len(data)):
            if i not in visited:
                min_val = data[curent][i]
                poz = i
                break
        for i in range(len(data)):
            if i not in visited and data[curent][i] < min_val:
                poz = i
                min_val = data[curent][i]
        curent = poz
        dist = dist + min_val
        nod = curent + 1
        path.append(nod)
        visited[curent] = 1
    dist = dist + data[path[0]-1][path[len(path)-1]-1]
    return path, dist


def rezolva(matrix, start, stop):
    arr = []
    for i in range(len(matrix)):
        arr.append(paths(matrix, i))

    rez = arr[0][0]
    min_dist = arr[0][1]
    if start == stop:
        for i in arr:
            if i[1] < min_dist:
                rez = i[0]
                min_dist = i[1]
        return rez, min_dist
    else:
        min_dist = arr[0][1]
        first = 0
        rez = []

        for tuplu in arr:
            aux = []
            min_dist_aux = 0
            lista = tuplu[0]
            for j in range(len(lista)):
                if lista[j] == start or lista[j] == stop:
                    first = j
                    break
            if lista[first] == start:
                aux.append(start)
                for j in range(first + 1, len(lista)):
                    if lista[j] != stop:
                        aux.append(lista[j])
                        min_dist_aux = min_dist_aux + matrix[lista[j]-1][lista[j-1]-1]
                    else:
                        aux.append(lista[j])
                        min_dist_aux = min_dist_aux + matrix[lista[j]-1][lista[j - 1]-1]
                        break
            else:
                aux.append(stop)
                for j in range(first + 1, len(lista)):
                    if lista[j] != start:
                        aux.append(lista[j])
                        min_dist_aux = min_dist_aux + matrix[lista[j]-1][lista[j - 1]-1]
                    else:
                        aux.append(lista[j])
                        min_dist_aux = min_dist_aux + matrix[lista[j]-1][lista[j - 1]-1]
                        break
            if min_dist_aux < min_dist:
                rez = aux
                min_dist = min_dist_aux
        if rez[0] == stop:
            rez.reverse()
        return rez, min_dist


def write_to_file(size1, arr1, value1, size2, arr2, value2):
    writer = open("out.txt", "w")
    writer.write(str(size1))
    writer.write("\n")
    writer.write(str(arr1))
    writer.write("\n")
    writer.write(str(value1))
    writer.write("\n")
    writer.write(str(size2))
    writer.write("\n")
    writer.write(str(arr2))
    writer.write("\n")
    writer.write(str(value2))
    writer.write("\n")
    writer.close()


def main():
    file_name = "in.txt"
    data = readfile(file_name)
    rez1 = rezolva(data[1], 0, 0)
    rez2 = rezolva(data[1], data[2], data[3])
    write_to_file(len(rez1[0]), rez1[0], rez1[1], len(rez2[0]), rez2[0], rez2[1])
    print(rezolva(data[1], data[2], data[3]))


main()
