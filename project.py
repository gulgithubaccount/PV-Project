import numpy as np

def linguistic_numerical(DE):
    """
    Transfer a linguistic preference matrix to a numerical one
    DE = [
        [M, G, VG, G],
        [],
        []
    ]
    """
    n = len(DE)
    m = len(DE[0])
    DE_numeriacl = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            if DE[i][j] == "AG":
                DE_numeriacl[i][j] = 1
            elif DE[i][j] == "VG":
                DE_numeriacl[i][j] = 0.85
            elif DE[i][j] == "G":
                DE_numeriacl[i][j] = 0.70
            elif DE[i][j] == "M":
                DE_numeriacl[i][j] = 0.50
            elif DE[i][j] == "P":
                DE_numeriacl[i][j] = 0.40
            elif DE[i][j] == "VP":
                DE_numeriacl[i][j] = 0.20
            else:
                DE_numeriacl[i][j] = 0.00
    return DE_numeriacl


def aggregated_decision_matrix(*DEs):
    AD = DEs[0]
    le = len(DEs)
    for i in range(1, le):
        AD = AD + DEs[i]
    return AD / le


def linguistic_numerical_criteria(C):
    """
    Transfer a linguistic criteria matrix to a numerical one
    C = [
        number of criteria
        [VH, AH, VH], number of the experts
        ...
        []
    ]
    """
    n = len(C)
    m = len(C[0])
    C_numeriacl = np.zeros((n, m, 2, 2))
    for i in range(n):
        for j in range(m):
            if C[i][j] == "AH":
                C_numeriacl[i][j] = [[0.85, 1.0], [0, 0]]
            elif C[i][j] == "VH":
                C_numeriacl[i][j] = [[0.75, 0.95], [0, 0.05]]
            elif C[i][j] == "H":
                C_numeriacl[i][j] = [[0.60, 0.80], [0.05, 0.20]]
            elif C[i][j] == "M":
                C_numeriacl[i][j] = [[0.40, 0.60], [0.15, 0.40]]
            elif C[i][j] == "L":
                C_numeriacl[i][j] = [[0.5, 0.20], [0.60, 0.80]]
            elif C[i][j] == "VL":
                C_numeriacl[i][j] = [[0, 0.05], [0.75, 0.95]]
            else:
                C_numeriacl[i][j] = [[0, 0], [0.85, 1]]
    return C_numeriacl


def aggregated_values(C_numerical):
    return np.mean(linguistic_numerical_criteria(C), axis=1)


def IVIFDWE(reals, IVIFS, epsilon):
    """
    reals: aggregated real numbers
    IVIFS: weights in the form of IVIFS
    epsilon: the customed parameter
    """
    result = np.zeros((2, 2))
    n = len(reals)
    s1 = 0
    s2 = 0
    s3 = 0
    s4 = 0
    for i in range(n):
        s1 = s1 + (1 - IVIFS[i][0][0]) * (((1 - reals[i]) / reals[i]) ** epsilon)
        s2 = s2 + (1 - IVIFS[i][0][1]) * (((1 - reals[i]) / reals[i]) ** epsilon)

        s3 = s3 + IVIFS[i][1][0] * (((1 - reals[i]) / reals[i]) ** epsilon)
        s4 = s4 + IVIFS[i][1][1] * (((1 - reals[i]) / reals[i]) ** epsilon)
    result[0][0] = (1 + s1 ** (1 / epsilon)) ** (-1)
    result[0][1] = (1 + s2 ** (1 / epsilon)) ** (-1)
    result[1][0] = 1 - (1 + s3 ** (1 / epsilon)) ** (-1)
    result[1][1] = 1 - (1 + s4 ** (1 / epsilon)) ** (-1)
    return result


def distance(I1, I2, p, t):
    """
    calculating the distance between two IVIFs I1 and I2
    p: the parameter in the Generalized Distance
    t: cardinality of the set
    """
    I1 = np.array(I1)
    I2 = np.array(I2)
    res = (np.sum(abs(I1 - I2) ** p) ** (1 / p)) / (4 * t)
    return res

# inputs
DE_1 = [
    ["M", "G", "VG", "G"],
    ["VG", "M", "G", "P"],
    ["VP", "P", "G", "G"],
    ["G", "VG", "VG", "G"],
    ["VP", "VP", "P", "M"],
    ["P", "M", "G", "M"]
]

DE_2 = [
    ["G", "P", "M", "M"],
    ["G", "P", "M", "VP"],
    ["VP", "M", "VG", "M"],
    ["VG", "G", "M", "VG"],
    ["P", "VP", "M", "VP"],
    ["VP", "M", "P", "P"]
]

DE_3 = [
    ["VG", "M", "G", "G"],
    ["M", "M", "P", "P"],
    ["VP", "P", "G", "M"],
    ["VG", "G", "G", "M"],
    ["P", "P", "G", "M"],
    ["P", "G", "M", "VP"]
]

C = [
    ["VH", "AH", "VH"],
    ["AH", "VH", "AH"],
    ["AH", "VH", "VH"],
    ["VH", "VH", "AH"],
]

# step 1: transfer the linguistic matrix into their numerical ones
DE_1_numerical = linguistic_numerical(DE_1)
DE_2_numerical = linguistic_numerical(DE_2)
DE_3_numerical = linguistic_numerical(DE_3)

C_numeriacl = linguistic_numerical_criteria(C)

# step 2: obtain AD
AD = aggregated_decision_matrix(DE_1_numerical, DE_2_numerical, DE_3_numerical)

# step 3: obtain the agregated values
values = aggregated_values(C_numeriacl)

# step 4: obtain DV
epsilon = 1
p = 2
n = len(AD)
DV = np.zeros((n,2,2))
for i in range(len(AD)):
    DV[i] = IVIFDWE(AD[i], values, epsilon)

# step 5: calculate the distance
p = 2
M = [[1,1],[0,0]]
N = [[0,0],[1,1]]
t = 1
dis1 = np.zeros((n))
dis2 = np.zeros((n))

for i in range(n):
    dis1[i] = distance(DV[i],M,p,t)
    dis2[i] = distance(DV[i],N,p,t)
dis3 = dis1 / (dis1 + dis2)


print("-----------------AD-----------------")
print(AD)

print("-----------------DV-------------------")
print(DV)

print("-----------------dis1-------------------")
print(dis1)

print("-----------------dis2-------------------")
print(dis2)

print("-----------------dis3-------------------")
print(dis3)