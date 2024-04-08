import numpy as np
import tqdm
import torch
import queue


class CNode:
    def __init__(
        self,
        nodeid=0,
        childPoint=[[]] * 8,
        parent=0,
        oct=0,
        pos=np.array([0, 0, 0]),
        octant=0,
        morton_pos=[],
    ) -> None:
        self.nodeid = nodeid
        self.childPoint = childPoint.copy()
        self.parent = parent
        self.oct = oct  # occupancyCode 1~255
        self.pos = pos
        self.octant = octant  # 1~8
        self.morton_pos = morton_pos


class COctree:
    def __init__(self, node=[], level=0) -> None:
        self.node = node.copy()
        self.level = level


def dec2bin(n, count=8):
    """returns the binary of integer n, using count number of digits"""
    return [int((n >> y) & 1) for y in range(count - 1, -1, -1)]


def dec2binAry(x, bits=8):
    mask = np.expand_dims(2 ** np.arange(bits - 1, -1, -1), 1).T
    return (np.bitwise_and(np.expand_dims(x, 1), mask) != 0).astype(int)


def dec2bin_ary_torch(x, bits=8):
    mask = (2 ** torch.arange(bits - 1, -1, -1)[True]).to(x.device)
    return (torch.bitwise_and(x[:, True], mask) != 0).int()


def bin2decAry(x):
    if x.ndim == 1:
        x = np.expand_dims(x, 0)
    bits = x.shape[1]
    mask = np.expand_dims(2 ** np.arange(bits - 1, -1, -1), 1)
    return x.dot(mask).astype(int)


def Morton(A):
    A = A.astype(int)
    n = np.ceil(np.log2(np.max(A) + 1)).astype(int)
    x = dec2binAry(A[:, 0], n)
    y = dec2binAry(A[:, 1], n)
    z = dec2binAry(A[:, 2], n)
    m = np.stack((x, y, z), 2)
    m = np.transpose(m, (0, 2, 1))
    mcode = np.reshape(m, (A.shape[0], 3 * n), order="F")
    return mcode


def DeOctree(Codes):
    Codes = np.squeeze(Codes)
    occupancyCode = np.flip(dec2binAry(Codes, 8), axis=1)
    codeL = occupancyCode.shape[0]
    N = np.ones((30), int)
    codcal = 0
    L = 0
    while codcal + N[L] <= codeL:
        L += 1
        try:
            N[L + 1] = np.sum(occupancyCode[codcal : codcal + N[L], :])
        except:
            assert 0
        codcal = codcal + N[L]
    Lmax = L
    Octree = [COctree() for _ in range(Lmax + 1)]
    proot = [np.array([0, 0, 0])]
    Octree[0].node = proot
    codei = 0
    for L in range(1, Lmax + 1):
        childNode = []  # the node of next level
        for currentNode in Octree[L - 1].node:  # bbox of currentNode
            code = occupancyCode[codei, :]
            for bit in np.where(code == 1)[0].tolist():
                newnode = currentNode + (
                    np.array(dec2bin(bit, count=3)) << (Lmax - L)
                )  # bbox of childnode
                childNode.append(newnode)
            codei += 1
        Octree[L].node = childNode.copy()
    points = np.array(Octree[Lmax].node)
    return points


def gen_K_parent_seq(octree, K):
    LevelNum = len(octree)
    nodeNum = octree[-1].node[-1].nodeid
    Seq = np.ones((nodeNum + 1, K), "int") * 256
    LevelOctant = np.zeros((nodeNum + 1, K, 2), "int")  # Level and Octant
    Pos = np.zeros((nodeNum + 1, K, 3), "int")  # padding 0
    ChildID = [[] for _ in range(nodeNum)]
    Seq[1, K - 1] = octree[0].node[0].oct
    LevelOctant[0, K - 1, 0] = 1
    LevelOctant[0, K - 1, 1] = 1
    Pos[0, K - 1, :] = octree[0].node[0].pos
    octree[0].node[0].parent = 0  # set to 1
    n = 0
    for L in range(0, LevelNum):
        for node in octree[L].node:
            Seq[n + 1, K - 1] = node.oct
            Seq[n + 1, 0 : K - 1] = Seq[node.parent, 1:K]
            LevelOctant[n + 1, K - 1, :] = [L + 1, node.octant]
            LevelOctant[n + 1, 0 : K - 1] = LevelOctant[node.parent, 1:K, :]
            Pos[n + 1, K - 1] = node.pos
            Pos[n + 1, 0 : K - 1, :] = Pos[node.parent, 1:K, :]
            if n == 0:
                Seq[n + 1, 0 : K - 1] = 256
                LevelOctant[n + 1, 0 : K - 1] = 0
                Pos[n + 1, 0 : K - 1, :] = 0
            if L == LevelNum - 1:
                pass
            n += 1
    assert n == nodeNum
    DataStruct = {
        "Seq": Seq[1:],
        "Level": LevelOctant[1:],
        "ChildID": ChildID,
        "Pos": Pos[1:],
    }
    return DataStruct


def get_pos(code, Lmax):
    if len(code) == 0:
        return np.array([0, 0, 0])
    mask = np.expand_dims(2 ** np.arange(Lmax - 1, -1, -1), 1).T
    code = np.unpackbits(code.astype(np.ubyte)).reshape(code.shape[0], -1)[:, -3:]
    return np.matmul(mask[:, :len(code)], code)


def GenOctree(points, Lmax=None):
    Codes = []
    mcode = Morton(points)
    if Lmax is None:
        Lmax = np.ceil(mcode.shape[1] / 3).astype(int)
    pointNum = mcode.shape[0]

    mcode2 = np.zeros((pointNum, Lmax))
    for n in range(Lmax):
        mcode2[:, n : n + 1] = bin2decAry(mcode[:, n * 3 : n * 3 + 3])

    pointID = list(range(0, pointNum))
    nodeid = 0
    proot = CNode(childPoint=[np.array(pointID)])
    Octree = [COctree() for _ in range(Lmax + 1)]
    Octree[0].node = [proot]
    for L in tqdm.trange(1, Lmax + 1):
        Octree[L].level = L
        for node in Octree[L - 1].node:
            for octant, ptid in enumerate(node.childPoint):
                if len(ptid) == 0:
                    continue
                nodeid += 1
                Node = CNode(nodeid=nodeid, parent=node.nodeid, pos=get_pos(mcode2[ptid[0], :L-1], Lmax))
                idn = mcode2[ptid, L - 1]
                for i in range(8):
                    Node.childPoint[i] = ptid[idn == i]
                occupancyCode = np.in1d(np.array(range(7, -1, -1)), idn).astype(int)
                Node.oct = int(bin2decAry(occupancyCode))
                Node.octant = octant + 1
                Codes.append(Node.oct)
                Octree[L].node.append(Node)
    del Octree[0]
    return Codes, Octree, Lmax


def mullevel_gen_octree(points, Lmax=None, morton_path=[0]):
    morton_path = np.array(morton_path)
    Codes = []
    mcode = Morton(points)
    idxs = np.where((mcode[:, :3*len(morton_path):3] == morton_path).sum(-1) == len(morton_path))[0]
    mcode = mcode[idxs]
    points = points[idxs]
    if Lmax is None:
        Lmax = np.ceil(mcode.shape[1] / 3).astype(int)
    pointNum = mcode.shape[0]

    mcode2 = np.zeros((pointNum, Lmax))
    for n in range(Lmax):
        mcode2[:, n : n + 1] = bin2decAry(mcode[:, n * 3 : n * 3 + 3])

    pointID = list(range(0, pointNum))
    nodeid = 0
    proot = CNode(childPoint=[np.array(pointID)])
    Octree = [COctree() for _ in range(Lmax + 1)]
    Octree[0].node = [proot]
    for L in tqdm.trange(1, Lmax + 1):
        Octree[L].level = L
        for node in Octree[L - 1].node:
            for octant, ptid in enumerate(node.childPoint):
                if len(ptid) == 0:
                    continue
                nodeid += 1
                Node = CNode(nodeid=nodeid, parent=node.nodeid, pos=get_pos(mcode2[ptid[0], :L-1], Lmax), morton_pos=mcode[ptid[0]][0::3])
                idn = mcode2[ptid, L - 1]
                for i in range(8):
                    Node.childPoint[i] = ptid[idn == i]
                occupancyCode = np.in1d(np.array(range(7, -1, -1)), idn).astype(int)
                Node.oct = int(bin2decAry(occupancyCode))
                Node.octant = octant + 1
                Codes.append(Node.oct)
                Octree[L].node.append(Node)
    del Octree[0]
    return Codes, Octree, Lmax, idxs


def gen_K_parent_seq_mullevel(octree, K):
    LevelNum = len(octree)
    nodeNum = octree[-1].node[-1].nodeid
    Seq = np.ones((nodeNum + 1, K), "int") * 256
    LevelOctant = np.zeros((nodeNum + 1, K, 2), "int")  # Level and Octant
    Pos = np.zeros((nodeNum + 1, K, 3), "int")  # padding 0
    ChildID = [[] for _ in range(nodeNum)]
    Seq[1, K - 1] = octree[0].node[0].oct
    LevelOctant[0, K - 1, 0] = 1
    LevelOctant[0, K - 1, 1] = 1
    Pos[0, K - 1, :] = octree[0].node[0].pos
    octree[0].node[0].parent = 0  # set to 1
    n = 0
    outer_node_ids = []

    for L in range(0, LevelNum):
        for node in octree[L].node:
            if node.morton_pos[0] == 1:
                outer_node_ids.append(n)
            Seq[n + 1, K - 1] = node.oct
            Seq[n + 1, 0 : K - 1] = Seq[node.parent, 1:K]
            LevelOctant[n + 1, K - 1, :] = [L + 1, node.octant]
            LevelOctant[n + 1, 0 : K - 1] = LevelOctant[node.parent, 1:K, :]
            Pos[n + 1, K - 1] = node.pos
            Pos[n + 1, 0 : K - 1, :] = Pos[node.parent, 1:K, :]
            if n == 0:
                Seq[n + 1, 0 : K - 1] = 256
                LevelOctant[n + 1, 0 : K - 1] = 0
                Pos[n + 1, 0 : K - 1, :] = 0
            if L == LevelNum - 1:
                pass
            n += 1
    outer_node_ids = np.array(outer_node_ids)
    # assert n == nodeNum
    DataStruct = {
        "Seq": Seq[1:n],
        "Level": LevelOctant[1:n],
        "ChildID": ChildID,
        "Pos": Pos[1:n],
        "outer": outer_node_ids,
    }
    # DataStruct = {
    #     "Seq": Seq[1:],
    #     "Level": LevelOctant[1:],
    #     "ChildID": ChildID,
    #     "Pos": Pos[1:],
    #     "outer": outer_node_ids,
    # }
    return DataStruct
