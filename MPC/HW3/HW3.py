import pandas as pd
import numpy as np

# returns (r, 'secret' Xor r)
def shr(secret, n=2):
    r = np.random.choice([0, 1])
    return [r, r ^ secret]

# reconstructs S with minimum of t shares
def recself(secret):
    return secret[0] ^ secret[1]

def rec(x, y):
    return recself(x)^recself(y)

# secret = (Xa,Xb), secret shares of a secret X
def openTo(secret):
    return secret[0] ^ secret[1]

def XOR(x, y):
    return [x[0]^y[0],x[1]^y[1]]

def XORconst(x, c):
    return [x[0]^c,x[1]]

def ANDconst(x, c):
    return [x[0]*c, x[1]*c]

def AND(x, y):
    d = openTo(XOR(x, dealer.u[dealer.counter]))
    e = openTo(XOR(y, dealer.v[dealer.counter]))
    
    z = XOR(dealer.w[dealer.counter],ANDconst(x,e))
    z = XOR(z,ANDconst(y,d))
    z = XORconst(z, e*d)
    dealer.counter += 1
    return z

def OR(x, y):
    return XOR(XOR(x,y),AND(x,y))

class Dealer:
    def __init__(self):
        uPre = [np.random.choice([0, 1]) for i in range(22)]
        vPre = [np.random.choice([0, 1]) for i in range(22)]
        wPre = [uPre[i]*vPre[i] for i in range(len(vPre))]

        self.counter = 0
        self.u = []
        self.v = []
        self.w = []

        for i in range(len(vPre)):
            self.u.append(shr(uPre[i]))
            self.v.append(shr(vPre[i]))
            self.w.append(shr(wPre[i]))

    # retruns uA,vA,wA vector for each AND gate in the circuit for Alice 
    # each row in arr is a vector of the shape (uA,vA,wA)
    def RandA(self):
        arr = []
        for i in range(len(self.u)):
            arr.append([self.u[i][0],self.v[i][0],self.w[i][0]])
        return arr
    
    # same as RandA but RandB returns uB,vB,wB vectors
    def RandB(self):
        arr = []
        for i in range(len(self.u)):
            arr.append([self.u[i][1],self.v[i][1],self.w[i][1]])
        return arr
    

class Alice:
    def __init__(self, x ,matrix):
        self.input = x
        self.beaverTriples = matrix

        # assumption: x is not greater than 3
        # getting wires for first number
        self.x1 = int(np.binary_repr(x[0],width=2)[1])      # get LSB
        self.x2 = int(np.binary_repr(x[0],width=2)[0])      # get MSB

        # getting wires for second number
        self.x3 = int(np.binary_repr(x[1],width=2)[1])      # get LSB
        self.x4 = int(np.binary_repr(x[1],width=2)[0])      # get MSB

class Bob:
    def __init__(self, y, matrix):
        self.input = y
        self.beaverTriples = matrix
        
        # assumption: x is not greater than 3
        # getting wires for first number
        self.y1 = int(np.binary_repr(y[0],width=2)[1])      # get LSB
        self.y2 = int(np.binary_repr(y[0],width=2)[0])      # get MSB

        # getting wires for second number
        self.y3 = int(np.binary_repr(y[1],width=2)[1])      # get LSB
        self.y4 = int(np.binary_repr(y[1],width=2)[0])      # get MSB


x = [1,2]   # x = [x2x1, x4x3] = [01,10]
y = [3,1]   # y = [y2y1, y4y3] = [11,01]

dealer = Dealer()
alice = Alice(x, dealer.RandA())
bob = Bob(y, dealer.RandB())

share1 = shr(1)
share2 = shr(1)
a = OR(share1,share2)
print(recself(a))


