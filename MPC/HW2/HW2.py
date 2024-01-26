import pandas as pd
import numpy as np


def shiftRows(arr_2d,r):
    #np.roll "rolls" the given array in a chosen axis a chosen amount of times 
    return np.roll(arr_2d, r, axis=0)

def shiftCols(arr_2d,c):
    #np.roll "rolls" the given array in a chosen axis a chosen amount of times 
    return np.roll(arr_2d, c, axis=1)

class Dealer:
    def __init__(self):
        # set truth table for equation 2, table is going to be a 4x4 numpy array 
        # rows = a's value, column = x's value
        self.table = pd.read_csv('truth_table.csv').to_numpy()
        
        # shifts[0] shifts for rows
        # shifts[1] shifts for columns
        shifts = np.array([np.random.choice(range(1,5)),np.random.choice(range(1,5))])
        
        #assigning shift values to the class' variables
        self.r = shifts[0]
        self.c = shifts[1]
        
        # shift table rows anc columns
        self.table = shiftRows(self.table,shifts[0])
        self.table = shiftCols(self.table,shifts[1])
        
        # randomized matrix the size of 4x4 (like the truth table for equation 2)
        # this matrix will be sent to Bob
        self.matrixB = [[np.random.choice([0, 1]) for _ in range(4)] for _ in range(4)]
        
        # compute matrixA for the dealer to send to Alice
        self.matrixA = np.bitwise_xor(self.table, self.matrixB)

        
    def RandA(self):
        # give Alice (r, Ma)
        return (self.r, self.matrixA)
    
    def RandB(self):
        # give Bob (c,Mb)
        return (self.c, self.matrixB)

class Alice:
    def __init__(self,input, dealerOut):
        self.input = input
        self.r = dealerOut[0]
        self.matrix = dealerOut[1]
        
    def Send(self):
        # Alice sends Bob u = x + r mod 4
        self.u = (self.input + self.r) % 4
        return self.u
        
    def Receive(self, bobOut):
        # Alice gets a message from Bob and calculates z = Ma[u][v] XOR Zb
        self.output = np.bitwise_xor(self.matrix[self.u][bobOut[0]], bobOut[1])
        
    def Output(self):
        return self.output
            
class Bob:
    def __init__(self,input, dealerOut):
        self.input = input
        self.c = dealerOut[0]
        self.matrix = dealerOut[1]
        
    def Send(self):
        # Bob sends Alice (v,Zb)
        return (self.v, self.zB)
        
    def Receive(self, u):
        # Bob gets message from Alice and calculates v = y + c mod 4, zB = Mb[u][v]
        self.v = (self.input + self.c) % 4
        self.zB = self.matrix[u][self.v]

arr = [[0,0,0,0],
       [0,0,0,0],
       [0,0,0,0],
       [0,0,0,0]]

for a in range(0,4):
    for x in range(0,4):
        # -------------------------- TESTS -------------------------- #
        dealer = Dealer()
        alice = Alice(x, dealer.RandA())
        bob = Bob(a, dealer.RandB())
        bob.Receive(alice.Send())
        alice.Receive(bob.Send())
        z = alice.Output()
        # ----------------------------------------------------------- #
        print("for a =",a, ", x =",x,", secure f(x,a) =",z)
        arr[a][x] = z

print("\nall in all, secure f will look like this:\n")
for i in range(len(arr)):
    print(arr[i])
print("a is rows, x is columns")