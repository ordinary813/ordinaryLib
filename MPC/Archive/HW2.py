import pandas as pd
import numpy as np


def shiftRows(arr_2d,r):
    return np.roll(arr_2d, r, axis=0)

def shiftCols(arr_2d,c):
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
        return (self.r, self.matrixA)
    
    def RandB(self):
        return (self.c, self.matrixB)

class Alice:
    def __init__(self,input, dealerOut):
        self.input = input
        self.r = dealerOut[0]
        self.matrix = dealerOut[1]
        
    def Send(self):
        self.u = (self.input + self.r) % 4
        return self.u
        
    def Receive(self, bobOut):
        self.output = np.bitwise_xor(self.matrix[self.u][bobOut[0]], bobOut[1])
        
    def Output(self):
        return self.output
            
class Bob:
    def __init__(self,input, dealerOut):
        self.input = input
        self.c = dealerOut[0]
        self.matrix = dealerOut[1]
        
    def Send(self):
        return (self.v, self.zB)
        
    def Receive(self, u):
        self.v = (self.input + self.c) % 4
        self.zB = self.matrix[u][self.v]

print("a and x are values of the range [0-3]")
print("function f(a,x) is 1 when ax>=4 and 0 otherwise")
while True:
    x = int(input("enter x: "))
    a = int(input("enter a: "))
    dealer = Dealer()
    alice = Alice(x, dealer.RandA())
    bob = Bob(a, dealer.RandB())
    bob.Receive(alice.Send())
    alice.Receive(bob.Send())
    z = alice.Output()
    print(z)