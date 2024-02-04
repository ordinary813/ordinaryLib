import pandas as pd
import numpy as np

NUMBER_OF_AND = 24

# returns (r, 'secret' Xor r)
def shr(secret):
    r = np.random.choice([0, 1])
    return [r, r ^ secret]

# reconstructs S with minimum of t shares

def recSelf(secret):
    return secret[0] ^ secret[1]

def rec(x, y):
    return recSelf(x)^recSelf(y)

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

# We will implement the 2-bit Multiplier that was used in HW1
# layer1c2 is defined as 2nd calculation at layer 1 etc.
def twoBitMulti(num1,num2):
    layer1c1 = AND(num1[1],num2[0])
    layer1c2 = AND(num1[1],num2[1]) #LSB
    layer1c3 = AND(num1[0],num2[1]) 
    layer1c4 = AND(num1[0],num2[0])
    layer2c1 = XOR(layer1c1,layer1c3) #2nd bit
    layer2c2 = AND(layer1c1,layer1c3)
    layer3c1 = XOR(layer1c4, layer2c2) #3rd bit
    layer3c2 = AND(layer1c4, layer2c2) #MSB
    return [layer3c2,layer3c1,layer2c1,layer1c2] # The 4-bit calculation from MSB to LSB

# We will implement the 4-bit Adder that was used in HW1
# layer1c2 is defined as 2nd calculation at layer 1 etc.
def fourBitAdder(num1,num2):
    layer1c1 = XOR(num1[3],num2[3]) #LSB
    layer1c2 = AND(num1[3],num2[3]) 
    layer1c3 = XOR(num1[2],num2[2]) 
    layer1c4 = AND(num1[2],num2[2])
    layer1c5 = XOR(num1[1],num2[1]) 
    layer1c6 = AND(num1[1],num2[1]) 
    layer1c7 = XOR(num1[0],num2[0]) 
    layer1c8 = AND(num1[0],num2[0])  
    layer2c1 = XOR(layer1c2,layer1c3) #2nd bit
    layer2c2 = AND(layer1c2,layer1c3)
    layer3c1 = OR(layer2c2,layer1c4)
    layer4c1 = XOR(layer3c1,layer1c5) #3rd bit
    layer4c2 = AND(layer3c1,layer1c5)
    layer5c1 = OR(layer4c2,layer1c6)
    layer6c1 = XOR(layer5c1,layer1c7) #MSB
    layer6c2 = AND(layer5c1,layer1c7)
    layer7c1 = OR(layer6c2,layer1c8) #Carry out
    return [layer7c1,layer6c1,layer4c1,layer2c1,layer1c1]
    
def boolianCircuit(a1,a2,x1,x2):

    # first we preform a1*x1, a2*x2 as specified in the equation
    product1 = twoBitMulti(a1,x1) 
    product2 = twoBitMulti(a2,x2) 

    # now we add the two products to get [CarryOut,b4,b3,b2,b1] Where b4 is MSB
    sum = fourBitAdder(product1,product2)
    
    # as we described in HW1: the sum is greater than 4 (threshold)
    # if at least one of {Carry,b4,b3} is 1
    z = OR(sum[0],OR(sum[1],sum[2]))
    
    # now we reconstruct the secret share of the calculation
    return recSelf(z)

class Dealer:
    def __init__(self):
        uPre = [np.random.choice([0, 1]) for i in range(NUMBER_OF_AND)]
        vPre = [np.random.choice([0, 1]) for i in range(NUMBER_OF_AND)]
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
    def __init__(self, a ,matrix):
        self.input = a
        self.beaverTriples = matrix

        # assumption: a is not greater than 3
        # getting wires for first number
        self.a1 = int(np.binary_repr(a[0],width=2)[1])      # get LSB
        self.a2 = int(np.binary_repr(a[0],width=2)[0])      # get MSB

        # getting wires for second number
        self.a3 = int(np.binary_repr(a[1],width=2)[1])      # get LSB
        self.a4 = int(np.binary_repr(a[1],width=2)[0])      # get MSB

class Bob:
    def __init__(self, x, matrix):
        self.input = x
        self.beaverTriples = matrix
        
        # assumption: y is not greater than 3
        # getting wires for first number
        self.x1 = int(np.binary_repr(x[0],width=2)[1])      # get LSB
        self.x2 = int(np.binary_repr(x[0],width=2)[0])      # get MSB

        # getting wires for second number
        self.x3 = int(np.binary_repr(x[1],width=2)[1])      # get LSB
        self.x4 = int(np.binary_repr(x[1],width=2)[0])      # get MSB


for a1 in range(4):
    for a2 in range(4):
        for x1 in range(4):
            for x2 in range(4):
                a = [a1,a2]
                x = [x1,x2]
                print(f'for a = {a}, x = {x}\n{a1} * {x1} + {a2} * {x2}')
                dealer = Dealer()
                alice = Alice(a, dealer.RandA())
                bob = Bob(x, dealer.RandB())

                # alice sends her secret sharings of her 2 numbers, same as bob,
                # the circuit computes equation 3
                secure = boolianCircuit([shr(alice.a2),shr(alice.a1)],
                            [shr(alice.a4),shr(alice.a3)],
                            [shr(bob.x2),shr(bob.x1)],
                            [shr(bob.x4),shr(bob.x3)])
                
                # test the boolean circuit
                if(a1*x1 + a2*x2 >= 4):
                    comp = 1
                else:
                    comp = 0

                print(f'secure computation worked: {comp == secure}\n')