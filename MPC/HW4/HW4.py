import numpy as np
import sys

NUMBER_OF_AND = 24
NUMBER_OF_TOUHCED_AND = 8


# returns (r, 'secret' Xor r)
def shr(value):
    r = np.random.choice([0, 1])
    return [r, r ^ value]

# reconstructs S with minimum of t=2 shares

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
    
    # z = [w] XOR (e * [x]) XOR (d* [y]) XOR (e*d)
    z = XOR(dealer.w[dealer.counter],ANDconst(x,e))
    z = XOR(z,ANDconst(y,d))
    z = XORconst(z, e*d)
    dealer.counter += 1
    
    return z

def OR(x, y):
    return XOR(XOR(x,y),AND(x,y))

# We will implement the 2-bit Multiplier that was used in HW1
# layer1c2 is defined as 2nd calculation at layer 1 etc.
def twoBitMulti(secret1,secret2):
    layer1c1 = AND(secret1[1],secret2[0])
    # check(ver(tagAliceA[0],))
    # check(ver(tagAliceX[0],))
    
    layer1c2 = AND(secret1[1],secret2[1]) #LSB
    # check(ver(tagAliceA[1],))
    # check(ver(tagAliceX[1],))
    
    layer1c3 = AND(secret1[0],secret2[1])
    # check(ver(tagAliceA[2],))
    # check(ver(tagAliceX[2],))
    
    layer1c4 = AND(secret1[0],secret2[0])
    # check(ver(tagAliceA[3],))
    # check(ver(tagAliceX[3],))
    
    layer2c1 = XOR(layer1c1,layer1c3) #2nd bit
    layer2c2 = AND(layer1c1,layer1c3)
    layer3c1 = XOR(layer1c4, layer2c2) #3rd bit
    layer3c2 = AND(layer1c4, layer2c2) #MSB
    return [layer3c2,layer3c1,layer2c1,layer1c2] # The 4-bit calculation from MSB to LSB

# We will implement the 4-bit Adder that was used in HW1
# layer1c2 is defined as 2nd calculation at layer 1 etc.
def fourBitAdder(secret1,secret2):
    layer1c1 = XOR(secret1[3],secret2[3]) #LSB
    layer1c2 = AND(secret1[3],secret2[3]) 
    layer1c3 = XOR(secret1[2],secret2[2]) 
    layer1c4 = AND(secret1[2],secret2[2])
    layer1c5 = XOR(secret1[1],secret2[1]) 
    layer1c6 = AND(secret1[1],secret2[1]) 
    layer1c7 = XOR(secret1[0],secret2[0]) 
    layer1c8 = AND(secret1[0],secret2[0])  
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
    
def boolianCircuit(a1SecretVec,a2SecretVec,x1SecretVec,x2SecretVec):
    # a1SecretVec = [ s(MSB of a1), s(LSB of a1)]
    # a2SecretVec = [ s(MSB of a2), s(LSB of a2)]
    # ... for x
    # first we preform a1*x1, a2*x2 as specified in the equation
    product1 = twoBitMulti(a1SecretVec,x1SecretVec) 
    product2 = twoBitMulti(a2SecretVec,x2SecretVec) 

    # now we add the two products to get [CarryOut,b4,b3,b2,b1] Where b4 is MSB
    sum = fourBitAdder(product1,product2)
    
    # as we described in HW1: the sum is greater than 4 (threshold)
    # if at least one of {Carry,b4,b3} is 1
    z = OR(sum[0],OR(sum[1],sum[2]))
    
    # now we reconstruct the secret share of the calculation
    return recSelf(z)

class Dealer:
    def __init__(self):
        # initialize 3 vectors of size NUMBER_OF_AND
        # u,v random - 0's or 1's
        # w is u[i]*v[i] mod 2, no need for mod 2 since its a binary circuit
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
        for i in range(NUMBER_OF_AND):
            arr.append([self.u[i][0],self.v[i][0],self.w[i][0]])
        return arr
    
    # same as RandA but RandB returns uB,vB,wB vectors
    def RandB(self):
        arr = []
        for i in range(NUMBER_OF_AND):
            arr.append([self.u[i][1],self.v[i][1],self.w[i][1]])
        return arr
    
class Alice:
    def __init__(self, a ,matrix):
        self.input = a
        # set number of keys for every and touched by the parties
        self.kArr = [gen() for _ in range(NUMBER_OF_TOUHCED_AND)]
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
        # set number of keys for every and touched by the parties
        self.kArr = [gen() for _ in range(NUMBER_OF_TOUHCED_AND)]
        self.beaverTriples = matrix
        
        # assumption: y is not greater than 3
        # getting wires for first number
        self.x1 = int(np.binary_repr(x[0],width=2)[1])      # get LSB
        self.x2 = int(np.binary_repr(x[0],width=2)[0])      # get MSB

        # getting wires for second number
        self.x3 = int(np.binary_repr(x[1],width=2)[1])      # get LSB
        self.x4 = int(np.binary_repr(x[1],width=2)[0])      # get MSB

# ---------------------- HW4 ---------------------------- #
def gen():
    return np.random.choice([0, 1]), np.random.choice([0, 1])
    
def tag(a, b, x):
    return  a*x + b

def ver(tag, a, b, x):
    return (tag == a*x + b)

def check(bool):
    if(bool):
        sys.exit("AND gate failed to authenticate.")
# ------------------------------------------------------- #
    
print("enter values for the equation:\na1*x1 +a2*x2")
a1 = int(input("a1 = "))
x1 = int(input("x1 = "))
a2 = int(input("a2 = "))
x2 = int(input("x2 = "))

a = [int(a1),int(a2)]
x = [int(x1),int(x2)]

print(f'for a = {a}, x = {x}\n{a1} * {x1} + {a2} * {x2}')
dealer = Dealer()
# initialize alice with beaver triples
alice = Alice(a, dealer.RandA())
# initialize bob with beaver triples
bob = Bob(x, dealer.RandB())

# vector of secret sharings of alice's values
sharesA = []
sharesA.append(shr(alice.a1))           # s(first bit of LSB of a1)
sharesA.append(shr(alice.a2))           # s(second bit of MSB of a1)
sharesA.append(shr(alice.a3))           # ...
sharesA.append(shr(alice.a4))

# vector of secret sharings of bob's values
sharesX = []
sharesX.append(shr(bob.x1))
sharesX.append(shr(bob.x2))
sharesX.append(shr(bob.x3))
sharesX.append(shr(bob.x4))

# vector of what alice holds of the secret sharings i.e the first cell
secretAliceA = []
for i in range(int(NUMBER_OF_TOUHCED_AND/2)):
    secretAliceA.append(sharesA[i][0])
secretAliceX = []
for i in range(int(NUMBER_OF_TOUHCED_AND/2)):
    secretAliceX.append(sharesX[i][0])
    
# vector of what bob holds of the secret sharings i.e the second cell
secretBobA = []
for i in range(int(NUMBER_OF_TOUHCED_AND/2)):
    secretBobA.append(sharesA[i][1])
secretBobX = []
for i in range(int(NUMBER_OF_TOUHCED_AND/2)):
    secretBobX.append(sharesX[i][1])

# compute the tags for every k, kArr = 2d vector of random 0's or 1's
tagAliceA = [tag(alice.kArr[i][0], alice.kArr[i][1], secretAliceA[i]) for i in range(int(NUMBER_OF_TOUHCED_AND/2))]
tagAliceX = [tag(alice.kArr[i+4][0], alice.kArr[i+4][1], secretAliceX[i]) for i in range(int(NUMBER_OF_TOUHCED_AND/2))]

# compute the tags for every k
tagBobA = [tag(bob.kArr[i][0], bob.kArr[i][1], secretBobA[i]) for i in range(int(NUMBER_OF_TOUHCED_AND/2))]
tagBobX = [tag(bob.kArr[i+4][0], bob.kArr[i+4][1], secretBobX[i]) for i in range(int(NUMBER_OF_TOUHCED_AND/2))]

# alice sends her secret sharings of her 2 numbers, same as bob,
# the circuit computes equation 3
# secure = boolianCircuit(
#             [shr(alice.a2),shr(alice.a1)],
#             [shr(alice.a4),shr(alice.a3)],
#             [shr(bob.x2),shr(bob.x1)],
#             [shr(bob.x4),shr(bob.x3)])

# each row is a vector of 2 secret shares - or 2 bits.
secure = boolianCircuit(
            [sharesA[1],sharesA[0]],
            [sharesA[3],sharesA[2]],
            [sharesX[1],sharesX[0]],
            [sharesX[3],sharesX[2]])

# test the boolean circuit
if(a1*x1 + a2*x2 >= 4):
    comp = 1
else:
    comp = 0

print(f'{comp}, {secure}: {comp == secure}\n')