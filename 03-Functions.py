# FUNCTIONS

print("a", "b", sep="_")


## Defining a function

def calculate(x):
    print(x * 2)

calculate(5)

###################
def summer(arg1,arg2):
    print(arg1+arg2)

summer(arg1=7, arg2=8)

##Docstring
def summer(arg1,arg2):
    """sum of two numbers..."""
        print(arg1+arg2)

summer(1,2)

### statement/body

def say_hi(string):
    print("Merhaba")
    print("Hi")
    print("Hello")

say_hi("miuul")


def multiplaction(a,b):
    c = a * b
    print(c)

##stores the elements in a list

list_store = []

def add_element(a,b):
    c = a * b
    list_store.append(c)
    print(list_store)

add_element(180,10)
add_element(100,2)
add_element(2,3)


##Default (Predefined) Parameters/Arguments
def divide(a,b=1):
    print(a/b)

divide(1,2)
divide(3)

##########################################
# RETURN

def calculate(varm, moisture, charge):
    print((varm + moisture) / charge)

calculate(98,12,78) *10 ## ERROR because:
type(calculate(98,12,78)) ## NoneType

def calculate(varm, moisture, charge):
    return (varm + moisture) / charge

calculate(98,12,78) *10


def calculate(varm, moisture, charge):
    varm = varm * 2
    moisture = moisture * 2
    charge = charge * 2
    output = (varm + moisture) / charge

    return varm, moisture, charge, output

varm, moisture, charge, output = calculate(98, 12, 78)

## calling a function inside a func

def calculate(varm, moisture, charge):
    return int((varm + moisture) / charge)

calculate(98,12,12)*10

###############################

def standardization(a, p):
    return a * 10 / 100 * p * p

standardization(45, 1)

def all_calculate(varm, moisture, charge, p):
    a = calculate(varm, moisture, charge)
    b = standardization(a, p)
    print(b * 10)

all_calculate(1,3,5,12)

######################
##local and global variables

list_store = [1, 2]

## here, c is a global vrb
def add_element(a,b):
    c = a * b
    list_store.append(c)
    print(list_store)

add_element(1, 9)

def square(a):
    return a**2

square("hello")

store = []

def add_vrbs(a,b):
    c = a**b
    store.append(c)
    print(store)

add_vrbs(4,3)