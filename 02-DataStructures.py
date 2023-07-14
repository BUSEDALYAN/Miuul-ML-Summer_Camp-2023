## DATA STRUCTURES

## - Introduction & Summary

### list, tuple, set and dictionary --> Python Collections (Arrays)

# - Numbers: int, float, complex

x = 46
type(x)


x = 46.1
type(x)

x = 2j+1
type(x)

a = 5
b = 10.5

a * b / 10
a**3

int(b) # float --> integer
float(a) # integer --> float


# - Strings: str

x = "Hello ai era"
type(x)

print("hello")
print('hello') # same

long_str = """Data Structures: Summary,
Numbers: int, float, complex"""

name = "John"
name[0] #indexs begin with 0
name[0:2] #slice
"Data" in long_str

dir(int) # to find the methods that can be used

###len
name = "john"
type(len)
len(name)

###upper() & lower()
"miuul".upper()
"MIUUL".lower()
# if a function is defined in class structure--> method
type(upper())

###replace
hi ="hello"
hi.replace("l","p")

###split
"Hello AI Era".split()

##strip
"ofofo".strip("o")

###capitalize
"foo foo".capitalize()

# - Boolean(TRUE-FALSE): bool

True
False
type(True)

5==4
type(5==4)
type(5==4)

# - Lists

x=["b","u","s","e"]
type(x)

notes=[1,2,3,4]

not_nam = [1,2,3,"a","b",True, [1,2,3]]
not_nam[6]
not_nam[6][1]

notes[0] = 99

not_nam[0:4]

len(not_nam)

###append
notes.append(100) ## adds a new argument

###pop
notes.pop(0) ## deletes according to index

###insert
notes.insert(2, 99)

# - Dictionary
        # key-value
        # they can be changed
        # not in ordered (after 3.7 --> ordered)

x={"name": "Peter", "Age":36} ##name --> key, age --> value
type(x)

dict = {"REG":"regression",
        "LOG": "logistic regression",
        "CART":"classification and Reg"}
dict["REG"]

"REG" in dict

dict.get("REG")

dict["REG"] = ["YSA",10] #change a value by giving its key

dict.keys()
dict.values()

dict.items()

dict.update({"REG": 11})

dict.update({"RF":10}) ## adds a new key-value
# - Tuple
        ##they can not be changed.
        ##ordered

x=("b","u","s","e")
type(x)

t = ("john","mark",1,2)

### to change an element
t = list(t) ## first convert it to a list
t[0] = 99 ## change the element
t= tuple(t) ## then convert it bact to s tuple


# - Set
        ## they can be changed.
        ## unique - ordered

x={"b","u","s","e"}
type(x)

set1 = set([1,3,5])
set2 = set([1,2,3])

set1.difference(set2)
set1 - set2

set1.symmetric_difference(set2)

set1.intersection(set2)
set1 & set2

set1.union(set2)


set1.isdisjoint(set2)

set3 = set([7,8,9])
set4 = set([5,6,7,8,9,10])

set3.issubset(set4) #subset
set4.issubset(set3)

set3.issuperset(set4)
set4.issuperset(set3)

a = 9** (1/2)
b= 10/5

a-b