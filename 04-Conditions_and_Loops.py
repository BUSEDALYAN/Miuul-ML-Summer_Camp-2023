### CONDITIONS

# - if

if 1 == 1:
    print("sth")

if 1 == 2:
    print("sth")


def nmb_check(number):
    if number == 10:
        print("number is 10")


nmb_check(10)


# - else

def nmb_check(number):
    if number == 10:
        print("number is 10")
    else:
        print("number is not 10")


nmb_check(12)


# - elif

def number_check(number):
    if number > 10:
        print("number is greater than 10")
    elif number < 10:
        print("number is smaller than 10")
    else:
        print("number is equal to 10")


number_check(10)

### LOOPS

# - for loop

students = ["John", "Mark", "Venessa", "Mariam"]

for student in students:
    print(student.upper())

salaries = [1000, 2000, 3000, 4000, 5000]

for salary in salaries:
    print(int(salary * 20 / 100 + salary))


def new_salary(salary, rate):
    return int(salary * rate / 100 + salary)


new_salary(1000, 20)

for salary in salaries:
    if salary >= 3000:
        print(new_salary(salary, 10))
    else:
        print(new_salary(salary, 20))


############################
# EXERCISE
# even index = uppercase
# odd index = lowercase

def low_upp(string):
    new_string = ""
    for i in range(len(string)):
        if i % 2 == 0:
            new_string += string[i].upper()
        else:
            new_string += string[i].lower()
    print(new_string)


low_upp("computer")

################################
# - break & continue & while

for salary in salaries:
    if salary == 3000:
        break
    print(salary)

for salary in salaries:
    if salary == 3000:
        continue
    print(salary)

number = 1
while number <5:
    print(number)
    number += 1

# - enumerate

students = ["John", "Mark","Venessa", "Mariam"]

for student in students:
    print(student)

for index, student in enumerate(students, 1):
    print(index, student)

A = []
B = []
for index, student in enumerate(students):
    if index % 2 == 0:
        A.append(student)
    else:
        B.append(student)
    print(A,B)

##################################
# EXERCISE

students = ["John", "Mark","Venessa", "Mariam"]

def divide_students(students):
    groups = [[],[]]
    for index, student in enumerate(students):
        if index %2 ==0:
            groups[0].append(student)
        else:
            groups[1].append(student)
    return groups
divide_students(students)

# -alternating function with enumerate

def alternating_enum(string):
    new_str = ""
    for i, letter in enumerate(string):
        if i % 2 ==0:
            new_str += letter.upper()
        else:
            new_str += letter.lower()
    print(new_str)
alternating_enum("hi my name is john and i am learning python")

# - Zip
students = ["John", "Mark","Venessa", "Mariam"]
deps =["math", "stat", "phys", "astro"]
ages =[23, 30, 26, 22]

list(zip(students, deps, ages))


# lambda, , filter,reduce
def summer(a,b):
    return a + b

summer(1,3) * 9

new_sum = lambda a, b: a+ b
new_sum(4,5)

# -map
salaries =[1000,2000,3000,4000,5000]

def new_salary(x):
    return x*20 / 100 + x

new_salary(5000)

for salary in salaries:
    print(new_salary(salary))

list(map(new_salary, salaries))

list(map(lambda x: x * 20 / 100 + x, salaries))

# - filter
list_store = [ 1,2,3,4,5,6,7,8,9,10]
list(filter(lambda x: x % 2 ==0, list_store))

# - reduce
from functools import reduce

list_store = [1,2,3,4]
reduce(lambda a, b: a+b, list_store)
# QUIZ
wages = [10,20,30,40,50]

new_wages =[]

for w in wages:
    if w <40:
        new_w = w * 1.10
        new_wages.append(new_w)
    else:
        new_wages.append(w)
print(new_wages)

###
wages = [700,800,900,1000]
[wage*1.1 if wage >950 else wage*1.2 for wage in wages]

###
students = ["Denise","Arsen","Tony","Audrey"]
[student[0].upper() if len(student) % 2 != 0 else student[0].lower() for student in students]

###
string = "abracadabra"
group = []

for index, letter in enumerate(string,1):
    if index * 2 %2 ==0:
        group.append(letter)

print(group)

###
city_name =["london","paris","berlin"]

def plate(cities):
    for index, city in enumerate(cities,1):
        print(f"{index} : {city}")
plate(city_name)