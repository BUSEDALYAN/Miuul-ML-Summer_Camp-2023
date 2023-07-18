###  COMPREHENSIONS

# - List Comprehension

salaries =[1000,2000,3000,4000,5000]

def new_salary(x):
    return x * 20 / 100 + x

for salary in salaries:
    print(new_salary(salary))


null_list = []

for salary in salaries:
    null_list.append(new_salary(salary))

for salary in salaries:
    if salary > 3000:
        null_list.append(new_salary(salary))
    else:
        null_list.append(new_salary(salary * 2))

[new_salary(salary * 2) if salary <3000 else new_salary(salary) for salary in salaries]

[salary * 2 for salary in salaries]

students = ["John", "Mark", "Venessa", "Mariam"]
students_no = ["John", "Venessa"]

[student.lower() if student in students_no else student.upper() for student in students]

[student.upper() if student not in  students_no else student.lower() for student in students]

# - Dict Comprehension

dict ={'a':1,'b':2,'c':3,'d':4}

dict.keys()
dict.values()
dict.items()


{k: v**2 for (k, v) in dict.items()}

{k.upper(): v for (k, v) in dict.items()}

{k.upper(): v*2 for (k, v) in dict.items()}

#############################################
### EXERCISE: take the square of numbs and then add a dictionary.

numbers = range(10)
new_dict = {}

for n in numbers:
    if n % 2 == 0:
        new_dict[n] = n**2

{n : n**2 for n in numbers if n % 2 == 0}


###List & Dictionary Comprehension Application

## 1. Changing the names in a dataset

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns
A =[]
for col in df.columns:
    A.append(col.upper())

df.columns = A

# w/ comprehension

df = sns.load_dataset("car_crashes")

df.columns = [col.upper() for col in df.columns]

## 2. if a name contains "INS" add FLAG to its beginning, otherwise NO_FLAG

["FLAG_"+ col for col in df.columns if "INS" in col ]

["FLAG_"+ col if "INS" in col else "NO_FLAG_" + col for col in df.columns]

# 3. creating a dict

df = sns.load_dataset("car_crashes")
df.columns

num_cols =[col for col in df.columns if df[col].dtype != "O"] # non-categoric

dict = {}
agg_list ={"mean","min","max","sum"}

for col in num_cols:
    dict[col] = agg_list

# w/ comprehension
new_dict = {col: agg_list for col in num_cols}

df[num_cols].head()

df[num_cols].agg(new_dict)

## QUIZ

###########################
names = ["denise", "jean", "fleur"]
ages = [ 20, 32,45]
cities = [ "lyon", "lille","nantes"]

list(zip(names,ages,cities))

############################
wages = [1000,2000,3000,4000,5000]
new_wages = lambda x: x*0.20 + x

list(map(new_wages, wages))

############################
students = ["Denise","Arsen","Tony","Audrey"]
low = lambda x: x[0].lower()
print(list(map(low, students)))


############################

dictn = {"Denise":10,"Arsen": 12,"Tony": 15,"Audrey":17 }
new_dict = {k: v * 2 + 3 for (k, v) in dictn.items()}

############################
numbs = range(1,10)
{n: n ** 2 for n in numbs if n %2 != 0  }