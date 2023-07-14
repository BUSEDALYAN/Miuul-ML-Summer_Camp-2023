print("hello world") #strings
print("Hello AI Era")
print(9) # integer
print(9.2) # float

type(9) #to reach an object's type
type(9.2)
type("Hello")

#Assignmetns and variables

a = 9
a

b= "hello ai era"
b

c= 10

a*c
d = a-c

## Virtual Environment and Package Management
# sanal ortamların listelenmesi: conda env list

# sanal ortam oluşturma: conda create -n ''isimlendirme''
# To activate this environment, use
#
#     $ conda activate my_env
#
# To deactivate an active environment, use
#
#     $ conda deactivate

# listing installed packages: conda list

# installing new packages: conda install ''packagename'' ''secondpackagename''

# uninstalling packages: conda remove ''packagename''

# installing new packages with a specified version: conda install ''packagename''=''desiredversion''

# upgrading a package: conda upgrade ''packagename''
# upgrading all packages: conda upgrade -all


##pip: pypi (python package index): package management tool

# installing a package: pip install ''packagename''
# installing new packages with a specified version: pip install ''packagename''==''desiredversion''
# dependence management: if we have an already installed package and then we install the same package
# with a specified version, it uninstalls the first package by itself. We don't have to remove a package first
# and install specified version.

# to create a list to share or save all the packages: conda env export > environment.yaml
# removing an environment after we export all packages, then creating a new one by using exported
# environment: conda env create -f environment.yaml