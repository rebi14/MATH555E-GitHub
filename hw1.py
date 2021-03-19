# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Question 1
import math
import numpy as np
 

x = int(input('Enter real number:'))

def calculate(x):
    a = float(2*math.sin(x^2) + np.log(abs(x)) + 1)
    print('result :',a)

calculate(x)

#Question 2
a = float(input('enter lowerlimit : '))
b = float(input('enter Upperlimit : '))
n = int(input('enter numberofSubdivisions : '))
def riemann(a, b, n):
    if a > b:
        a,b = b,a
    dx = (b-a)/n
    #n = int((b - a) / dx)
    s = 0.0
    x = a
    for i in range(n):
        f_i = float(x*x)
        s += float(f_i)
        x += dx
    return s * dx  
   
riemann(a,b,n)

#Question 3
import random

def randomArray():
 a = [0.0,1.0]
 for i in range(100):
    b = float(random.random())
    a.append(b)
    c = np.array([a])
 c.sort(axis=1)
 return c

def CalculateDerivative():
 y=randomArray()
 dy=[]
 for i in range(101):
    fark = y[0,i+1] - y[0,i]
    dy.append(fark)
    arr_y = np.array([dy])
 return arr_y
    
#Question 4
def randomArray2():
 a = []
 for i in range(100):
    b = float(random.random())
    a.append(b)
    c = np.array([a])
 c.sort(axis=1)
 return c

dx = randomArray2()
mu, sigma = 0, 0.1 
dy = np.random.normal(mu, sigma, 100)
import matplotlib.pyplot

matplotlib.pyplot.scatter(dx,dy,color='red')

matplotlib.pyplot.show() 

#Question 5
matrix = np.random.standard_normal(size=(100, 100))

