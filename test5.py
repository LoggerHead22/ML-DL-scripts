# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 19:31:31 2020

@author: Logge
"""

def foo(x):
  x = x or 42
  return x

y=0
for i in range(11):
  y+=foo(i)
  
print(y)


result = []

base = [1,2,3,4]

for x in base:
  if x > sum(result) /2:
    result.append(x)
    
print(sum(result))

a = [ 2 & 1, 2&2 ,2&3 ,2|1, 2|2 ,2|3, 2^1,2^2,2^3]

result = 0
for x in a:
  result+= x
print(result)


def foo(depth, value):
  if depth:
    foo(depth -1, value*2)
  else:
    print(value)
    
foo(10,1)



s =''

base_sing = 'helloword'

for char in base_sing:
  if char < 'h':
    s+= char
  if char < 'e':
    s+= char
  if char < 'd':
    s+= char
    
print(s)




from datetime import datetime 

print(int((datetime(2010,1,1,13,10,0) - datetime(2010,1,1,13,4,0))).total_seconds())
    













