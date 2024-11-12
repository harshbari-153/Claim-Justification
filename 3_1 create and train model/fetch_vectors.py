# Enrollment No: p23ds004
# College: NIT Surat, Gujarat, India.
# Course: M. Tech in Data Science (2023-2025)
# Guide: Krupa K. Jariwala
# Final Year Dissertation
# Topic: Claim Justification



################ Import Libraries ###############
import numpy as np
#################################################

'''
def normalize_values(a, b, c):
  if a < b and b < c:
    return 0, (b-a)/(c-a), 1
  
  elif c < b and b < a:
    return 1, (b-c)/(a-c), 0
    
  elif b < a and a < c:
    return (a-b)/(c-b), 0, 1
    
  elif c < a and a < b:
    return (a-c)/(b-c), 1, 0
    
  elif a < c and c < b:
    return 0, 1, (c-a)/(b-a)
    
  # b < c and c < a
  else:
    if a-c == 0:
      return 0, 0, 0
    return 1, 0, (c-b)/(a-c)
    
  


def normalize_vector(nums):
  i = 0
  while i < 5:
    #print(str(i*3+5)+" "+str(i*3+6)+" "+str(i*3+7))
    nums[i*3+5], nums[i*3+6], nums[i*3+7] = normalize_values(nums[i*3+5], nums[i*3+6], nums[i*3+7])
    i += 1
    
  return nums
'''


def get_array(file_name, d):
  # Read the text file
  with open(file_name, 'r') as file:
      lines = file.readlines()
  
  # Initialize an empty list to store the vectors
  vectors = []
  
  # Process each line
  for line in lines:
      nums = list(map(float, line.split()))

      if len(nums) > d:
        nums = nums[:d]
      
      if len(nums) < d:
        nums.extend([0] * (d - len(nums)))
        
      #nums = normalize_vector(nums)
      vectors.append(nums)
  
  # Return the list of vectors into a numpy array
  return np.array(vectors)