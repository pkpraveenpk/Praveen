#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os


# In[25]:


def file_checker(folder_path):
    totalFiles = 0
    Json=0
    pdf=0
    for base, directory, files in os.walk(folder_path):
        print('Searching files in : ',base)
        for item in files:
            name,extension=os.path.splitext(os.path.abspath(item))
            if(extension==".json"):
                Json+=1
            elif(extension==".pdf"):
                pdf+=1
            totalFiles += 1
            if(totalFiles>2):
                return "Only 2 files allowed"

    print('Total number of files',totalFiles)
    if(Jso==1 and pdf==1):
        print("One JSON file and one PDF file")
   
    
    


# In[26]:


folder_path=input("Enter path =")
file_checker(folder_path)

