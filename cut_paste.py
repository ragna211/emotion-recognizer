# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 18:41:43 2018

@author: HP
"""
import os
import shutil
src="C:/Users/HP/Desktop/downloads/output"
des="C:/Users/HP/Desktop/downloads/test"
way=os.listdir(src)
for file in way:   
    path=src+'/'+file
    d_path=des+'/'+file
    os.makedirs(d_path)
    way1=os.listdir(path)    #go into that particular directory
    cop=way1[0:4]
    for file1 in cop:
        f_src=path+'/'+file1
        f_des=d_path+'/'+file1
        shutil.move(f_src,f_des) #moes files from source to destination

        