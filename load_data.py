# -*- coding: utf-8 -*-
"""
Created on Tue May  2 22:33:15 2023

@author: Tejas Rao
"""
import numpy as np 
import os
from PIL import Image
import matplotlib.pyplot as plt


# Returns a list containing the light field of given scene path 
def load_lf(lf_dir):
    
    num_view_points = 25 
    num_vert_imgs = 434 // 64 
    num_hor_imgs = 625 // 64 
    lf = []
    
    im_width = 434
    im_height = 625
    block_size = 64 
    
    num_cropped_imgs = (im_width//64)*(im_height//64)
    
    for v in range(num_cropped_imgs):
        lf.append(np.zeros((25,64, 64, 3)))
        
    
    
    for file_name in os.listdir(lf_dir):
        
        file_path = os.path.join(lf_dir, file_name)
        img_pos = file_path[-8:-4]
        u = int(img_pos[0:2])
        v = int(img_pos[2:])
        img_no = (u-6) * 5 + (v-6)
        
        img_load = Image.open(file_path)
        img_load = np.asarray(img_load)
        tick = 0
        for ver_id in range(num_vert_imgs):
            for hor_id in range(num_hor_imgs):
                im_crop = img_load[ver_id*64:(ver_id+1)*64, hor_id*64:(hor_id+1)*64,:]
                        
                lf[tick][img_no,:,:,:] = im_crop
                tick += 1
        
    return lf


def load_fs(fs_dir):
    num_view_points = 25 
    num_vert_imgs = 434 // 64 
    num_hor_imgs = 625 // 64 
    fs = []
    im_width = 434
    im_height = 625
    block_size = 64 
    
    num_cropped_imgs = (im_width//64)*(im_height//64)
    
    for v in range(num_cropped_imgs):
        fs.append(np.zeros((3,64, 64, 3)))
        

    for file_name in os.listdir(fs_dir):
        
        file_path = os.path.join(fs_dir, file_name)
        
        img_load = Image.open(file_path)
        img_load = np.asarray(img_load)
        img_pos = int(file_path[-5])-1
        tick = 0
        for ver_id in range(num_vert_imgs):
            for hor_id in range(num_hor_imgs):
                im_crop = img_load[ver_id*64:(ver_id+1)*64, hor_id*64:(hor_id+1)*64,:]
                        
                fs[tick][img_pos,:,:,:] = im_crop
                tick += 1
        
    return fs

    
    


# Loads the light field and focal stack

def load_data(scenes, path_to_lf, path_to_fs): 
    
    im_width = 434
    im_height = 625
    block_size = 64 
    
    num_cropped_imgs = (im_width//64)*(im_height//64)
    
    num_view_points = 25 
    num_depths = 3 
    num_scenes = len(scenes)
    num_channels = 3
    
    
    dataset_size =   num_channels * num_cropped_imgs * num_scenes
    
    lf_stack = np.zeros((dataset_size, num_view_points, block_size, block_size))
    fs_stack = np.zeros((dataset_size, num_depths, block_size, block_size))    
    
    r_start_ind = 0 
    g_start_ind = num_cropped_imgs * num_scenes
    b_start_ind = num_cropped_imgs * num_scenes * 2
    
    tick_lf = 0
    tick_fs = 0
    for scene_id in range(len(scenes)):
        
        lf_dir = os.path.join(path_to_lf, scenes[scene_id])
        fs_dir = os.path.join(path_to_fs, scenes[scene_id])
        
        lf = load_lf(lf_dir)
        fs = load_fs(fs_dir)
        
        
        
        for n in range(num_cropped_imgs):
            lf_stack[scene_id * num_cropped_imgs + n + r_start_ind,: , :, :] = lf[n][:,:,:,0] 
            lf_stack[scene_id * num_cropped_imgs + n + g_start_ind,:, :, :] = lf[n][:,:,:,1] 
            lf_stack[scene_id * num_cropped_imgs + n + b_start_ind,:, :, :] = lf[n][:,:,:,2] 
        
            
            fs_stack[scene_id * num_cropped_imgs + n + r_start_ind,: , :, :] = fs[n][:,:,:,0] 
            fs_stack[scene_id * num_cropped_imgs + n + g_start_ind,: , :, :] = fs[n][:,:,:,1] 
            fs_stack[scene_id * num_cropped_imgs + n + b_start_ind,: , :, :] = fs[n][:,:,:,2] 
        
        tick_fs += 1
    
    return lf_stack, fs_stack
    


if __name__=="__main__":
    
    
    scenes = ['magnets']
    path_to_lf = r"C:\Users\Tejas Rao\Documents\coder_aperture\light_fields"
    path_to_fs = r"C:\Users\Tejas Rao\Documents\coder_aperture\focal_stacks"
    lf_stack, fs_stack = load_data(scenes, path_to_lf, path_to_fs)
    
    for i in range(lf_stack.shape[0]):
        v = np.random.randint(0,2)
        img = fs_stack[i,v, :,: ]
        plt.imshow(img)
        plt.pause(0.001)
        
    
    