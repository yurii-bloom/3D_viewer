
#%%
#import plotly.graph_objects as go
#import itk
#from itkwidgets import view
import pydicom
import time
start = time.time()
import matplotlib.colors as clr
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import numpy as np
#from sklearn.cluster import DBSCAN
import cv2
#from scipy.interpolate import splprep, splev
#from collections import OrderedDict
#import os
import napari
#import itk
def crop_3d(video,cut):
    height, width = video.shape[1], video.shape[2]
    video_cropped = video[:,int(height*cut):int(height-height*cut),int(width*cut):int(width-width*cut)]
    return video_cropped
def threshold_3d(video,min=0,max=255):
    video[video<min] = 0
    video[video>max] = max
    return video
def crop_coord(video,coords:list,adjust:int):
    video_cropped = video[:,coords[1]-adjust:coords[1]+coords[3]+adjust,coords[0]-adjust:coords[0]+coords[2]+adjust]
    return video_cropped
#%%
#Opening, preprocessing, cropping, thresholding
base = r'C:\Users\jemmi\Flywheel\Dicoms\Local'
pass_file = r'20221213155524___.mp4'
if pass_file[-3:]=='dcm':
    file = pydicom.data.data_manager.get_files(base,pass_file)[0]
    ds = pydicom.dcmread(file)
    video = ds.pixel_array
    video = video[:,:,:,0]
    #print(video.shape)
else:
    frames=[]
    file = os.path.join(base, pass_file)
    cap = cv2.VideoCapture(file)
    for i in range(int(cap.get(7))):
        _,frame = cap.read()
        frame = frame[:,:,0]
        frames.append(frame)
    cap.release()
    video = np.stack(frames,axis=0)
#video_cropped = threshold_3d(video_cropped,0,255)
#print(video_thr.shape)
#%%
video_cropped = crop_3d(video,0.1)
stack = np.zeros((video_cropped.shape[1],video_cropped.shape[2]))
for i in range(video_cropped.shape[0]):
    stack+=video_cropped[i,:,:]
stack_final = stack/video_cropped.shape[0]
plt.imshow(stack_final,cmap='gray')
plt.show()
#%%
mode = 'manual'
if mode == 'auto':
    _, thresh = cv2.threshold(stack_final,40,255, cv2.THRESH_BINARY)
    plt.imshow(thresh,cmap='gray')
    plt.show()
    thresh = np.expand_dims(thresh,-1)
    thresh = thresh.astype(np.uint8)
    #print(thresh.dtype)
    cnt,_ = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    objects = [i for i in cnt if cv2.contourArea(i) > 600]
    coords = []
    for object in objects:
        x,y,w,h = cv2.boundingRect(object)
        coords.append([x,y,w,h])
        #cv2.rectangle(img_contours,(x,y),(x+w,y+h),(255,255,255),2)
    y_s = [i[1] for i in coords]
    final_coords = coords[y_s.index(min(y_s))]
    #final_coords = [i+50 for i in semifinal_coords]
else:
    final_coords = [750,400,650,600]
adjust = 0
stack_cnt = stack_final
cv2.rectangle(stack_cnt,(final_coords[0]-adjust,final_coords[1]-adjust), \
(final_coords[0]+final_coords[2]+adjust,final_coords[1]+final_coords[3]+adjust),(255,255,255),2)
#cv2.drawContours(stack_final,cnt,-1,(255,255,255),3)
cv2.imwrite('./contours.png', stack_cnt)
crop = crop_coord(video_cropped,final_coords,adjust)
#print(crop.shape)
plt.imshow(crop[100,:,:],cmap='gray')
plt.show()
#%%
#final = np.zeros(crop.shape)
frames = []
for i in range(0,crop.shape[0]):
    frame = crop[i,:,:]
    _, fr_thresh = cv2.threshold(frame,50,255, cv2.THRESH_BINARY)
    cnt_1,_ = cv2.findContours(fr_thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    areas = [cv2.contourArea(c) for c in cnt_1]
    heart_ind = areas.index(np.max(areas))
    #heart = cnt_1[heart_ind]
    artefacts = list(cnt_1)
    del artefacts[heart_ind]
    mask = np.zeros(frame.shape)
    cv2.drawContours(mask,artefacts,-1,(255,255,255))
    for artefact in artefacts:
        cv2.fillPoly(mask, pts = [artefact], color = (255,255,255))
    mask = mask.astype(np.uint8)
    #img_contours_1 = np.zeros(frame.shape)
    #cv2.drawContours(img_contours_1,heart,-1,(255,255,255),1)
    #cv2.fillPoly(img_contours_1, pts = [heart], color = (255,255,255))
    #img_contours_bl = cv2.GaussianBlur(img_contours_1,(7,7),4)
    cleaned = cv2.subtract(frame,mask)
    cleaned = cv2.GaussianBlur(cleaned,(15,15),7)
    frames.append(cleaned)
    '''if i%10==0:
        plt.title('contours')
        plt.imshow(img_contours_bl, cmap='gray')
        plt.show()
        plt.title('frame')
        plt.imshow(frame, cmap='gray')
        plt.show()
        plt.title('artefacts_mask')
        plt.imshow(mask, cmap='gray')
        plt.show()
        plt.title('cleaned')
        plt.imshow(cleaned, cmap='gray')
        plt.show()
        print(frame.dtype)
        print(mask.dtype)'''
final = np.stack(frames, axis=0)
#%%
#%gui qt
#viewer = napari.Viewer()
finish = time.time()
runtime = finish-start
print(f'Runtime is {runtime}')
viewer = napari.view_image(crop, colormap='plasma', rendering='iso', \
     ndisplay = 3)
viewer.add_image(video)
#viewer.add_image(video_cropped)
#viewer.add_image(video_thr)
#viewer.layers['crop'].scale = [1,1,1]
napari.run()
# %%
