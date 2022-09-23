import albumentations as A
import numpy as np
import cv2

__all__= ['cutmix_aug', 'mixup_aug', 'mosaic_aug', 'copy_paste', 'box_channel_drop']

def cutmix_aug(window_size,
               img_1, mask_1, 
               img_2, mask_2):
    """
    img: numpy array of shape (height, width,channel)
    mask: numpy array of shape (height, width,channel)
    """
    def get_cut_transform(cut_size):
        return A.Compose([
                    A.PadIfNeeded(min_height=window_size, min_width=window_size,border_mode=0, p=1),
                    A.RandomCrop(window_size, window_size, p=1),
                    A.OneOf([
                        A.HueSaturationValue(hue_shift_limit=100, sat_shift_limit=15, val_shift_limit=15, p=0.7),
                        A.CLAHE(clip_limit=2, p=0.3),
                        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                    ], p=0.9),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit= 10,
                                                    interpolation=cv2.INTER_LINEAR, border_mode=0, p=0.5),
                    A.RandomCrop(cut_size, cut_size, p=1)])

    cut_size= np.random.randint(int(window_size*0.1), int(window_size*0.5))
    aug= get_cut_transform(cut_size)
    transform= aug(image= img_2, mask= mask_2)
    img_2, mask_2= transform['image'], transform['mask']

    while True:
        rand_xcoord= np.random.randint(img_1.shape[0])
        rand_ycoord= np.random.randint(img_1.shape[1])
        try:
            img_1[rand_xcoord:rand_xcoord+cut_size, rand_ycoord:rand_ycoord+cut_size]= img_2
            mask_1[rand_xcoord:rand_xcoord+cut_size, rand_ycoord:rand_ycoord+cut_size]= mask_2
            break
        except:
            pass
    return img_1, mask_1


def mixup_aug(window_size,
              img_1, mask_1, 
              img_2, mask_2):
    """
    img: numpy array of shape (height, width,channel)
    mask: numpy array of shape (height, width,channel)
    """
    def get_mix_transform():
        return A.Compose([
                    A.PadIfNeeded(min_height=window_size, min_width=window_size,border_mode=0, p=1),
                    A.RandomCrop(window_size, window_size, p=1),
                    A.OneOf([
                        A.HueSaturationValue(hue_shift_limit=100, sat_shift_limit=15, val_shift_limit=15, p=0.7),
                        A.CLAHE(clip_limit=2, p=0.3),
                        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                    ], p=0.9),
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit= 10,
                                                    interpolation=cv2.INTER_LINEAR, border_mode=0, p=0.5)])
    
    aug= get_mix_transform()
    transform= aug(image= img_2, mask= mask_2)
    img_2, mask_2= transform['image'], transform['mask']
    
    ## mixup
    weight= np.random.beta(a=2, b=2)
    img= img_1*weight + img_2*(1-weight)
    mask= mask_1*weight + mask_2*(1-weight)
    return img, mask


def mosaic_aug(window_size,
               img_1, mask_1, 
               img_2, mask_2, 
               img_3, mask_3, 
               img_4, mask_4):
    """
    img: numpy array of shape (height, width,channel)
    mask: numpy array of shape (height, width,channel)
    """
    def get_mos_transform():
        return A.Compose([
                    A.PadIfNeeded(min_height=window_size, min_width=window_size,border_mode=0, p=1),
                    A.RandomCrop(window_size, window_size, p=1),
                    A.OneOf([
                        A.HueSaturationValue(hue_shift_limit=100, sat_shift_limit=15, val_shift_limit=15, p=0.7),
                        A.CLAHE(clip_limit=2, p=0.3),
                        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                    ], p=0.9),
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit= 10,
                                                    interpolation=cv2.INTER_LINEAR, border_mode=0, p=0.5)])
    
    aug= get_mos_transform()
    transform= aug(image= img_2, mask= mask_2)
    img_2, mask_2= transform['image'], transform['mask']
    transform= aug(image= img_3, mask= mask_3)
    img_3, mask_3= transform['image'], transform['mask']
    transform= aug(image= img_4, mask= mask_4)
    img_4, mask_4= transform['image'], transform['mask']
    
    ## choose center point
    center_point_x= np.random.randint( int(window_size*0.25), int(window_size*0.75) )
    center_point_y= np.random.randint( int(window_size*0.25), int(window_size*0.75) )
    
    ## fill img
    img_1[:center_point_y, center_point_x:]= img_2[:center_point_y, center_point_x:]
    img_1[center_point_y:, :center_point_x]= img_3[center_point_y:, :center_point_x]
    img_1[center_point_y:, center_point_x:]= img_4[center_point_y:, center_point_x:]
    
    ## fill mask
    mask_1[:center_point_y, center_point_x:]= mask_2[:center_point_y, center_point_x:]
    mask_1[center_point_y:, :center_point_x]= mask_3[center_point_y:, :center_point_x]
    mask_1[center_point_y:, center_point_x:]= mask_4[center_point_y:, center_point_x:]
    
    return img_1, mask_1


def copy_paste(window_size,
               img_1, mask_1,
               img_2, mask_2):
    """
    img: numpy array of shape (height, width,channel)
    mask: numpy array of shape (height, width,channel)
    """
    def get_copypaste_transform():
        return A.Compose([
                    A.PadIfNeeded(min_height=window_size, min_width=window_size,border_mode=0, p=1),
                    A.RandomCrop(window_size, window_size, p=1),
                    A.OneOf([
                        A.HueSaturationValue(hue_shift_limit=100, sat_shift_limit=15, val_shift_limit=15, p=0.7),
                        A.CLAHE(clip_limit=2, p=0.3),
                        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                    ], p=0.9),
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit= 10,
                                       interpolation=cv2.INTER_LINEAR, border_mode=0, p=0.5)])
    
    aug= get_copypaste_transform()
    transform= aug(image= img_2, mask= mask_2)
    img_2, mask_2= transform['image'], transform['mask']
    
    bool_mask= np.asarray(mask_2, dtype= bool)
    if bool_mask.shape[2]!=3:
        bool_mask_3d= np.zeros_like(img_1)
        bool_mask_3d[..., 0:1]= bool_mask
        bool_mask_3d[..., 1:2]= bool_mask
        bool_mask_3d[..., 2:3]= bool_mask
        bool_mask_3d= np.asarray(bool_mask_3d, dtype= bool)
    else:
        bool_mask_3d= bool_mask
    
    img_1[bool_mask_3d]= 0
    mask_1[bool_mask]= 0
    img_2[~bool_mask_3d]= 0
    img_1+= img_2
    mask_1+= mask_2
    
    return img_1, mask_1


def box_channel_drop(img_size, window_size, img):
    box_size= np.random.randint(30, int(img.shape[0])-50)
    drop_channel= np.random.randint(3)
    
    for i in range(2):
        while True:
            rand_xcoord= np.random.randint(img.shape[0])
            rand_ycoord= np.random.randint(img.shape[1])
            try:
                img[rand_xcoord:rand_xcoord+box_size, rand_ycoord:rand_ycoord+box_size, drop_channel]=0
                break
            except:
                pass
    return img