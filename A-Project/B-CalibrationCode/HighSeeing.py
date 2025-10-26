import numpy as np

def HighSeeing_Filter(fileName,maskNumber):
    """
    MaskNumber: Should be mask1 if on mask1 and mask2 if on the second mask group
    """
    Exceptions=np.load("/Users/gonzalo/Desktop/Code/Gonzalo_Binospec/code/files/BadSeeing_Exposures.npy")
    MaskExceptions=[]
    MaskDic={}
    MaskDic["Mask1"]="mask1"
    MaskDic["Mask2_1"]="mask2"
    MaskDic["Mask2_2"]="mask2"
    MaskDic["Mask2_3"]="mask2"
    for exc in Exceptions:
        if exc.find(MaskDic[maskNumber])>0:
            MaskExceptions.append(exc)
            
    if maskNumber=="Mask1":
        for exc in MaskExceptions:
            splitted=exc.split("/")
            ExpBatch=splitted[13]
            ExpNum=splitted[14].split("_")[1]
            if fileName.find(ExpBatch)>0 and fileName.find(ExpNum)>0:
                return False
        
    if maskNumber=="Mask2_1"or maskNumber=="Mask2_2" or maskNumber=="Mask2_3":
        for exc in MaskExceptions:
            splitted=exc.split("/")
            ExpBatch=splitted[12]
            ExpNum=splitted[13].split("_")[1]
            if fileName.find(ExpBatch)>0 and fileName.find(ExpNum)>0:
                return False
            
        return True

    return None