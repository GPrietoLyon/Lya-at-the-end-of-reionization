#Specific Tools

import numpy as np
import matplotlib.pyplot as plt
import sTools

try:
    from grizli.aws import db
    from grizli import utils
except:
    x=0
    
from astropy.io import fits
import matplotlib.colors as colors


def FindModule(exp):
    name=exp["dataset"]
    if "nrcblong" in name:
        return "B"
    if "nrcalong" in name:
        return "A"
    

def ExposureModules(coord,Verbose=False,showPlot=False):
    exp=fits.open("../data/large_files/FrescoDatabase/fresco_exposure_footprints.fits")[1].data

    has_point = np.array([utils.SRegion(fp).path[0].contains_point(coord)
                        for fp in exp['footprint']])

    modules=[sTools.FindModule(e) for e in exp[has_point]]

    if Verbose==True:
        print(f'{has_point.sum()} exposures cover the point {coord}')


    if showPlot==True:
        fig, ax = plt.subplots(1,1,figsize=(5,5))
        ax.scatter(exp['crval1'][has_point], exp['crval2'][has_point])
        for fp in exp['footprint'][has_point]:
            sr = utils.SRegion(fp)
            
            for p in sr.patch(fc='b', ec='None', alpha=0.1):
                ax.add_patch(p)
                
        ax.scatter(*coord, marker='x', color='r', zorder=100, label='test point')
        ax.legend()

        ax.grid()
        ax.set_aspect(1./np.cos(coord[1]/180*np.pi))
        plt.show()


    QuantifiedModules={"A":-1,"B":1}
    modNum=0
    for mod in modules:
        modNum+=QuantifiedModules[mod]

    modNum=modNum/len(modules)


    return modules,modNum

def Merge_data(data, dz,vo,crd,sn,zsp,id):
    All_d,All_z,All_v,All_c,All_sn,All_zsp,All_id=[],[],[],[],[],[],[]
    for k in data.keys():
        if k=="Ha":
            continue
        All_d=All_d+data[k]
        All_z=All_z+dz[k]
        All_v=All_v+vo[k]
        All_id=All_id+id[k]
        All_c=All_c+crd[k]
        All_sn=All_sn+sn[k]
        All_zsp=All_zsp+zsp[k]

    return All_d,All_z,All_v,All_c,All_sn,All_zsp,All_id

def bin_data(x_data, y_data, bin_edges):
    # Initialize an array to store the binned data
    binned_data = [[] for _ in range(len(bin_edges) - 1)]

    # Iterate over each point in the data
    for x, y in zip(x_data, y_data):
        # Find the bin index for the current point
        bin_index = np.digitize(x, bin_edges) - 1

        # If the bin index is within the valid range, add the point to the corresponding bin
        if 0 <= bin_index < len(binned_data):
            binned_data[bin_index].append(y)

    # Return the binned data
    return binned_data

def bin_edges_to_centers(bin_edges):
    bin_centers = []
    for i in range(len(bin_edges) - 1):
        center = (bin_edges[i] + bin_edges[i + 1]) / 2
        bin_centers.append(center)
    return bin_centers

def GiveLine(z,linesDic,id):
    rangeF=[39000,49000]
    visible="X"
    for l in linesDic.keys():
        if rangeF[0]<linesDic[l]*(1+z)<rangeF[1]:
            print(id,z,l)
            visible=l
    if visible=="X":
        return [np.nan,np.nan]
    
    return [visible,linesDic[visible]*(1+z)] 

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def masker(array,mask):
    return np.array(array)[mask]

