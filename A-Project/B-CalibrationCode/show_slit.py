import matplotlib.pyplot as plt

def show_slit(slit):
    plt.rcParams["figure.figsize"] = (20, 2)
    dat=slit.data
    plt.imshow(dat,aspect='auto',cmap="gray",vmin=-1.2e-18,vmax=3e-18)
    print("a")
    plt.show()