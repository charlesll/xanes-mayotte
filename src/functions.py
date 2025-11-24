########## Calling relevant libraries ##########
import numpy as np
from scipy.optimize import minimize
from scipy.special import erf

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import rampy as rp

# and now larch
from larch import Group
from larch.xafs import pre_edge

##############
# FOR SULFUR #
##############

def index_nearest(array, value):
    """
    return index of array *nearest* to value
    >>> ix = index_nearest(array, value)
    Arguments
    ---------
    array  (ndarray-like):  array to find index in
    value  (float): value to find index of
    Returns
    -------
    integer for index in array nearest value
    """
    return np.abs(array-value).argmin()

def S_bkg(x,a, b, c, d, e):
    """S K-edge background function"""
    return 0.5*(1 + erf((x-a)/b)) + rp.gaussian(x,c,d,e)

def S_model(x, a, b, c, d, e, 
            a1, f1, l1, 
            a2, f2, l2, 
            a3, f3, l3, 
            a4, f4, l4, 
            a5, f5, l5, details=False):
    """model of the sulfur spectra"""
    bkg_erf = 0.5*(1 + erf((x-a)/b))
    bgk_gauss = rp.gaussian(x,c,d,e)
    p_sulfide = rp.gaussian(x,a1,f1,l1)
    p_S2 = rp.gaussian(x,a2,f2,l2)
    p_S4 = rp.gaussian(x,a3,f3,l3)
    p_S6 = rp.gaussian(x,a4,f4,l4)
    p_Sio = rp.gaussian(x,a5,f5,l5)
    if details == True:
        return bkg_erf, bgk_gauss, p_sulfide, p_S2, p_S4, p_S6, p_Sio
    else:
        return bkg_erf + bgk_gauss + p_sulfide + p_S2 + p_S4 + p_S6 + p_Sio

def S_treatment(dat=None, nom=None, spectrum_type=0, 
                roi_pre = np.array([[2455., 2462.]]), 
                roi_post = np.array([[2485.,2487],[2510, 2525]]), 
                nnorm=1, nvict=0, path="../xas/sulfur/", 
                view_figure=False, output_flat_dat = False):
    """Treat S K-edge spectra
    
    Parameters
    ----------
    nom : string
        path to spectrum file
    spectrum_type : int
        0 for partial data, 1 for full spectrum
    
    """
    if dat == None:
        # import data and save them in a Larch group
        sp = np.genfromtxt(path+nom, encoding = 'cp1252')    
        dat = Group()
        dat.e = sp[:,0]
        dat.mu = sp[:,4]/sp[:,1]
    
    # start the figure
    plt.figure(figsize=(8,4), dpi=150)
    if spectrum_type == 1:
        plt.subplot(1,2,1)
    plt.plot(dat.e, dat.mu, ".-", linewidth=0.2, label="raw data")
    plt.xlabel("Energy, eV")
    plt.ylabel("$\mu(E)$, observed")
    
    if spectrum_type == 1: # full spectrum to be treated and peak-fitted
        # do pre-processing steps, here XAFS pre-edge removal
        #pre_edge(dat.e,dat.mu,group=dat,
        #         e0 = 2472.4,
        #         pre1 = pre1,
        #         pre2 = pre2,
        #         norm1 = norm1,
        #         norm2 = norm2,
        #         nnorm = nnorm,
        #         nvict = nvict,
        #         _larch = my_larch)
        #mback(dat.energy, dat.mu, group=dat, z=16, edge='K', order=4)
    
        # find e0 as half the edge step
        ie0 = index_nearest(dat.mu[dat.e<2478], 0.5*np.max(dat.mu))
        dat.e0 = dat.e[ie0]
        
        # Manual fit
        _, base_pre = rp.baseline(dat.e, dat.mu, roi_pre, "poly", polynomial_order = 1)
        _, base_post = rp.baseline(dat.e, dat.mu, roi_post, "poly", polynomial_order = 1)
        dat.pre_edge = base_pre.ravel()
        dat.post_edge = base_post.ravel()

        # update figure with normalization stuffs
        #plt.plot([dat.e0+pre1, dat.e0+pre1],[0,np.max(dat.mu)],"--")
        #plt.plot([dat.e0+pre2, dat.e0+pre2],[0,np.max(dat.mu)],"--")
        #plt.plot([dat.e0+norm1, dat.e0+norm1],[0,np.max(dat.mu)],"--")
        #plt.plot([dat.e0+norm2, dat.e0+norm2],[0,np.max(dat.mu)],"--")
        plt.plot([roi_pre[0], roi_pre[0]],[0,np.max(dat.mu)],"b--")
        for iki in range(len(roi_post)):
            plt.plot([roi_post[iki], roi_post[iki]],[iki,np.max(dat.mu)],"c--")
        plt.plot(dat.e, dat.pre_edge, linewidth=0.5)
        plt.plot(dat.e, dat.post_edge, linewidth=0.5)
    
        # normalize and flatten
        edge_step = dat.post_edge[ie0] - dat.pre_edge[ie0]
        dat.norm = (dat.mu-dat.pre_edge)/edge_step

        _, base_norm = rp.baseline(dat.e, dat.norm, roi_post, "poly", polynomial_order = 1)

        dat.flat = dat.norm - (base_norm.ravel()  - base_norm.ravel()[ie0])
        dat.flat[:ie0] = dat.norm[:ie0]
    
        # model the spectra
        x = dat.e
        y = dat.flat

        p0 = [2485.0, 8.00, 0.17, 2500., 8., # BKG
              0.1, 2470., 1, # sulfide
              1.5, 2476., 3, # S2
              0.1, 2477.8, 0.5, # S4
              1.0, 2482., 2, # S6
              1.0, 2485., 2] # Sio

        lb_p0 = [2484.99, 7.99, 0, 2492., 0., # BKG
                 0., 2460.0, 0,  # sulfide
                 0., 2475.0, 0,  # S2
                 0., 2477.0, 0,  # S4
                 0., 2480.0, 0,  # S6
                 0., 2483.5, 0.] # Sio

        hb_p0 = [2485.01, 8.01, 15, 2530., 10., # BKG
                 15, 2472.0, 2, # sulfide
                 15, 2477.0, 4, # S2
                 15, 2479.0, 1, # S4
                 15, 2483.0, 3, # S6
                 15, 2486.0, 4] # Sio

        #popt=p0.copy()
        popt, pcov = curve_fit(S_model, x, y, 
                           p0 = p0,
                           bounds = (lb_p0, hb_p0), method="trf")

        ####
        # Calculate sulfur speciation
        ####
        a_SF = popt[5]
        f_SF = popt[6]
        l_SF = popt[7]
        a_S2 = popt[8]
        f_S2 = popt[9]
        l_S2 = popt[10]
        a_S4 = popt[11]
        f_S4 = popt[12]
        l_S4 = popt[13]
        a_S6 = popt[14]
        f_S6 = popt[15]
        l_S6 = popt[16]

        area_S2, _ = rp.gaussianarea(a_S2, l_S2)
        area_S6, _ = rp.gaussianarea(a_S6, l_S6)

        area_S6_ratio = area_S6/(area_S2+area_S6)

        # Lerner et al., 2021, in review
        redox_6_over_TOT_L2021 = 0.1733*area_S6_ratio**2 + 0.8343*area_S6_ratio
        
        # Jugo2010 and our revision
        I_6_ratio, redox_6_over_TOT_J2010, redox_6_over_TOT_LL2023 = Red_I_determination(dat.e, dat.flat)

        # finish figure
        plt.subplot(1,2,2)
        plt.plot(x, y, "k-", linewidth=0.5, label="data")
        plt.plot(x, S_model(x, *popt), "r-", linewidth=0.5, label="model")

        plt.xlabel("Energy, eV")
        plt.ylabel("Intensity, normalised")

        peaks = S_model(x, *popt, details=True)
        labels = ["background_erf", "background_gauss", "Sulfide", "S$^{2-}$", "S$^{4+}$", "S$^{6+}$", "S ionization"]
        it = 0
        for j in peaks:
            plt.plot(x, j, label=labels[it])
            it += 1
    
    plt.legend()
    plt.tight_layout()
    plt.savefig("../figures/Sulfur/"+nom+".pdf")
    if view_figure == False:
        plt.close()
    
    if spectrum_type == 1: # full spectrum, output redox params
        if output_flat_dat == True:
            return area_S6_ratio, I_6_ratio, redox_6_over_TOT_L2021, redox_6_over_TOT_J2010, redox_6_over_TOT_LL2023, dat
        else:
            return area_S6_ratio, I_6_ratio, redox_6_over_TOT_L2021, redox_6_over_TOT_J2010, redox_6_over_TOT_LL2023 
    elif spectrum_type == 0: # part spectrum, output NaN
        # we still use the method of Jugo 2010
        I_6_ratio, redox_6_over_TOT_J2010, redox_6_over_TOT_LL2023 = Red_I_determination(dat.e, dat.mu)
        if output_flat_dat == True:
            return np.nan, I_6_ratio, np.nan, redox_6_over_TOT_J2010, redox_6_over_TOT_LL2023, dat
        else:
            return np.nan, I_6_ratio, np.nan, redox_6_over_TOT_J2010, redox_6_over_TOT_LL2023
    
def Red_I_determination(x, y):
    signal_S6 = rp.extract_signal(x, y, np.array([[2481.0, 2485]]))
    signal_S2 = rp.extract_signal(x, y, np.array([[2475, 2477]]))
    I_S6 = np.max(signal_S6[:,1])#np.trapz(signal_S6[:,1], signal_S6[:,0])
    I_S2 = np.median(signal_S2[:,1])#np.trapz(signal_S2[:,1], signal_S2[:,0])
    I_6_ratio = I_S6/(I_S6 + I_S2)
    # Jugo2010
    P_ = np.array([  9.22616886, -20.50903574,  17.98046302,  -6.46184204, 0.79052444])
    redox_6_over_TOT_J2010 = np.polyval(P_, I_6_ratio)
    #redox_6_over_TOT_J2010 = -0.81 * np.log((I_6_ratio-1.2427)/-0.94911)
    redox_6_over_TOT_LL2023 = 1.69815916*I_6_ratio-0.69360041

    return I_6_ratio, redox_6_over_TOT_J2010, redox_6_over_TOT_LL2023

def get_S_spectrum(nom, path="../xas/sulfur/",
                   
                   roi_pre = np.array([[2455., 2462.]]), 
                   roi_post = np.array([[2485.,2487],[2510, 2525]]), 
                   nnorm=1, nvict=0):
    """Treat S K-edge spectra
    
    Parameters
    ----------
    nom : string
        path to spectrum file
    spectrum_type : int
        0 for partial data, 1 for full spectrum
    
    """
    
    # import data and save them in a Larch group
    sp = np.genfromtxt(path+nom, encoding = 'cp1252')    
    dat = Group()
    dat.e = sp[:,0]
    dat.mu = sp[:,4]/sp[:,1]
    
    # find e0 as half the edge step
    ie0 = index_nearest(dat.mu[dat.e<2478], 0.5*np.max(dat.mu))
    dat.e0 = dat.e[ie0]
        
    # Manual fit
    _, base_pre = rp.baseline(dat.e, dat.mu, roi_pre, "poly", polynomial_order = 1)
    _, base_post = rp.baseline(dat.e, dat.mu, roi_post, "poly", polynomial_order = 1)
    dat.pre_edge = base_pre.ravel()
    dat.post_edge = base_post.ravel()
    
    # normalize and flatten
    edge_step = dat.post_edge[ie0] - dat.pre_edge[ie0]
    dat.norm = (dat.mu-dat.pre_edge)/edge_step

    _, base_norm = rp.baseline(dat.e, dat.norm, roi_post, "poly", polynomial_order = 1)

    dat.flat = dat.norm - (base_norm.ravel()  - base_norm.ravel()[ie0])
    dat.flat[:ie0] = dat.norm[:ie0]
    
    return dat.e, dat.flat


##############
## FOR IRON ##
##############

def pv(x,A,F,L,m):
    """pseudo-voigt function"""
    Lor = A/(1+((x-F)/L)**2)
    Gauss = A*np.exp(-np.log(2)*((x-F)/L)**2)
        
    return m*Gauss + (1-m)*Lor

def bkg(x, a, b, xo):
    """background"""
    return a*np.exp(b*(x-xo))

def model(x, a, b, xo, a1, f1, a2, f2, l, e):
    """pre-edge model"""
    return bkg(x, a, b, xo) + pv(x,a1,f1,l,e) + pv(x, a2, f2, l, e)

def area_pv(h,L,eta):
    """Pseudo-voigt area calculation"""
    return ((1-eta)*np.sqrt(np.pi*np.log(2))+eta*np.pi)*h*L

def C_to_Fe3(C, C_Fe2, method="F2017"):
    """Convert centroid to Fe3+ fraction using literature calibrations.
    
    Parameters
    ----------
    C: centroid position
    C_Fe2: Fe2+ centroid position
    method: calibration to use
        "F2017": Fiege et al. 2017 Am. Min. 102:369
        "W2005": Wilke et al. 2005 Chem. Geol 220:143
    """
    if method == "W2005":
        return (-0.028 + np.sqrt((0.000784 + 0.00052 * (C_Fe2-C))))/-0.00026
    elif method == "F2017":
        # figure...
        a = 32.7237839
        b = -43.3064978
        c = 55.84432
        x = C-C_Fe2
        return a*x**3 + b*x**2 + c*x
    
def glass_treatment(nom, pre1 = -50, pre2 = -15, norm1 = 110, norm2 = 129, nnorm = 1, nvict = 4):
    
    sp = np.genfromtxt("../xas/iron/"+nom, encoding = 'cp1252')
    
    dat = Group()
    dat.e = sp[:,0]
    dat.mu = sp[:,4]/sp[:,1]
    
    # do pre-processing steps, here XAFS pre-edge removal
    pre_edge(dat.e,dat.mu,group=dat,
             pre1 = pre1,
             pre2 = pre2,
             norm1 = norm1,
             norm2= norm2,
             nnorm= nnorm,
             nvict= nvict,
             )
    
    roi_preedge = np.array([[7105, 7119]])
    roi_bkg_preedge = np.array([[7105,7110],[7117, 7119]])

    plt.figure(figsize=(8,4), dpi=150)
    plt.subplot(1,2,1)
    plt.plot(dat.e, dat.mu, linewidth=0.5)
    plt.plot(dat.e, dat.pre_edge, linewidth=0.5)
    plt.plot(dat.e, dat.post_edge, linewidth=0.5)

    preedge = rp.extract_signal(dat.e, dat.flat, roi_preedge)

    x = preedge[:,0]
    y = preedge[:,1] - np.min(preedge[:,1])
    
    popt, pcov = curve_fit(model, x, y, 
                       p0 = [1, 1, 7122., 0.05, 7112.33, 0.05, 7114.16, 0.9, 0.2],
                       bounds = ([-np.inf, -np.inf, 7111, 0, 7112.20, 0, 7113.5, 0.5, 0],
                                 [np.inf, np.inf, 7130, 1, 7112.40, 1, 7114.20, 1.5, 1]))
    
    plt.xlabel("Energy, eV")
    plt.ylabel("Intensity, measured")
    
    plt.subplot(1,2,2)
    plt.plot(x, y, "k-", linewidth=0.5, label="data")
    plt.plot(x, model(x, *popt), "r-", linewidth=0.5, label="model")
    
    a1 = popt[3]
    f1 = popt[4]
    a2 = popt[5]
    f2 = popt[6]
    l = popt[7]
    e = popt[8]
    
    plt.plot(x, bkg(x, popt[0], popt[1], popt[2]), "--", linewidth=0.5, label="background")
    plt.plot(x, pv(x, a1, f1, l, e) + bkg(x, popt[0], popt[1], popt[2]), ":", linewidth=0.5, label="Fe$^{2+}$")
    plt.plot(x, pv(x, a2, f2, l, e) + bkg(x, popt[0], popt[1], popt[2]), ":", linewidth=0.5, label="Fe$^{3+}$")
    
    plt.xlabel("Energy, eV")
    plt.ylabel("Intensity, normalised")
    plt.legend(loc="best")
    
    area_Fe2 = area_pv(a1, l, e)
    area_Fe3 = area_pv(a2, l, e)
    area_tot = area_Fe2 + area_Fe3
    
    area_Fe2 = area_Fe2/area_tot
    area_Fe3 = area_Fe3/area_tot
    
    centroid = f1 *area_Fe2 + f2*area_Fe3
    
    centroid_2 = rp.centroid(x,y - bkg(x, popt[0], popt[1], popt[2]))

    plt.tight_layout()
    
    plt.savefig("../figures/Iron/"+nom+".pdf")
    plt.close()
    return x, y, f1, f2, area_Fe3/(area_Fe2+area_Fe3), centroid, centroid_2

def index_nearest(array, value):
    """
    return index of array *nearest* to value
    >>> ix = index_nearest(array, value)
    Arguments
    ---------
    array  (ndarray-like):  array to find index in
    value  (float): value to find index of
    Returns
    -------
    integer for index in array nearest value
    """
    return np.abs(array-value).argmin()

def get_Fe_spectrum(nom, pre1 = -50, pre2 = -15, norm1 = 110, norm2 = 129, nnorm = 1, nvict = 4):
    
    sp = np.genfromtxt("../xas/iron/"+nom, encoding = 'cp1252')
    
    dat = Group()
    dat.e = sp[:,0]
    dat.mu = sp[:,4]/sp[:,1]
    
    # do pre-processing steps, here XAFS pre-edge removal
    pre_edge(dat.e,dat.mu,group=dat,
             pre1 = pre1,
             pre2 = pre2,
             norm1 = norm1,
             norm2= norm2,
             nnorm= nnorm,
             nvict= nvict,
             )
    
    roi_preedge = np.array([[7105, 7119]])
    roi_bkg_preedge = np.array([[7105,7110],[7117, 7119]])

    preedge = rp.extract_signal(dat.e, dat.flat, roi_preedge)

    x = preedge[:,0]
    y = preedge[:,1] - np.min(preedge[:,1])
    
    return dat.e, dat.flat, x, y


# Fe redox calculations with centroid

def calculate_centroid_noFeadjust(x, w):
    f2 = 0.0 
    f3 = 7114.13-7112.32
    return (1-x)*f2 + x*f3 + (x*(1-x))**(w+x)

def calculate_centroid(x, f2 = 7112.32, f3 = 7114.13, w = 0.515):
    return (1-x)*f2 + x*f3 + (x*(1-x))**(w+x)

def objective_f(x_redox, centroid_meas):
    centroid_calc = calculate_centroid(x_redox)
    return (centroid_calc - centroid_meas)**2

def provide_redox(centroid_meas):
    """Provide the redox from the centroid provided.
    Performs a grid search with a resolution of 0.01"""
    x_redox = np.arange(0,1,0.01)
    centroid_calc = calculate_centroid(x_redox)
    diff = (centroid_calc - centroid_meas)**2
    return x_redox[diff == np.min(diff)]

def provide_redox2(centroid_meas):
    """Provide the redox from the centroid provided.
    Performs a grid search with a resolution of 0.01"""
    res = minimize(objective_f, 0.2, args=(centroid_meas,), bounds=((0,1),))
    return res.x