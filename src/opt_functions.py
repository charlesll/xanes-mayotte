import numpy as np
import pandas as pd
import subprocess
#import emcee
import csv

def QFM(T, P=1.0):
    """Frost 1991 equation, RiMG 25"""
    return -25096.3/T + 8.735 + 0.11*(P-1.0)/T

def list_oxides():
    """returns the list of oxydes handled by i-Melt codes

    Returns
    -------
    out : list
        list of the oxides in the good order that are handled in i-Melt codes
    """
    return [
        "sio2",
        "tio2",
        "al2o3",
        "feo",
        "fe2o3",
        "mno",
        "na2o",
        "k2o",
        "mgo",
        "cao",
        "p2o5",
        "h2o",
    ]

def molarweights():
    """returns a partial table of molecular weights for elements and oxides that can be used in other functions

    Uses molar weights from IUPAC Periodic Table 2016

    Returns
    =======
    w : dictionary
        containing the molar weights of elements and oxides:

        - si, ti, al, fe, h, li, na, k, mg, ca, ba, sr, ni, mn, p, o (no upper case, symbol calling)

        - sio2, tio2, al2o3, fe2o3, feo, h2o, li2o, na2o, k2o, mgo, cao, sro, bao, nio, mno, p2o5 (no upper case, symbol calling)

    """
    w = {"si": 28.085}

    # in g/mol
    w["ti"] = 47.867
    w["al"] = 26.982
    w["fe"] = 55.845
    w["h"] = 1.00794
    w["li"] = 6.94
    w["na"] = 22.990
    w["k"] = 39.098
    w["mg"] = 24.305
    w["ca"] = 40.078
    w["ba"] = 137.327
    w["sr"] = 87.62
    w["o"] = 15.9994

    w["ni"] = 58.6934
    w["mn"] = 54.938045
    w["p"] = 30.973762

    # oxides
    w["sio2"] = w["si"] + 2 * w["o"]
    w["tio2"] = w["ti"] + 2 * w["o"]
    w["al2o3"] = 2 * w["al"] + 3 * w["o"]
    w["fe2o3"] = 2 * w["fe"] + 3 * w["o"]
    w["feo"] = w["fe"] + w["o"]
    w["h2o"] = 2 * w["h"] + w["o"]
    w["li2o"] = 2 * w["li"] + w["o"]
    w["na2o"] = 2 * w["na"] + w["o"]
    w["k2o"] = 2 * w["k"] + w["o"]
    w["mgo"] = w["mg"] + w["o"]
    w["cao"] = w["ca"] + w["o"]
    w["sro"] = w["sr"] + w["o"]
    w["bao"] = w["ba"] + w["o"]

    w["nio"] = w["ni"] + w["o"]
    w["mno"] = w["mn"] + w["o"]
    w["p2o5"] = w["p"] * 2 + w["o"] * 5

    return w  # explicit return

def wt_mol(data, basis_1O= False):
    """convert weights in mol fraction

    Parameters
    ==========
    data: Pandas DataFrame
        containing the fields sio2, tio2, al2o3, feo, mno, na2o, k2o, mgo, cao, p2o5, h2o

    Returns
    =======
    chemtable: Pandas DataFrame
        contains the fields sio2, tio2, al2o3, feo, mno, na2o, k2o, mgo, cao, p2o5, h2o in mol%
    """
    chemtable = data.copy()
    w = molarweights()

    if basis_1O == True:
        # conversion to mol in 100 grammes
        sio2 = chemtable["sio2"] / (0.5*w["sio2"])
        tio2 = chemtable["tio2"] / (w["tio2"]/2)
        al2o3 = chemtable["al2o3"] / (w["al2o3"]/3)
        fe2o3 = chemtable["fe2o3"] / (w["fe2o3"]/3)
        feo = chemtable["feo"] / w["feo"]
        mno = chemtable["mno"] / w["mno"]
        na2o = chemtable["na2o"] / (w["na2o"]/2)
        k2o = chemtable["k2o"] / (w["k2o"]/2)
        mgo = chemtable["mgo"] / w["mgo"]
        cao = chemtable["cao"] / w["cao"]
        p2o5 = chemtable["p2o5"] / (w["p2o5"]/5)
        h2o = chemtable["h2o"] / (w["h2o"]/2)
    else:
        # conversion to mol in 100 grammes
        sio2 = chemtable["sio2"] / w["sio2"]
        tio2 = chemtable["tio2"] / w["tio2"]
        al2o3 = chemtable["al2o3"] / w["al2o3"]
        fe2o3 = chemtable["fe2o3"] / w["fe2o3"]
        feo = chemtable["feo"] / w["feo"]
        mno = chemtable["mno"] / w["mno"]
        na2o = chemtable["na2o"] / w["na2o"]
        k2o = chemtable["k2o"] / w["k2o"]
        mgo = chemtable["mgo"] / w["mgo"]
        cao = chemtable["cao"] / w["cao"]
        p2o5 = chemtable["p2o5"] / w["p2o5"]
        h2o = chemtable["h2o"] / w["h2o"]

    # renormalisation
    tot = sio2 + tio2 + al2o3 + fe2o3 + feo + mno + na2o + k2o + mgo + cao + p2o5 + h2o
    chemtable["sio2"] = sio2 / tot
    chemtable["tio2"] = tio2 / tot
    chemtable["al2o3"] = al2o3 / tot
    chemtable["fe2o3"] = fe2o3 / tot
    chemtable["feo"] = feo / tot
    chemtable["mno"] = mno / tot
    chemtable["na2o"] = na2o / tot
    chemtable["k2o"] = k2o / tot
    chemtable["mgo"] = mgo / tot
    chemtable["cao"] = cao / tot
    chemtable["p2o5"] = p2o5 / tot
    chemtable["h2o"] = h2o / tot

    return chemtable

def chimie_control(data):
    """check that all needed oxides are there and setup correctly the Pandas datalist.

    Parameters
    ----------
    data : Pandas dataframe
        the user input list.

    Returns
    -------
    out : Pandas dataframe
        the output list with all required oxides.
    """
    list_ox = list_oxides()
    datalist = data.copy()  # safety net

    for i in list_ox:
        try:
            oxd = datalist[i]
        except:
            datalist[i] = 0.0

    # we calculate the sum and store it
    sum_oxides = (
        datalist["sio2"]
        + datalist["tio2"]
        + datalist["al2o3"]
        + datalist["fe2o3"]
        + datalist["feo"]
        + datalist["mno"]
        + datalist["na2o"]
        + datalist["k2o"]
        + datalist["mgo"]
        + datalist["cao"]
        + datalist["p2o5"]
        + datalist["h2o"]
    )

    # renormalisation of each element
    datalist["sio2"] = datalist["sio2"] / sum_oxides
    datalist["tio2"] = datalist["tio2"] / sum_oxides
    datalist["al2o3"] = datalist["al2o3"] / sum_oxides
    datalist["fe2o3"] = datalist["fe2o3"] / sum_oxides
    datalist["feo"] = datalist["feo"] / sum_oxides
    datalist["mno"] = datalist["mno"] / sum_oxides
    datalist["na2o"] = datalist["na2o"] / sum_oxides
    datalist["k2o"] = datalist["k2o"] / sum_oxides
    datalist["mgo"] = datalist["mgo"] / sum_oxides
    datalist["cao"] = datalist["cao"] / sum_oxides
    datalist["p2o5"] = datalist["p2o5"] / sum_oxides
    datalist["h2o"] = datalist["h2o"] / sum_oxides

    # we calculate again the sum and store it in the dataframe
    datalist["sum"] = (
        datalist["sio2"]
        + datalist["tio2"]
        + datalist["al2o3"]
        + datalist["fe2o3"]
        + datalist["feo"]
        + datalist["mno"]
        + datalist["na2o"]
        + datalist["k2o"]
        + datalist["mgo"]
        + datalist["cao"]
        + datalist["p2o5"]
        + datalist["h2o"]
    )

    return datalist.copy()

def redox_B2018(chimie, fo2, T):
    """return the ration of Fe3+ over total iron for a melt at given T and fO2, 1 atm pressure

    It uses the Borisov et al. (2018) parametric model

    Parameters
    ==========
    chimie : pandas dataframe
        the melt chemical composition in mol%

    fo2 : float or ndarray
        the oxygen fugacity, float or array of size n

    T : ndarray
        the temperature in K, size n

    Options
    -------
    model : string
        The model that should be used. Choose between KC1991, B2018

    Returns
    =======
    fe3_fe3pfe2 : ndarray
        the ratio of Fe3+ over (Fe3+ + Fe2+) in the melt, size n

    To Do
    =====
    Raise error if fo2 and T are of different sizes
    """

    # get the major element composition
    if not isinstance(chimie, pd.DataFrame):
        raise TypeError("chimie should be a pandas DataFrame")
    
    chimie_maj = chimie.loc[:, ["sio2", "tio2", "al2o3", "feo", "mno", "na2o", "k2o", "mgo", "cao", "p2o5"]].copy()

    # re-normalize the composition
    sum_maj = chimie_maj.sum(axis=1)
    chimie_maj = chimie_maj.div(sum_maj, axis=0)


    a = 0.207
    b = 4633.3
    c = -1.852
    d_sio2 = -0.445
    d_tio2 = -0.900
    d_mgo = 1.532
    d_cao = 0.314
    d_na2o = 2.030
    d_k2o = 3.355
    d_p2o5 = -4.851
    d_sial = -3.081
    d_simg = -4.370

    xdi = (chimie_maj["sio2"]*d_sio2 +
          chimie_maj["tio2"]*d_tio2 +
          chimie_maj["mgo"]*d_mgo +
          chimie_maj["cao"]*d_cao +
          chimie_maj["na2o"]*d_na2o +
          chimie_maj["k2o"]*d_k2o +
          chimie_maj["p2o5"]*d_p2o5 +
          chimie_maj["sio2"]*chimie_maj["al2o3"]*d_sial +
          chimie_maj["sio2"]*chimie_maj["mgo"]*d_simg)

    log_feo1p5_feo = a*np.log10(fo2) + b/T + c + xdi # 1 atm equation

    fe3_fe3pfe2 = 10**log_feo1p5_feo/((1+10**log_feo1p5_feo)) #

    return fe3_fe3pfe2

# Attempt with direct calculation
def fo2_B2018(chimie, fe3_fe3pfe2, T):
    """return the ration of Fe3+ over total iron for a melt at given T and fO2, 1 atm pressure

    It uses the Borisov et al. (2018) parametric model

    Parameters
    ==========
    chimie : pandas dataframe
        the melt chemical composition in mol%

    fe3_fe3pfe2 : float or ndarray
        the ratio of Fe3+ over (Fe3+ + Fe2+) in the melt, size n

    T : ndarray
        the temperature in K, size n

    Returns
    =======
    log_fo2 : ndarray
        the oxygen fugacity, float or array of size n

    To Do
    =====
    Raise error if fo2 and T are of different sizes
    """


    # get the major element composition
    if not isinstance(chimie, pd.DataFrame):
        raise TypeError("chimie should be a pandas DataFrame")
    
    chimie_maj = chimie.loc[:, ["sio2", "tio2", "al2o3", "feo", "mno", "na2o", "k2o", "mgo", "cao", "p2o5"]].copy()

    # re-normalize the composition
    sum_maj = chimie_maj.sum(axis=1)
    chimie_maj = chimie_maj.div(sum_maj, axis=0)


    a = 0.207
    b = 4633.3
    c = -1.852
    d_sio2 = -0.445
    d_tio2 = -0.900
    d_mgo = 1.532
    d_cao = 0.314
    d_na2o = 2.030
    d_k2o = 3.355
    d_p2o5 = -4.851
    d_sial = -3.081
    d_simg = -4.370

    xdi = (chimie_maj["sio2"]*d_sio2 +
          chimie_maj["tio2"]*d_tio2 +
          chimie_maj["mgo"]*d_mgo +
          chimie_maj["cao"]*d_cao +
          chimie_maj["na2o"]*d_na2o +
          chimie_maj["k2o"]*d_k2o +
          chimie_maj["p2o5"]*d_p2o5 +
          chimie_maj["sio2"]*chimie_maj["al2o3"]*d_sial +
          chimie_maj["sio2"]*chimie_maj["mgo"]*d_simg)

    #log_feo1p5_feo = a*np.log10(fo2) + b/T + c + xdi # 1 atm equation
    log_feo1p5_feo = np.log10(fe3_fe3pfe2 / (1 - fe3_fe3pfe2))
    log10_fo2 = (log_feo1p5_feo - b/T - c - xdi)/a

    return log10_fo2

def redox_KC1991(chimie,fo2, T):
    """return the ration of Fe3+ over total iron for a melt at given T and fO2, 1 atm pressure

    It uses the Kress and Carmnichael (1991) parametric model

    Parameters
    ==========
    chimie : pandas dataframe
        the melt chemical composition in mol%

    fo2 : float or ndarray
        the oxygen fugacity, float or array of size n

    T : ndarray
        the temperature in K, size n

    Returns
    =======
    fe3_fe3pfe2 : ndarray
        the ratio of Fe3+ over (Fe3+ + Fe2+) in the melt, size n

    To Do
    =====
    Raise error if fo2 and T are of different sizes
    """

    # get the major element composition
    if not isinstance(chimie, pd.DataFrame):
        raise TypeError("chimie should be a pandas DataFrame")
    
    chimie_maj = chimie.loc[:, ["sio2", "tio2", "al2o3", "feo", "mno", "na2o", "k2o", "mgo", "cao", "p2o5"]].copy()

    # re-normalize the composition
    sum_maj = chimie_maj.sum(axis=1)
    chimie_maj = chimie_maj.div(sum_maj, axis=0)

    a = 0.196
    b = 1.1492*10**4
    c = -6.675
    d_al2o3 = -2.243
    d_feo = -1.828
    d_cao = 3.201
    d_na2o = 5.854
    d_k2o = 6.215
    e = -3.36
    f = -7.01*10**-7
    g = -1.54*10**-10
    h = 3.85*10**-17

    xdi = chimie_maj["al2o3"]*d_al2o3 + chimie_maj["feo"]*d_feo + chimie_maj["cao"]*d_cao + chimie_maj["na2o"]*d_na2o + chimie_maj["k2o"]*d_k2o

    ln_fe2o3_feo = a*np.log(fo2) + b/T + c + xdi # 1 atm equation

    fe3_fe3pfe2 = np.exp(ln_fe2o3_feo+np.log(2))/((1+np.exp(ln_fe2o3_feo+np.log(2)))) # np.log(2)factor two because I want XFeO and XFeO1.5 to have Fe3+/Fe2+

    return fe3_fe3pfe2

def forward(theta, data_):

    nb_val = len(data_)
    # Corrected DataFrame construction using a dictionary
    db = pd.DataFrame({
        "SiO2": data_.sio2,
        "TiO2": data_.tio2,
        "Al2O3": data_.al2o3,
        "Fe2O3": np.zeros(nb_val),
        "Cr2O3": np.zeros(nb_val),
        "FeO": data_.feo,
        "MnO": data_.mno,
        "MgO": data_.mgo,
        "CaO": data_.cao,
        "Na2O": data_.na2o,
        "K2O": data_.k2o,
        "P2O5": data_.p2o5,
        "H2O": data_.h2o,
        "S_tot_ppm": data_.S_,
        "so2": np.zeros(nb_val),
        "T(C)": data_.loc[:,"T_start"],
        "Pbar": 350.0*np.ones(nb_val),
        "xossi_fo2": theta[0:13],
        "fs2": theta[13:26],
        "index_author": np.zeros(nb_val),
        "kflag": np.zeros(nb_val),
        "wmol": 0.01*np.ones(nb_val),
        "kir": np.zeros(nb_val)
    })
    db.to_csv("COMPO.txt",index=False,float_format="%.5g")

    # now we add the appropriate header
    with open('INPUT.txt', 'w', newline='') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow([str(nb_val),"","","","","","","","","","","","","","","","","","","","","",""])

        with open('COMPO.txt', 'r', newline='') as incsv:
            reader = csv.reader(incsv)
            writer.writerows(row + [0.0] for row in reader)

    # run the code
    subprocess.run("./ctsfg6")

    # retrieve results
    results = pd.read_csv("./ctsfg6.jet", skiprows=1, names=["T(K)","P(bar)","logfO2(in)",
                                                                "logfO2(calc)","logfS2","aossi",
                                                                "TotAni","TotCat","nO=","nO-",
                                                                "nO0","NBO/T","Kpol","Stot(obs wt%)",
                                                                "Stot(calc wt%)","S_as_S2-(wt%)","S_as_S6+(wt%)",
                                                                "S6+/tot","log(KSO4/KS2)","Redox","Redoz",
                                                                "actFe2+","cost_FeO","kflag"])

    # get the ratio of Fe2/Fe3, log10 unit
    FE2_FE3 = results["Redox"].values

    # convert that into a Fe3/(Fe2 + Fe3) ratio
    FE3_FETOT = 1-(10**FE2_FE3)/(1+(10**FE2_FE3))

    # get the sulfur redox
    S6_STOT = results.loc[:,"S6+/tot"].values

    # Get the sulphur concentration in ppm
    # convertion coefficient from wt% to ppm
    # 1 wt% = 10000 ppm
    # so we multiply by 10000 to get ppm
    S_TOT = results.loc[:,"Stot(calc wt%)"].values*10000

    return FE3_FETOT, S6_STOT,  S_TOT

def lognormal(x,mu,std):
    var = std**2
    return -0.5*((x-mu)**2/var + np.log(2.0*np.pi*var))

def logprior(theta):
    """log prior probability function for the parameters
    
    This implementation uses a uniform distribution
    
    Parameters
    ==========
    theta : list
        the parameters
        
    Returns
    =======
    logprob_prior : the estimated probability of the prior values
    
    """
    xossi_fo2 = theta[0:13]
    fs2 = theta[13:26]
    
    # Any value outside the ranges below will return an -np.inf prior (so a probability of 0 = model not valid)
    #if ((900. < T_)&(T_ < 1200.)).all() \
    if ((-15. < xossi_fo2)&(xossi_fo2 < -7.)).all() and ((fs2 > -10.0)&(fs2 < 1.0)).all():
        # Gaussian priors on T, P, fO2
        #P_prior = lognormal(Pbar, 350.0, 10.0)
        #T_prior = lognormal(T_, [1058, 1133, 1106, 1069, 952, 977, 986, 1058], 10.0)
        #fO2_prior = lognormal(xossi_fo2, -11.5, 1.0)
        #fS2_prior = lognormal(fs2, -1.1, 1.5)
        #prior_probability = np.sum(fO2_prior) + np.sum(fS2_prior)
        return 0.0
    else:
        return -np.inf
    
def loglike(theta, x, fe3, s6, stot, fe3_err, s6_err, stot_err):
    """log likelyhood function
    
    This implementation uses the log of a gaussian distribution
    
    Parameters
    ==========
    theta : list
        the parameters
    x : ndarray
        the X variable
    y : ndarray
        the y observations
    yerr : ndarray
        the y errors
        
    Returns
    =======
    ln_likely : float
        the estimated likelyhood of the a model compared to observations
    
    """
    fe3_calc, s6_calc, stot_calc = forward(theta, x)

    # Catch model failures
    try:
        if np.any(np.isnan(fe3_calc)):  
            return -np.inf
        if np.any(np.isnan(s6_calc)):
            return -np.inf
        if np.any(np.isnan(stot_calc)):
            return -np.inf
    except: # in case we have non numerical values
        return -np.inf
    
    # if no failure, we move forward to calculate the likelihood
    misfit_fe3 = np.sum(lognormal(fe3_calc, fe3, fe3_err)) 
    misfit_s6 = np.sum(lognormal(s6_calc, s6, s6_err))
    misfit_stot = 0.0#np.sum(lognormal(stot_calc, stot, stot_err))

    ln_likely = misfit_fe3 + misfit_s6 + misfit_stot
    
    return ln_likely 

def logjoint(theta, x, fe3, s6, stot, fe3_err, s6_err, stot_err):
    """joint probability function
    
    This implementation uses the log of a gaussian distribution
    
    Parameters
    ==========
    theta : list
        the parameters
    x : ndarray
        the X variable
    y : ndarray
        the y observations
    yerr : ndarray
        the y errors
        
    Returns
    =======
    ln_prob : float
        the total likelyhood of a model compared to observations
    
    """
    lp = logprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglike(theta, x, fe3, s6, stot, fe3_err, s6_err, stot_err)

if __name__ == "__main__":

    #
    # import data
    #
    data_ = pd.read_excel("Results_synthese.xlsx")
    nb_points = len(data_)

    #
    # Get the values
    #
    fe3_ = data_.loc[:,"Fe3"].values
    s6_ = data_.loc[:,"S6"].values
    stot_ = data_.loc[:,"S_"].values
    fe3_err_ = 0.02*fe3_
    fe3_err_[fe3_err_ == 0.] = 1. # if 0 => we put unity value = no effect and no error too!
    s6_err_ = 0.05*s6_
    s6_err_[s6_err_ == 0.] = 1. # if 0 => we put unity value = no effect and no error too!
    stot_err_ = data_.loc[:,"ese_S_"].values
    s6_err_[s6_err_ == 0.] = 1. # if 0 => we put unity value = no effect and no error too!

    #
    # Initialize the walkers
    #
    # We initialize them near an optimal solution retrieved using L-BFGS-B
    #
    theta_lbfgs = [-9.40710155,  -8.25738031,  -8.72281798,  -8.69657139,  -9.32128826,
 -10.31300445, -10.77015558,  -9.18409839,  -8.21109899,  -8.25078313,
  -8.36866221,  -7.72439141, -10.35645552,  -0.81210794,  -1.45899957,
  -1.27287393,  -2.70956334 , -8.00431023,  -7.60455323,  -2.51195498,
  -1.77882123,  -2.97986669 , -3.78520847,  -2.03489078,  -5.37979777,
  -6.37930583]
    nb_walkers = 100
    theta_start = []
    for i in range(nb_walkers):
        theta_i = theta_lbfgs + np.random.randn(len(theta_lbfgs))*0.01
        theta_start.append(theta_i)

    theta_start = np.array(theta_start)
    # xossi_fo2 = np.random.random_sample((nb_walkers, nb_points)) + -11.5
    # fs2 = 0.5*np.random.randn(nb_walkers, nb_points) - 2
    # theta_start = np.hstack((xossi_fo2, fs2))
    nwalkers, ndim = theta_start.shape
    print(theta_start.shape)
    print(logprior(theta_start[0,:]))

    # Set up the backend
    # Don't forget to clear it in case the file already exists
    filename = "sampler_new.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)

    # Initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, 
                                    ndim, 
                                    logjoint, 
                                    backend=backend, 
                                    args=(data_, fe3_, s6_, stot_, fe3_err_, s6_err_, stot_err_)
    )

    # Run the sampling
    sampler.run_mcmc(theta_start, 2000, progress=True)