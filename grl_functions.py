# Functions for GRL 2025 paper

import numpy as np
from netCDF4 import Dataset
from wrf import getvar, disable_xarray
from scipy import stats

## Read WRF variable ######################################################
def var_wrfread(infile, varname):
    disable_xarray()
    ncfile = Dataset(infile)
    var = getvar(ncfile, varname)
    ncfile.close()
    var = np.squeeze(var)
    return var

## Calculate confidence intervals ######################################################
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = a.shape[0]
    # print(n)
    m, se = np.mean(a, axis=0), stats.sem(a, axis=0) # specifically for mem x time
    num = stats.t.ppf((1 + confidence) / 2., n-1)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

## Saturation vapor pressure ######################################################

# ; PURPOSE:
# ;       compute saturation vapor pressure given temperature in K or C
# ; INPUTS:
# ;       T       SCALAR OR VECTOR OF TEMPERATURES IN CELSIUS OR K
# ; OUTPUTS:
# ;       returns the saturation vapor pressure in Pa
# ; MODIFICATION HISTORY:
# ;  Dominik Brunner (brunner@atmos.umnw.ethz.ch), Feb 2000
# ;       A good reference is Gibbins, C.J., Ann. Geophys., 8, 859-886, 1990

# James Ruppert (jruppert@ou.edu), converted to python and placed here, June 2022
#   Converted all input/output to SI units, June 2022
# 
# ; WMO reference formula is that of Goff and Gratch (1946), slightly
# ; modified by Goff in 1965:

def esat(T):

    # WMO, 2008
    # http://cires1.colorado.edu/~voemel/vp.html
    # https://library.wmo.int/doc_num.php?explnum_id=7450

    esat=611.2*np.exp((17.67*T)/(T+243.5))

    return esat


## Relative humidity  ######################################################

# ; PURPOSE:
# ;       Convert mixing ratio (kg H2O per kg of dry air) at given
# ;       temperature and pressure into relative humidity (%)
# ; INPUTS:
# ;       MIXR: Float or FltArr(n) H2O mixing ratios in kg H2O per kg dry air
# ;       p   : Float or FltArr(n) ambient pressure in Pa
# ;       T   : Float or FltArr(n) ambient Temperature in C or K
# ; OUTPUTS:
# ;       returns the relative humidity over liquid water
# ; MODIFICATION HISTORY:
# ;  Dominik Brunner (brunner@atmos.umnw.ethz.ch), August 2001

# James Ruppert (jruppert@ou.edu), converted to python and placed here, June 2022
#   Converted all input/output to SI units, June 2022
#   Added switch and if-statement for ice, June 2022

# ;  Derivation:
# ;                                      Mw*e              e
# ;  W (mixing ratio) = m_h2o/m_dry = -------- = Mw/Md * ---
# ;                                    Md*(p-e)           p-e
# ;
# ;  RH (rel. hum.)    = e/esat(T)*100.
def relh(MIXR,p,T):
    
    es=esat(T) # T must be in Celsius
    
    Mw=18.0160 # molecular weight of water [g/mol]
    Md=28.9660 # molecular weight of dry air [g/mol]

    fact=MIXR*(Md/Mw)
    top = (Md / Mw) * MIXR * p
    bottom = es * (1 + ((Md / Mw) * MIXR))
    rh = (top / bottom) * 100.
    return rh 

## Density moist ######################################################

# Calculate density for an array in pressure coordinates
#   tmpk - temp [K]
#   qv   - water vapor mixing ratio [kg/kg]
#   pres - pressure [Pa]
def density_moist(T, qv, pres):
    
    if np.max(pres) < 1e4:
        pres*=1e2 # Convert to Pa
    
    if np.min(T) < 105.: # degC or K?
        T0=273.16
    else:
        T0=0.
    T+=T0
    
    rd=287.04
    # rv=461.5
    # eps_r=rv/rd
    # return pres / ( rd * T * (1. + qv*eps_r)/(1.+qv) )
    return pres / ( rd * T * (1. + 0.61*qv) )

## Equivalent potential temperature ##############################################

# ; BASED ON (34) OF BRYAN & FRITSCH 2002, OR (2.31) OF MARKOWSKI AND RICHARDSON
# ; (2002), which is the "wet equivalent potential temperature" (BF02) or simply
# ; "equiv pot temp" (MR02).
# ;
# ; INPUTS:
#     T - temp [K]
#     rv   - water vapor mixing ratio [kg/kg]
#     pres - pressure [Pa]
# ; XX  RTOT: TOTAL WATER (VAPOR+HYDROMETEOR) MIXING RATIO (KG/KG)
# ; 
# ; RETURNS:
# ; 
# ;   EQUIVALENT OTENTIAL TEMPERATURE (K)
# ; 
# ; James Ruppert, jruppert@ou.edu
# ; 8/4/14
# ; Converted to python, June 2022
# 
def theta_equiv(T, rv, rtot, pres):
    
    if np.max(pres) < 1e4:
        pres*=1e2 # Convert to Pa
    
    if np.min(T) < 105.: # degC or K?
        T0=273.16
    else:
        T0=0.
    T+=T0
    
  # ;CONSTANTS
    R=287.    # J/K/kg
    lv0=2.5e6 # J/kg
    cp=1004.  # J/K/kg
    cpl=4186. # J/k/kg
    cpv=1885. # J/K/kg
    eps=18.0160/28.9660 # Mw / Md (source: Brunner scripts)

  # ;LATENT HEAT OF VAPORIZATION
    lv = lv0 - (cpl-cpv)*(T-273.15)

  # ;DRY AIR PRESSURE
    e = pres / ((eps/rv) + 1.)
    p_d = pres-e

  # ;CALCULATE THETA-E
    c_term = cp + cpl*rtot
    th_e = T * (1e5/p_d)**(R/c_term) * np.exp( lv*rv / (c_term*T) )

    return th_e

    ## Virtual potential temp ######################################################

# Calculate virtual potential temperature
#   tmpk - temp [K]
#   qv   - vapor mixing ratio [kg/kg]
#   pres - pressure [Pa]
def theta_virtual(T, qv, pres):
    
    if np.max(pres) < 1e4:
        pres*=1e2 # Convert to Pa
    
    if np.min(T) < 105.: # degC or K?
        T0=273.16
    else:
        T0=0.
    T+=T0
    
    p0=1.e5 # Pa
    rd=287.04 # J/K/kg
    cp=1004. # J/K/kg
    rocp = rd/cp
    virt_corr = (1. + 0.61*qv)
    return T * virt_corr * ( p0 / pres ) ** rocp