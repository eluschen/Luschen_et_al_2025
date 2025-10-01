#!/usr/bin/env python
# coding: utf-8

# ### Functions for CFADs
# 
# James Ruppert  
# jruppert@ou.edu  
# 5/16/23
# updated by Emily Luschen 10 Jan 2025


from netCDF4 import Dataset
import numpy as np
from grl_functions import density_moist, theta_equiv, theta_virtual, relh
from calc_vort_conv import calc_kin_prop
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker




def mask_edges(array):
    # Returns a masked array with edges masked
    # Last dimensions of input array must be x1,x2
    #   It is otherwise versatile
    buffer=80
    array = np.ma.array(array, mask=False, copy=False)
    array[...,0:buffer,:]=np.ma.masked
    array[...,-buffer:,:]=np.ma.masked
    array[...,:,0:buffer]=np.ma.masked
    array[...,:,-buffer:]=np.ma.masked
    # array = np.ma.filled(array, fill_value=np.nan)
    return array


# Variable settings

def cfads_var_settings(ivar_plot):

    if ivar_plot == 'thv':
        
        # Virtual potential temperature

        # Bin settings
        nbin=50
        fmax=10 #5 #; fmin=-5
        #step=(fmax-fmin)/nbin
        step=fmax*2/nbin
        bins=np.arange(0,fmax,step)+step
        bins=np.concatenate((-1.*np.flip(bins),bins))
    
        # Figure settings
        fig_title=r"$\theta_v$"
        fig_tag='thv'
        units_var='K'
    
        # For mean var
        scale_mn=1.
        units_mn=units_var
        xrange_mn=(-1,1)
        # xrange_mn2=(-0.1,0.2)
        xrange_mn2=(-0.3,0.6) # TEST - CTL

    elif ivar_plot == 'the':
        
        # Equivalent potential temperature

        # Bin settings
        nbin=50
        fmax=10 #; fmin=-10
        #step=(fmax-fmin)/nbin
        step=fmax*2/nbin
        bins=np.arange(0,fmax,step)+step
        bins=np.concatenate((-1.*np.flip(bins),bins))
    
        # Figure settings
        fig_title=r"$\theta_e$"
        fig_tag='the'
        units_var='K'
    
        # For mean var
        scale_mn=1.#e3
        units_mn=units_var
        xrange_mn=(-1,1)
        # xrange_mn2=(-0.5,0.25)
        xrange_mn2=(-0.3,0.6) # TEST - CTL


    elif ivar_plot == 'lq':
        
        # Latent energy, calculated as MSE - DSE

        # Bin settings
        nbin=50
        fmax=1e4 #; fmin=-10
        #step=(fmax-fmin)/nbin
        step=fmax*2/nbin
        bins=np.arange(0,fmax,step)+step
        bins=np.concatenate((-1.*np.flip(bins),bins))
    
        # Figure settings
        fig_title='$L_vq$'
        fig_tag='lvq'
        units_var='J kg$^{-1}$'
    
        # For mean var
        scale_mn=1.#e3
        units_mn=units_var
        xrange_mn=(-2e3,4e3)
        # xrange_mn2=(-15,100)
        xrange_mn2=(-100,25) # TEST - CTL


    elif ivar_plot == 'qv':
        
        # Water vapor mixing ratio

        # Bin settings
        nbin=50
        fmax=10 #; fmin=-10
        #step=(fmax-fmin)/nbin
        step=fmax*2/nbin
        bins=np.arange(0,fmax,step)+step
        bins=np.concatenate((-1.*np.flip(bins),bins))
    
        # Figure settings
        fig_title='$q_v$'
        fig_tag='qv'
        units_var='g kg$^{-1}$'

        # For mean var
        scale_mn=1.#e3
        units_mn=units_var
        xrange_mn=(-10,10)
        xrange_mn2=(-0.05,0.05)

    elif ivar_plot == 'vmf':
        
        # Vertical mass flux

        # Bin settings
        bins=np.logspace(-3,1.1,num=20)
        # bins=np.logspace(-3.5,0.7,num=20)
        bins=np.concatenate((-1.*np.flip(bins),bins))

        # Figure settings
        fig_title='VMF'
        fig_tag='vmf'
        units_var='kg m$^{-2}$ s$^{-1}$'
        # fig_title='$w$'
        # fig_tag='w'
        # units_var='m s$^{-1}$'

        # For mean var
        scale_mn=1e2
        units_mn='$10^{-2}$ '+units_var
        xrange_mn=(-20,20)
        # xrange_mn2=(-1,1.5)
        xrange_mn2=(-1.5,1)


    elif ivar_plot == 'rh':
        
        # Rel Hum

        # Bin settings
        nbin=45
        fmax=125; fmin=-10
        step=(fmax-fmin)/nbin
        bins=np.arange(fmin,fmax,step)

        # Figure settings
        fig_title='RH'
        fig_tag='rh'
        units_var='%'

        # For mean var
        scale_mn=1
        units_mn=units_var
        xrange_mn=(-1,105)
        xrange_mn2=(-2,2)
      
    elif ivar_plot == 'qrad':
        
        # Radiative heating

        # Bin settings
        nbin=60
        fmax=20 #; fmin=-10
        #step=(fmax-fmin)/nbin
        step=fmax*2/nbin
        bins=np.arange(0,fmax,step)+step
        bins=np.concatenate((-1.*np.flip(bins),bins))

        # Figure settings
        fig_title='$Q_R$'
        fig_tag='qrad'
        units_var='K d$^{-1}$'

        # For mean var
        scale_mn=1.
        units_mn=units_var
        xrange_mn=(-8,3)
        xrange_mn2=(-10,7)

    elif ivar_plot == 'lh':
        
        # Latent heating

        # Bin settings
        nbin=60
        fmax=20 #; fmin=-10
        #step=(fmax-fmin)/nbin
        step=fmax*2/nbin
        bins=np.arange(0,fmax,step)+step
        bins=np.concatenate((-1.*np.flip(bins),bins))

        # Figure settings
        fig_title='$Q_L$'
        fig_tag='lheat'
        units_var='K hr$^{-1}$'

        # For mean var
        scale_mn=1.
        units_mn=units_var
        xrange_mn=(-1,1)
        xrange_mn2=(-0.5,0.5)

    elif 'wpth' in ivar_plot:
        
        # Eddy vertical theta-v flux

        # Bin settings
        bins=np.logspace(-2,3,num=20)
        bins=np.concatenate((-1.*np.flip(bins),bins))

        # Figure settings
        if ivar_plot == 'wpthp':
          fig_title=r"$w'\theta_v'$"
        elif ivar_plot == 'wpthep':
          fig_title=r"$w'\theta_e'$"
        
        fig_tag=ivar_plot
        units_var='K m s$^{-1}$'

        # For mean var
        scale_mn=1
        units_mn=units_var
        xrange_mn=(-4,4)
        xrange_mn2=(-0.1,0.1)

    elif ivar_plot == 'vordiv':
        # Bin settings
        nbin=60
        fmax=20 #; fmin=-10
        step=fmax*2/nbin
        bins=np.arange(0,fmax,step)+step
        bins=np.concatenate((-1.*np.flip(bins),bins))

        # Figure settings
        fig_title='$VortDiv$'
        fig_tag='vdiv'
        units_var='10$^3$ s$^{-2}$'

        # For mean var
        scale_mn=1
        units_mn=units_var
        xrange_mn=(-4,4)
        xrange_mn2=(-0.1,0.1)

    elif ivar_plot == 'vor':
        # Bin settings
        nbin=60
        fmax=20 #; fmin=-10
        step=fmax*2/nbin
        bins=np.arange(0,fmax,step)+step
        bins=np.concatenate((-1.*np.flip(bins),bins))

        # Figure settings
        fig_title='$Vorticity$'
        fig_tag='vor'
        units_var='10$^3$ s$^{-2}$'

        # For mean var
        scale_mn=1
        units_mn=units_var
        xrange_mn=(-4,4)
        xrange_mn2=(-0.1,0.1)
    
    return bins, fig_title, fig_tag, units_var, scale_mn, units_mn, xrange_mn, xrange_mn2



# Calculate variables

def cfads_var_calc(ivar_plot, datdir, pres, t0, t1, lat, lon):

    # Mixing ratio
    varfil_main = Dataset(datdir+'QVAPOR_HiRes.nc')
    qv = varfil_main.variables['QVAPOR'][t0:t1,:,:,:] # kg/kg
    varfil_main.close()

    # Temperature
    varfil_main = Dataset(datdir+'T_HiRes.nc')
    tmpk = varfil_main.variables['T'][t0:t1,:,:,:] # K
    varfil_main.close()

    if ivar_plot == 'thv':
      
      # Virtual potential temp
      
      var = theta_virtual(tmpk,qv,(pres[np.newaxis,:,np.newaxis,np.newaxis])*1e2) # K
    
    elif ivar_plot == 'the': 
      
      # Equiv potential temp
      
      var = theta_equiv(tmpk,qv,qv,(pres[np.newaxis,:,np.newaxis,np.newaxis])*1e2) # K
    
    elif ivar_plot == 'lq': 
      
      # Latent energy, calculated as MSE - DSE
      
      # varfil_main = Dataset(datdir+'mse_diag.nc')
      varfil_main = Dataset(datdir+'mse_HiRes.nc')
      mse = varfil_main.variables['mse'][t0:t1,:,:,:] # J/kg
      varfil_main = Dataset(datdir+'dse_HiRes.nc')
      dse = varfil_main.variables['dse'][t0:t1,:,:,:] # J/kg
      varfil_main.close()
      var = mse - dse
      
      # nt, nz, nx1, nx2 = mse.shape
      # nz+=1
      # var = np.zeros([nt,nz,nx1,nx2])
      # var[:,nz-1,:,:]=np.nan
      # var[:,0:nz-1,:,:]=mse-dse
    
    elif ivar_plot == 'qv': 
      
      # Water vapor mixing ratio
      
      varfil_main = Dataset(datdir+'QVAPOR.nc')
      qv = varfil_main.variables['QVAPOR'][t0:t1,:,:,:] # kg/kg
      varfil_main.close()
      var *= 1e3 # kg/kg --> g/kg
    
    elif ivar_plot == 'vmf':
      
      # Vertical mass flux
      
      # Density
      # rho = density_moist(tmpk,qv,(pres[np.newaxis,:,np.newaxis,np.newaxis])*1e2) # kg/m3
      varfil = Dataset(datdir+'rho_HiRes.nc')
      rho = varfil.variables['rho'][t0:t1,:,:,:]
      varfil = Dataset(datdir+'W_HiRes.nc') # this opens the netcdf file
      var = varfil.variables['W'][t0:t1,:,:,:] # m/s
      varfil.close()
      # vmf_copy=np.copy(var)
      var *= rho

    elif ivar_plot == 'rh':
      
      # Humidity
      
      var = relh(qv,pres[np.newaxis,:,np.newaxis,np.newaxis]*1e2,tmpk,ice=1) # %

    elif ivar_plot == 'qrad':
      
      # Radiation
      
      varfil = Dataset(datdir+'RTHRATLW.nc') # this opens the netcdf file
      var = varfil.variables['RTHRATLW'][t0:t1,:,:,:]*3600*24 # K/s --> K/d
      varfil.close()
      varfil = Dataset(datdir+'RTHRATSW.nc') # this opens the netcdf file
      var += varfil.variables['RTHRATSW'][t0:t1,:,:,:]*3600*24 # K/s --> K/d
      varfil.close()

    elif ivar_plot == 'lh':
      
      # Latent heat
      
      varfil = Dataset(datdir+'H_DIABATIC.nc') # this opens the netcdf file
      var = varfil.variables['H_DIABATIC'][t0:t1,:,:,:]*3600 # K/s --> K/hr
      varfil.close()

    elif ivar_plot == 'wpthp':
      
      # W'Thv'
      
      thp = theta_virtual(tmpk,qv,(pres[np.newaxis,:,np.newaxis,np.newaxis])*1e2) # K
      # Density
      varfil = Dataset(datdir+'W.nc') # this opens the netcdf file
      www = varfil.variables['W'][t0:t1,:,:,:] # m/s
      varfil.close()

    elif ivar_plot == 'wpthep':
      
      # W'The'

      thp = theta_equiv(tmpk,qv,qv,(pres[np.newaxis,:,np.newaxis,np.newaxis])*1e2) # K
      # Density
      varfil = Dataset(datdir+'W.nc') # this opens the netcdf file
      www = varfil.variables['W'][t0:t1,:,:,:] # m/s
      varfil.close()

    elif ivar_plot == 'vordiv':
       
       # Vorticity convergence

       varfil = Dataset(datdir+'U.nc')
       uuu = varfil.variables['U'][t0:t1,:,:,:]
       varfil.close()
       varfil = Dataset(datdir+'V.nc')
       vvv = varfil.variables['V'][t0:t1,:,:,:]
       varfil.close()
       varfil = Dataset(datdir+'AVOR.nc')
       avor = varfil.variables['AVOR'][t0:t1,:,:,:]
       varfil.close()

       div, vort = calc_kin_prop(uuu, vvv, axis=(2,3)) 
       var = (-1)*div*avor*1e3 # s^-2

    elif ivar_plot == 'vor':
       
       # Vorticity

       varfil = Dataset(datdir+'AVOR_HiRes.nc')
       var = varfil.variables['AVOR'][t0:t1,:,:,:]
       varfil.close()

    return var

def plt_diff_compare(varplot):
  yvalues=np.flip(np.linspace(50,1000,39))
  lw=2
  # ['ncrf36h', 'STRATANVIL_OFF', 'STRATANVIL_ON', 'STRAT_OFF']

  fig =plt.figure(facecolor='white',figsize=(6,8))
  ax = fig.subplots()
  ax.plot(varplot, yvalues[:],label='NCRF', color='red',lw=lw)
  # ax.plot(VMFDiff_plt[1,:], yvalues[:],label='SA-NCRF', color='blue',lw=lw)
  # ax.plot(VMFDiff_plt[2,:], yvalues[:],label='C-NCRF', color='green', linestyle='dashed',lw=lw)
  # ax.plot(VMFDiff_plt[3,:], yvalues[:],label='S-NCRF', color='orange',lw=lw)
  ax.vlines(0,ymin=0,ymax=1000, color='black',alpha=0.5)
  ax.set_yscale('log')
  ax.invert_yaxis()
  ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
  ax.tick_params(axis='both',length=7)
  ytick_loc=np.arange(1000,0,-100)
  plt.yticks(ticks=ytick_loc, size=12)
  plt.ylim(1000, 100)#np.min(pres))
  plt.xticks(ticks=np.arange(-1,1.25,0.25), size=12)
  plt.legend()
  ax.set_xlabel('Difference [10$\mathbf{^3}$ s$\mathbf{^{-2}}$]', size=14, weight='bold')
  ax.set_ylabel('Pressure [hPa]', size=14, weight='bold')
  ax.set_title('VMF Difference (CTL - TEST)(Precip)',pad=10, size=14, weight='bold')
  plt.show()
  # plt.savefig('profile__diff_1mem_ncrf.png',dpi=200, facecolor='white', \
  #                   bbox_inches='tight', pad_inches=0.2)