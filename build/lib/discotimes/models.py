from theano.tensor import *
from theano import tensor
from arviz import *
import pickle
import pandas as pd
import numpy as np
import pymc3 as pm

                    
class discotimes_model(pm.model.Model):
    # 1) override init
    
    """model for changepoint detection (discontinuities and trend changes)
    type pm.model.Model
    """
    
    def __init__(self, observed=None,name='', model=None,change_trend=False,n_changepoints=5,offsets_std=1,p_=0.1,
                      sigma_noise=1.,trend_inc_sigma=0.01,annual_cycle=False,change_offsets=True,
                 estimate_offset_sigma=False,estimate_trend_inc_sigma=False,post_seismic=False,
                 AR1=False,distribute_offsets=False,robust_reg=False,initial_values={},**kwargs):
        
        """Requires parameters as defined in model_settings
        
        
        Parameters
        ----------
        
        observed : discotimes.observed
            observed object
        name : str
            Name of the data/model
        change_trend : bool
            Turn on/off changing trend fits
        n_changepoints : int
            Maximum allowed number of change points
        offsets_std : float
            prior: offsets std
        p_ : 0.1,
            prior: Bernoulli's initial p (probability of a change point 0.1 ==10%)
        sigma_noise : float
            prior: white-noise sigma       
        trend_inc_sigma : float
            prior: trend-increments hyperparameter sigma       
        annual_cycle : bool
            Turn on/off annual cycle estimation       
        change_offsets : bool
            Turn on/off estimating discontinuities (named here offsets)
        estimate_offset_sigma : bool
            Turn on/off estimating hyperparameter offset sigma        
        estimate_trend_inc_sigma : bool
            Turn on/off estimating hyperparameter trend-inc. sigma   
        post_seismic : bool
            Turn on/off post-seismic module ! to be implemented
        AR1 : bool
            Turn on/off AR1 model, if false, only white noise is estimated
        distribute_offsets : bool
            Turn on/off equally distribute offsets in the beginning        
        robust_reg : bool
            Turn on/off ! future implementation to deal with strong outliers       
        initial_values : dict
            dictionary of initial conditions e.g. 
            {'n_changepoints':n_changepoints,'p_':p_,'positions':positions,'offsets':offsets}
        
        """
        super().__init__(name,model)

        
        x=observed.x
        y=observed.y
            
        xmin=np.min(x)
        xmax=np.max(x)
        
        
        X_mat=observed.X_mat
        
        if 'offsets' in initial_values:
            # integrate pre-defined offsets
            initialize=True
            print('manually initialize with: ')
            print(initial_values)
            estimate_offset_sigma=False
            offsets_mu=initial_values['offsets']
            p_=initial_values['p_']
            start = {'offsets': initial_values['offsets'], 'positions': initial_values['positions']}

        else:
            offsets_mu=0
            start={}

        # Priors for model parameters
        offset = pm.Normal('offset', mu=0, sigma=1)
        trend = pm.Normal('trend', mu=0, sigma=1) 
        sigma = pm.HalfNormal('sigma', sigma=sigma_noise)   
        act_number = pm.Bernoulli('act_number', p = p_, shape=n_changepoints)

        if not change_offsets:
            mult_offsets=0.
        else:
            mult_offsets=1. 

        if estimate_offset_sigma: # estimate one distribution for multiple offsets
            offset_sigma = pm.HalfNormal('offset_sigma', sigma=offsets_std)   
            offsets = pm.Normal('offsets', mu=offsets_mu, sigma=offset_sigma,
                                shape=n_changepoints)*mult_offsets   
            
        else: # estimate multiple distributions for multiple offsets
            offsets = pm.Normal('offsets', mu=offsets_mu, sigma=offsets_std,
                                shape=n_changepoints)*mult_offsets  

        mult=pm.Deterministic('mult', (act_number> 0.5)*1) # array with 1,0 defining changep =True/False
        offsets=offsets*mult
        if distribute_offsets:
            # equally distributed offset initial positions
            mup=np.linspace(xmin,xmax,n_changepoints)
            mu_pos=pm.Uniform('mu_pos',mup, lower=xmin, upper=xmax, shape=n_changepoints) 
        else: 
            # randomly distributed offsets
            mu_pos=pm.Uniform('mu_pos', lower=xmin, upper=xmax, shape=n_changepoints) 

        s = pm.Normal('positions',  mu=mu_pos, sigma=5, shape=n_changepoints)

        A = (x[:, None] >= s) * 1
        offset_change = det_dot(A, offsets)

        if change_trend:
            if estimate_trend_inc_sigma:
                # estimate one hyperparameter distribution from which trend increments can stem from
                trend_inc_sigma_est = pm.HalfNormal('trend_inc_sigma_est', sigma=trend_inc_sigma)  
                trend_inc = pm.Normal('trend_inc', mu=0, sigma=trend_inc_sigma_est,shape=n_changepoints)
            else:
                # no hyperparameter distribution estimation
                trend_inc = pm.Normal('trend_inc', mu=0, sigma=trend_inc_sigma,shape=n_changepoints)
            trend_inc=trend_inc*mult
            gamma = -s* trend_inc

            trend_inc=det_dot(A, trend_inc)
            trend=trend+trend_inc
            A_gamma=det_dot(A, gamma)
            offset_change=offset_change+A_gamma
        if post_seismic:
            # !!! not implemented yet !!!
            # add exponential term to every trend section
            c_constant=pm.Normal('c_constant', mu=0, sigma=2,shape=n_changepoints)
            etau=pm.HalfNormal('etau', sigma=3,shape=n_changepoints)  
            t_vec=(A*x[:,None])-s
            inside_function=((t_vec-t_vec*[t_vec<0]).squeeze()/etau)              
            post_seismic=sum(c_constant*(1-np.exp(-((t_vec-t_vec*[t_vec<0]).squeeze()/etau)))*mult,axis=1)
            offset_change=offset_change+post_seismic 
        if annual_cycle:
            m_coeffs=pm.Normal('m_coeffs', mu=0, sigma=1,shape=12)
            annual=pm.Deterministic("annual", det_dot(X_mat,m_coeffs)) 
            mu = pm.Deterministic("mu", offset_change + trend*x + offset + annual)   
        else:
            mu = pm.Deterministic("mu", offset_change + trend*x + offset)            
        if robust_reg:
            # !!! not implemented yet !!!
            sigma_s = pm.HalfNormal('sigma_s', sigma=sigma_noise) 
            nu = pm.InverseGamma("nu", alpha=1, beta=1)
            mu = pm.StudentT("mu", mu=offset_change + trend*x + offset,
                             sigma=sigma_s, nu=nu)
      
        else:
            if AR1: # first order autoregressive ...
                beta = pm.HalfNormal('beta', sigma=0.4)
                likelihood = pm.AR('AR1_coeff', beta, sigma=sigma, observed=y-mu) 
            else:    
                Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y)


def det_dot(a, b):
    """

    """
    return (a * b[None, :]).sum(axis=-1)