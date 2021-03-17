from sealeveltools.sl_class import *
from load_files import *
from working_scripts.vlad_globcoast_functions import *
from theano.tensor import *
from theano import tensor
from arviz import *
import pickle


                    
class vlam_exp_independent(pm.model.Model):
    # 1) override init
    
    """
    model which doesn't rely on poisson distributions but on beta distributions (which are independent)
    
    Parameters
    ----------
    robust_reg: bool
        use student's t likelihood distribution (instead of normal)
    observed: vlame.obs
        Observed dependent variable
    """
    
    def __init__(self, observed=None,name='', model=None,change_trend=False,n_changepoints=5,number_mu=0.1,
                     offsets_opt='normal',offsets_std=1,p_=0.1,
                      add_to_num=0.,sigma_noise=1.,trend_inc_sigma=0.01,annual_cycle=False,
                      change_offsets=True,estimate_offset_sigma=False,estimate_trend_inc_sigma=False,
                      estimate_number_mu=False,n_samples=500,target_accept=0.8,post_seismic=True,
                      AR1=False,studentst_noise=False,distribute_offsets=False,robust_reg=False,initial_values={},**kwargs):

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
            estimate_number_mu=False
            estimate_offset_sigma=False
            offsets_mu=initial_values['offsets']
            p_=initial_values['p_']
            start = {'offsets': initial_values['offsets'], 'positions': initial_values['positions']}
            
            #trace = pm.sample(10, step=step, start=start,
        else:
            offsets_mu=0
            start={}
            
            
            # {'n_changepoints':n_changepoints,'number_mu':number_mu,'positions':positions,'offsets':offsets}
        
        # Priors for unknown model parameters
        offset = pm.Normal('offset', mu=0, sigma=1)
        trend = pm.Normal('trend', mu=0, sigma=1) 
        sigma = pm.HalfNormal('sigma', sigma=sigma_noise)   

        if estimate_number_mu:
            act_number_sigma = pm.Gamma('act_number_sigma', 1.0, 1.0)
            act_number = pm.Beta('act_number', 1.0, act_number_sigma, shape=n_changepoints)
        else:
            #act_number = pm.Beta('act_number', 1.0, 3.0, shape=n_changepoints)
            
            act_number = pm.Bernoulli('act_number', p = p_, shape=n_changepoints)

        if not change_offsets:
            mult_offsets=0.
        else:
            mult_offsets=1. 
        if offsets_opt=='normal':
            if estimate_offset_sigma:
                offset_sigma = pm.HalfNormal('offset_sigma', sigma=offsets_std)   
                offsets = pm.Normal('offsets', mu=offsets_mu, sigma=offset_sigma,
                                    shape=n_changepoints)*mult_offsets              
            else:
                offsets = pm.Normal('offsets', mu=offsets_mu, sigma=offsets_std,
                                    shape=n_changepoints)*mult_offsets  

                
                
                
                
        #mult=np.zeros(n_changepoints)
        #arr=np.arange(n_changepoints)+0.5
        mult=pm.Deterministic('mult', (act_number> 0.5)*1) 
        #mult = (act_number> 0.5)*1 #(arr<act_number) *1 #[arr<act_number]=1 # array with 1,0 defining changep =True/False
        offsets=offsets*mult
        if distribute_offsets:
            # initial distribution
            mup=np.linspace(xmin,xmax,n_changepoints)
            mu_pos=pm.Uniform('mu_pos',mup, lower=xmin, upper=xmax, shape=n_changepoints) 

        else:
            mu_pos=pm.Uniform('mu_pos', lower=xmin, upper=xmax, shape=n_changepoints) 

        s = pm.Normal('positions',  mu=mu_pos, sigma=5, shape=n_changepoints)

        A = (x[:, None] >= s) * 1
        offset_change = det_dot(A, offsets)

        if change_trend:
            if estimate_trend_inc_sigma:
                trend_inc_sigma_est = pm.HalfNormal('trend_inc_sigma_est', sigma=trend_inc_sigma)  
                trend_inc = pm.Normal('trend_inc', mu=0, sigma=trend_inc_sigma_est,shape=n_changepoints)

            else:

                trend_inc = pm.Normal('trend_inc', mu=0, sigma=trend_inc_sigma,shape=n_changepoints)
            trend_inc=trend_inc*mult
            gamma = -s* trend_inc

            trend_inc=det_dot(A, trend_inc)
            trend=trend+trend_inc
            A_gamma=det_dot(A, gamma)
            offset_change=offset_change+A_gamma
        if post_seismic:
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


