#    GPLv3 License

#    DiscOTimeS: Automated estimation of trends, discontinuities and nonlinearities
#    in geophysical time series
#    Copyright (C) 2021  Julius Oelsmann

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

#    main discotimes class


import pandas as pd
import numpy as np
import xarray as xr
from theano.tensor import *
from theano import tensor
from arviz import *
import pickle
from matplotlib import pyplot as plt
import matplotlib
import pymc3 as pm
import arviz as az
import os
import datetime
import copy
from .models import *


    
    
    

class discotimes:
    """ estimates of DIScontinuities in TIME Series """
    
    def __init__(self, timser,settings={},sample_specs={},
                 name='ilame',testing=False,test_model=None):
        
        """Estimate discontinuities (i.e. jumps), trend changes,
        seasonal signals and AR1 noise in time series

        Parameters
        ----------
        timser : pd.Series, xr.Dataarray
            Time series to be fit with the model
        settings : dict
            Dictionary of settings 
        name : str
            Name of the data
        testing : bool
            Use user-defined pm.model         
        test_model : pm.model
            User-defined test-model 
            
        Attributes
        ----------
        obs : discotimes.observed
            object contains information on input data
        specs : dict
            model settings
        model : pm.model
            model design to fit data
        trace :  InferenceData, dict
            Either full (InferenceData) or compressed (dict) trace,
            Compressed data provide the mean and std computed over all iterations per chain
        compressed : bool
            state of trace compression
        chain_stats : dict
            'stats' contains a comparison based on 'loo';
            Indices represent different chain selections
        convergence_stats : dict
            convergence of trend and sigma-noise (should be within +-1)
        random : list
            list of selected states (posterior means) of the un-compressed chains
        name : str
            Name of the data
        initial_values : dict
            Dictionary containing information of known dicontinuity positions and sizes

        """
        self.obs=observed(timser)
        self.specs = settings['model_settings']
        self.sample_specs=settings['run_settings']
        self.specs['model_type'] = 'standard'
        if testing: 
            self.model = test_model(observed=self.obs,**self.specs)
            self.specs['model_type'] = 'test_model'            
            # ! U may not be able to use all of the functions (like plotting) with this option
        else:
            self.model = discotimes_model(observed=self.obs,**self.specs)            
        self.trace = None
        self.compressed=False
        self.chain_stats = {}
        self.chain_stats_update = {} # delete
        self.convergence_stats = {}
        self.random=[]
        self.name = name        
        self.initial_values=settings['model_settings']['initial_values'] # usually empty
        
    def run(self,n_samples=8000,tune=2000,cores=8,nuts={'target_accept':0.9},
            return_inferencedata=True,compress=True,keep_trace=False,**kwargs):
            
        """starts sampling
        see https://docs.pymc.io/api/inference.html

        Parameters
        ----------
        n_samples : int
            number of samples to draw
        tune : int
            number of tunings steps
        cores : int
            number of cores to use in parallel = number of chains            
        nuts : dict
            NUTS settings
        return_inferencedata : bool
            returns Inferencedata object
        compress : bool
            compress chains afterwards   
        **kwargs : dict
            parameter for pm.sample
        """
        
        if 'offsets' in self.initial_values:
            initialize=True
            print('manually initialize with: ')
            print(self.initial_values)
            start = {'offsets': self.initial_values['offsets'],
                     'positions': self.initial_values['positions']}
        else:
            start={}

        with self.model:
            self.trace = pm.sample(n_samples,tune=tune,nuts=nuts,cores=cores,
                                   return_inferencedata=return_inferencedata,start=start,**kwargs)            
        self.check_convergence()   # check for parameter convergence
        
        if compress:
            self.compress(keep_trace=keep_trace)    
        return self
    

    def save(self,save_dir='',allow_pickle=True):
        """
        save object
        
        !!!!! include csv output
        
        Parameters
        ----------
        
        save_dir: str
            output directory
        
        """
        
        if self.name == '' or save_dir=='':
            raise Exception('Define Object.name and save_dir before saving!')
        else:
            if allow_pickle:
                with open(save_dir+self.name+'.dt', 'wb') as ilame_file:
                    pickle.dump(self, ilame_file, pickle.HIGHEST_PROTOCOL)
            else:
                self.trace.to_nectdf(save_dir+self.name)

    def load(save_dir='',name='',allow_pickle=True):
        """ load object
        
        Parameters
        ----------
        
        save_dir: str
            output directory
        
        name : str
            name of object (without ending .dt)
            
        """
        if name == '' or save_dir=='':
            raise Exception('Define name and save_dir before loading!')
        else:
            if allow_pickle:
                with open(save_dir+name+'.dt', 'rb') as ilame_file:
                    self = pickle.load(ilame_file)  
            else:
                self=np.load(save_dir+name+'.npy',allow_pickle=True)
            if not hasattr(self, 'model_type'):
                self.specs['model_type'] = 'exp'
            else:
                self.specs['model_type'] = self.model_type
            return self
        
    def check_convergence(self,parameters=['trend','sigma']):
        """checks geweke scores for paramater
        
        References:
        Geweke (1992)
        
        statistic should oscillate between +- 1
        to indicate convergence.
        This can only be done before compression!
        
        Parameters
        ----------
        
        parameters : list
            list of parameters to test
        """
        stats={}
        for par in parameters:
            sub_set=[]
            for i in self.trace.posterior.chain.values:
                sub_set.append(pm.geweke(self.trace.posterior[par][i,:], intervals=50)[:,1])
            stats[par]=pd.DataFrame(np.asarray(sub_set).T,columns=self.trace.posterior.chain.values)
        self.convergence_stats = stats
        
                
    def compress(self,burn=0.,random=50,how='mean',keep_trace=False):
        """compress trace, 
        reduces data-size by factor ~n_samples
        
        computes mean and std-dev along draw dimensions
        1. derive statistics options
        2. random draws from chains
        3. mean, std of trace
        4. set compressed to True
        
        Parameters
        ----------
        
        burn : int
            reject samples at the beginning     
        random : int
            step size for selecting selected samples from chains
        how : str
            'mean' or 'median' as expectation 
        keep_trace : bool
            keep old full trace 
        """
        
        self.get_best_model(return_=False) # set and safe statistics

        RANDOM=[]
        if 'mu' in self.trace.posterior:
            for chain in self.trace.posterior['chain'].values:
                RANDOM.append(self.trace.posterior.mu[chain,:random,:])         
            self.random=RANDOM

        COMPRESSED_TRACE={} # compress trace
        for op in [how,'std']:
            elements=[]
            for element in ['posterior','sample_stats','log_likelihood']:
                elements.append(getattr(getattr(self.trace,element),op)(dim='draw'))

            COMPRESSED_TRACE[op]=az.InferenceData(posterior=elements[0],sample_stats=elements[1],
                                    log_likelihood=elements[2])            
        if keep_trace:
            COMPRESSED_TRACE['full_trace']=copy.deepcopy(self.trace)
        self.trace=COMPRESSED_TRACE

        self.compressed = True
        print('successfully compressed trace')
    
    def get_best_model(self,crit='standard',return_stats=False,return_=True):
        """computes model statistics and returns different chain selections
        
        Parameters
        ----------
        
        crit : int
            chain selection criterion:
            - 'standard' (default): Select best model (according to loo) when the number 
                of estimated offset is equal to the average number of all chains 
            - 'all': Select best model (according to loo) among all chains
            - 'lowest_p': Select best model based on the lowest p_value 
                (lowest estimated effective number of parameter)          
        return_stats : bool
            returns statistics
        return_ : bool
            return data
        
        Returns
        -------
        
        stats : pd.DataFrame
            pm.compare - statistics
            see Vehtari A, Gelman A, Gabry J (2017) Practical bayesian 
                model evaluation using leave-one-out cross-validation and
                waic. Statistics and Computing 27, DOI 10.8931007/s11222-016-9696-4
            or https://docs.pymc.io/notebooks/model_comparison.html
        
        """
        
        all_={}
        nums=[]
        mean_act_number = np.round(self.trace.posterior.mean(dim='draw').mean(dim='chain')['act_number'])
        # average number of changepoints across all realizations
        for i in range(len(self.trace.posterior.chain)):
            new=self.trace.sel(chain=[i]) 
            act_numb=np.sum(((self.trace.posterior['mult'].mean(dim='draw')[i,:]> 0.5)*1).values)
            all_[str(i)]=new
            nums.append(act_numb)  
        stats=pm.compare(all_)


        stats['chain_index']=np.asarray(stats.index.values).astype(int)
        stats=stats.sort_values(by='chain_index')
        stats['num']=nums
        stats=stats.sort_values(by='rank')
        # crit='standard'   # is ensemble mean of changepoints and then best p_loo

        i=stats['num'].mean().round()
        while stats.where(stats['num']==i).dropna().empty:
            i=i-1
        best_chain_standard=int(stats.where(stats['num']==i).dropna().iloc[0].name)

        # crit='all' # is best p_loo
        best_chain_all=int(stats.iloc[0].name)

        # crit='lowest_p' # is lowest number of effective parameters
        best_chain_lowest_p=int(stats.where(stats['p_loo']==stats['p_loo'].min()).dropna().iloc[0].name)

        self.chain_stats={'stats':stats,
                          'avg_best_loo':best_chain_standard,
                          'best_loo':best_chain_all,
                          'lowest_p_loo':best_chain_lowest_p}
        if return_:
            if return_stats:
                return best_chain,stats
            else: 
                return best_chain

        
    def plot_chain(self,ax,chains='all',num=50,burn=1000):
        """


        """    
        #self.trace.posterior['mu']=self.trace.posterior.mu*self.obs.std
        #self.trace.posterior['sigma']=self.trace.posterior['sigma']*self.obs.std

        x=self.obs.series_clean.index
        if chains=='all':
            if 'mu' in self.trace['mean'].posterior:
            
                for chain in self.trace['mean'].posterior['chain'].values:
                    ax.plot(np.repeat(x[:,np.newaxis], num, axis=1).T,
                            self.random[chain]*self.obs.std,color='blue',alpha=0.1)     

                y_mod=self.trace['mean'].posterior.mu[:,:].mean(dim='chain')*self.obs.std
                ci=self.trace['mean'].posterior.sigma[:].mean(dim='chain')*2*self.obs.std


                ax.fill_between(x, (y_mod-ci), (y_mod+ci), color='b', alpha=.1,label='randomized model realizations \n and 2-sigma confidence bounds')
        return ax      

    def plot_chain_clean(self,ax,chain,plt_text=True,
                         lines_max=False,alpha=1.,**kwargs):
        """
        plot derived model (chain)
        
        Parameters:
        -------------------
        ax: axis to plot on
        
        chain: int, chain selector
        
        color, label: plot properties
        -------------------
        
        returns ax
        
        """ 
        
        
        y_model_data=self.ymod(chain,denormalize=True,**self.specs)
        #{'ymod':y_mod,'trend':trend,'post':post,'offset_change':offset_change} 
        y_model=y_model_data['ymod']
        y_mean=np.mean(y_model)
        y_std=np.std(y_model)
        
        index=self.obs.series_clean.index
        off_x=(index[-1]-index[0])/100


        ax.plot(index,y_model,alpha=alpha,**kwargs)
        
        #alpha_in=0.2
        #if kwargs['label'] == 'best_loo':
        alpha_in=alpha
        
        if isinstance(y_model_data['trend'], float):
            trends=y_model_data['trend']
            ax.text(index[0]+off_x,y_mean, str(np.round(trends,3)),alpha=alpha_in)
        else:
            positions=np.sort(y_model_data['positions_v'][:1+int(y_model_data['act_num'])])            
            trends=pd.DataFrame(y_model_data['trend']).drop_duplicates().values.squeeze()
            if plt_text:
                if trends.shape == ():
                    ax.text(index[0]+off_x,y_mean, str(np.round(trends,3)),alpha=alpha_in)                
                else:
                    if len(positions)==len(trends):
                        add_=0
                    else:
                        add_=1
                    sign=-1
                    for i in range(len(trends)):
                        
                        ax.text(positions[i]+off_x,y_mean+0.5*y_std*sign, str(np.round(trends[i],3)),alpha=alpha_in)
                        sign=sign*-1
                    sign=-1
                    ax.text(positions[0]+off_x,y_mean+0.5*y_std*sign, str(np.round(trends[0],3)),alpha=alpha_in) 

        #ax.plot(self.obs.x,y_model_data['trend']*self.obs.x+y_model_data['offset_change'],color='orange',alpha=0.5)
            if lines_max:
                ax.vlines(positions[1:], -1,1,alpha=alpha_in,linestyles='dashed')
            else:
                ax.vlines(positions[1:], np.min(y_model),np.max(y_model),alpha=alpha_in,linestyles='dashed')            

        return ax
    
    def plot(self,chains='all',other=None,
             label_other='gps',normalize=False,save=False,
             plot_all_chains=True,save_dir='',save_name='',crit='best_loo'):
        """
        Parameters 
        ----------
        chains: str, default: 'all'
            which chains to plot; 'all','best'

        other: pd.Series, default: None
            external time series, units should be m
        """
        if self.compress == False:
            raise Exception('Compress object before plotting! ( use object.compress() )')
            
        fig, axs = plt.subplots(nrows=2, ncols=1,sharex=False,
                                       gridspec_kw={'height_ratios': [2, 1]},
                                       figsize=(10, 8))
        fig.set_tight_layout({'rect': [0, 0, 1, 0.95], 'pad': 1.5, 'h_pad': 1.5})

        fig.suptitle('Observed and modelled time series')

        axs[0]=self.plot_chain(axs[0],chains=chains)
        self.obs.series_clean.plot(ax=axs[0],color='tomato',label='observed',alpha=0.85)
        
        if other is not None:   
            if isinstance(other, dict):
                for key in other:
                    item=other[key]
                    item=item-np.mean(item)+np.nanmean(self.obs.series_clean)    
                    item.plot(ax=axs[0],alpha=1.,label=key,linewidth=4,color='orange')                
            else:
                other=other-np.mean(other)+np.nanmean(self.obs.series_clean)    
                other.plot(ax=axs[0],alpha=0.8,label=label_other,color='lime',linewidth=4)                
        
        #axs[0].plot(self.obs.series_clean,color='orange',label='observed')
        kwargs={'color':'blue','label':'best chain'}
        #axs[1]=self.plot_chain_clean(axs[1],self.chain_stats[crit],alpha=1.,**kwargs)
        plt_text=True
        if plot_all_chains:
            for i,alpha in zip(self.chain_stats['stats'].index.values.astype(int),np.linspace(1,0.1,len(self.chain_stats['stats']))):
                if self.chain_stats[crit]!=i:
                    plt_text=False
                kwargs={'color':'blue','label':'chain '+str(i),'alpha':alpha,'plt_text':plt_text}
                axs[1]=self.plot_chain_clean(axs[1],i,**kwargs)
        axs[0].legend()
        axs[1].legend(ncol=5)

        axs[1].set_xlabel('year')
        axs[0].set_ylabel('Height')
        axs[1].set_ylabel('Height')
        if save:
            plt.savefig(save_dir+save_name)
    
    def positions_to_date(self,positions):
        """converts numeric positions to dates

        Returns
        --------
        vector : 
        
        first element is starting date of observations
        the rest are changepoint-positions or not fitted positions 
        (if number of changepoints is less then vector-length )
        
        """
        x=self.obs.x
        args_m=[]
        args_m.append(0)
        for pos in positions:
            args_m.append(np.argmin(abs(x-pos)))
        return self.obs.series_clean.index[args_m]

    def ymod(self,chain,n_changepoints=5,denormalize=True,change_trend=True,
             trend_inc=None,change_offsets=True,post_seismic=False,
             trend_independent=False,annual_cycle=False,**kwargs):

        """convert model samples to dictionary of parameters

        Parameters
        ----------

        chain : int
            number of chain to look at
        n_changepoints : int 
            maximum allowed number of changepoints
        denormalize : bool
            re-scale data

        ... other model settings parameters (see model_settings)

        Returns
        --------
        
        dictionary of parameters

        """

        data=self.trace['mean'].sel(chain=[chain]).posterior.squeeze(dim='chain')
        data_std=self.trace['std'].sel(chain=[chain]).posterior.squeeze(dim='chain')        
        mult = ((data['mult']> 0.5)*1).values # 1,0 vector of changepoints
        num = np.sum(mult)

        offsets=data.offsets.values*mult
        offset=data.offset
        x=self.obs.x

        if not change_offsets:
            offsets=offsets*0
        positions = data.positions.values
        A = (x[:, None] >= positions) * 1
        offset_change = elem_matrix_vector_product(A, offsets)
        trend=data.trend.values
        trend_err=data_std.trend.values
        trend_v=trend
        trend_err_v=trend_err
        if change_trend:      
            trend_inc=data.trend_inc.values*mult
            trend_inc_err=data_std.trend_inc.values*mult 

            if trend_independent: # no autocorrelation in trend itself
                trend_v=copy.deepcopy(np.append(trend_v, (trend_inc+trend_v)*mult))
                trend_err_v=copy.deepcopy(np.append(trend_err_v, trend_inc_err))
                gamma=-positions* trend_inc+np.roll(np.diff(positions,append=s[-1])*trend_inc,1).cumsum()
                A = (x[:, None] >= s) * 1
                A_alt=np.diff(A,axis=1,append=0)*-1
                trend_inc=elem_matrix_vector_product(A_alt, trend_inc)
                A_gamma=elem_matrix_vector_product(A_alt, gamma)
                trend=trend+trend_inc
            else:
                trend_v=copy.deepcopy(np.append(trend_v, (trend_inc+trend_v)*mult))
                trend_err_v=copy.deepcopy(np.append(trend_err_v, trend_inc_err))

                gamma = -positions * trend_inc
                trend_inc=elem_matrix_vector_product(A, trend_inc)
                trend=trend+trend_inc
                A_gamma=elem_matrix_vector_product(A, gamma)

            offset_change=offset_change+A_gamma
            trend_v_sorted=pd.DataFrame(trend).drop_duplicates()
            start_pos=self.obs.series_clean.index[trend_v_sorted.index.values]
            end_pos=self.obs.series_clean.index[np.roll(trend_v_sorted.index.values-1,shift=-1)]
            diff=end_pos-start_pos

            # trend uncertainty and offsets
            trend_inc_err=data_std.trend_inc.values*mult 
            ff = pd.DataFrame(np.vstack([positions*mult,trend_inc_err*mult,offsets*mult]).T,columns=['pos','trend_inc_err','offsets'])
            x=self.obs.x
            args_m=[]
            for pos in positions:
                args_m.append(np.argmin(abs(x-pos)))
            ff['real_p']=self.obs.series_clean.index[args_m]
            ff=ff[ff['pos']!=0].sort_values(by='pos')
            ff2=copy.deepcopy(ff)
            if len(ff)==0:
                full_err = pd.DataFrame(np.vstack([0,trend_err,offset,start_pos[0]]).T,columns=['pos','trend_inc_err','offsets','real_p'])

            else:
                ff2.iloc[0]=np.vstack([0,trend_err,offset,start_pos[0]]).flatten()
                full_err=pd.concat([ff,ff2.iloc[0:1]]).sort_values(by='pos')



        if post_seismic:
            new=(A*x[:,None])-s
            #new[new<0]=0
            post=np.nansum(data.c_constant.values*(1-np.exp(-((new-new*[new<0]).squeeze()/data.etau.values)))*mult,axis=1)
            y_mod=offset_change + trend*x + data.offset.values +post
        else:
            post=np.nan
            y_mod=offset_change + trend*x + data.offset.values 
        if annual_cycle:
            annual=data.annual.values
            y_mod=y_mod+annual

        if change_trend and change_offsets:
            trend_v =np.append(trend_v[0],trend_v[1:][mult==1])
            positions = positions[mult==1]
            trend_err_v=np.append(trend_err_v[0],trend_err_v[1:][mult==1])
            positions_v=self.positions_to_date(positions)

        elif not change_trend and not change_offsets:
            trend_v =np.asarray(trend)
            trend_err_v=np.asarray(trend_err_v)
            positions_v=self.obs.series_clean.index[0]  
            mult=mult*0
            num=0
            trend_v_sorted=pd.DataFrame([trend])
            start_pos=self.obs.series_clean.index[0]
            end_pos=self.obs.series_clean.index[-1]
            diff=end_pos-start_pos        
            full_err = pd.DataFrame(np.vstack([0,trend_err,offset,start_pos]).T,columns=['pos','trend_inc_err','offsets','real_p'])

        else:
            positions_v=self.positions_to_date(positions*mult)

        if denormalize:
            return {'ymod':y_mod*self.obs.std,'trend':trend*self.obs.std,
                    'post':post*self.obs.std,'offset_change':(offset_change+data.offset.values)*self.obs.std,
                   'trend_v':trend_v*self.obs.std,'trend_err_v':trend_err_v*self.obs.std,'act_num':np.round(num),
                    'positions_v':positions_v,'mult':mult,'trend_v_sorted':trend_v_sorted.values.flatten()*self.obs.std,
                    'start_pos':start_pos,'end_pos':end_pos,'diff':diff,
                    'trend_un':full_err['trend_inc_err'].values*self.obs.std,
                    'offsets':full_err['offsets'].values*self.obs.std}          
        else:
            return {'ymod':y_mod,'trend':trend,'post':post,'offset_change':offset_change+data.offset.values}  
        
        
    def approximate_initial_offsets(self,detection_threshold=15,
                                detection_resolution=4):
        
        """function to make a first guess of positions and sizes of offsets:

        * Based on consecutive differences
        * offsets are sorted by their probability
        * offsets are detected when a abs(cons.difference) is higher than
          detection_threshold times the median of all cons.differences
        * choose detection_threshold=15 to detect very obvious offsets
        
        Parameters
        ----------
        detection_threshold : int
            fraction of minimum consecutive difference to detect cp 
            (relative to all cons. differences)
        detection_resolution : int
            smoothing of the data before computing average cons. diff. ranges

        """
        new_series=pd.Series(self.obs.y,index=self.obs.series_clean.index)

        x=self.obs.x

        Frequency=pd.infer_freq(self.obs.series.index)[:1]
        time_res=str(detection_resolution)+Frequency

        DIFF=abs(new_series.diff())
        DIFFS=(new_series.diff()/DIFF.median())
        #time_res=periods=detection_resolution
        DET_max=DIFFS.resample(time_res).max()
        DET_min=DIFFS.resample(time_res).min()    

        DET_max=DET_max.where((DET_max>detection_threshold)).dropna()
        DET_min=DET_min.where((DET_min<-detection_threshold)).dropna()

        DETECTED=DIFFS.where(np.isin(DIFFS,DET_max) | np.isin(DIFFS,DET_min)).dropna()
        
        if len(DETECTED)>0:
            xmin=new_series.min()
            xmax=new_series.max()

            #sorted_index=abs(DETECTED).sort_values(ascending=False).index

            pos_approx=self.obs.x[np.isin(self.obs.series_clean.index,DETECTED.index)]
            DETECTED=pd.DataFrame(DETECTED.rename('offsets'))
            DETECTED['abs']=abs(DETECTED)
            DETECTED['x_index']=pos_approx
            DETECTED.sort_values('abs',ascending=False,inplace=True)

            offsets_approx=DETECTED['offsets'].values
            pos_approx=DETECTED['x_index'].values
            number_mu=len(pos_approx)
            n_changepoints=number_mu+3

            p_ = np.ones([n_changepoints])*0.1
            p_[:number_mu]=0.8

            positions=np.random.uniform(low=np.min(x), high=np.max(x), size=n_changepoints)
            offsets=np.random.normal(loc=0.0, scale=self.specs['offsets_std'], size=n_changepoints)

            offsets[:number_mu]=offsets_approx
            positions[:number_mu]=pos_approx
            if self.specs['model_type'] =='standard':
                INIT_VALS={'n_changepoints':n_changepoints,'p_':p_,'positions':positions,'offsets':offsets}
                if self.initial_values=={}:       # if not defined yet
                    self.initial_values=INIT_VALS # these will automatically be included in initialization
                self.specs['n_changepoints']=n_changepoints
                self.specs['p_']=p_
                self.specs['initial_values']=INIT_VALS
                self.specs['estimate_number_mu']=False
                self.specs['estimate_offset_sigma']=False   
                self.model = discotimes_model(observed=self.obs,**self.specs)
    
    def to_nc(self):
        """converts output to xr.DataSet ~ netcdf 
        
        """
        ALL_DATA=np.empty([21,1,self.sample_specs['cores'],self.specs['n_changepoints']+1])*np.nan
        iindex=0
        all_stats=self.chain_stats    
        bchain=all_stats['best_loo']
        bchain_lowp=all_stats['lowest_p_loo']
        bchain_standard=all_stats['avg_best_loo']        
        stats=all_stats['stats']
        sta=stats.sort_index()
        last_obs=(self.obs.series_clean.index[-1]-pd.Timestamp('1950')).days
        first_obs=(self.obs.series_clean.index[0]-pd.Timestamp('1950')).days 
        for chain in np.arange(self.sample_specs['cores']):
            ymod=self.ymod(chain,**self.specs)
            trend_v=pd.DataFrame(ymod['trend']).drop_duplicates().values.flatten()

            ALL_DATA[0, iindex, chain, :len(trend_v)]=trend_v # ymod['trend_v']
            ALL_DATA[1, iindex, chain, :len(ymod['trend_err_v'])]=ymod['trend_err_v']    
            ALL_DATA[2, iindex, chain, 0]=ymod['act_num']      
            ALL_DATA[3, iindex, chain, :len(ymod['positions_v'])]=ymod['positions_v']  

            ALL_DATA[5, iindex, chain, 0]=sta['loo'][chain]   
            ALL_DATA[6, iindex, chain, 0]=sta['p_loo'][chain]
            # new values bro!
            ALL_DATA[10, iindex, chain, :len(ymod['trend_v_sorted'])]=ymod['trend_v_sorted']
            ALL_DATA[11, iindex, chain, :len(ymod['trend_v_sorted'])]=(ymod['start_pos']-pd.Timestamp('1950')).days.astype(float)
            ALL_DATA[12, iindex, chain, :len(ymod['trend_v_sorted'])]=(ymod['end_pos']-pd.Timestamp('1950')).days.astype(float)
            ALL_DATA[13, iindex, chain, :len(ymod['trend_v_sorted'])]=ymod['diff'].days.astype(float)

            ALL_DATA[19, iindex, chain, :len(ymod['trend_un'])]=ymod['trend_un']
            ALL_DATA[20, iindex, chain, :len(ymod['offsets'])]=ymod['offsets']

        ALL_DATA[7, iindex, 0, 0]=last_obs
        ALL_DATA[8, iindex, 0, 0]=first_obs      
        ALL_DATA[4, iindex, 0, 0]=bchain   
        ALL_DATA[14, iindex, 0, 0]=bchain_lowp   
        ALL_DATA[17, iindex, 0, 0]=bchain_standard
        ALL_DATA[9, iindex, :, 0]=(self.trace['mean'].posterior['sigma']*self.obs.std).values

        ALL_DATA[18, iindex, :, 0]=self.convergence_stats['trend'].iloc[-1].values  # geweke convergence stats


        if 'accepted' in self.trace['mean'].sample_stats:
            ALL_DATA[15, iindex, :, 0]= self.trace['mean'].sample_stats['accepted'].values
        elif 'mean_tree_accept' in self.trace['mean'].sample_stats:
            ALL_DATA[15, iindex, :, 0]= self.trace['mean'].sample_stats['mean_tree_accept'].values

        ALL_DATA[16, iindex, :, 0]= self.trace['mean'].sample_stats['diverging'].values

        ds = xr.Dataset({'trend': (['x','chain', 'v_dim'],  ALL_DATA[10,:,:,:]), 
                     'trend_err': (['x','chain', 'v_dim'],  ALL_DATA[19,:,:,:]),                     
                     'offsets': (['x','chain', 'v_dim'],  ALL_DATA[20,:,:,:]),                                                                  
                     'start_pos': (['x','chain', 'v_dim'],  ALL_DATA[11,:,:,:]), 
                     'end_pos': (['x','chain', 'v_dim'],  ALL_DATA[12,:,:,:]),
                     'diff': (['x','chain', 'v_dim'],  ALL_DATA[13,:,:,:]), 
                     'number_cp': (['x','chain'],  ALL_DATA[2,:,:,0]),
                     'best_loo': (['x'],  ALL_DATA[4,:,0,0]), 
                     'lowest_p_loo': (['x'],  ALL_DATA[14,:,0,0]),
                     'avg_best_loo': (['x'],  ALL_DATA[17,:,0,0]),
                     'loo': (['x','chain'],  ALL_DATA[5,:,:,0]), 
                     'p_loo': (['x','chain'],  ALL_DATA[6,:,:,0]),
                     'last_obs': (['x',],  ALL_DATA[7,:,0,0]), 
                     'first_obs': (['x'],  ALL_DATA[8,:,0,0]),
                     'sigma_noise': (['x','chain'],  ALL_DATA[9,:,:,0]),
                     'accepted': (['x','chain'],  ALL_DATA[15,:,:,0]),
                     'geweke_converge': (['x','chain'],  ALL_DATA[18,:,:,0]),    
                     'diverging': (['x','chain'],  ALL_DATA[16,:,:,0])},

                      coords={'name': (['x'], [self.name]),
                        'chain':np.arange(self.sample_specs['cores'])})  

        t_attrs={'units' : "days since 1950-01-01 00:00:00",'calendar' : "proleptic_gregorian"}
        ds['start_pos'].attrs=t_attrs
        ds['end_pos'].attrs=t_attrs
        ds['last_obs'].attrs=t_attrs
        ds['first_obs'].attrs=t_attrs
        ds['trend'].attrs={'units' : "input unit/year",'long_name' : "Piecewise Trend"}
        ds['trend_err'].attrs={'units' : "input unit/year",'long_name' : "Trend uncertainty (1-sigma)"}
        ds['offsets'].attrs={'units' : "input unit",'long_name' : "discontinuity size"}
        ds['number_cp'].attrs={'units' : "",'long_name' : "number of change points"}

        ds['diff'].attrs={'units' : "days",'long_name' : "position difference"}


        ds.attrs=self.specs
        ds.attrs['info']="Data created with DiscoTimeS on "+str(datetime.date.today())
        return ds

class observed():
    
    """ class to handle input data"""

    def __init__(self, timser):
        """Normalizes input data

        Parameters
        ----------
        timser : pd.Series, xr.Dataarray
            Time series to be fit with the model
        
        
        Attributes
        ----------
        x : np.array
            numeric 'time steps' in years; starts from zero
        y : np.array
            normalized data       
        std : float
            standard deviation of original data
        index : np.array
            absolute 'time step' position of original data
        series_clean : pd.Series
            normalized and de-NaNed time series
        X_mat : np.array
            Matrix used to fit multi-year monthly means
        freq : str
            time series frequency; 'D','W','M','Y'
        """  
        
        self.series = timser
        self.x,self.y,self.std,self.index,self.series_clean,self.X_mat,self.freq = self.normalise()

    def normalise(self,normalize=True,rmv_nan=True):
        """ normalizes time-series for Bayesian model
        divides by 600D rolling median std 
        and shifts starting point to zero    
        
        Parameters
        ----------
        normalize : bool
            set normalize    
        rmv_nan : bool
            remove nans
      
        Returns
        ----------  
        x,y,std_data,index,series,X_mat,freq 
        """
        # remove nans from start and end
        if isinstance(self.series, pd.Series):
            df=self.series
        else:
            df=pd.Series(self.series.values.squeeze(),index=self.series.time.values)
        freq=pd.infer_freq(df.index)[:1]
        print('Normalize data')
        print('Frequency: '+freq)
        first_idx = df.first_valid_index()
        last_idx = df.last_valid_index()
        print('first: ', first_idx,' last: ', last_idx)
        series=df.loc[first_idx:last_idx]
        # normalize
        if normalize:
            y = series.values
            first_data = y[0]

            std_data = np.nanstd(y)
            std_data = df.rolling('600D').std().median()
            y = (y - first_data) / std_data
            series=series-first_data
        else:
            std_data=1.
        index=np.arange(len(y))
        
        if freq=='M':
            divider=12.
        elif freq=='W':
            divider=52.
        elif freq=='D':
            divider=365.25
        elif freq=='Y':
            divider=1.
        
        x=index/divider    
        # remove all nans
        if rmv_nan:
            y_notnan=~np.isnan(y)
            x=x[y_notnan]
            y=y[y_notnan]
            index=index[y_notnan]
            series=series.dropna()
        
        # matrix for annual cyle
        month=series.index.month
        months=np.repeat((np.arange(12)+1)[np.newaxis,:], len(series), axis=0)
        X_mat=(months == np.repeat(month[np.newaxis,:], 12, axis=0).T)*1
        return x,y,std_data,index,series,X_mat,freq     
    

def file_reader(file,variable='auto',resample='D'):
    
    """ read different file types
    
    'txt','nectdf (.nc)','tenv3', 'txyz2'
    
    Parameters
    ----------
    
    file : str
        location of file
    variable : str
        variable to read from file 
        (default: 'auto', selects the first availabe variable/column)
        
    """

    ending = file.split(".",1) 

    if len(ending)==1:
        ending ='txt'
    else:
        ending=ending[1]
    if ending =='txt' or ending =='':
        if variable=='auto':
            variable='Height'
        data=pd.read_csv(file,delim_whitespace=True)
        data['Year']=pd.to_datetime(data['Year']-1970.,unit='Y')
        data.set_index('Year',inplace=True)
        data=data.resample(resample).mean()[variable] 
    elif ending =='tenv3':
        if variable=='auto':
            variable='____up(m)'
        data=pd.read_csv(file,delim_whitespace=True,header=None)
        data['Year']=pd.to_datetime(data[2]-1970.,unit='Y')
        data.set_index('Year',inplace=True)
        data=data.resample(resample).mean()[variable]   
    elif ending =='txyz2':
        if variable=='auto':
            variable=3
        file = 'discotimes/examples/HOB2'
        data=pd.read_csv(file,delim_whitespace=True,header=None)
        data['Year']=pd.to_datetime(data[2]-1970.,unit='Y')
        data.set_index('Year',inplace=True)
        data=data.resample(resample).mean()[variable]
    elif ending =='nc':
        data = xr.open_dataset(file)
    else:
        raise Exception('File of type *.'+ending+' not implemented')
    return data    