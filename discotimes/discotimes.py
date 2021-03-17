from sealeveltools.sl_class import *
from load_files import *
from working_scripts.vlad_globcoast_functions import *
from theano.tensor import *
from theano import tensor
from arviz import *
import pickle

from working_scripts.vlame_sub_models import vlam_exp_independent

"""
changes:

- changepoint position definition


todos:

- enhanced model selection!! take care how many traces!
- seasonal component
- change how trends are added ...
- adopt better model

- test better definition of changepoint positions # its clear that vector with shape make independet variables
- + offset amplitudes
"""


class ilame():
    """
    Irregular land motion estimator based on bayesian modeling
    """
    
    def __init__(self, timser,specs={},sample_specs={},name='ilame',testing=False,test_model='exp'):
        self.obs=observed(timser)
        self.specs = specs
        self.specs['model_type'] = test_model
        if testing: 
            if test_model=='exp':          # !! different tested models!
                self.model = vlam_exp(observed=self.obs,**self.specs)

            if test_model =='exp_independent_cp':
                self.model = vlam_exp_independent(observed=self.obs,**self.specs)

        else:
            self.model = vlam(observed=self.obs,**self.specs)
            
        self.trace = None
        self.compressed=False
        self.chain_stats = {}
        self.chain_stats_update = {} # delete
        self.convergence_stats = {}
        self.random=[]
        self.name = name        
        self.initial_values={}
        self.model_type=test_model
        
    def run(self,n_samples=4000,tune=2000,cores=8,
            nuts={'target_accept':0.9},return_inferencedata=True,compress=True,**kwargs):

        if 'offsets' in self.initial_values:
            initialize=True
            print('manually initialize with: ')
            print(self.initial_values)
            start = {'offsets': self.initial_values['offsets'],
                     'positions': self.initial_values['positions']}
        else:
            start={}
            
        #if self.model_type =='exp_independent_cp':    # change that again!
        #    with self.model:
                # draw 500 posterior samples

        #        self.trace = pm.sample(n_samples,tune=tune,cores=cores,
        #                               return_inferencedata=return_inferencedata,start=start,**kwargs)
        #else:
        with self.model:
            # draw 500 posterior samples

            self.trace = pm.sample(n_samples,tune=tune,nuts=nuts,cores=cores,
                                   return_inferencedata=return_inferencedata,start=start,**kwargs)            
        self.check_convergence()   # check for parameter convergence
        
        if compress:
            self.compress()    
        return self
    

    def save(self,save_dir='',allow_pickle=True):
        """
        save object
        """
        if self.name == '' or save_dir=='':
            raise Exception('Define Object.name and save_dir before saving!')
        else:
            if allow_pickle:
                with open(save_dir+self.name+'.il', 'wb') as ilame_file:
                    pickle.dump(self, ilame_file, pickle.HIGHEST_PROTOCOL)
            else:
                self.trace.to_nectdf(save_dir+self.name)

    def load(save_dir='',name='',allow_pickle=True):
        """
        load object
        """
        if name == '' or save_dir=='':
            raise Exception('Define name and save_dir before loading!')
        else:
            if allow_pickle:
                with open(save_dir+name+'.il', 'rb') as ilame_file:
                    self = pickle.load(ilame_file)  
            else:
                self=np.load(save_dir+name+'.npy',allow_pickle=True)
            if not hasattr(self, 'model_type'):
                self.specs['model_type'] = 'exp'
            else:
                self.specs['model_type'] = self.model_type
            return self
        
    def check_convergence(self,parameters=['trend','sigma']):
        """
        check geweke scores for paramater
        
        References

        Geweke (1992)
        
        statistic should oscillate between +- 1
        """
        stats={}
        for par in parameters:
            sub_set=[]
            for i in self.trace.posterior.chain.values:
                sub_set.append(pm.geweke(self.trace.posterior[par][i,:], intervals=50)[:,1])
            stats[par]=pd.DataFrame(np.asarray(sub_set).T,columns=self.trace.posterior.chain.values)
        self.convergence_stats = stats
        
        
    def check_convergence(self,parameters=['trend','sigma']):
        """
        check geweke scores for paramater
        """
        stats={}
        for par in parameters:
            sub_set=[]
            for i in self.trace.posterior.chain.values:
                sub_set.append(pm.geweke(self.trace.posterior[par][i,:], intervals=50)[:,1])
            stats[par]=pd.DataFrame(np.asarray(sub_set).T,columns=self.trace.posterior.chain.values)
        self.convergence_stats = stats
                
    def compress(self,burn=0.,random=50,how='mean'):
        """
        compress data
        compute mean and std-dev along draw dimension
        1. derive statistics options
        2. random draws
        3. mean, std of trace
        4. set compressed to True
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

        self.trace=COMPRESSED_TRACE

        self.compressed = True
        print('successfully compressed trace')
    
    def correct_best_model(self,crit='standard',return_stats=False,return_=True):
        """
        to be deleted this corrects the old wrong stats and adds new
        
        returns best chain or mean of best chains according to different criteria

        criteria:

        crit = 
        'standard': Select best model (according to waic) when the number 
            of estimated offset is equal to the average number of all chains
        'all': Select best model (according to waic) among all chains

        'lowest_p': Select best model based on the lowest p_value

        """
        all_={}
        nums=[]
        # average number of changepoints across all realizations
        for i in range(len(self.trace['mean'].posterior.chain)):  
            act_numb=np.sum(((self.trace['mean'].posterior['mult'][i,:]> 0.5)*1).values)
            nums.append(act_numb)  
        stats=copy.deepcopy(self.chain_stats['stats'])
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

        self.chain_stats_update={'stats':stats,
                          'standard':best_chain_standard,
                          'all':best_chain_all,
                          'lowest_p':best_chain_lowest_p}
        if return_:
            best_chain=self.chain_stats_update['crit']

            if return_stats:
                return best_chain,stats
            else: 
                return best_chain
            
    def get_best_model_update(self,crit='standard',return_stats=False,return_=True):   # also update this
        """
        returns best chain or mean of best chains according to different criteria

        criteria:

        crit = 
        'standard': Select best model (according to waic) when the number 
            of estimated offset is equal to the average number of all chains
        'all': Select best model (according to waic) among all chains

        'lowest_p': Select best model based on the lowest p_value

        """
        all_={}
        nums=[]

        # average number of changepoints across all realizations
        for i in range(len(self.trace.posterior.chain)):
            new=self.trace.sel(chain=[i])    
            act_numb=np.sum(((self.trace.sel(chain=[i]).posterior['act_number'].mean(dim='draw')> 0.5)*1).values)
            all_[str(i)]=new
            nums.append(act_numb)  
        stats=pm.compare(all_)
        stats['num']=nums

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
                          'standard':best_chain_standard,
                          'all':best_chain_all,
                          'lowest_p':best_chain_lowest_p}
        if return_:
            best_chain=self.chain_stats['crit']

            if return_stats:
                return best_chain,stats
            else: 
                return best_chain

    def get_best_model(self,crit='standard',return_stats=False,return_=True):
        """
        returns best chain or mean of best chains according to different criteria

        criteria:

        crit = 
        'standard': Select best model (according to waic) when the number 
            of estimated offset is equal to the average number of all chains
        'all': Select best model (according to waic) among all chains

        'lowest_p': Select best model based on the lowest p_value

        """
        all_={}
        nums=[]
        mean_act_number = np.round(self.trace.posterior.mean(dim='draw').mean(dim='chain')['act_number'])
        # average number of changepoints across all realizations
        for i in range(len(self.trace.posterior.chain)):
            new=self.trace.sel(chain=[i])    
            act_numb=np.round(self.trace.sel(chain=[i]).posterior['act_number'].mean().values)
            all_[str(i)]=new
            nums.append(act_numb)  
        stats=pm.compare(all_)
        stats['num']=nums

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
                          'standard':best_chain_standard,
                          'all':best_chain_all,
                          'lowest_p':best_chain_lowest_p}
        if return_:
            best_chain=self.chain_stats['crit']

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
        index=self.obs.series_clean.index
        off_x=(index[-1]-index[0])/100


        ax.plot(index,y_model,**kwargs)
        
        alpha_in=0.2
        if kwargs['label'] == 'best chain':
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
                    for i in range(len(trends)):

                        ax.text(positions[i]+off_x,y_mean, str(np.round(trends[i],3)),alpha=alpha_in)
                    ax.text(positions[0]+off_x,y_mean, str(np.round(trends[0],3)),alpha=alpha_in) 

        #ax.plot(self.obs.x,y_model_data['trend']*self.obs.x+y_model_data['offset_change'],color='orange',alpha=0.5)
            if lines_max:
                ax.vlines(positions[1:], -1,1,alpha=alpha_in,linestyles='dashed')
            else:
                ax.vlines(positions[1:], np.min(y_model),np.max(y_model),alpha=alpha_in,linestyles='dashed')            

        return ax
    
    def plot(self,chains='all',other=None,
             label_other='gps',normalize=False,save=False,
             plot_all_chains=True,save_dir='',save_name='',crit='standard'):
        """
        Parameters 
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
        axs[1]=self.plot_chain_clean(axs[1],self.chain_stats[crit],**kwargs)
        if plot_all_chains:
            for i in range(len(self.trace['mean'].posterior.chain)):
                kwargs={'color':'blue','label':'chain '+str(i),'alpha':0.2}
                axs[1]=self.plot_chain_clean(axs[1],i,**kwargs)
        axs[0].legend()
        axs[1].legend(ncol=5)


        
        axs[1].set_xlabel('year')
        axs[0].set_ylabel('Height')
        axs[1].set_ylabel('Height')
        if save:
            plt.savefig(save_dir+save_name)
    
    def trend_average(self,opt='first'):
        """
        to be testes


        returns different trend estimates based on different assumptions

        1. solely first linear part
        2. combined linear parts
        3. combined linear part + nonlinear part with nonlinear part explaining less than 1 mm/year


        """   

        best_chain=self.chain_stats['standard']
        y_model_data=self.ymod(best_chain,denormalize=True,**self.specs)
        y_model=y_model_data['ymod']
        y_mean=np.mean(y_model)
        if opt=='first':
            off_x=0.2
            trend_err=self.read_chain('trend',best_chain,func='mean')
            positions=np.sort(self.read_chain('positions',best_chain))
            if isinstance(y_model_data['trend'], float):
                trends=y_model_data['trend']
                trend1,trend2=trends
                # only one trend 
            else:
                trends=pd.DataFrame(y_model_data['trend']).drop_duplicates().values.squeeze()
                trends_err=self.read_chain('trend_inc',best_chain,func='std')
                #1. option 
                print(trends)
                if trends.shape == ():
                    trend1 = trends          
                else:
                    trend1 = trends[0]
                """
                etau_=self.read_chain('etau',best_chain,func='mean')
                c_=self.read_chain('c_constant',best_chain,func='mean')

                for i range(len(trends[1:])):
                    c=c_[i-1]
                    etau=etau_[i-1]
                    nonlin = ps_func(c,etau,1)*self.obs.std 
                    if nonlin < 0.00101:
                """
            return trend1
        elif opt=='av':
            trends=y_model_data['trend_v'][y_model_data['trend_v']!=0]
            err=y_model_data['trend_err_v'][y_model_data['trend_err_v']!=0]
            weights=(1/err)/np.sum(1/err)
            av_trend=np.average(trends, weights=weights)
            return av_trend
    
    def read_chain(self,parameter,chain,func='mean'):
        data=self.trace['func'].sel(chain=[chain]).posterior.squeeze(dim='chain')
        out=data[parameter].values.squeeze()

        if data[parameter].shape != ():
            print(out)
            print(hasattr(out, '__len__'))
            act_number=data.act_number.values
            out=out[:int(np.round(act_number))]
        return out
    
    def positions_to_date(self,positions):
        """
        return vector:
        
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
             trend_inc=None,change_offsets=True,post_seismic=False,annual_cycle=False,trend_independent=False,
             **kwargs):

        data=self.trace['mean'].sel(chain=[chain]).posterior.squeeze(dim='chain')
        data_std=self.trace['std'].sel(chain=[chain]).posterior.squeeze(dim='chain')        
        
        if self.specs['model_type'] == 'exp_independent_cp':
            mult = ((data['mult']> 0.5)*1).values
            num = np.sum(mult)
        else:    
            mult=np.zeros(n_changepoints)  
            arr=np.arange(n_changepoints)+0.5

            num=data.act_number.values
            mult[arr<num]=1
        offsets=data.offsets.values*mult
        offset=data.offset
        x=self.obs.x

        if not change_offsets:
            offsets=offsets*0
        s = data.positions.values
        positions=s
        A = (x[:, None] >= s) * 1
        offset_change = det_dot(A, offsets)
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
                trend_inc=det_dot(A_alt, trend_inc)
                A_gamma=det_dot(A_alt, gamma)
                trend=trend+trend_inc
            else:
                trend_v=copy.deepcopy(np.append(trend_v, (trend_inc+trend_v)*mult))
                trend_err_v=copy.deepcopy(np.append(trend_err_v, trend_inc_err))

                gamma = -positions * trend_inc
                trend_inc=det_dot(A, trend_inc)
                trend=trend+trend_inc
                A_gamma=det_dot(A, gamma)

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
        #plt.plot(x,y_mod)
        #plt.plot(x,y)

        if self.specs['model_type'] == 'exp_independent_cp' and change_trend and change_offsets:
            trend_v =np.append(trend_v[0],trend_v[1:][mult==1])
            positions = positions[mult==1]
            trend_err_v=np.append(trend_err_v[0],trend_err_v[1:][mult==1])
            positions_v=self.positions_to_date(positions)

        elif self.specs['model_type'] == 'exp_independent_cp' and not change_trend and not change_offsets:
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
                                detection_resolution=4,
                                method='diff'):
        """
        function to make a first guess of number and positions of offsets:

        * Based on consecutive differences
        * offsets are sorted by their probability
        * offsets are detected when a abs(cons.difference) is higher than
          detection_threshold times the median of all cons.differences
        * choose detection_threshold=15 to detect very obvious offsets

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
            if self.specs['model_type']=='exp':          # !! different tested models!
                
                INIT_VALS={'n_changepoints':n_changepoints,'number_mu':number_mu,'positions':positions,'offsets':offsets}
                self.initial_values=INIT_VALS #these will automatically be included in initialization
                self.specs['n_changepoints']=n_changepoints
                self.specs['number_mu']=number_mu
                self.specs['initial_values']=INIT_VALS
                self.specs['estimate_number_mu']=False
                self.specs['estimate_offset_sigma']=False   
                self.model = vlam_exp(observed=self.obs,**self.specs)

            if self.specs['model_type'] =='exp_independent_cp':
                
                INIT_VALS={'n_changepoints':n_changepoints,'p_':p_,'positions':positions,'offsets':offsets}
                self.initial_values=INIT_VALS #these will automatically be included in initialization
                self.specs['n_changepoints']=n_changepoints
                self.specs['p_']=p_
                self.specs['initial_values']=INIT_VALS
                self.specs['estimate_number_mu']=False
                self.specs['estimate_offset_sigma']=False   
                self.model = vlam_exp_independent(observed=self.obs,**self.specs)


class observed():
    def __init__(self, timser):
        self.series = timser
        self.x,self.y,self.std,self.index,self.series_clean,self.X_mat,self.freq = self.normalise()

    def normalise(self,normalize=True,rmv_nan=True):
        """
        normalizes time-series for state-space model approach
        divide by std and shift starting point to zero    
        
        Parameter:

        data: xr with monthly resolution

        returns: x,y,norm_factor

        """
        # remove nans from start and end
        if isinstance(self.series, pd.Series):
            df=self.series
        else:
            df=pd.Series(self.series.values.squeeze(),index=self.series.time.values)
        freq=pd.infer_freq(df.index)[:1]
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
    
    
                
class vlam_manuela(pm.model.Model):
    # 1) override init
    """
    different offset properties
    
    """
    def __init__(self, observed=None,name='', model=None,change_trend=False,n_changepoints=5,number_mu=0.1,
                     offsets_opt='normal',offsets_std=1,
                      add_to_num=0.,sigma_noise=1.,trend_inc_sigma=0.01,
                      change_offsets=True,estimate_offset_sigma=False,estimate_trend_inc_sigma=False,
                      estimate_number_mu=False,n_samples=500,target_accept=0.8,post_seismic=True):

        super().__init__(name,model)

        
        x=observed.x
        y=observed.y
            
        xmin=np.min(x)
        xmax=np.max(x)

        # Priors for unknown model parameters
        offset = pm.Normal('offset', mu=0, sigma=1)
        trend = pm.Normal('trend', mu=0, sigma=1) 
        sigma = pm.HalfNormal('sigma', sigma=sigma_noise)   

        if estimate_number_mu:
            act_number_sigma = pm.HalfNormal('act_number_sigma', sigma=number_mu)  
            act_number=pm.Poisson('act_number', mu=act_number_sigma) -add_to_num
        else:
            act_number=pm.Poisson('act_number', mu=number_mu) -add_to_num

        if not change_offsets:
            mult_offsets=0.
        else:
            mult_offsets=1. 
        if offsets_opt=='normal':
            # positive and negative offsets ...
            vec_ones=np.asarray([1,-1]*20)[:n_changepoints]

            
            
            if estimate_offset_sigma:
                offset_sigma = pm.HalfNormal('offset_sigma', sigma=offsets_std)   
                offsets = pm.Normal('offsets', mu=1, sigma=offset_sigma, shape=n_changepoints)*mult_offsets*vec_ones              
            else:
                offsets = pm.Normal('offsets', mu=1, sigma=offsets_std, shape=n_changepoints)*mult_offsets*vec_ones   

        mult=np.zeros(n_changepoints)
        arr=np.arange(n_changepoints)+0.5

        mult = (arr<act_number) *1 #[arr<act_number]=1
        offsets=offsets*mult
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
        mu = pm.Deterministic("mu", offset_change + trend*x + offset)        
        Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y)
 

                
class vlam_2(pm.model.Model):
    """
    2. development:
    
    changes:
    
    """
    def __init__(self, observed=None,name='', model=None,change_trend=False,n_changepoints=5,number_mu=0.1,
                     offsets_opt='normal',offsets_std=1,
                      add_to_num=0.,sigma_noise=1.,trend_inc_sigma=0.01,annual_cycle=False,
                      change_offsets=True,estimate_offset_sigma=False,estimate_trend_inc_sigma=False,
                      estimate_number_mu=False,n_samples=500,target_accept=0.8,post_seismic=True):

        super().__init__(name,model)

        
        x=observed.x
        y=observed.y
            
        xmin=np.min(x)
        xmax=np.max(x)
        
        
        X_mat=observed.X_mat
        
        # Priors for unknown model parameters
        offset = pm.Normal('offset', mu=0, sigma=1)
        trend = pm.Normal('trend', mu=0, sigma=1) 
        sigma = pm.HalfNormal('sigma', sigma=sigma_noise)   

        if estimate_number_mu:
            act_number_sigma = pm.HalfNormal('act_number_sigma', sigma=number_mu)  
            act_number=pm.Poisson('act_number', mu=act_number_sigma) -add_to_num
        else:
            act_number=pm.Poisson('act_number', mu=number_mu) -add_to_num

        if not change_offsets:
            mult_offsets=0.
        else:
            mult_offsets=1. 
        if offsets_opt=='normal':
            if estimate_offset_sigma:
                offset_sigma = pm.HalfNormal('offset_sigma', sigma=offsets_std)   
                offsets = pm.Normal('offsets', mu=0, sigma=offset_sigma, shape=n_changepoints)*mult_offsets              
            else:
                offsets = pm.Normal('offsets', mu=0, sigma=offsets_std, shape=n_changepoints)*mult_offsets  

        mult=np.zeros(n_changepoints)
        arr=np.arange(n_changepoints)+0.5

        mult = (arr<act_number) *1 #[arr<act_number]=1
        offsets=offsets*mult
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
        Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y)
     
    
class w_vlam(pm.model.Model):
    # 1) override init
    """
    model in (working mode)
    changes: std normalizing
    
    
    """
    def __init__(self, observed=None,name='', model=None,change_trend=False,n_changepoints=5,number_mu=0.1,
                     offsets_opt='normal',offsets_std=1,
                      add_to_num=0.,sigma_noise=1.,trend_inc_sigma=0.01,annual_cycle=False,
                      change_offsets=True,estimate_offset_sigma=False,estimate_trend_inc_sigma=False,
                      estimate_number_mu=False,n_samples=500,target_accept=0.8,post_seismic=True,AR1=False,
                      trend_independent=False):

        super().__init__(name,model)

        
        x=observed.x
        y=observed.y
            
        xmin=np.min(x)
        xmax=np.max(x)
        
        
        X_mat=observed.X_mat
        
        # Priors for unknown model parameters
        offset = pm.Normal('offset', mu=0, sigma=1)
        trend = pm.Normal('trend', mu=0, sigma=1) 
        sigma = pm.HalfNormal('sigma', sigma=sigma_noise)   

        if estimate_number_mu:
            act_number_sigma = pm.HalfNormal('act_number_sigma', sigma=number_mu)  
            act_number=pm.Poisson('act_number', mu=act_number_sigma) -add_to_num
        else:
            act_number=pm.Poisson('act_number', mu=number_mu) -add_to_num

        if not change_offsets:
            mult_offsets=0.
        else:
            mult_offsets=1. 
        if offsets_opt=='normal':
            if estimate_offset_sigma:
                offset_sigma = pm.HalfNormal('offset_sigma', sigma=offsets_std)   
                offsets = pm.Normal('offsets', mu=0, sigma=offset_sigma, shape=n_changepoints)*mult_offsets              
            else:
                offsets = pm.Normal('offsets', mu=0, sigma=offsets_std, shape=n_changepoints)*mult_offsets  

        mult=np.zeros(n_changepoints)
        arr=np.arange(n_changepoints)+0.5

        mult = (arr<act_number) *1 #[arr<act_number]=1
        offsets=offsets*mult
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
            
            if trend_independent:
                
                trend_correct_offsets = tensor.roll((tensor.roll(s,shift=-1)-s)*trend_inc,shift=1).cumsum()
                trend_correct_offsets=trend_correct_offsets-trend_correct_offsets[0]
                #gamma=-s* trend_inc+np.roll(np.diff(alt_s,append=alt_s[-1])*trend_inc,1).cumsum()
                gamma=-s* trend_inc+trend_correct_offsets
                
                s_tens=tensor.vector('s_tens')
                dd=s_tens
                ffs = theano.function([s_tens], dd)
                sou=ffs(s)

                A_alt=np.diff(((x[:, None] >= sou) * 1),axis=1,append=0)*-1
                trend_inc=det_dot(A_alt, trend_inc)
                A_gamma=det_dot(A_alt, gamma)
                trend=trend+trend_inc
                
            else:    
                
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
        if AR1: # first order autoregressive ...
            beta = pm.HalfNormal('beta', sigma=0.4)
            likelihood = pm.AR('AR1_coeff', beta, sigma=sigma, observed=y-mu) 
        else:    
            Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y)
 
                    
class vlam(pm.model.Model):
    # 1) override init
    def __init__(self, observed=None,name='', model=None,change_trend=False,n_changepoints=5,number_mu=0.1,
                     offsets_opt='normal',offsets_std=1,
                      add_to_num=0.,sigma_noise=1.,trend_inc_sigma=0.01,annual_cycle=False,
                      change_offsets=True,estimate_offset_sigma=False,estimate_trend_inc_sigma=False,
                      estimate_number_mu=False,n_samples=500,target_accept=0.8,post_seismic=True,
                      AR1=False,studentst_noise=False,distribute_offsets=False):


        super().__init__(name,model)

        
        x=observed.x
        y=observed.y
            
        xmin=np.min(x)
        xmax=np.max(x)
        
        
        X_mat=observed.X_mat
        
        # Priors for unknown model parameters
        offset = pm.Normal('offset', mu=0, sigma=1)
        trend = pm.Normal('trend', mu=0, sigma=1) 
        sigma = pm.HalfNormal('sigma', sigma=sigma_noise)   

        if estimate_number_mu:
            act_number_sigma = pm.HalfNormal('act_number_sigma', sigma=number_mu)  
            act_number=pm.Poisson('act_number', mu=act_number_sigma) -add_to_num
        else:
            act_number=pm.Poisson('act_number', mu=number_mu) -add_to_num

        if not change_offsets:
            mult_offsets=0.
        else:
            mult_offsets=1. 
        if offsets_opt=='normal':
            if estimate_offset_sigma:
                offset_sigma = pm.HalfNormal('offset_sigma', sigma=offsets_std)   
                offsets = pm.Normal('offsets', mu=0, sigma=offset_sigma, shape=n_changepoints)*mult_offsets              
            else:
                offsets = pm.Normal('offsets', mu=0, sigma=offsets_std, shape=n_changepoints)*mult_offsets  

        mult=np.zeros(n_changepoints)
        arr=np.arange(n_changepoints)+0.5

        mult = (arr<act_number) *1 #[arr<act_number]=1
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
        if AR1: # first order autoregressive ...
            beta = pm.HalfNormal('beta', sigma=0.4)
            likelihood = pm.AR('AR1_coeff', beta, sigma=sigma, observed=y-mu) 
        else:    
            Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y)

                    
class vlam_exp(pm.model.Model):
    # 1) override init
    """
    cabable to deal with offset inputs ...
    
    """
    
    
    
    
    def __init__(self, observed=None,name='', model=None,change_trend=False,n_changepoints=5,number_mu=0.1,
                     offsets_opt='normal',offsets_std=1,
                      add_to_num=0.,sigma_noise=1.,trend_inc_sigma=0.01,annual_cycle=False,
                      change_offsets=True,estimate_offset_sigma=False,estimate_trend_inc_sigma=False,
                      estimate_number_mu=False,n_samples=500,target_accept=0.8,post_seismic=True,
                      AR1=False,studentst_noise=False,distribute_offsets=False,initial_values={},**kwargs):

        super().__init__(name,model)

        
        x=observed.x
        y=observed.y
            
        xmin=np.min(x)
        xmax=np.max(x)
        
        
        X_mat=observed.X_mat
        
        if 'number_mu' in initial_values:
            initialize=True
            print('manually initialize with: ')
            print(initial_values)
            estimate_number_mu=False
            estimate_offset_sigma=False
            offsets_mu=initial_values['offsets']
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
            act_number_sigma = pm.HalfNormal('act_number_sigma', sigma=number_mu)  
            act_number=pm.Poisson('act_number', mu=act_number_sigma) -add_to_num
        else:
            act_number=pm.Poisson('act_number', mu=number_mu) -add_to_num

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

        mult=np.zeros(n_changepoints)
        arr=np.arange(n_changepoints)+0.5

        mult = (arr<act_number) *1 #[arr<act_number]=1
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
        if AR1: # first order autoregressive ...
            beta = pm.HalfNormal('beta', sigma=0.4)
            likelihood = pm.AR('AR1_coeff', beta, sigma=sigma, observed=y-mu) 
        else:    
            Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y)

