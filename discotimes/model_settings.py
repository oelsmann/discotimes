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



# Define the model settings here


def set_settings(external_settings={}):
    """
    Make/Change default model settings 
    
    Parameters
    ----------
    external_settings: dict,
        can contain dicts: 'model_settings', 'run_settings', 'initial_run_settings'
    """
    
    settings={}
    settings['model_settings']   = model_settings(external_settings=external_settings)
    settings['run_settings']   = run_settings(external_settings=external_settings)        
    settings['initial_run_settings']   = initial_run_settings(external_settings=external_settings)  
    
    return settings

def run_settings(external_settings={}):
    
    specs={'n_samples':8000,'tune':2000,
                               'cores':8,'nuts':{'target_accept':0.9},
                               'return_inferencedata':True,'compress':True}
    if 'run_settings' in external_settings:
        for item in external_settings['run_settings']:
            specs[item]=external_settings['run_settings'][item]  
    return specs 

def initial_run_settings(external_settings={}):
    
    specs={'detection_threshold':15,
                    'detection_resolution':4,
                    'method':'diff'}
    if 'initial_run_settings' in external_settings:
        for item in external_settings['initial_run_settings']:
            specs[item]=external_settings['initial_run_settings'][item]  
    return specs          

def model_settings(external_settings={}):
    
    specs={}
    specs['n_changepoints']=5
    specs['offsets_std']=20.     
    specs['name']=''
    specs['change_trend']=True
    specs['change_offsets']=True
    specs['n_samples']=8000
    specs['trend_inc_sigma']=1.
    specs['post_seismic']=False
    specs['estimate_offset_sigma']=True
    specs['estimate_trend_inc_sigma']=True  #
    specs['annual_cycle']=True    
    specs['AR1']=True
    specs['p_']=0.1
    specs['initial_values'] = {}
    if 'model_settings' in external_settings:
        for item in external_settings['model_settings']:
            specs[item]=external_settings['model_settings'][item]  
    return specs