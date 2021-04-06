# Define custom model settings file
# Example file:
# note that when u use the custom settings, they will be stored in the output files
# so you don't need to store this information by yourself

custom_settings={}

custom_settings['run_settings']      =   {'n_samples':16000, # increase sample size from 8000 to 16000
                                            'tune':2000,
                                            'cores':8,
                                            'nuts':{'target_accept':0.9},
                                            'return_inferencedata':True,
                                            'compress':True}

custom_settings['initial_run_settings']   = {'detection_threshold':25} 

custom_settings['model_settings']   = {'n_changepoints':10, # e.g. change maximum possible number of changepoints from 5 to 10
                                            'change_trend':False} # disable trend changes      





"""

These are the standard settings


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
    specs['offsets_opt']='normal'
    specs['offsets_std']=20.
    specs['add_to_num'] = 0.        
    specs['model']=None
    specs['name']=''
    specs['change_trend']=True
    specs['change_offsets']=True
    specs['n_samples']=8000
    specs['trend_inc_sigma']=1.
    specs['target_accept']=0.9
    specs['add_to_num']=0
    specs['post_seismic']=False
    specs['estimate_offset_sigma']=True
    specs['estimate_trend_inc_sigma']=True  #
    specs['annual_cycle']=True    
    specs['AR1']=True
    specs['p_']=0.1
    if 'model_settings' in external_settings:
        for item in external_settings['model_settings']:
            specs[item]=external_settings['model_settings'][item]  
    return specs
    
"""