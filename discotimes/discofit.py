# MIT License

# Copyright (c) 2021 Julius Oelsmann

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# command line interface for discotimes


from discotimes import discotimes as dt
from model_settings import *
import os
import sys
import argparse
import importlib.util


#from setup import get_version

def cli_input():

    parser = argparse.ArgumentParser(description='Settings to run discotimes',
                                     usage='use "python %(prog)s --help" for more information',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # version
    parser.version = '0.0.1'#get_version()
    
    # Arguments
    parser.add_argument('files',action='store',nargs='+')
    
    # Options
    parser.add_argument('-p', '--plot',action='store_true',help='turn off plotting', default=False)
    parser.add_argument('-o', '--output_directory',action='append',type=str,help='output_directory',required=True)
    parser.add_argument('-c', '--concatenate',action='store_true',help='concatenate all files', default=True)
    parser.add_argument('-t', '--output_type',action='append',
                        type=str,help='output type: csv, netcdf or dt', default='netcdf')    
    parser.add_argument('-s', '--setting_file',action='store',type=str,
                        help='define location of custom settings file')
    parser.add_argument('-i', action='version')
    
    # Execute the parse_args() method
    args = parser.parse_args()
    print(args)
    return args

"""
def run_model():
    
    named=dirout + name+'.il'
    if not os.path.exists(named):
        model = ilame(synt.series,specs=specs_model,name=name,testing=True,test_model=model_version)
        model.approximate_initial_offsets(detection_threshold=15,
                                    detection_resolution=4,
                                    method='diff')    

        model.run(n_samples=8000,tune=4000,cores=8,
                nuts={'target_accept':0.9},return_inferencedata=True,
                compress=True) #,**{'init':'adapt_diag'}advi+adapt_diag,**{'init':'advi+adapt_diag'}
        model.save(save_dir=dirout)
        model.plot(save=True,plot_all_chains=True,crit='all',
                   save_dir=plot_dir,save_name='chains_'+name) 
    else:
        print('run exists already!')
"""

if __name__ == "__main__":
    
    
    args = cli_input()
    if args.setting_file is not None:
        spec = importlib.util.spec_from_file_location("custom_settings", args.setting_file)
        custom_settings = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_settings)
        #foo.MyClass()
        
    print(args)

    print(custom_settings.custom_settings)
    settings=set_settings(external_settings=custom_settings.custom_settings)
    print(settings)
        
    
    
    
    
    




    
    