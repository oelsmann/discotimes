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



#    <program>  Copyright (C) <year>  <name of author>
#    This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
#    This is free software, and you are welcome to redistribute it
#    under certain conditions; type `show c' for details.


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
        
    
    
    
    
    




    
    