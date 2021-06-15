#!/usr/bin/env python

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


# command line interface for discotimes

from discotimes import discotimes as dt
from model_settings import *
from discotimes import file_reader
import os
import sys
import argparse
import importlib.util
import xarray as xr
import pandas as pd


#from setup import get_version

def cli_input():
    
    usage_text = 'use "python %(prog)s --help" for more information \n \
      \n \
      \n \
      Example: \n \
      \n \
      python discofit.py examples/*.txt -o tests/test_output/ -s tests/custom_settings.py \n \
      \n       DiscoTimeS Copyright (C) 2021 Julius Oelsmann \n \
      This program comes with ABSOLUTELY NO WARRANTY; \n \
      This is free software, and you are welcome to redistribute it \n \
      under certain conditions; \n'


    
    parser = argparse.ArgumentParser(description='Settings to run discotimes',
                                     usage=usage_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # version
    parser.version = '0.0.1'#get_version()
    
    # Arguments
    parser.add_argument('files',action='store',nargs='+')
    
    # Options
    parser.add_argument('-p', '--plot',action='store_true',help='turn on plotting', default=False)
    parser.add_argument('-v', '--variable',action='store',type=str,help='variable name', default='auto')
    parser.add_argument('-r', '--resample',action='store',type=str,help='resampling frequency, e.g. \
                        D, W, M, default: D', default='D')
    parser.add_argument('-o', '--output_directory',action='store',type=str,help='output_directory',required=True)
    parser.add_argument('-c', '--concatenate',action='store_true',help='concatenate all files', default=False)
    parser.add_argument('-t', '--output_type',action='store',
                        type=str,help='output type: netcdf or dt', default='netcdf')    
    parser.add_argument('-s', '--setting_file',action='store',type=str,
                        help='define location of custom settings file')
    parser.add_argument('-i', action='version')
    
    # Execute the parse_args() method
    args = parser.parse_args()
    print(args)
    return args

def run_multiple_fits(args,settings):
    """Fit multiple files
    

    Parameters
    ----------    
    concatenate: bool
        concatenate all output files, default: True
    
    files: list
        list of file-locations to fit
    output_directory: str
        output directory
    output_type: str
        Output type. default: 'netcdf'; *.csv or *.dt
    plot: bool
        Plot results, default: False
    setting_file: str
        optional, file containing additional settings
    variable: str
        optional, variable name, default: 'auto'
    resample: str
        optional, resamlping frequency default: 'D'
            
    """
    
    if args.output_type=='netcdf' or args.output_type=='csv':
        if args.concatenate:
            first_file = os.path.basename(args.files[0]).split(".",1)[0]
            last_file = os.path.basename(args.files[-1]).split(".",1)[0]
                        
            DATA = []
            for file in args.files:
                fileout = run_fit(file,args,settings)
                if fileout != None:
                    DATA.append(fileout)
            DATA_concat = xr.concat(DATA,dim='x')
            DATA_concat.to_netcdf(args.output_directory+'DT_data_'+first_file+'_to_'+last_file+'.nc')
        else:
            for file in args.files:
                run_fit(file,args,settings)
    else:
        args.concatenate = False
        for file in args.files:
            run_fit(file,args,settings)
    
    
def run_fit(file,args,settings):
    """Fit single time series
    
    """
    base=os.path.basename(file)
    name = base.split(".",1)[0]
    
    series = file_reader(file,variable=args.variable,resample=args.resample)
    dt_model = dt(series,settings=settings,name=name)
    failed=False
    try:
        dt_model.run(**settings['run_settings'])

    except Exception:
        pass
        failed=True
        print("chain failed: "+name)
        failf = pd.Series('chain failed')
        failf.to_csv(args.output_directory+name+'.dt')
        # save failed file to block re-running
    if not failed:    
        if args.plot:
            print('plot results')
            dt_model.plot(save=True,save_dir = args.output_directory)     
        if args.output_type=='netcdf' or args.output_type=='csv':
            dt_model_out = dt_model.to_nc()
            if args.concatenate:
                return dt_model_out
            else:
                dt_model_out.to_netcdf(args.output_directory+name+'.nc')
        elif args.output_type=='dt':
            dt_model.save(save_dir = args.output_directory)    
            

    
def check_if_files_exist(args,settings):
    """Check if files exist already in output directory
    """
    file_exists = False
    filenames = []
    if args.concatenate:
        if args.output_type=='netcdf' or args.output_type=='csv':
            first_file = os.path.basename(args.files[0]).split(".",1)[0]
            last_file = os.path.basename(args.files[-1]).split(".",1)[0]
            filename = args.output_directory+'DT_data_'+first_file+'_to_'+last_file+'.nc'
            if os.path.isfile(filename):
                filenames.append(filename)
                file_exists=True 
    else:
        if args.output_type=='netcdf':
            ending = '.nc'
        else:
            ending = '.dt'            
        for file in args.files:
            base=os.path.basename(file)
            name = base.split(".",1)[0]
            filename = args.output_directory + name + ending
            if os.path.isfile(filename):
                filenames.append(filename)
                file_exists=True 
    if file_exists:
        raise ValueError("The following target files \n \
         "+ str(filenames) +" \n \
         already exist in "+args.output_directory+' output directory.')
        
        
    
    

if __name__ == "__main__":
    
    
    args = cli_input()
    external_settings={}
    if args.setting_file is not None:        
        spec = importlib.util.spec_from_file_location("custom_settings", args.setting_file)
        custom_settings = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_settings)
        external_settings=custom_settings.custom_settings

    settings=set_settings(external_settings=external_settings)
    
    check_if_files_exist(args,settings)
    run_multiple_fits(args,settings)
    
    

        
    
    
    
    
    




    
    