# KrakenSDR Signal Processor
#
# Copyright (C) 2018-2021  Carl Laufer, Tamás Pető
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#
# - coding: utf-8 -*-

# Import built-in modules
import logging
import os
import sys
import queue 
import time
import subprocess
import orjson
# Import third-party modules

#Testing dash_devices

#import dash
#from dash.dependencies import Input, Output, State

#import plotly.io as pio
#pio.renderers.default = 'iframe'

import dash_core_components as dcc
import dash_html_components as html

import dash_devices as dash
from dash_devices.dependencies import Input, Output, State
#from dash import html


from dash.exceptions import PreventUpdate
from dash.dash import no_update
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from configparser import ConfigParser
from numba import njit, jit

#from waitress import serve
#from gevent.pywsgi import WSGIServer

from threading import Timer

# Import Kraken SDR modules
current_path          = os.path.dirname(os.path.realpath(__file__))
root_path             = os.path.dirname(os.path.dirname(current_path))
receiver_path         = os.path.join(root_path, "_receiver")
signal_processor_path = os.path.join(root_path, "_signal_processing")
ui_path               = os.path.join(root_path, "_UI")

sys.path.insert(0, receiver_path)
sys.path.insert(0, signal_processor_path)
sys.path.insert(0, ui_path)

daq_subsystem_path    = os.path.join(
                            os.path.join(os.path.dirname(root_path), 
                            "heimdall_daq_fw"), 
                        "Firmware")
daq_preconfigs_path   = os.path.join(
                            os.path.join(os.path.dirname(root_path), 
                            "heimdall_daq_fw"), 
                        "config_files")
daq_config_filename   = os.path.join(daq_subsystem_path, "daq_chain_config.ini")
daq_stop_filename     = "daq_stop.sh"
daq_start_filename    = "daq_start_sm.sh"
#daq_start_filename    = "daq_synthetic_start.sh"
sys.path.insert(0, daq_subsystem_path)

import ini_checker
import save_settings as settings
from krakenSDR_receiver import ReceiverRTLSDR
from krakenSDR_signal_processor import SignalProcessor

import tooltips

class webInterface():

    def __init__(self):
        self.user_interface = None       
        
        logging.basicConfig(level=settings.logging_level*10)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(settings.logging_level*10)
        self.logger.info("Inititalizing web interface ")
        if not settings.settings_found:
            self.logger.warning("Web Interface settings file is not found!")
        
        #############################################
        #  Initialize and Configure Kraken modules  #
        #############################################

        # Web interface internal 
        self.disable_tooltips = settings.disable_tooltips
        self.page_update_rate = 1     
        self._avg_win_size = 10
        self._update_rate_arr = None

        self.sp_data_que = queue.Queue(1) # Que to communicate with the signal processing module
        self.rx_data_que = queue.Queue(1) # Que to communicate with the receiver modules

        # Instantiate and configure Kraken SDR modules
        self.module_receiver = ReceiverRTLSDR(data_que=self.rx_data_que, data_interface=settings.data_interface, logging_level=settings.logging_level*10)
        self.module_receiver.daq_center_freq   = settings.center_freq*10**6
        self.module_receiver.daq_rx_gain       = [settings.gain_1, settings.gain_2]
        self.module_receiver.rec_ip_addr       = settings.default_ip

        self.module_signal_processor = SignalProcessor(data_que=self.sp_data_que, module_receiver=self.module_receiver, logging_level=settings.logging_level*10)        
        self.module_signal_processor.en_PR    = settings.en_pr
        self.module_signal_processor.PR_clutter_cancellation = settings.clutter_cancel_algo
        self.module_signal_processor.max_bistatic_range = settings.max_bistatic_range
        self.module_signal_processor.max_doppler = settings.max_doppler
        self.en_persist = settings.en_pr_persist
        self.pr_persist_decay = settings.pr_persist_decay
        self.pr_dynamic_range_min = settings.pr_dynrange_min
        self.pr_dynamic_range_max = settings.pr_dynrange_max

        self.module_signal_processor.start()
        
        #############################################
        #       UI Status and Config variables      #
        #############################################

        # DAQ Subsystem status parameters
        self.daq_conn_status       = 0
        self.daq_cfg_iface_status  = 0 # 0- ready, 1-busy        
        self.daq_restart           = 0 # 1-restarting
        self.daq_update_rate       = 0
        self.daq_frame_sync        = 1 # Active low
        self.daq_frame_index       = 0
        self.daq_frame_type        = "-"
        self.daq_power_level       = 0
        self.daq_sample_delay_sync = 0
        self.daq_iq_sync           = 0
        self.daq_noise_source_state= 0
        self.daq_center_freq       = settings.center_freq
        self.daq_adc_fs            = "-"
        self.daq_fs                = "-"
        self.daq_cpi               = "-"
        self.daq_if_gains          ="[,,,,]"
        self.en_advanced_daq_cfg   = False #settings.en_advanced_daq_cfg 
        self.daq_ini_cfg_params    = read_config_file()
        self.active_daq_ini_cfg    = self.daq_ini_cfg_params[0] #"Default" # Holds the string identifier of the actively loaded DAQ ini configuration
        self.tmp_daq_ini_cfg       = "Default"
        self.daq_cfg_ini_error     = ""

        # DSP Processing Parameters and Results  
        self.spectrum              = None
        self.daq_dsp_latency       = 0 # [ms]
        self.logger.info("Web interface object initialized")        

        # Passive Radar Data
        self.RD_matrix = None

        self.dsp_timer = None
        self.update_time = 9999

        self.pathname = ""
        self.reset_doa_graph_flag = False
        self.reset_spectrum_graph_flag = False
        self.CAFMatrixPersist = None

        # Basic DAQ Config
        self.decimated_bandwidth = 12.5

        if self.daq_ini_cfg_params is not None: 
            self.logger.info("Config file found and read succesfully")        
            """
             Set the number of channels in the receiver module because it is required 
             to produce the initial gain configuration message (Only needed in shared-memory mode)
            """
            self.module_receiver.M = self.daq_ini_cfg_params[1]


    def save_configuration(self):
        data = {}

        # DAQ Configuration
        data["center_freq"]    = self.module_receiver.daq_center_freq/10**6
        data["gain_1"]   = self.module_receiver.daq_rx_gain[0]
        data["gain_2"]   = self.module_receiver.daq_rx_gain[1]
        data["data_interface"] = settings.data_interface
        data["default_ip"]     = settings.default_ip
        
        # DOA Estimation
        data["en_pr"]          = self.module_signal_processor.en_PR

        data["clutter_cancel_algo"] = self.module_signal_processor.PR_clutter_cancellation
        data["max_bistatic_range"]          = self.module_signal_processor.max_bistatic_range
        data["max_doppler"]          = self.module_signal_processor.max_doppler
        data["en_pr_persist"]          = self.en_persist
        data["pr_persist_decay"]          = self.pr_persist_decay
        data["pr_dynrange_min"]          = self.pr_dynamic_range_min
        data["pr_dynrange_max"]          = self.pr_dynamic_range_max

        # Web Interface
        data["en_hw_check"]         = settings.en_hw_check
        data["en_advanced_daq_cfg"] = self.en_advanced_daq_cfg
        data["logging_level"]       = settings.logging_level
        data["disable_tooltips"]    = settings.disable_tooltips

        settings.write(data)
    def start_processing(self):
        """
            Starts data processing

            Parameters:
            -----------
            :param: ip_addr: Ip address of the DAQ Subsystem

            :type ip_addr : string e.g.:"127.0.0.1"
        """
        self.logger.info("Start processing request")
        self.first_frame = 1
        #self.module_receiver.rec_ip_addr = "0.0.0.0" 
        self.module_signal_processor.run_processing=True 
    def stop_processing(self):
        self.module_signal_processor.run_processing=False
        while self.module_signal_processor.is_running: time.sleep(0.01) # Block until signal processor run_processing while loop ends
    def close_data_interfaces(self):
        self.module_receiver.eth_close()
    def close(self):
        pass
    """
    def config_doa_in_signal_processor(self):
        if self._doa_method == 0:
            self.module_signal_processor.en_DOA_Bartlett = True
            self.module_signal_processor.en_DOA_Capon    = False
            self.module_signal_processor.en_DOA_MEM      = False
            self.module_signal_processor.en_DOA_MUSIC    = False
        elif self._doa_method == 1:
            self.module_signal_processor.en_DOA_Bartlett = False
            self.module_signal_processor.en_DOA_Capon    = True
            self.module_signal_processor.en_DOA_MEM      = False
            self.module_signal_processor.en_DOA_MUSIC    = False
        elif self._doa_method == 2:
            self.module_signal_processor.en_DOA_Bartlett = False
            self.module_signal_processor.en_DOA_Capon    = False
            self.module_signal_processor.en_DOA_MEM      = True
            self.module_signal_processor.en_DOA_MUSIC    = False
        elif self._doa_method == 3:
            self.module_signal_processor.en_DOA_Bartlett = False
            self.module_signal_processor.en_DOA_Capon    = False
            self.module_signal_processor.en_DOA_MEM      = False
            self.module_signal_processor.en_DOA_MUSIC    = True
    """
    def config_squelch_value(self, squelch_threshold_dB):
        """
            Configures the squelch thresold both on the DAQ side and 
            on the local DoA DSP side.
        """        
        #NOT USING THIS ANYMORE
        self.daq_cfg_iface_status = 1
        #self.module_signal_processor.squelch_threshold = 10**(squelch_threshold_dB/20)
        #self.module_receiver.set_squelch_threshold(squelch_threshold_dB)
        #webInterface_inst.logger.info("Updating receiver parameters")
        #webInterface_inst.logger.info("Squelch threshold : {:f} dB".format(squelch_threshold_dB))
    def config_daq_rf(self, f0, gain):
        """
            Configures the RF parameters in the DAQ module
        """
        self.daq_cfg_iface_status = 1
        self.module_receiver.set_center_freq(int(f0*10**6))
        self.module_receiver.set_if_gain(gain)

        webInterface_inst.logger.info("Updating receiver parameters")
        webInterface_inst.logger.info("Center frequency: {:f} MHz".format(f0))
        #webInterface_inst.logger.info("Gain: {:f} dB".format(gain))
        webInterface_inst.logger.info("Gain: " + ' '.join(str(x) for x in gain) + " dB")




def read_config_file(config_fname=daq_config_filename):    
    parser = ConfigParser()
    found = parser.read([config_fname])
    param_list = []
    if not found:            
        return None
    param_list.append(parser.get('meta', 'config_name'))

    param_list.append(parser.getint('hw', 'num_ch'))

    param_list.append(parser.getint('daq','daq_buffer_size'))
    param_list.append(parser.getint('daq','sample_rate'))
    param_list.append(parser.getint('daq','en_noise_source_ctr'))

    param_list.append(parser.getint('squelch','en_squelch'))
    param_list.append(parser.getfloat('squelch','amplitude_threshold'))

    param_list.append(parser.getint('pre_processing', 'cpi_size'))
    param_list.append(parser.getint('pre_processing', 'decimation_ratio'))
    param_list.append(parser.getfloat('pre_processing', 'fir_relative_bandwidth'))
    param_list.append(parser.getint('pre_processing', 'fir_tap_size'))
    param_list.append(parser.get('pre_processing','fir_window'))
    param_list.append(parser.getint('pre_processing','en_filter_reset'))

    param_list.append(parser.getint('calibration','corr_size'))
    param_list.append(parser.getint('calibration','std_ch_ind'))
    param_list.append(parser.getint('calibration','en_iq_cal'))
    param_list.append(parser.getint('calibration','gain_lock_interval'))
    param_list.append(parser.getint('calibration','require_track_lock_intervention'))
    param_list.append(parser.getint('calibration','cal_track_mode'))
    param_list.append(parser.get('calibration','amplitude_cal_mode'))
    param_list.append(parser.getint('calibration','cal_frame_interval'))
    param_list.append(parser.getint('calibration','cal_frame_burst_size'))
    param_list.append(parser.getint('calibration','amplitude_tolerance'))
    param_list.append(parser.getint('calibration','phase_tolerance'))
    param_list.append(parser.getint('calibration','maximum_sync_fails'))

    param_list.append(parser.get('data_interface','out_data_iface_type'))   

    return param_list
   
def write_config_file(param_list):
    webInterface_inst.logger.info("Write config file: {0}".format(param_list))
    parser = ConfigParser()
    found = parser.read([daq_config_filename])
    if not found:            
        return -1
    
    parser['meta']['config_name']=str(param_list[0])

    parser['hw']['num_ch']=str(param_list[1])

    parser['daq']['daq_buffer_size']=str(param_list[2])
    parser['daq']['sample_rate']=str(param_list[3])
    parser['daq']['en_noise_source_ctr']=str(param_list[4])
    parser['daq']['center_freq']=str(int(webInterface_inst.module_receiver.daq_center_freq))

    parser['squelch']['en_squelch']=str(param_list[5])
    parser['squelch']['amplitude_threshold']=str(param_list[6])

    parser['pre_processing']['cpi_size']=str(param_list[7])
    parser['pre_processing']['decimation_ratio']=str(param_list[8])
    parser['pre_processing']['fir_relative_bandwidth']=str(param_list[9])
    parser['pre_processing']['fir_tap_size']=str(param_list[10])
    parser['pre_processing']['fir_window']=str(param_list[11])
    parser['pre_processing']['en_filter_reset']=str(param_list[12])

    parser['calibration']['corr_size']=str(param_list[13])
    parser['calibration']['std_ch_ind']=str(param_list[14])
    parser['calibration']['en_iq_cal']=str(param_list[15])
    parser['calibration']['gain_lock_interval']=str(param_list[16])
    parser['calibration']['require_track_lock_intervention']=str(param_list[17])
    parser['calibration']['cal_track_mode']=str(param_list[18])
    parser['calibration']['amplitude_cal_mode']=str(param_list[19])
    parser['calibration']['cal_frame_interval']=str(param_list[20])
    parser['calibration']['cal_frame_burst_size']=str(param_list[21])
    parser['calibration']['amplitude_tolerance']=str(param_list[22])
    parser['calibration']['phase_tolerance']=str(param_list[23])
    parser['calibration']['maximum_sync_fails']=str(param_list[24])

    ini_parameters = parser._sections
    error_list = ini_checker.check_ini(ini_parameters, settings.en_hw_check)
    if len(error_list):
        for e in error_list:
            webInterface_inst.logger.error(e)
        return -1, error_list
    else:
        with open(daq_config_filename, 'w') as configfile:
            parser.write(configfile)
        return 0,[]

def get_preconfigs(config_files_path):
    parser = ConfigParser()
    preconfigs = []
    preconfigs.append([daq_config_filename, "Current"])
    for root, dirs, files in os.walk(config_files_path):
        if len(files):
            config_file_path = os.path.join(root, files[0])
            parser.read([config_file_path])
            parameters = parser._sections
            preconfigs.append([config_file_path, parameters['meta']['config_name']])
    return preconfigs


#############################################
#          Prepare Dash application         #
############################################
webInterface_inst = webInterface()


#############################################
#       Prepare component dependencies      #
#############################################

trace_colors = px.colors.qualitative.Plotly
trace_colors[3] = 'rgb(255,255,51)'
valid_fir_windows = ['boxcar', 'triang', 'blackman', 'hamming', 'hann', 'bartlett', 'flattop', 'parzen' , 'bohman', 'blackmanharris', 'nuttall', 'barthann'] 
valid_sample_rates = [0.25, 0.900001, 1.024, 1.4, 1.8, 1.92, 2.048, 2.4, 2.56, 3.2]
valid_daq_buffer_sizes = (2**np.arange(10,21,1)).tolist()
calibration_tack_modes = [['No tracking',0] , ['Periodic tracking',2]]
doa_trace_colors =	{
  "DoA Bartlett": "#00B5F7",
  "DoA Capon"   : "rgb(226,26,28)",
  "DoA MEM"     : "#1CA71C",
  "DoA MUSIC"   : "rgb(257,233,111)"
}
figure_font_size = 20

y=np.random.normal(0,1,2**3)
x=np.arange(2**3)

fig_layout = go.Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)', 
        template='plotly_dark',
        showlegend=True,
        margin=go.layout.Margin(
            t=0 #top margin
        )    
    )

fig_dummy = go.Figure(layout=fig_layout)
#fig_dummy.add_trace(go.Scatter(x=x, y=y, name = "Avg spectrum"))

for m in range(0, webInterface_inst.module_receiver.M+1): #+1 for the auto decimation window selection
    fig_dummy.add_trace(go.Scatter(x=x,
                             y=y,
                             name="Channel {:d}".format(m),
                             line = dict(color = trace_colors[m],
                                         width = 2)
                             ))


fig_dummy.update_xaxes(title_text="Frequency [MHz]")
fig_dummy.update_yaxes(title_text="Amplitude [dB]")

option = [{"label":"", "value": 1}]

spectrum_fig = go.Figure(layout=fig_layout)

for m in range(0, webInterface_inst.module_receiver.M):
    spectrum_fig.add_trace(go.Scattergl(x=x,
                             y=y,
                             name="Channel {:d}".format(m),
                             line = dict(color = trace_colors[m],
                                         width = 2)
                             ))


waterfall_init = [[-80] * webInterface_inst.module_signal_processor.spectrum_window_size] * 50
waterfall_fig = go.Figure(layout=fig_layout)
waterfall_fig.add_trace(go.Heatmapgl(
                         z=waterfall_init,
                         zsmooth=False,
                         showscale=False,
                         hoverinfo='skip',
                         colorscale=[[0.0, '#000020'],
                         [0.0714, '#000030'],
                         [0.1428, '#000050'],
                         [0.2142, '#000091'],
                         [0.2856, '#1E90FF'],
                         [0.357, '#FFFFFF'],
                         [0.4284, '#FFFF00'],
                         [0.4998, '#FE6D16'],
                         [0.5712, '#FE6D16'],
                         [0.6426, '#FF0000'],
                         [0.714, '#FF0000'],
                         [0.7854, '#C60000'],
                         [0.8568, '#9F0000'],
                         [0.9282, '#750000'],
                         [1.0, '#4A0000']]))


waterfall_fig.update_xaxes(tickfont_size=1)
waterfall_fig.update_yaxes(tickfont_size=1)
waterfall_fig.update_layout(margin=go.layout.Margin(t=5))



pr_init = [[-80] * 128] * 128
y_range = list(range(-128, 128))

pr_fig = go.Figure(layout=fig_layout)
pr_fig.add_trace(go.Heatmap(
                         z=pr_init,
                         y=y_range,
                         zsmooth='best', #False,
                         #zsmooth=False, #False,
                         showscale=False,
                         #hoverinfo='skip',
                         colorscale=[[0.0, '#000020'],
                         [0.0714, '#000030'],
                         [0.1428, '#000050'],
                         [0.2142, '#000091'],
                         [0.2856, '#1E90FF'],
                         [0.357, '#FFFFFF'],
                         [0.4284, '#FFFF00'],
                         [0.4998, '#FE6D16'],
                         [0.5712, '#FE6D16'],
                         [0.6426, '#FF0000'],
                         [0.714, '#FF0000'],
                         [0.7854, '#C60000'],
                         [0.8568, '#9F0000'],
                         [0.9282, '#750000'],
                         [1.0, '#4A0000']]))


#app = dash.Dash(__name__, suppress_callback_exceptions=True, compress=True, update_title="") # cannot use update_title with dash_devices
app = dash.Dash(__name__, suppress_callback_exceptions=True, compress=True)

# app_log = logger.getLogger('werkzeug')
# app_log.setLevel(settings.logging_level*10)
# app_log.setLevel(30) # TODO: Only during dev time
app.layout = html.Div([
    dcc.Location(id='url', children='/config',refresh=False),

    html.Div([html.Img(src="assets/kraken_interface_bw_pr.png", style={"display": "block", "margin-left": "auto", "margin-right": "auto", "height": "60px"})]),
    html.Div([html.A("Configuration", className="header_active"   , id="header_config"  ,href="/config"),
            html.A("Spectrum"       , className="header_inactive" , id="header_spectrum",href="/spectrum"),   
            html.A("Passive Radar" , className="header_inactive" , id="header_doa"     ,href="/pr"),
            ], className="header"),

    html.Div(id="placeholder_start"                , style={"display":"none"}),
    html.Div(id="placeholder_stop"                 , style={"display":"none"}),
    html.Div(id="placeholder_save"                 , style={"display":"none"}),
    html.Div(id="placeholder_update_rx"            , style={"display":"none"}),
    html.Div(id="placeholder_recofnig_daq"         , style={"display":"none"}),
    html.Div(id="placeholder_update_daq_ini_params", style={"display":"none"}),
    html.Div(id="placeholder_update_freq"          , style={"display":"none"}),
    html.Div(id="placeholder_update_dsp"           , style={"display":"none"}),
    html.Div(id="placeholder_update_squelch"       , style={"display":"none"}),
    html.Div(id="placeholder_config_page_upd"      , style={"display":"none"}),
    html.Div(id="placeholder_spectrum_page_upd"    , style={"display":"none"}),
    html.Div(id="placeholder_doa_page_upd"         , style={"display":"none"}),
    html.Div(id="dummy_output"                     , style={"display":"none"}),

    html.Div(id='page-content')
])
def generate_config_page_layout(webInterface_inst):   
    # Read DAQ config file
    daq_cfg_params = webInterface_inst.daq_ini_cfg_params    
    
    if daq_cfg_params is not None:
        en_noise_src_values       =[1] if daq_cfg_params[4]  else []
        en_squelch_values         =[1] if daq_cfg_params[5]  else []
        en_filter_rst_values      =[1] if daq_cfg_params[12] else []
        en_iq_cal_values          =[1] if daq_cfg_params[15] else []
        en_req_track_lock_values  =[1] if daq_cfg_params[17] else []

        #daq_data_iface_type       = daq_cfg_params[25]
    
        # Read available preconfig files
        preconfigs = get_preconfigs(daq_preconfigs_path)
    
    en_persist_values     =[1] if webInterface_inst.en_persist else []
    en_pr_values          =[1] if webInterface_inst.module_signal_processor.en_PR else []
    
    en_advanced_daq_cfg   =[1] if webInterface_inst.en_advanced_daq_cfg                       else []
    # Calulcate spacings
    wavelength= 300 / webInterface_inst.daq_center_freq
    
    ant_spacing_wavelength = webInterface_inst.module_signal_processor.DOA_inter_elem_space
    ant_spacing_meter = round(wavelength * ant_spacing_wavelength, 3)
    ant_spacing_feet  = ant_spacing_meter*3.2808399
    ant_spacing_inch  = ant_spacing_meter*39.3700787

    cfg_decimated_bw = ((daq_cfg_params[3]) / daq_cfg_params[8]) / 10**3
    cfg_data_block_len = ( daq_cfg_params[7] / (cfg_decimated_bw) )
    cfg_recal_interval =  (daq_cfg_params[20] * (cfg_data_block_len/10**3)) / 60 

    if daq_cfg_params[18] == 0: #If set to no tracking
        cfg_recal_interval = 1

    gain_list = [
                {'label': '0 dB',    'value': 0},
                {'label': '0.9 dB',  'value': 0.9},
                {'label': '1.4 dB',  'value': 1.4},
                {'label': '2.7 dB',  'value': 2.7},
                {'label': '3.7 dB',  'value': 3.7},
                {'label': '7.7 dB',  'value': 7.7},
                {'label': '8.7 dB',  'value': 8.7},
                {'label': '12.5 dB', 'value': 12.5},
                {'label': '14.4 dB', 'value': 14.4},
                {'label': '15.7 dB', 'value': 15.7},
                {'label': '16.6 dB', 'value': 16.6},
                {'label': '19.7 dB', 'value': 19.7},
                {'label': '20.7 dB', 'value': 20.7},
                {'label': '22.9 dB', 'value': 22.9},
                {'label': '25.4 dB', 'value': 25.4},
                {'label': '28.0 dB', 'value': 28.0},
                {'label': '29.7 dB', 'value': 29.7},
                {'label': '32.8 dB', 'value': 32.8},
                {'label': '33.8 dB', 'value': 33.8},
                {'label': '36.4 dB', 'value': 36.4},
                {'label': '37.2 dB', 'value': 37.2},
                {'label': '38.6 dB', 'value': 38.6},
                {'label': '40.2 dB', 'value': 40.2},
                {'label': '42.1 dB', 'value': 42.1},
                {'label': '43.4 dB', 'value': 43.4},
                {'label': '43.9 dB', 'value': 43.9},
                {'label': '44.5 dB', 'value': 44.5},
                {'label': '48.0 dB', 'value': 48.0},
                {'label': '49.6 dB', 'value': 49.6},
                ]


    #-----------------------------
    #   Start/Stop Configuration Card
    #-----------------------------
    start_stop_card = \
    html.Div([
        html.Div([html.Div([html.Button('Start Processing', id='btn-start_proc', className="btn_start", n_clicks=0)], className="ctr_toolbar_item"),
              html.Div([html.Button('Stop Processing', id='btn-stop_proc', className="btn_stop", n_clicks=0)], className="ctr_toolbar_item"),
              html.Div([html.Button('Save Configuration', id='btn-save_cfg', className="btn_save_cfg", n_clicks=0)], className="ctr_toolbar_item")
            ], className="ctr_toolbar"),
    ])

    #-----------------------------
    #   DAQ Configuration Card
    #-----------------------------
    # -- > Main Card Layout < --
    daq_config_card_list = \
    [
        html.H2("RF Receiver Configuration", id="init_title_c"),
        html.Div([
                html.Div("Center Frequency [MHz]", className="field-label"),
                dcc.Input(id='daq_center_freq', value=webInterface_inst.module_receiver.daq_center_freq/10**6, type='number', debounce=True, className="field-body-textbox")
                ], className="field"),
        html.Div([
                html.Div("Receiver gain", className="field-label"), 
                dcc.Dropdown(id='daq_rx_gain',
                        options=gain_list,
                    value=webInterface_inst.module_receiver.daq_rx_gain[0], clearable=False, style={"display":"inline-block", "padding-bottom":"5px"}, className="field-body"),
                #], className="field"),

                html.Div("Receiver 2 gain", className="field-label"), 
                dcc.Dropdown(id='daq_rx_gain_2',
                        options=gain_list,
                    value=webInterface_inst.module_receiver.daq_rx_gain[1], clearable=False, style={"display":"inline-block"}, className="field-body"),
                ], className="field"),


        html.Div([
            html.Button('Update Receiver Parameters', id='btn-update_rx_param', className="btn"),
        ], className="field"),


        html.Div([
            html.Div("Preconfigured DAQ Files", className="field-label"), 
            #TEST
            #dcc.Input(id='daq_cfg_files', value=cfg_data_block_len, type='number', debounce=True, className="field-body"),

            dcc.Dropdown(id='daq_cfg_files',
                    options=[
                        {'label': str(i[1]), 'value': i[0]} for i in preconfigs
                    ],
            clearable=False,
            value=preconfigs[0][0],
            placeholder="Select Configuration File",
            persistence=True,
            style={"display":"inline-block"},
            className="field-body-wide"),
        ], className="field"),
        html.Div([
            html.Div("Active Configuration: " + webInterface_inst.active_daq_ini_cfg, id="active_daq_ini_cfg", className="field-label"),
        ], className="field"),
        html.Div([
                html.Div(webInterface_inst.daq_cfg_ini_error , id="daq_ini_check", className="field-label", style={"color":"red"}),
        ], className="field"),

        html.Div([html.Div("Basic Custom DAQ Configuration", id="label_en_basic_daq_cfg"     , className="field-label")]),
            html.Div([
                html.Div("Data Block Length (ms):", className="field-label"),                                         
                dcc.Input(id='cfg_data_block_len', value=cfg_data_block_len, type='number', debounce=True, className="field-body-textbox")
            ], className="field"),
            html.Div([
                html.Div("Decimated Bandwidth (kHz):", className="field-label"),                                         
                dcc.Input(id='cfg_decimated_bw', value=cfg_decimated_bw, type='number', debounce=True, className="field-body-textbox")
            ], className="field"),
            html.Div([
                html.Div("Recalibration Interval (mins):", className="field-label"),                                         
                dcc.Input(id='cfg_recal_interval', value=cfg_recal_interval, type='number', debounce=True, className="field-body-textbox")
            ], className="field"),

        html.Div([html.Div("Advanced Custom DAQ Configuration", id="label_en_advanced_daq_cfg"     , className="field-label"),
                dcc.Checklist(options=option     , id="en_advanced_daq_cfg"     ,  className="field-body", value=en_advanced_daq_cfg),
        ], className="field"),        
    ]
    
    # --> Optional DAQ Subsystem reconfiguration fields <--   
    #if len(en_advanced_daq_cfg):
    #    if daq_cfg_params is not None:
    daq_subsystem_reconfiguration_options = [ \
        html.Div([

            html.H2("DAQ Subsystem Reconfiguration", id="init_title_reconfig"),
            html.H3("HW", id="cfg_group_hw"),
            html.Div([
                    html.Div("# RX Channels:", className="field-label"),                                         
                    dcc.Input(id='cfg_rx_channels', value=daq_cfg_params[1], type='number', debounce=True, className="field-body-textbox")
            ], className="field"),
            html.H3("DAQ", id="cfg_group_daq"),
            html.Div([
                    html.Div("DAQ Buffer Size:", className="field-label", id="label_daq_buffer_size"),                                                                 
                    dcc.Dropdown(id='cfg_daq_buffer_size',
                                options=[
                                        {'label': i, 'value': i} for i in valid_daq_buffer_sizes
                                ],
                                value=daq_cfg_params[2], style={"display":"inline-block"},className="field-body"),
            ], className="field"),
            html.Div([
                html.Div("Sample Rate [MHz]:", className="field-label", id="label_sample_rate"),
                dcc.Dropdown(id='cfg_sample_rate',
                        options=[
                            {'label': i, 'value': i} for i in valid_sample_rates                                
                            ],
                    value=daq_cfg_params[3]/10**6, style={"display":"inline-block"},className="field-body")
            ], className="field"),
            html.Div([
                    html.Div("Enable Noise Source Control:", className="field-label", id="label_en_noise_source_ctr"),                                         
                    dcc.Checklist(options=option     , id="en_noise_source_ctr"   , className="field-body", value=en_noise_src_values),
            ], className="field"),
            #html.H3("Squelch"),
            #html.Div([
            #        html.Div("Enable DAQ Squelch (NOT ACTIVE):", className="field-label", id="label_en_squelch"),                                                                 
            #        dcc.Checklist(options=option     , id="en_squelch_mode"   , className="field-body", value=en_squelch_values),
            #], className="field"),
            #html.Div([
            #        html.Div("Initial Threshold:", className="field-label", id="label_squelch_init_threshold"),                                         
            #        dcc.Input(id='cfg_squelch_init_th', value=daq_cfg_params[6], type='number', debounce=True, className="field-body")
            #], className="field"),
            html.H3("Pre Processing"),
            html.Div([
                    html.Div("CPI Size [sample]:", className="field-label", id="label_cpi_size"),                                         
                    dcc.Input(id='cfg_cpi_size', value=daq_cfg_params[7], type='number', debounce=True, className="field-body-textbox")
            ], className="field"),
            html.Div([
                    html.Div("Decimation Ratio:", className="field-label", id="label_decimation_ratio"),                                         
                    dcc.Input(id='cfg_decimation_ratio', value=daq_cfg_params[8], type='number', debounce=True, className="field-body-textbox")
            ], className="field"),
            html.Div([
                    html.Div("FIR Relative Bandwidth:", className="field-label", id="label_fir_relative_bw"),                                         
                    dcc.Input(id='cfg_fir_bw', value=daq_cfg_params[9], type='number', debounce=True, className="field-body-textbox")
            ], className="field"),
            html.Div([
                    html.Div("FIR Tap Size:", className="field-label", id="label_fir_tap_size"),                                         
                    dcc.Input(id='cfg_fir_tap_size', value=daq_cfg_params[10], type='number', debounce=True, className="field-body-textbox")
            ], className="field"),
            html.Div([
                html.Div("FIR Window:", className="field-label", id="label_fir_window"),
                dcc.Dropdown(id='cfg_fir_window',
                        options=[
                            {'label': i, 'value': i} for i in valid_fir_windows
                            ],
                    value=daq_cfg_params[11], style={"display":"inline-block"},className="field-body")
            ], className="field"),
            html.Div([
                    html.Div("Enable Filter Reset:", className="field-label", id="label_en_filter_reset"),                                         
                    dcc.Checklist(options=option     , id="en_filter_reset"   , className="field-body", value=en_filter_rst_values),
            ], className="field"),
            html.H3("Calibration"),
            html.Div([
                    html.Div("Correlation Size [sample]:", className="field-label", id="label_correlation_size"),
                    dcc.Input(id='cfg_corr_size', value=daq_cfg_params[13], type='number', debounce=True, className="field-body-textbox")
            ], className="field"),
            html.Div([
                    html.Div("Standard Channel Index:", className="field-label", id="label_std_ch_index"),                                         
                    dcc.Input(id='cfg_std_ch_ind', value=daq_cfg_params[14], type='number', debounce=True, className="field-body-textbox")
            ], className="field"),
            html.Div([
                    html.Div("Enable IQ Calibration:", className="field-label", id="label_en_iq_calibration"),                                         
                    dcc.Checklist(options=option     , id="en_iq_cal"   , className="field-body", value=en_iq_cal_values),
            ], className="field"),
            html.Div([
                    html.Div("Gain Lock Interval [frame]:", className="field-label", id="label_gain_lock_interval"),                                         
                    dcc.Input(id='cfg_gain_lock', value=daq_cfg_params[16], type='number', debounce=True, className="field-body-textbox")
            ], className="field"),
            html.Div([
                    html.Div("Require Track Lock Intervention (For Kerberos):", className="field-label", id="label_require_track_lock"),                                         
                    dcc.Checklist(options=option     , id="en_req_track_lock_intervention"   , className="field-body", value=en_req_track_lock_values),
            ], className="field"),
            html.Div([
                    html.Div("Calibration Track Mode:", className="field-label", id="label_calibration_track_mode"),                  
                    dcc.Dropdown(id='cfg_cal_track_mode',
                                options=[
                                        {'label': i[0], 'value': i[1]} for i in calibration_tack_modes
                                ],
                                value=daq_cfg_params[18], style={"display":"inline-block"},className="field-body"),                                        
            ], className="field"),
            html.Div([
                    html.Div("Amplitude Calibration Mode :", className="field-label", id="label_amplitude_calibration_mode"),                  
                    dcc.Dropdown(id='cfg_amplitude_cal_mode',
                                options=[
                                        {'label': 'default', 'value': 'default'},
                                        {'label': 'disabled', 'value': 'disabled'},
                                        {'label': 'channel_power', 'value': 'channel_power'}
                                ],
                                value=daq_cfg_params[19], style={"display":"inline-block"},className="field-body"),                                        
            ], className="field"),
            html.Div([
                    html.Div("Calibration Frame Interval:", className="field-label", id="label_calibration_frame_interval"),                                         
                    dcc.Input(id='cfg_cal_frame_interval', value=daq_cfg_params[20], type='number', debounce=True, className="field-body-textbox")
            ], className="field"),
            html.Div([
                    html.Div("Calibration Frame Burst Size:", className="field-label", id="label_calibration_frame_burst_size"),                                         
                    dcc.Input(id='cfg_cal_frame_burst_size', value=daq_cfg_params[21], type='number', debounce=True, className="field-body-textbox")
            ], className="field"),
            html.Div([
                    html.Div("Amplitude Tolerance [dB]:", className="field-label", id="label_amplitude_tolerance"),                                         
                    dcc.Input(id='cfg_amplitude_tolerance', value=daq_cfg_params[22], type='number', debounce=True, className="field-body-textbox")
            ], className="field"),
            html.Div([
                    html.Div("Phase Tolerance [deg]:", className="field-label", id="label_phase_tolerance"),                                         
                    dcc.Input(id='cfg_phase_tolerance', value=daq_cfg_params[23], type='number', debounce=True, className="field-body-textbox")
            ], className="field"),
            html.Div([
                    html.Div("Maximum Sync Fails:", className="field-label", id="label_max_sync_fails"),                                         
                    dcc.Input(id='cfg_max_sync_fails', value=daq_cfg_params[24], type='number', debounce=True, className="field-body-textbox")
            ], className="field"),
        ], style={'width': '100%'}, id='adv-cfg-container'),

        # Reconfigure Button
        html.Div([
            html.Button('Reconfigure & Restart DAQ chain', id='btn_reconfig_daq_chain', className="btn"),
        ], className="field"),
    ]
    for i in range(len(daq_subsystem_reconfiguration_options)):
        daq_config_card_list.append(daq_subsystem_reconfiguration_options[i])

    daq_config_card = html.Div(daq_config_card_list, className="card")
    #-----------------------------
    #       DAQ Status Card
    #-----------------------------
    daq_status_card = \
    html.Div([
        html.H2("DAQ Subsystem Status", id="init_title_s"),
        html.Div([html.Div("Update rate:"              , id="label_daq_update_rate"   , className="field-label"), html.Div("- ms"        , id="body_daq_update_rate"   , className="field-body")], className="field"),
        html.Div([html.Div("Latency:"                  , id="label_daq_dsp_latency"   , className="field-label"), html.Div("- ms"        , id="body_daq_dsp_latency"   , className="field-body")], className="field"),
        html.Div([html.Div("Frame index:"              , id="label_daq_frame_index"   , className="field-label"), html.Div("-"           , id="body_daq_frame_index"   , className="field-body")], className="field"),
        html.Div([html.Div("Frame type:"               , id="label_daq_frame_type"    , className="field-label"), html.Div("-"           , id="body_daq_frame_type"    , className="field-body")], className="field"),
        html.Div([html.Div("Frame sync:"               , id="label_daq_frame_sync"    , className="field-label"), html.Div("LOSS"        , id="body_daq_frame_sync"    , className="field-body", style={"color": "red"})], className="field"),                
        html.Div([html.Div("Power level:"              , id="label_daq_power_level"   , className="field-label"), html.Div("-"           , id="body_daq_power_level"   , className="field-body")], className="field"),
        html.Div([html.Div("Connection status:"        , id="label_daq_conn_status"   , className="field-label"), html.Div("Disconnected", id="body_daq_conn_status"   , className="field-body", style={"color": "red"})], className="field"),
        html.Div([html.Div("Sample delay snyc:"        , id="label_daq_delay_sync"    , className="field-label"), html.Div("LOSS"        , id="body_daq_delay_sync"    , className="field-body", style={"color": "red"})], className="field"),
        html.Div([html.Div("IQ snyc:"                  , id="label_daq_iq_sync"       , className="field-label"), html.Div("LOSS"        , id="body_daq_iq_sync"       , className="field-body", style={"color": "red"})], className="field"),
        html.Div([html.Div("Noise source state:"       , id="label_daq_noise_source"  , className="field-label"), html.Div("Disabled"    , id="body_daq_noise_source"  , className="field-body", style={"color": "green"})], className="field"),
        html.Div([html.Div("RF center frequecy [MHz]:" , id="label_daq_rf_center_freq", className="field-label"), html.Div("- MHz"       , id="body_daq_rf_center_freq", className="field-body")], className="field"),
        html.Div([html.Div("Sampling frequency [MHz]:" , id="label_daq_sampling_freq" , className="field-label"), html.Div("- MHz"       , id="body_daq_sampling_freq" , className="field-body")], className="field"),
        html.Div([html.Div("Data block length [ms]:"   , id="label_daq_cpi"           , className="field-label"), html.Div("- ms"        , id="body_daq_cpi"           , className="field-body")], className="field"),
        html.Div([html.Div("IF gains [dB]:"            , id="label_daq_if_gain"       , className="field-label"), html.Div("[,] dB"      , id="body_daq_if_gain"       , className="field-body")], className="field"),
    ], className="card")

    #-----------------------------
    #    DSP Confugartion Card
    #-----------------------------

    dsp_config_card = \
    html.Div([
        html.H2("Passive Radar Configuration", id="init_title_d"),
        html.Div([html.Div("Enable Passive Radar", id="label_en_pr"     , className="field-label"),
                dcc.Checklist(options=option     , id="en_pr_check"     , className="field-body", value=en_pr_values),
        ], className="field"),

        html.Div([html.Div("Clutter Cancellation:"              , id="label_clutter_cancellation"   , className="field-label"),
        dcc.Dropdown(id='clutter_cancel_algo',
            options=[
                {'label': 'OFF', 'value': "OFF"},
                {'label': 'Wiener MRE'   , 'value': "Wiener MRE"},
                ],
        value=webInterface_inst.module_signal_processor.PR_clutter_cancellation, style={"display":"inline-block"},className="field-body")
        ], className="field"),

        html.Div([html.Div("Max Bistatic Range [km]:"             , id="label_max_bistatic_range"  , className="field-label"), 
            dcc.Dropdown(id='max_bistatic_range',
                options=[
                    {'label': '16', 'value': 16},
                    {'label': '32'   , 'value': 32},
                    {'label': '64'   , 'value': 64},
                    {'label': '128'   , 'value': 128},
                    {'label': '256'   , 'value': 256},
                    {'label': '512'   , 'value': 512},
                    {'label': '1024'   , 'value': 1024},
                    ],
            value=webInterface_inst.module_signal_processor.max_bistatic_range, style={"display":"inline-block"},className="field-body")
            ], className="field"),

        html.Div([html.Div("Max Doppler [Hz]:"             , id="label_max_doppler"  ,style={"display":"inline-block"}, className="field-label"), 
            dcc.Input(id="max_doppler", value=webInterface_inst.module_signal_processor.max_doppler, type='number', style={"display":"inline-block"}, debounce=True, className="field-body-textbox")
            ], style={'display':'inline-block'}, className="field"),

        html.Div([html.Div("PR Persist", id="label_en_persist"     , className="field-label"),
                dcc.Checklist(options=option     , id="en_persist_check"     , className="field-body", value=en_persist_values),
        ], className="field"),

        html.Div([html.Div("Persist Decay:"             , id="label_persist_decay"  ,style={"display":"inline-block"}, className="field-label"), 
            dcc.Input(id="persist_decay", value=webInterface_inst.pr_persist_decay, type='number', style={"display":"inline-block"}, debounce=True, className="field-body-textbox")
            ], style={'display':'inline-block'}, className="field"),


        html.Div([html.Div("Dynamic Range (Min):"             , id="label_dynrange_min"  ,style={"display":"inline-block"}, className="field-label"), 
            dcc.Input(id="dynrange_min", value=webInterface_inst.pr_dynamic_range_min, type='number', style={"display":"inline-block"}, debounce=True, className="field-body-textbox")
            ], style={'display':'inline-block'}, className="field"),

        html.Div([html.Div("Dynamic Range (Max):"             , id="label_dynrange_max"  ,style={"display":"inline-block"}, className="field-label"), 
            dcc.Input(id="dynrange_max", value=webInterface_inst.pr_dynamic_range_max, type='number', style={"display":"inline-block"}, debounce=True, className="field-body-textbox")
            ], style={'display':'inline-block'}, className="field"),




    ], className="card")

    #-----------------------------
    #    Display Options Card
    #-----------------------------
    #config_page_component_list = [daq_config_card, daq_status_card, dsp_config_card, display_options_card,squelch_card]
    config_page_component_list = [start_stop_card, daq_config_card, daq_status_card, dsp_config_card]

    if not webInterface_inst.disable_tooltips:
        config_page_component_list.append(tooltips.dsp_config_tooltips)
        #if len(en_advanced_daq_cfg):
            #if daq_cfg_params is not None:
        config_page_component_list.append(tooltips.daq_ini_config_tooltips)

    return html.Div(children=config_page_component_list)

        
spectrum_page_layout = html.Div([   
    html.Div([
    dcc.Graph(
        id="spectrum-graph",
        style={'width': '100%', 'height': '45%'},
        figure=fig_dummy #spectrum_fig #fig_dummy #spectrum_fig #fig_dummy
    ),
    dcc.Graph(
        id="waterfall-graph",
        style={'width': '100%', 'height': '65%'},
        figure=waterfall_fig #waterfall fig remains unchanged always due to slow speed to update entire graph #fig_dummy #spectrum_fig #fig_dummy
    ),
], style={'width': '100%', 'height': '80vh'}), #className="monitor_card"),
])

def generate_pr_page_layout(webInterface_inst):
    pr_page_layout = html.Div([
        html.Div([
        dcc.Graph(
            style={"height": "inherit", "width" : "100%"},
            id="pr-graph",
            figure=pr_fig, #fig_dummy #doa_fig #fig_dummy
        )], style={'width': '100%', 'height': '80vh'}), #className="monitor_card"),
    ])
    return pr_page_layout

#============================================
#          CALLBACK FUNCTIONS
#============================================  
@app.callback_connect
def func(client, connect):
    #print(client, connect, len(app.clients))
    if connect and len(app.clients)==1:
        fetch_dsp_data()
    elif not connect and len(app.clients)==0:
        webInterface_inst.dsp_timer.cancel()

def fetch_dsp_data():
    daq_status_update_flag = 0
    spectrum_update_flag   = 0
    doa_update_flag        = 0
    freq_update            = no_update
    #############################################
    #      Fetch new data from back-end ques    #
    #############################################
    try:
        # Fetch new data from the receiver module
        que_data_packet = webInterface_inst.rx_data_que.get(False)
        for data_entry in que_data_packet:
            if data_entry[0] == "conn-ok":
                webInterface_inst.daq_conn_status = 1
                daq_status_update_flag = 1
            elif data_entry[0] == "disconn-ok":
                webInterface_inst.daq_conn_status = 0
                daq_status_update_flag = 1
            elif data_entry[0] == "config-ok":
                webInterface_inst.daq_cfg_iface_status = 0
                daq_status_update_flag = 1
    except queue.Empty:
        # Handle empty queue here
        webInterface_inst.logger.debug("Receiver module que is empty")
    else:
        pass
        # Handle task here and call q.task_done()
    if webInterface_inst.daq_restart: # Set by the restarting script
        daq_status_update_flag = 1
    try:
        # Fetch new data from the signal processing module
        que_data_packet  = webInterface_inst.sp_data_que.get(False)
        for data_entry in que_data_packet:
            if data_entry[0] == "iq_header":
                webInterface_inst.logger.debug("Iq header data fetched from signal processing que")
                iq_header = data_entry[1]
                # Unpack header
                webInterface_inst.daq_frame_index = iq_header.cpi_index
                
                if iq_header.frame_type == iq_header.FRAME_TYPE_DATA:
                    webInterface_inst.daq_frame_type  = "Data"
                elif iq_header.frame_type == iq_header.FRAME_TYPE_DUMMY:
                    webInterface_inst.daq_frame_type  = "Dummy"
                elif iq_header.frame_type == iq_header.FRAME_TYPE_CAL:
                    webInterface_inst.daq_frame_type  = "Calibration"
                elif iq_header.frame_type == iq_header.FRAME_TYPE_TRIGW:
                    webInterface_inst.daq_frame_type  = "Trigger wait"
                else:
                    webInterface_inst.daq_frame_type  = "Unknown"

                webInterface_inst.daq_frame_sync        = iq_header.check_sync_word()            
                webInterface_inst.daq_power_level       = iq_header.adc_overdrive_flags
                webInterface_inst.daq_sample_delay_sync = iq_header.delay_sync_flag
                webInterface_inst.daq_iq_sync           = iq_header.iq_sync_flag
                webInterface_inst.daq_noise_source_state= iq_header.noise_source_state
                
                if webInterface_inst.daq_center_freq != iq_header.rf_center_freq/10**6: 
                    freq_update = 1
                
                webInterface_inst.daq_center_freq       = iq_header.rf_center_freq/10**6                
                webInterface_inst.daq_adc_fs            = iq_header.adc_sampling_freq/10**6
                webInterface_inst.daq_fs                = iq_header.sampling_freq/10**6
                webInterface_inst.daq_cpi               = int(iq_header.cpi_length*10**3/iq_header.sampling_freq)
                gain_list_str=""
                for m in range(iq_header.active_ant_chs):
                    gain_list_str+=str(iq_header.if_gains[m]/10)
                    gain_list_str+=", "
                webInterface_inst.daq_if_gains          =gain_list_str[:-2]
                daq_status_update_flag = 1
            elif data_entry[0] == "update_rate":
                webInterface_inst.daq_update_rate = data_entry[1]
                # Set absoluth minimum
                #if webInterface_inst.daq_update_rate < 0.1: webInterface_inst.daq_update_rate = 0.1
                if webInterface_inst._update_rate_arr is None:
                    webInterface_inst._update_rate_arr = np.ones(webInterface_inst._avg_win_size)*webInterface_inst.daq_update_rate
                webInterface_inst._update_rate_arr[0:webInterface_inst._avg_win_size-2] = \
                webInterface_inst._update_rate_arr[1:webInterface_inst._avg_win_size-1]                
                webInterface_inst._update_rate_arr[webInterface_inst._avg_win_size-1] = webInterface_inst.daq_update_rate
                #webInterface_inst.page_update_rate = np.average(webInterface_inst._update_rate_arr)*0.8
            elif data_entry[0] == "latency":
                webInterface_inst.daq_dsp_latency = data_entry[1] + webInterface_inst.daq_cpi
            elif data_entry[0] == "spectrum":
                webInterface_inst.logger.debug("Spectrum data fetched from signal processing que")
                spectrum_update_flag = 1
                webInterface_inst.spectrum = data_entry[1]
            elif data_entry[0] == "RD_matrix":
                webInterface_inst.logger.debug("Passive Radar RD Matrix data fetched from signal processing que")
                doa_update_flag = 1
                webInterface_inst.RD_matrix = data_entry[1]
            else:                
                webInterface_inst.logger.warning("Unknown data entry: {:s}".format(data_entry[0]))
    except queue.Empty:
        # Handle empty queue here
        webInterface_inst.logger.debug("Signal processing que is empty")
    else:
        pass
        # Handle task here and call q.task_done()

    if (webInterface_inst.pathname == "/config" or webInterface_inst.pathname == "/") and daq_status_update_flag:
        update_daq_status()
    elif webInterface_inst.pathname == "/spectrum" and spectrum_update_flag:
        plot_spectrum()
    elif (webInterface_inst.pathname == "/pr" and doa_update_flag): #or (webInterface_inst.pathname == "/doa" and webInterface_inst.reset_doa_graph_flag):
        plot_pr()

    webInterface_inst.dsp_timer = Timer(.01, fetch_dsp_data)
    webInterface_inst.dsp_timer.start()


def update_daq_status():

    #############################################
    #      Prepare UI component properties      #
    #############################################

    if webInterface_inst.daq_conn_status == 1:

        if not webInterface_inst.daq_cfg_iface_status:
            daq_conn_status_str = "Connected"
            conn_status_style={"color": "green"}
        else: # Config interface is busy
            daq_conn_status_str = "Reconfiguration.."
            conn_status_style={"color": "orange"}
    else:
        daq_conn_status_str = "Disconnected"
        conn_status_style={"color": "red"}

    if webInterface_inst.daq_restart:
        daq_conn_status_str = "Restarting.."
        conn_status_style={"color": "orange"}

    if webInterface_inst.daq_update_rate < 1:
        daq_update_rate_str    = "{:d} ms".format(round(webInterface_inst.daq_update_rate*1000))
    else:
        daq_update_rate_str    = "{:.2f} s".format(webInterface_inst.daq_update_rate)

    daq_dsp_latency        = "{:d} ms".format(webInterface_inst.daq_dsp_latency)
    daq_frame_index_str    = str(webInterface_inst.daq_frame_index)

    daq_frame_type_str =  webInterface_inst.daq_frame_type
    if webInterface_inst.daq_frame_type == "Data":
        frame_type_style   = frame_type_style={"color": "green"}
    elif webInterface_inst.daq_frame_type == "Dummy":
        frame_type_style   = frame_type_style={"color": "white"}
    elif webInterface_inst.daq_frame_type == "Calibration":
        frame_type_style   = frame_type_style={"color": "orange"}
    elif webInterface_inst.daq_frame_type == "Trigger wait":
        frame_type_style   = frame_type_style={"color": "yellow"}
    else:
        frame_type_style   = frame_type_style={"color": "red"}

    if webInterface_inst.daq_frame_sync:
        daq_frame_sync_str = "LOSS"
        frame_sync_style={"color": "red"}
    else:
        daq_frame_sync_str = "Ok"
        frame_sync_style={"color": "green"}
    if webInterface_inst.daq_sample_delay_sync:
        daq_delay_sync_str     = "Ok"
        delay_sync_style={"color": "green"}
    else:
        daq_delay_sync_str     = "LOSS"
        delay_sync_style={"color": "red"}

    if webInterface_inst.daq_iq_sync:
        daq_iq_sync_str        = "Ok"
        iq_sync_style={"color": "green"}
    else:
        daq_iq_sync_str        = "LOSS"
        iq_sync_style={"color": "red"}

    if webInterface_inst.daq_noise_source_state:
        daq_noise_source_str   = "Enabled"
        noise_source_style={"color": "red"}
    else:
        daq_noise_source_str   = "Disabled"
        noise_source_style={"color": "green"}

    if webInterface_inst.daq_power_level:
        daq_power_level_str = "Overdrive"
        daq_power_level_style={"color": "red"}
    else:
        daq_power_level_str = "OK"
        daq_power_level_style={"color": "green"}

    daq_rf_center_freq_str = str(webInterface_inst.daq_center_freq)
    daq_sampling_freq_str  = str(webInterface_inst.daq_fs)
    daq_cpi_str            = str(webInterface_inst.daq_cpi)


    app.push_mods({
           'body_daq_update_rate': {'children': daq_update_rate_str},
           'body_daq_dsp_latency': {'children': daq_dsp_latency},
           'body_daq_frame_index': {'children': daq_frame_index_str},
           'body_daq_frame_sync': {'children': daq_frame_sync_str},
           'body_daq_frame_type': {'children': daq_frame_type_str},
           'body_daq_power_level': {'children': daq_power_level_str},
           'body_daq_conn_status': {'children': daq_conn_status_str },
           'body_daq_delay_sync': {'children': daq_delay_sync_str},
           'body_daq_iq_sync': {'children': daq_iq_sync_str},
           'body_daq_noise_source': {'children': daq_noise_source_str},
           'body_daq_rf_center_freq': {'children': daq_rf_center_freq_str},
           'body_daq_sampling_freq': {'children': daq_sampling_freq_str},
           'body_daq_cpi': {'children': daq_cpi_str},
           'body_daq_if_gain': {'children': webInterface_inst.daq_if_gains},
    })

    app.push_mods({
           'body_daq_frame_sync': {'style': frame_sync_style},
           'body_daq_frame_type': {'style': frame_type_style},
           'body_daq_power_level': {'style': daq_power_level_style},
           'body_daq_conn_status': {'style': conn_status_style},
           'body_daq_delay_sync': {'style': delay_sync_style},
           'body_daq_iq_sync': {'style': iq_sync_style},
           'body_daq_noise_source': {'style': noise_source_style},
    })

@app.callback(
    Output(component_id="placeholder_update_freq", component_property="children"),
    [Input(component_id ="btn-update_rx_param"   , component_property="n_clicks")],
    [State(component_id ="daq_center_freq"       , component_property='value'),
    State(component_id ="daq_rx_gain"           , component_property='value'),
    State(component_id ="daq_rx_gain_2"           , component_property='value')],
)
def update_daq_params(input_value, f0, gain, gain_2):
    webInterface_inst.daq_center_freq = f0
    webInterface_inst.config_daq_rf(f0, [gain, gain_2] ) # CARL: TO CHANGE THIS TO AUTO POPULATE EACH GAIN UP TO M RECEIVERS?
    return 1

@app.callback([Output("page-content"   , "children"),
              Output("header_config"  ,"className"),  
              Output("header_spectrum","className"),
              Output("header_doa"     ,"className")],
              [Input("url"            , "pathname")])
def display_page(pathname):    
    global spectrum_fig
    global doa_fig
    webInterface_inst.pathname = pathname

    if pathname == "/":
        webInterface_inst.module_signal_processor.en_spectrum = False
        return [generate_config_page_layout(webInterface_inst), "header_active", "header_inactive", "header_inactive"]
    elif pathname == "/config":
        webInterface_inst.module_signal_processor.en_spectrum = False
        return [generate_config_page_layout(webInterface_inst), "header_active", "header_inactive", "header_inactive"]
    elif pathname == "/spectrum":
        webInterface_inst.module_signal_processor.en_spectrum = True
        spectrum_fig = None # Force reload of graphs as axes may change etc
        #time.sleep(1)
        return [spectrum_page_layout, "header_inactive", "header_active", "header_inactive"]
    elif pathname == "/pr":
        webInterface_inst.module_signal_processor.en_spectrum = False
        plot_pr()
        return [generate_pr_page_layout(webInterface_inst), "header_inactive", "header_inactive", "header_active"]
    return Output('dummy_output', 'children', '') #[no_update, no_update, no_update, no_update]

    #return [no_update, no_update, no_update, no_update]


@app.callback_shared(
    None,
    [Input(component_id='btn-start_proc', component_property='n_clicks')],
)
def start_proc_btn(input_value):
    webInterface_inst.logger.info("Start pocessing btn pushed")
    webInterface_inst.start_processing()

@app.callback_shared(
    None,
    [Input(component_id='btn-stop_proc', component_property='n_clicks')],
)
def stop_proc_btn(input_value):
    webInterface_inst.logger.info("Stop pocessing btn pushed")
    webInterface_inst.stop_processing()

@app.callback_shared(
    None,
    [Input(component_id='btn-save_cfg'     , component_property='n_clicks')],
)
def save_config_btn(input_value):
    webInterface_inst.logger.info("Saving DAQ and DSP Configuration")
    webInterface_inst.save_configuration()

def plot_pr():
    global pr_fig

    if webInterface_inst.RD_matrix is not None:

        CAFMatrix = np.abs(webInterface_inst.RD_matrix)

        CAFMatrix = CAFMatrix  / 50 #/  np.amax(CAFMatrix)  # Noramlize with the maximum value

        if webInterface_inst.CAFMatrixPersist is None or webInterface_inst.CAFMatrixPersist.shape != CAFMatrix.shape or not webInterface_inst.en_persist:
            webInterface_inst.CAFMatrixPersist = CAFMatrix
        else:
            webInterface_inst.CAFMatrixPersist = np.maximum(webInterface_inst.CAFMatrixPersist, CAFMatrix)*webInterface_inst.pr_persist_decay #webInterface_inst.CAFMatrixPersist * 0.5 + CAFMatrix * 0.5

        CAFMatrixLog = 20 * np.log10(webInterface_inst.CAFMatrixPersist)

        CAFDynRange = webInterface_inst.pr_dynamic_range_min
        CAFMatrixLog[CAFMatrixLog < CAFDynRange] = CAFDynRange

        CAFDynRange = webInterface_inst.pr_dynamic_range_max
        CAFMatrixLog[CAFMatrixLog > CAFDynRange] = CAFDynRange

        y_height = CAFMatrixLog.shape[0]
        y_range = list(np.arange(-y_height/2, y_height/2))

        app.push_mods({
            'pr-graph': {'extendData': [dict(z = [CAFMatrixLog], y = [y_range]), [0], len(CAFMatrixLog)]}
        })

def plot_spectrum():
    global spectrum_fig
    global waterfall_fig
    if spectrum_fig == None:
        spectrum_fig = go.Figure(layout=fig_layout)

        x=webInterface_inst.spectrum[0,:] + webInterface_inst.daq_center_freq*10**6

        # Plot traces
        for m in range(np.size(webInterface_inst.spectrum, 0)-1):
            spectrum_fig.add_trace(go.Scattergl(x=x,
                                     y=y, #webInterface_inst.spectrum[m+1, :],
                                     name="Channel {:d}".format(m),
                                     line = dict(color = trace_colors[m],
                                                 width = 1)
                                    ))

        spectrum_fig.update_xaxes( #title_text=freq_label,
                    color='rgba(255,255,255,1)',
                    title_font_size=20,
                    tickfont_size= 15, #figure_font_size,
                    range=[np.min(x), np.max(x)],
                    rangemode='normal',
                    mirror=True,
                    ticks='outside',
                    showline=True)
        spectrum_fig.update_yaxes(title_text="Amplitude [dB]",
                    color='rgba(255,255,255,1)',
                    title_font_size=20,
                    tickfont_size=figure_font_size,
                    range=[-90, 0],
                    mirror=True,
                    ticks='outside',
                    showline=True)


        spectrum_fig.update_layout(margin=go.layout.Margin(b=5, t=0))

        webInterface_inst.reset_spectrum_graph_flag = False
        app.push_mods({
               'spectrum-graph': {'figure': spectrum_fig},
        #       'waterfall-graph': {'figure': waterfall_fig}
        })
    
    else:
        update_data = []
        for m in range(1, np.size(webInterface_inst.spectrum, 0)): #webInterface_inst.module_receiver.M+1):
            update_data.append(dict(x=webInterface_inst.spectrum[0,:] + webInterface_inst.daq_center_freq*10**6, y=webInterface_inst.spectrum[m, :]))

        x_app = []
        y_app = []
        for m in range(1, np.size(webInterface_inst.spectrum, 0)): #webInterface_inst.module_receiver.M+1):
            x_app.append(webInterface_inst.spectrum[0,:] + webInterface_inst.daq_center_freq*10**6)
            y_app.append(webInterface_inst.spectrum[m, :])

        update_data = dict(x=x_app, y=y_app)

        app.push_mods({
            'spectrum-graph': {'extendData': [update_data, list(range(0,len(webInterface_inst.spectrum)-1)), len(webInterface_inst.spectrum[0,:])]},
            'waterfall-graph': {'extendData': [dict(z =[[webInterface_inst.spectrum[1, :]]]), [0], 50]}
        })


@app.callback(
    None,
    [Input(component_id ="placeholder_update_freq"       , component_property='children'),
    Input(component_id ="en_pr_check"       , component_property='value'),
    Input(component_id ="en_persist_check"       , component_property='value'),
    Input(component_id ="persist_decay"       , component_property='value'),
    Input(component_id ="max_bistatic_range"           , component_property='value'),
    Input(component_id ="max_doppler"           , component_property='value'),
    Input(component_id ="clutter_cancel_algo"           , component_property='value'),
    Input(component_id ="dynrange_max"           , component_property='value'),
    Input(component_id ="dynrange_min"           , component_property='value')]
)
def update_dsp_params(update_freq, en_pr, en_persist, persist_decay, max_bistatic_range, max_doppler, clutter_cancel_algo, dynrange_max, dynrange_min): #, input_value):

    if en_pr is not None and len(en_pr):
        webInterface_inst.logger.debug("Passive Radar enabled")
        webInterface_inst.module_signal_processor.en_PR = True
    else:
        webInterface_inst.module_signal_processor.en_PR = False

    if en_persist is not None and len(en_persist):
        webInterface_inst.en_persist = True
    else:
        webInterface_inst.en_persist = False

    webInterface_inst.module_signal_processor.PR_clutter_cancellation = clutter_cancel_algo
    webInterface_inst.module_signal_processor.max_bistatic_range = max_bistatic_range
    webInterface_inst.module_signal_processor.max_doppler = max_doppler
    webInterface_inst.pr_persist_decay = persist_decay
    webInterface_inst.pr_dynamic_range_min = dynrange_min
    webInterface_inst.pr_dynamic_range_max = dynrange_max

@app.callback(
    None,
    [Input('cfg_rx_channels'         ,'value'),
    Input('cfg_daq_buffer_size'      ,'value'),
    Input('cfg_sample_rate'          ,'value'),
    Input('en_noise_source_ctr'      ,'value'),
    #Input('en_squelch_mode'          ,'value'),
    #Input('cfg_squelch_init_th'      ,'value'),
    Input('cfg_cpi_size'             ,'value'),
    Input('cfg_decimation_ratio'     ,'value'),
    Input('cfg_fir_bw'               ,'value'),
    Input('cfg_fir_tap_size'         ,'value'),
    Input('cfg_fir_window'           ,'value'),
    Input('en_filter_reset'          ,'value'),
    Input('cfg_corr_size'            ,'value'),
    Input('cfg_std_ch_ind'           ,'value'),
    Input('en_iq_cal'                ,'value'),
    Input('cfg_gain_lock'            ,'value'),
    Input('en_req_track_lock_intervention','value'),
    Input('cfg_cal_track_mode'       ,'value'),
    Input('cfg_amplitude_cal_mode'   ,'value'),
    Input('cfg_cal_frame_interval'   ,'value'),
    Input('cfg_cal_frame_burst_size' ,'value'),
    Input('cfg_amplitude_tolerance'  ,'value'),
    Input('cfg_phase_tolerance'      ,'value'),
    Input('cfg_max_sync_fails'       ,'value'),
    Input('cfg_data_block_len'       ,'value'),
    Input('cfg_decimated_bw'         ,'value'),
    Input('cfg_recal_interval'       ,'value'),
    Input('daq_cfg_files'            , 'value'),

]
)
def update_daq_ini_params(
                    cfg_rx_channels,cfg_daq_buffer_size,cfg_sample_rate,en_noise_source_ctr, \
                    #en_squelch_mode,cfg_squelch_init_th,cfg_cpi_size,cfg_decimation_ratio, \
                    cfg_cpi_size,cfg_decimation_ratio, \
                    cfg_fir_bw,cfg_fir_tap_size,cfg_fir_window,en_filter_reset,cfg_corr_size, \
                    cfg_std_ch_ind,en_iq_cal,cfg_gain_lock,en_req_track_lock_intervention, \
                    cfg_cal_track_mode,cfg_amplitude_cal_mode,cfg_cal_frame_interval, \
                    cfg_cal_frame_burst_size, cfg_amplitude_tolerance,cfg_phase_tolerance, \
                    cfg_max_sync_fails, cfg_data_block_len, cfg_decimated_bw, cfg_recal_interval, config_fname):
    # TODO: Use disctionarry instead of parameter list 

    ctx = dash.callback_context
    component_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if ctx.triggered:
        if len(ctx.triggered) == 1: # User manually changed one parameter
            webInterface_inst.tmp_daq_ini_cfg = "Custom"

        # If is was the preconfig changed, just update the preconfig values
        if component_id == 'daq_cfg_files':
            webInterface_inst.daq_ini_cfg_params = read_config_file(config_fname)
            webInterface_inst.tmp_daq_ini_cfg = webInterface_inst.daq_ini_cfg_params[0]

            daq_cfg_params = webInterface_inst.daq_ini_cfg_params

            if daq_cfg_params is not None:
                en_noise_src_values       =[1] if daq_cfg_params[4]  else []
                en_squelch_values         =[1] if daq_cfg_params[5]  else []
                en_filter_rst_values      =[1] if daq_cfg_params[12] else []
                en_iq_cal_values          =[1] if daq_cfg_params[15] else []
                en_req_track_lock_values  =[1] if daq_cfg_params[17] else []


            en_persist_values     =[1] if webInterface_inst.en_persist else []
            en_pr_values          =[1] if webInterface_inst.module_signal_processor.en_PR else []

            en_advanced_daq_cfg   =[1] if webInterface_inst.en_advanced_daq_cfg                       else []

            cfg_decimated_bw = ((daq_cfg_params[3]) / daq_cfg_params[8]) / 10**3
            cfg_data_block_len = ( daq_cfg_params[7] / (cfg_decimated_bw) )
            cfg_recal_interval =  (daq_cfg_params[20] * (cfg_data_block_len/10**3)) / 60

            if daq_cfg_params[18] == 0: #If set to no tracking
                cfg_recal_interval = 1

            app.push_mods({
                'cfg_data_block_len': {'value': cfg_data_block_len},
                'cfg_decimated_bw': {'value': cfg_decimated_bw},
                'cfg_recal_interval': {'value': cfg_recal_interval},

                'cfg_rx_channels': {'value': daq_cfg_params[1]},
                'cfg_daq_buffer_size': {'value': daq_cfg_params[2]},
                'cfg_sample_rate': {'value': daq_cfg_params[3]/10**6},
                'en_noise_source_ctr': {'value': en_noise_src_values},
                'cfg_cpi_size': {'value': daq_cfg_params[7]},
                'cfg_decimation_ratio': {'value': daq_cfg_params[8]},
                'cfg_fir_bw': {'value': daq_cfg_params[9]},
                'cfg_fir_tap_size': {'value': daq_cfg_params[10]},
                'cfg_fir_window': {'value': daq_cfg_params[11]},
                'en_filter_reset': {'value': en_filter_rst_values},
                'cfg_cal_frame_interval': {'value': daq_cfg_params[20]},
                'cfg_corr_size': {'value': daq_cfg_params[13]},
                'cfg_std_ch_ind': {'value': daq_cfg_params[14]},
                'en_iq_cal': {'value': en_iq_cal_values},
                'cfg_gain_lock': {'value': daq_cfg_params[16]},
                'en_req_track_lock_intervention': {'value': en_req_track_lock_values},
                'cfg_cal_track_mode': {'value': daq_cfg_params[18]},
                'cfg_amplitude_cal_mode': {'value': daq_cfg_params[19]},
                'cfg_cal_frame_interval': {'value': daq_cfg_params[20]},
                'cfg_cal_frame_burst_size': {'value': daq_cfg_params[21]},
                'cfg_amplitude_tolerance': {'value': daq_cfg_params[22]},
                'cfg_phase_tolerance': {'value': daq_cfg_params[23]},
                'cfg_max_sync_fails': {'value': daq_cfg_params[24]},
            })

            return Output('dummy_output', 'children', '') #[no_update, no_update, no_update, no_update]




        # If the input was from basic DAQ config, update the actual DAQ params
        if component_id == "cfg_data_block_len" or component_id == "cfg_decimated_bw" or component_id == "cfg_recal_interval":
            if not cfg_data_block_len or not cfg_decimated_bw or not cfg_recal_interval:
                return Output('dummy_output', 'children', '') #[no_update, no_update, no_update, no_update]

            cfg_daq_buffer_size = 262144 # This is a reasonable DAQ buffer size to use
            cfg_corr_size = 32768 # Reasonable value that never has problems calibrating
            en_noise_source_ctr = [1]
            en_squelch_mode = [] # NOTE: For checkboxes, zero is defined by an empty array
            cfg_fir_bw = 1
            cfg_fir_window = 'hann'
            en_filter_reset = []
            cfg_std_ch_ind = 0
            en_iq_cal = [1]
            en_req_track_lock_intervention = []
            cfg_amplitude_cal_mode = 'channel_power'
            cfg_cal_frame_burst_size = 10
            cfg_amplitude_tolerance = 2
            cfg_phase_tolerance = 2
            cfg_max_sync_fails = 10

            cfg_decimation_ratio = round( (cfg_sample_rate*10**6) / (cfg_decimated_bw*10**3) )

            cfg_cpi_size = round( (cfg_data_block_len / 10**3) * cfg_decimated_bw*10**3 )
            cfg_cal_frame_interval = round((cfg_recal_interval*60) / (cfg_data_block_len/10**3))

            while cfg_decimation_ratio * cfg_cpi_size < cfg_daq_buffer_size:
                cfg_daq_buffer_size = (int) (cfg_daq_buffer_size / 2)

            cfg_corr_size = (int) (cfg_daq_buffer_size / 2)

            # Choose a tap size larger than the decimation ratio
            cfg_fir_tap_size = (int)(cfg_decimation_ratio * 1.2) + 8

            if cfg_decimation_ratio == 1:
                cfg_fir_tap_size = 1

            cfg_cal_track_mode = 0
            if cfg_cal_frame_interval > 1:
                cfg_cal_track_mode = 2 #[{'label': calibration_tack_modes[1], 'value': calibration_tack_modes[1]}]
            else:
                cfg_cal_track_mode = 0 


    param_list = []
    param_list.append("Custom")
    param_list.append(cfg_rx_channels)
    param_list.append(cfg_daq_buffer_size)
    param_list.append(int(cfg_sample_rate*10**6))
    if en_noise_source_ctr is not None and len(en_noise_source_ctr):
        param_list.append(1)
    else:
        param_list.append(0)
    
    #if en_squelch_mode is not None and len(en_squelch_mode):
    #    param_list.append(1)
    #else:
    param_list.append(0) #en_squelch placeholder

    #param_list.append(cfg_squelch_init_th)
    param_list.append(0) #cfg_squelch_init_th placeholder

    param_list.append(cfg_cpi_size)
    param_list.append(cfg_decimation_ratio)
    param_list.append(cfg_fir_bw)
    param_list.append(cfg_fir_tap_size)
    param_list.append(cfg_fir_window)
    if en_filter_reset is not None and len(en_filter_reset):
        param_list.append(1)
    else:
        param_list.append(0)
    param_list.append(cfg_corr_size)
    param_list.append(cfg_std_ch_ind)
    if en_iq_cal is not None and len(en_iq_cal):
        param_list.append(1)
    else:
        param_list.append(0)
    param_list.append(cfg_gain_lock)
    if en_req_track_lock_intervention is not None and len(en_req_track_lock_intervention):
        param_list.append(1)
    else:
        param_list.append(0)
    param_list.append(cfg_cal_track_mode)
    param_list.append(cfg_amplitude_cal_mode)
    param_list.append(cfg_cal_frame_interval)
    param_list.append(cfg_cal_frame_burst_size)
    param_list.append(cfg_amplitude_tolerance)
    param_list.append(cfg_phase_tolerance)
    param_list.append(cfg_max_sync_fails)
    param_list.append(webInterface_inst.daq_ini_cfg_params[25]) # Preserve data interface information
    webInterface_inst.daq_ini_cfg_params = param_list
    
    if ctx.triggered:
        # If we updated advanced daq, update basic DAQ params
        if component_id  == "cfg_sample_rate" or component_id == "cfg_decimation_ratio" or component_id == "cfg_cpi_size" or component_id == "cfg_cal_frame_interval":
            if not cfg_sample_rate or not cfg_decimation_ratio or not cfg_cpi_size:
                return Output('dummy_output', 'children', '') #[no_update, no_update, no_update, no_update]

            cfg_decimated_bw = ((int(cfg_sample_rate*10**6)) / cfg_decimation_ratio) / 10**3
            cfg_data_block_len = ( cfg_cpi_size  / (cfg_decimated_bw) )
            cfg_recal_interval =  (cfg_cal_frame_interval * (cfg_data_block_len/10**3)) / 60

            app.push_mods({
               'cfg_data_block_len': {'value': cfg_data_block_len},
               'cfg_decimated_bw': {'value': cfg_decimated_bw},
               'cfg_recal_interval': {'value': cfg_recal_interval},
            })
        # If we updated basic DAQ, update advanced DAQ
        elif component_id == "cfg_data_block_len" or component_id == "cfg_decimated_bw" or component_id == "cfg_recal_interval":
            app.push_mods({
               'cfg_decimation_ratio': {'value': cfg_decimation_ratio},
               'cfg_cpi_size': {'value': cfg_cpi_size},
               'cfg_cal_frame_interval': {'value': cfg_cal_frame_interval},
               'cfg_fir_tap_size': {'value': cfg_fir_tap_size},
               'cfg_sample_rate': {'value': cfg_sample_rate},
               'cfg_daq_buffer_size': {'value': cfg_daq_buffer_size},
               'cfg_corr_size': {'value': cfg_corr_size},
               'en_noise_source_ctr': {'value': en_noise_source_ctr},
               #'en_squelch_mode': {'value': en_squelch_mode},
               'cfg_fir_bw': {'value': cfg_fir_bw},
               'cfg_fir_window': {'value': cfg_fir_window},
               'en_filter_reset': {'value': en_filter_reset},
               'cfg_std_ch_ind': {'value': cfg_std_ch_ind},
               'en_iq_cal': {'value': en_iq_cal},
               'en_req_track_lock_intervention': {'value': en_req_track_lock_intervention},
               'cfg_amplitude_cal_mode': {'value': cfg_amplitude_cal_mode},
               'cfg_cal_frame_burst_size': {'value': cfg_cal_frame_burst_size},
               'cfg_amplitude_tolerance': {'value': cfg_amplitude_tolerance},
               'cfg_phase_tolerance': {'value': cfg_phase_tolerance},
               'cfg_max_sync_fails': {'value': cfg_max_sync_fails},
            })



@app.callback(Output('adv-cfg-container', 'style'),
             [Input("en_advanced_daq_cfg", "value")]
)
def toggle_adv_daq(toggle_value):
    webInterface_inst.en_advanced_daq_cfg = toggle_value
    if toggle_value:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    None,
    [Input(component_id="btn_reconfig_daq_chain"    , component_property="n_clicks")],
    [State(component_id ="daq_center_freq"       , component_property='value'),
    State(component_id ="daq_rx_gain"           , component_property='value')]
)
def reconfig_daq_chain(input_value, freq, gain):

    if input_value is None:
        return Output('dummy_output', 'children', '') #[no_update, no_update, no_update, no_update]

        #return [no_update]
#        raise PreventUpdate

    # TODO: Check data interface mode here !
    #    Update DAQ Subsystem config file

    config_res, config_err = write_config_file(webInterface_inst.daq_ini_cfg_params)
    if config_res:
        webInterface_inst.daq_cfg_ini_error = config_err[0]
        return Output("placeholder_recofnig_daq", "children", '-1')
    else:
        webInterface_inst.logger.info("DAQ Subsystem configuration file edited")

#    time.sleep(2)

    webInterface_inst.daq_restart = 1
    
    #    Restart DAQ Subsystem
    
    # Stop signal processing
    webInterface_inst.stop_processing()
 #   time.sleep(2)
    webInterface_inst.logger.debug("Signal processing stopped")

    # Close control and IQ data interfaces
    webInterface_inst.close_data_interfaces()
    webInterface_inst.logger.debug("Data interfaces are closed")

    os.chdir(daq_subsystem_path)
    # Kill DAQ subsystem
    daq_stop_script = subprocess.Popen(['bash', daq_stop_filename])#, stdout=subprocess.DEVNULL)
    daq_stop_script.wait()
    webInterface_inst.logger.debug("DAQ Subsystem halted")

    # Start DAQ subsystem
    daq_start_script = subprocess.Popen(['bash', daq_start_filename])#, stdout=subprocess.DEVNULL)
    daq_start_script.wait()
    webInterface_inst.logger.debug("DAQ Subsystem restarted")

    os.chdir(root_path)

    # Reinitialize receiver data interface
    #if webInterface_inst.module_receiver.init_data_iface() == -1:
    #    webInterface_inst.logger.critical("Failed to restart the DAQ data interface")
    #    webInterface_inst.daq_cfg_ini_error = "Failed to restart the DAQ data interface"
    #    return [-1]

    # Reset channel number count
    #webInterface_inst.module_signal_processor.first_frame = 1

    en_PR    = webInterface_inst.module_signal_processor.en_PR
    PR_clutter_cancellation = webInterface_inst.module_signal_processor.PR_clutter_cancellation
    max_bistatic_range = webInterface_inst.module_signal_processor.max_bistatic_range
    max_doppler = webInterface_inst.module_signal_processor.max_doppler
    en_persist = webInterface_inst.en_persist
    pr_persist_decay = webInterface_inst.pr_persist_decay
    pr_dynamic_range_min = webInterface_inst.pr_dynamic_range_min
    pr_dynamic_range_max = webInterface_inst.pr_dynamic_range_max

    # Recreate and reinit the receiver and signal processor modules from scratch, keeping current setting values
    daq_center_freq = webInterface_inst.module_receiver.daq_center_freq
    daq_rx_gain = webInterface_inst.module_receiver.daq_rx_gain
    rec_ip_addr = webInterface_inst.module_receiver.rec_ip_addr

    webInterface_inst.module_receiver = ReceiverRTLSDR(data_que=webInterface_inst.rx_data_que, data_interface=settings.data_interface, logging_level=settings.logging_level*10)
    webInterface_inst.module_receiver.daq_center_freq   = daq_center_freq
    webInterface_inst.module_receiver.daq_rx_gain       = daq_rx_gain #settings.uniform_gain #daq_rx_gain
    webInterface_inst.module_receiver.rec_ip_addr       = rec_ip_addr

    webInterface_inst.module_signal_processor = SignalProcessor(data_que=webInterface_inst.sp_data_que, module_receiver=webInterface_inst.module_receiver, logging_level=settings.logging_level*10)
    webInterface_inst.module_signal_processor.en_PR    = en_PR
    webInterface_inst.module_signal_processor.PR_clutter_cancellation = PR_clutter_cancellation
    webInterface_inst.module_signal_processor.max_bistatic_range = max_bistatic_range
    webInterface_inst.module_signal_processor.max_doppler = max_doppler
    webInterface_inst.en_persist = en_persist
    webInterface_inst.pr_persist_decay = pr_persist_decay
    webInterface_inst.pr_dynamic_range_min = pr_dynamic_range_min
    webInterface_inst.pr_dynamic_range_max = pr_dynamic_range_max

    webInterface_inst.module_signal_processor.start()

    # This must be here, otherwise the gains dont reinit properly?
    webInterface_inst.module_receiver.M = webInterface_inst.daq_ini_cfg_params[1]

    # Restart signal processing
    webInterface_inst.start_processing()
    webInterface_inst.logger.debug("Signal processing started")
    webInterface_inst.daq_restart = 0

    webInterface_inst.module_receiver.iq_header.active_ant_chs = webInterface_inst.daq_ini_cfg_params[1]

    webInterface_inst.daq_cfg_ini_error = ""
    webInterface_inst.active_daq_ini_cfg = webInterface_inst.daq_ini_cfg_params[0] #webInterface_inst.tmp_daq_ini_cfg


    return Output("daq_cfg_files", "value", daq_config_filename), Output("active_daq_ini_cfg", "children", "Active Configuration: " + webInterface_inst.active_daq_ini_cfg)


if __name__ == "__main__":    
    # For Development only, otherwise use gunicorn    
    # Debug mode does not work when the data interface is set to shared-memory "shmem"! 
    app.run_server(debug=False, host="0.0.0.0", port=8080)
    #waitress #serve(app.server, host="0.0.0.0", port=8050)

"""
html.Div([
    html.H2("System Logs"),
    dcc.Textarea(
        placeholder = "Enter a value...",
        value = "System logs .. - Curently NOT used",
        style = {"width": "100%", "background-color": "#000000", "color":"#02c93d"}
    )
], className="card")
"""
