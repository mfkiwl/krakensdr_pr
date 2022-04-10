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
import sys
import os
import time
import logging
import threading
import queue
import math

# Import optimization modules
import numba as nb
from numba import jit, njit
from functools import lru_cache

# Math support
import numpy as np
import numpy.linalg as lin
#from numba import jit
import pyfftw

# Signal processing support
import scipy
from scipy import fft
from scipy import signal
from scipy.signal import correlate
from scipy.signal import convolve

from pyapril import channelPreparation as cp
from pyapril import clutterCancellation as cc
from pyapril import detector as det

c_dtype = np.complex64

#import socket
# UDP is useless to us because it cannot work over mobile internet

# Init UDP
#server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
#server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
# Enable broadcasting mode
#server.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
# Set a timeout so the  socket does not block
# indefinitely when trying to receive data.
#server.settimeout(0.2)

class SignalProcessor(threading.Thread):
    
    def __init__(self, data_que, module_receiver, logging_level=10):

        """
            Parameters:
            -----------
            :param: data_que: Que to communicate with the UI (web iface/Qt GUI)
            :param: module_receiver: Kraken SDR DoA DSP receiver modules
        """        
        super(SignalProcessor, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging_level)

        root_path      = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        #doa_res_file_path = os.path.join(os.path.join(root_path,"_android_web","DOA_value.html"))        
        #self.DOA_res_fd = open(doa_res_file_path,"w+")

        self.module_receiver = module_receiver
        self.data_que = data_que
        self.en_spectrum = False
        self.en_record = False
        self.en_DOA_estimation = True
        self.first_frame = 1 # Used to configure local variables from the header fields
        self.processed_signal = np.empty(0)        

        # Squelch feature
        self.data_ready = False
        self.en_squelch = False
        self.squelch_threshold = 0.1
        self.squelch_trigger_channel = 0
        self.raw_signal_amplitude = np.empty(0)
        self.filt_signal = np.empty(0)
        self.squelch_mask = np.empty(0)
                
        # DOA processing options
        self.en_DOA_Bartlett = False
        self.en_DOA_Capon    = False
        self.en_DOA_MEM      = False
        self.en_DOA_MUSIC    = False
        self.en_DOA_FB_avg   = False
        self.DOA_offset      = 0
        self.DOA_inter_elem_space = 0.5
        self.DOA_ant_alignment    = "ULA"
        self.DOA_theta =  np.linspace(0,359,360)

        # PR processing options
        self.PR_clutter_cancellation = "Wiener MRE"
        self.max_bistatic_range = 128
        self.max_doppler = 256
        self.en_PR = True
            

        # Processing parameters        
        self.spectrum_window_size = 2048 #1024
        self.spectrum_window = "hann"
        self.run_processing = False
        self.is_running = False 


        self.channel_number = 4  # Update from header
        
        # Result vectors
        self.DOA_Bartlett_res = np.ones(181)
        self.DOA_Capon_res = np.ones(181)
        self.DOA_MEM_res = np.ones(181)
        self.DOA_MUSIC_res = np.ones(181)
        self.DOA_theta = np.arange(0,181,1)

        self.max_index = 0
        self.max_frequency = 0
        self.fft_signal_width = 0

        self.DOA_theta =  np.linspace(0,359,360)

        self.spectrum = None #np.ones((self.channel_number+2,N), dtype=np.float32)
        self.spectrum_upd_counter = 0


    def run(self):
        """
            Main processing thread        
        """

        pyfftw.config.NUM_THREADS = 4
        scipy.fft.set_backend(pyfftw.interfaces.scipy_fft)
        pyfftw.interfaces.cache.enable()

        while True:
            self.is_running = False
            time.sleep(1)
            while self.run_processing:  
                self.is_running = True

                que_data_packet = []
                start_time = time.time()

                #-----> ACQUIRE NEW DATA FRAME <-----
                self.module_receiver.get_iq_online()

                # Check frame type for processing
                en_proc = (self.module_receiver.iq_header.frame_type == self.module_receiver.iq_header.FRAME_TYPE_DATA)# or \
                          #(self.module_receiver.iq_header.frame_type == self.module_receiver.iq_header.FRAME_TYPE_CAL)# For debug purposes
                """
                    You can enable here to process other frame types (such as call type frames)
                """

                que_data_packet.append(['iq_header',self.module_receiver.iq_header])
                self.logger.debug("IQ header has been put into the data que entity")

                # Configure processing parameteres based on the settings of the DAQ chain
                if self.first_frame:
                    self.channel_number = self.module_receiver.iq_header.active_ant_chs
                    self.spectrum_upd_counter = 0
                    self.spectrum = np.ones((self.channel_number+1, self.spectrum_window_size), dtype=np.float32)
                    self.first_frame = 0

                decimation_factor = 1

                self.data_ready = False

                if en_proc:
                    self.processed_signal = self.module_receiver.iq_samples
                    self.data_ready = True

                    first_decimation_factor = 1 #480

                    # TESTING: DSP side main decimation - significantly slower than NE10 but it works ok-ish
                    #decimated_signal = signal.decimate(self.processed_signal, first_decimation_factor, n = 584, ftype='fir', zero_phase=True) #first_decimation_factor * 2, ftype='fir')
                    #self.processed_signal = decimated_signal #.copy()
                    #spectrum_signal = decimated_signal.copy()

                    max_amplitude = -100

                    #max_ch = np.argmax(np.max(self.spectrum[1:self.module_receiver.iq_header.active_ant_chs+1,:], axis=1)) # Find the channel that had the max amplitude
                    max_amplitude = 0 #np.max(self.spectrum[1+max_ch, :]) #Max amplitude out of all 5 channels
                    #max_spectrum = self.spectrum[1+max_ch, :] #Send max ch to channel centering

                    que_data_packet.append(['max_amplitude',max_amplitude])

                    #-----> SQUELCH PROCESSING <-----

                    if self.en_squelch:                    
                        self.data_ready = False

                        self.processed_signal, decimation_factor, self.fft_signal_width, self.max_index = \
                                               center_max_signal(self.processed_signal, self.spectrum[0,:], max_spectrum, self.module_receiver.daq_squelch_th_dB, self.module_receiver.iq_header.sampling_freq)

                        #decimated_signal = []
                        #if(decimation_factor > 1):
                        #    decimated_signal = signal.decimate(self.processed_signal, decimation_factor, n = decimation_factor * 2, ftype='fir')
                        #    self.processed_signal = decimated_signal #.copy()


                        #Only update if we're above the threshold
                        if max_amplitude > self.module_receiver.daq_squelch_th_dB:
                            self.data_ready = True

                     
                    #-----> SPECTRUM PROCESSING <----- 
                    
                    if self.en_spectrum and self.data_ready:

                        spectrum_samples = self.module_receiver.iq_samples #spectrum_signal #self.processed_signal #self.module_receiver.iq_samples #self.processed_signal

                        N = self.spectrum_window_size

                        N_perseg = 0
                        N_perseg = min(N, len(self.processed_signal[0,:])//25)
                        N_perseg = N_perseg // 1
                       
                        for m in range(self.channel_number):
                            # Get power spectrum
                            f, Pxx_den = signal.welch(self.processed_signal[m, :], self.module_receiver.iq_header.sampling_freq//first_decimation_factor,
                                                nperseg=N_perseg,
                                                nfft=N,
                                                noverlap=0, #int(N_perseg*0.25),
                                                detrend=False,
                                                return_onesided=False,
                                                window= ('tukey', 0.25), #tukey window gives better time resolution for squelching #self.spectrum_window, #('tukey', 0.25), #self.spectrum_window, 
                                                #window=self.spectrum_window,
                                                scaling="spectrum")

                            self.spectrum[1+m, :] = np.fft.fftshift(10*np.log10(Pxx_den))
                        #self.spectrum[1:self.module_receiver.iq_header.active_ant_chs+1,:] = np.fft.fftshift(10*np.log10(Pxx_den))

                        self.spectrum[0,:] = np.fft.fftshift(f)


                        # Create signal window for plot
#                        signal_window = np.ones(len(self.spectrum[1,:])) * -100
 #                       signal_window[max(self.max_index - self.fft_signal_width//2, 0) : min(self.max_index + self.fft_signal_width//2, len(self.spectrum[1,:]))] = max(self.spectrum[1,:])
                        #signal_window = np.ones(len(max_spectrum)) * -100
                        #signal_window[max(self.max_index - self.fft_signal_width//2, 0) : min(self.max_index + self.fft_signal_width//2, len(max_spectrum))] = max(max_spectrum)

                        #self.spectrum[self.channel_number+1, :] = signal_window #np.ones(len(spectrum[1,:])) * self.module_receiver.daq_squelch_th_dB # Plot threshold line
                        que_data_packet.append(['spectrum', self.spectrum])

                    #-----> Passive Radar <----- 
                    conf_val = 0
                    theta_0 = 0
                    if self.en_PR and self.data_ready and self.channel_number > 1:

                        ref_ch = self.module_receiver.iq_samples[0,:]
                        surv_ch = self.module_receiver.iq_samples[1,:]

                        td_filter_dimension = self.max_bistatic_range

                        start = time.time()

                        if self.PR_clutter_cancellation == "Wiener MRE":
                            surv_ch, w = Wiener_SMI_MRE(ref_ch, surv_ch, td_filter_dimension)
                            #surv_ch, w = cc.Wiener_SMI_MRE(ref_ch, surv_ch, td_filter_dimension)

                        surv_ch = det.windowing(surv_ch, "Hamming") #surv_ch * signal.tukey(surv_ch.size, alpha=0.25) #det.windowing(surv_ch, "hamming")

                        max_Doppler = self.max_doppler #256
                        max_range = self.max_bistatic_range

                        #RD_matrix = det.cc_detector_ons(ref_ch, surv_ch, self.module_receiver.iq_header.sampling_freq, max_Doppler, max_range, verbose=0, Qt_obj=None)
                        RD_matrix = cc_detector_ons(ref_ch, surv_ch, self.module_receiver.iq_header.sampling_freq, max_Doppler, max_range)

                        end = time.time()
                        #print("Time: " + str((end-start) * 1000))

                        que_data_packet.append(['RD_matrix', RD_matrix])

                    # Record IQ samples
                    if self.en_record:
                        # TODO: Implement IQ frame recording
                        self.logger.error("Saving IQ samples to npy is obsolete, IQ Frame saving is currently not implemented")

                stop_time = time.time()
                que_data_packet.append(['update_rate', stop_time-start_time])
                que_data_packet.append(['latency', int(stop_time*10**3)-self.module_receiver.iq_header.time_stamp])

                # If the que is full, and data is ready (from squelching), clear the buffer immediately so that useful data has the priority
                if self.data_que.full() and self.data_ready:
                    try:
                        #self.logger.info("BUFFER WAS NOT EMPTY, EMPTYING NOW")
                        self.data_que.get(False) #empty que if not taken yet so fresh data is put in
                    except queue.Empty:
                        #self.logger.info("DIDNT EMPTY")
                        pass

                # Put data into buffer, but if there is no data because its a cal/trig wait frame etc, then only write if the buffer is empty
                # Otherwise just discard the data so that we don't overwrite good DATA frames.
                try:
                    self.data_que.put(que_data_packet, False) # Must be non-blocking so DOA can update when dash browser window is closed
                except:
                    # Discard data, UI couldn't consume fast enough
                    pass

                """
                start = time.time()
                end = time.time()
                thetime = ((end - start) * 1000)
                print ("Time elapsed: ", thetime)
                """
@jit(fastmath=True)
def Wiener_SMI_MRE(ref_ch, surv_ch, K):
    """
        Description:
        ------------
            Performs Wiener filtering with applying the Minimum Redundance Estimation (MRE) technique. 
            When using MRE, the autocorrelation matrix is not fully estimated, but only the first column.
            With this modification the required calculations can be reduced from KxK to K element.
            
        Parameters:
        -----------
            :param K      : Filter tap number
            :param ref_ch : Reference signal array
            :param surv_ch: Surveillance signal array
            
            :type K      : int
            :type ref_ch : 1 x N complex numpy array
            :type surv_ch: 1 x N complex numpy array
        Return values:
        --------------
            :return filt: Filtered surveillance channel
            :rtype filt: 1 x N complex numpy array
            
            :return None: Input parameters are not consistent
    """

    N = ref_ch.shape[0]  # Number of time samples
    R, r = pruned_correlation(ref_ch, surv_ch, K, N)
    R_mult = R_eye_memoize(K)
    w = fast_w(R, r, K, R_mult)

    #return surv_ch - np.convolve(ref_ch, w)[0:N], w  # subtract the zero doppler clutter
    return surv_ch - signal.oaconvolve(ref_ch, w)[0:N], w  # subtract the zero doppler clutter #oaconvolve saves us about 100-200 ms

@njit(fastmath=True, parallel=True, cache=True)
def fast_w(R, r, K, R_mult):
    # Complete the R matrix based on its Hermitian and Toeplitz property

    for k in range(1, K):
        R[:, k] = shift(R[:, 0], k)
    #R[:, K] = shift(R[:,0], K)

    R += np.transpose(np.conjugate(R))
    R *= R_mult #(np.ones(K) - np.eye(K) * 0.5)

    #w = np.dot(lin.inv(R), r)  # weight vector
    w = lin.inv(R) @ r #np.dot(lin.inv(R), r)  # weight vector #matmul (@) may be slightly faster that np.dot for 1D, 2D arrays.
    # inverse and dot product run time : 1.1s for 2048*2048 matrix

    return w

#Memoize ~50ms speedup?
@lru_cache(maxsize=2)
def R_eye_memoize(K):
    return (np.ones(K) - np.eye(K) * 0.5)

#Modified pruned correlation, returns R and r directly and saves one FFT
@jit(fastmath=True, cache=True)
def pruned_correlation(ref_ch, surv_ch, clen, N):
    """
        Description:
        -----------
        Calculates the part of the correlation function of arrays with same size
        The total length of the cross-correlation function is 2*N-1, but this
        function calculates the values of the cross-correlation between [N-1 : N+clen-1]
        Parameters:
        -----------
        :param x : input array
        :param y : input array
        :param clen: correlation length

        :type x: 1 x N complex numpy array
        :type y: 1 x N complex numpy array
        :type clen: int
        Return values:
        --------------
        :return corr : part of the cross-correlation function
        :rtype  corr : 1 x clen complex numpy array

        :return None : inconsistent array size
    """
    R = np.zeros((clen, clen), dtype=c_dtype)  # Autocorrelation mtx.

    # --calculation--
    # set up input matrices pad zeros if not multiply of the correlation length
    cols = clen - 1 #(clen = Filter drowsimension)
    rows = np.int32(N / (cols)) + 1

    zeropads = cols * rows - N
    x = np.pad(ref_ch, (0, zeropads))

    # shaping inputs into matrices
    xp = np.reshape(x, (rows, cols))

    # padding matrices for FFT
    ypp = np.vstack([xp[1:, :], np.zeros(cols, dtype=c_dtype)]) #vstack appears to be faster than pad
    yp = np.concatenate([xp, ypp], axis=1)

    #print("pruned corr xp shape: " + str(xp.shape))
    #print("pruned corr yp shape: " + str(yp.shape))

    # execute FFT on the matrices
    xpw = fft.fft(xp, n = 2*cols, axis=1, workers=4, overwrite_x=True)
    bpw = fft.fft(yp, axis=1, workers=4, overwrite_x=True)

    # magic formula which describes the unified equation of the universe
   # corr_batches = np.fliplr(fft.fftshift(fft.ifft(corr_mult(xpw, bpw), axis=1, workers=4, overwrite_x=True)).conj()[:, 0:clen])
    corr_batches = fft.fftshift(fft.ifft(corr_mult(xpw, bpw), axis=1, workers=4, overwrite_x=True)).conj()[:, 0:clen]

    # sum each value in a column of the batched correlation matrix
    R[:,0] = np.fliplr([np.sum(corr_batches, axis=0)])[0]

    #calc r
    y = np.pad(surv_ch, (0, zeropads))
    yp = np.reshape(y, (rows, cols))
    ypp = np.vstack([yp[1:, :], np.zeros(cols, dtype=c_dtype)]) #vstack appears to be faster than pad
    yp = np.concatenate([yp, ypp], axis=1)
    bpw = fft.fft(yp, axis=1, workers=4, overwrite_x=True)
    #corr_batches = np.fliplr(fft.fftshift(fft.ifft(corr_mult(xpw, bpw), axis=1, workers=4, overwrite_x=True)).conj()[:, 0:clen])
    corr_batches = fft.fftshift(fft.ifft(corr_mult(xpw, bpw), axis=1, workers=4, overwrite_x=True)).conj()[:, 0:clen]
    #r = np.sum(corr_batches, axis=0)
    r = np.fliplr([np.sum(corr_batches, axis=0)])[0]

    return R, r

@njit(fastmath=True, cache=True)
def shift(x, i):
    """
        Description:
        -----------
        Similar to np.roll function, but not circularly shift values
        Example:
        x = |x0|x1|...|xN-1|
        y = shift(x,2)
        x --> y: |0|0|x0|x1|...|xN-3|
        Parameters:
        -----------
        :param:x : input array on which the roll will be performed
        :param i : delay value [sample]
        
        :type i :int
        :type x: N x 1 complex numpy array
        Return values:
        --------------
        :return shifted : shifted version of x
        :rtype shifted: N x 1 complex numpy array
    """

    N = x.shape[0]
    if np.abs(i) >= N:
        return np.zeros(N, dtype=c_dtype)
    if i == 0:
        return x
    shifted = np.roll(x, i)
    if i < 0:
        shifted[np.mod(N + i, N):] = np.zeros(np.abs(i), dtype=c_dtype)
    if i > 0:
        shifted[0:i] = np.zeros(np.abs(i), dtype=c_dtype)
    return shifted


@njit(fastmath=True, parallel=True, cache=True)
def resize_and_align(no_sub_tasks, ref_ch, surv_ch, fs, fD_max, r_max):
    surv_ch_align = np.reshape(surv_ch,(no_sub_tasks, r_max))  # shaping surveillance signal array into a matrix
    pad_zeros = np.expand_dims(np.zeros(r_max, dtype=c_dtype), axis=0)
    surv_ch_align = np.vstack((surv_ch_align, pad_zeros))  # padding one row of zeros into the surv matrix
    surv_ch_align = np.concatenate((surv_ch_align[0 : no_sub_tasks,:], surv_ch_align[1 : no_sub_tasks +1, :]), axis = 1)

    ref_ch_align = np.reshape(ref_ch, (no_sub_tasks, r_max))  # shaping reference signal array into a matrix
    pad_zeros = np.zeros((no_sub_tasks, r_max),dtype = c_dtype)
    ref_ch_align = np.concatenate((ref_ch_align, pad_zeros),axis = 1)  # shaping

    return ref_ch_align, surv_ch_align

@njit(fastmath=True, cache=True)
def corr_mult(surv_fft, ref_fft):
    return np.multiply(surv_fft, ref_fft.conj())

@jit(fastmath=True, cache=True)
def cc_detector_ons(ref_ch, surv_ch, fs, fD_max, r_max):
    """
    Parameters:
    -----------
        :param N: Range resolution - N must be a divisor of the input length
        :param F: Doppler resolution, F has a theoretical limit. If you break the limit, the output may repeat
                    itself and get wrong results. F should be less than length/N otherwise use other method!
    Return values:
    --------------
        :return None: Improper input parameters

    """
    N = ref_ch.size

    # --> Set processing parameters
    fD_step = fs / (2 * N)  # Doppler frequency step size (with zero padding)
    Doppler_freqs_size = int(fD_max / fD_step)
    no_sub_tasks = N // r_max

    # Allocate range-Doppler maxtrix
    mx = np.zeros((2*Doppler_freqs_size+1, r_max),dtype = c_dtype) #memoize_zeros((2*Doppler_freqs_size+1, r_max), c_dtype) #np.zeros((2*Doppler_freqs_size+1, r_max),dtype = nb.c8)

    ref_ch_align, surv_ch_align = resize_and_align(no_sub_tasks, ref_ch, surv_ch, fs, fD_max, r_max)

    #print("ref_ch_align shape: " + str(ref_ch_align.shape))
    #print("surv_ch_align shape: " + str(surv_ch_align.shape))


    # row wise fft on both channels
    ref_fft = fft.fft(ref_ch_align, axis = 1, overwrite_x=True, workers=4) #pyfftw.interfaces.numpy_fft.fft(ref_ch_align_a, axis = 1, overwrite_input=True, threads=4) #fft.fft(ref_ch_align_a, axis = 1, overwrite_x=True, workers=4)
    surv_fft = fft.fft(surv_ch_align, axis = 1, overwrite_x=True, workers=4) #pyfftw.interfaces.numpy_fft.fft(surv_ch_align_a, axis = 1, overwrite_input=True, threads=4) #fft.fft(surv_ch_align_a, axis = 1, overwrite_x=True, workers=4)

    corr = corr_mult(surv_fft, ref_fft) #np.multiply(surv_fft, ref_fft.conj())

    corr = fft.ifft(corr,axis = 1, workers=4, overwrite_x=True)

    corr_a = pyfftw.empty_aligned(np.shape(corr), dtype=c_dtype)
    corr_a[:] = corr #.copy()

    # This is the most computationally intensive part ~120ms, overwrite_x=True gives a big speedup, not sure if it changes the result though...
    corr = fft.fft(corr_a, n=2* no_sub_tasks,  axis = 0, workers=4, overwrite_x=True) # Setting the output size with "n=.." is faster than doing a concat first.

    # crop and fft shift
    mx[ 0 : Doppler_freqs_size, 0 : r_max] = corr[2*no_sub_tasks - Doppler_freqs_size : 2*no_sub_tasks, 0 : r_max]
    mx[Doppler_freqs_size : 2 * Doppler_freqs_size+1, 0 : r_max] = corr[ 0 : Doppler_freqs_size+1 , 0 : r_max]

    return mx

