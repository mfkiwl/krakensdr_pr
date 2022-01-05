# Kraken SDR Passive Radar

NOTE still under development.

## Installation

1. Install the prerequisites

``` bash
sudo apt update
sudo apt install libfftw3-3
sudo apt install libfftw3-dev
```

2. Install Heimdall DAQ

If not done already, first, follow the instructions at https://github.com/krakenrf/heimdall_daq_fw/tree/development to install the Heimdall DAQ Firmware.

3. Set up Miniconda environment

You will have created a Miniconda environment during the Heimdall DAQ install phase.

Please run the installs in this order as we need to ensure a specific version of dash is installed.

``` bash
conda activate kraken

conda install pip
conda install quart
conda install pandas
conda install orjson
conda install matplotlib

pip3 install dash_bootstrap_components
pip3 install quart_compress
pip3 install dash_devices
pip3 install pyapril
pip3 install cython
pip3 install pyfftw

conda install dash==1.20.0
```

4. Install the krakensdr_pr software

```bash
cd ~/krakensdr
git clone https://github.com/krakenrf/krakensdr_pr
cd krakensdr_pr
git checkout development
```

Copy the the *krakensdr_doa/util/kraken_doa_start.sh* and the *krakensdr_doa/util/kraken_doa_stop.sh* scripts into the krakensdr root folder of the project.
```bash
cd ~/krakensdr
cp krakensdr_pr/util/kraken_pr_start.sh .
cp krakensdr_pr/util/kraken_pr_stop.sh .
```

## Running

### Local operation (Recommended)

```bash
./kraken_pr_start.sh
```

Please be patient on the first run, at it can take 1-2 minutes for the JIT numba compiler to compile the numba optimized functions, and during this compilation time it may appear that the software has gotten stuck. On subsqeuent runs this loading time will be much faster as it will read from cache.

Make sure to set an appropriate passive radar config ini for the Heimdall DAQ. (MORE INFO TO BE ADDED SOON, but basically, set decimation and FIR filtering to 1, set the sample rate to the max of 2.56 MHz, set the nubmer of RX channels to 2, DAQ buffer size 262144, CPI size to 524288 or larger (must be a power of 2). Then use channel 1 to connect your reference antenna, and channel 2 for your surveillance antenna.)

### Remote operation

*UNTESTED*

1. Start the DAQ Subsystem either remotely. (Make sure that the *daq_chain_config.ini* contains the proper configuration) 
    (See:https://github.com/krakenrf/heimdall_daq_fw/Documentation)
2. Set the IP address of the DAQ Subsystem in the settings.json, *default_ip* field.
3. Start the DoA DSP software by typing:
`./gui_run.sh`
4. To stop the server and the DSP processing chain run the following script:
`./kill.sh`

<p1> After starting the script a web based server opens at port number 8088, which then can be accessed by typing "KRAKEN_IP:8080/" in the address bar of any web browser. You can find the IP address of the KrakenSDR Pi4 wither via your routers WiFi management page, or by typing "ip addr" into the terminal. You can also use the hostname of the Pi4 in place of the IP address, but this only works on local networks, and not the internet, or mobile hotspot networks. </p1>
