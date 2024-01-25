#!/bin/bash

# Clear pycache before starting if the -c flag is given
while getopts c flag
do
    case "${flag}" in
        c) sudo py3clean . ;;
    esac
done

#./kraken_doa_stop.sh
#sleep 2

cd /home/krakenrf/heimdall_daq_fw/Firmware
#sudo ./daq_synthetic_start.sh
sudo -E ./daq_start_sm.sh
sleep 2
cd /home/krakenrf/krakensdr_pr
sudo -E ./gui_run.sh
