#!/bin/bash

sudo ip addr add 192.168.0.10/24 dev enp5s0 # add my pc's ip to the ethernetwork
sudo ip link set enp5s0 up # i dunno. do more add stuff.
ping 192.168.0.2 # check if we can talk to powersupply