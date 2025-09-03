#!/bin/bash

sudo ip addr add 192.168.0.10/24 dev enp5s0                     
sudo ip link set enp5s0 up                                      
ping 192.168.0.2     