#!/bin/bash

# Runs the python scripts to generate data for report

files='1 2 3 4 5 6'

for file in $files
do
	#
	/usr/bin/time -v python3 parprog.py -i Example-Netlists/$file -o Results/$file > Performance/$file
done


files='b_50_50 b_100_100 b_400_400 b_600_1000 b_900_800 b_1000_1200 b_1200_1500 b_1500_1500 b_2000_2000'

for file in $files
do
	#
	/usr/bin/time -v python3 parprog.py -i Benchmarks2/$file -o Results/$file > Performance/$file
done
