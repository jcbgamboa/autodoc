#! /bin/bash

# Tells which class refers to each number

count=0
cd $1
for d in * ; do
	#echo "$d $count"
	echo "$d"
	count=$((count+1))
done

