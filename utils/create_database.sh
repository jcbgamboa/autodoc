#! /bin/bash

# Generates the input file for Caffe

count=0
cd $1
for d in */ ; do
	files=`ls $d | wc -l`

	#for ((i = 0; i < $files; i++));
	for f in `ls $d`
	do
		if [ $f == "Thumbs.db" ]
		then
			continue
		fi

		echo "$d$f $count"
	done
	count=$((count+1))
done

