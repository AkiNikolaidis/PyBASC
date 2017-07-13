#!/bin/bash

sublist='/data/rockland_sample/sub.txt'
subs=`cat $sublist`


for sub in $subs
do
        #fslmaths /Users/aki.nikolaidis/Desktop/NKI_SampleData/$sub/bandpassed_demeaned_filtered_antswarp.nii.gz -subsamp2 /Users/aki.nikolaidis/Desktop/NKI_SampleData/$sub/downsamp_bandpassed_demeaned_filtered_antswarp.nii.gz
	mkdir -p /data/Projects/anikolai/rockland_downsampled/$sub

        flirt -in /data/rockland_sample/$sub/functional_mni/*/bandpassed_demeaned_filtered_antswarp.nii.gz -ref /data/rockland_sample/MNI152_T1_3mm_brain.nii.gz -applyxfm -init /data/rockland_sample/xfm_idt.mat -out /data/Projects/anikolai/rockland_downsampled/$sub/3mm_resampled.nii.gz

done
