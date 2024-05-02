#!/bin/bash

# List of files to download
years=("2018")
device_ids=(002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023)

# Output directory
output_dir="./big_data"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Loop over the years
for year in "${years[@]}"
do
    # Loop over the devices
    for device_id in "${device_ids[@]}"
    do
        # Download the files from S3
        aws s3 cp --no-sign-request s3://grillo-openeew/records/country_code=mx/device_id=${device_id}/year=${year} "$output_dir/device${device_id}" --recursive
        # Run the Python preprocessing script on the files in the directory
        # python preprocess.py "$output_dir/device${device_id}" "$output_dir/device${device_id}_preprocessed"
        # Delete the downloaded data directory
        # rm -rf "$output_dir/device${device_id}"
    done
done