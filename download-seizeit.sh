#!/bin/bash

BUCKET="s3://openneuro.org/ds005873"
DESTINATION="/home/ruben/Documents/1. Studie/INF/Jaar 3/AI in Healthcare/capstone-project/data/raw/seizeit2/"

for i in {1..125}; do
    # Format the number with leading zeros (e.g., 001, 002)
    SUBJECT=$(printf "sub-%03d" $i)

    echo "Downloading $SUBJECT..."

    aws s3 sync \
        "$BUCKET/$SUBJECT/" \
        "$DESTINATION$SUBJECT/" \
        --exclude "*" \
        --include "*eeg*" \
        --include "*ecg*" \
        --no-sign-request
done