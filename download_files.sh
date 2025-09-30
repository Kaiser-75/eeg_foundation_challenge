#!/bin/bash
for i in {4..11}; do
  R="R$i"
  echo "Downloading $R..."
  aws s3 cp --recursive s3://nmdatasets/NeurIPS25/${R}_mini_L100_bdf competition_data/${R} --no-sign-request
done