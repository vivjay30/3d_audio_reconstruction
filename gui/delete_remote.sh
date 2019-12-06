#!/bin/bash
rmCommand="rm /projects/grail/vjayaram/d3audiorecon/gui/output/*"

# delete file before running this command
rm output.json
rm unet_cqt_output.wav
rm output/*
ssh vjayaram@lungo.cs.washington.edu $rmCommand

