#!/bin/bash
# forwardcmd="source ~/.zshrc; CUDA_VISIBLE_DEVICES=1 python /projects/grail/vjayaram/d3audiorecon/network/inference.py /projects/grail/vjayaram/d3audiorecon/data/checkpoints/resnet_direction_Dec3/resnet_4.pt 0 /projects/grail/vjayaram/d3audiorecon/gui/output --output-dir=/projects/grail/vjayaram/d3audiorecon/gui/output"
# forwardcmd1="source ~/.zshrc; CUDA_VISIBLE_DEVICES=0 python /projects/grail/vjayaram/d3audiorecon/network/inference.py /projects/grail/vjayaram/d3audiorecon/data/checkpoints/unet_cqt_Dec3/unet_3.pt 1 /projects/grail/vjayaram/d3audiorecon/gui/output --output-dir=/projects/grail/vjayaram/d3audiorecon/gui/output"
# ssh vjayaram@lungo.cs.washington.edu $forwardcmd &
# ssh vjayaram@lungo.cs.washington.edu $forwardcmd1;
# scp vjayaram@lungo.cs.washington.edu:/projects/grail/vjayaram/d3audiorecon/gui/output/output.json . &
# scp vjayaram@lungo.cs.washington.edu:/projects/grail/vjayaram/d3audiorecon/gui/output/unet_cqt_output.wav .

forwardcmd="source ~/.zshrc; CUDA_VISIBLE_DEVICES=1 python /projects/grail/vjayaram/d3audiorecon/network/inference.py /projects/grail/vjayaram/d3audiorecon/data/checkpoints/unet_direction_Dec3/unet_5.pt 2 /projects/grail/vjayaram/d3audiorecon/gui/output --output-dir=/projects/grail/vjayaram/d3audiorecon/gui/output"
ssh vjayaram@lungo.cs.washington.edu $forwardcmd1;