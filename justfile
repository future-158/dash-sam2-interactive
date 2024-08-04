install:   
    #!/bin/bash
    set -exo pipefail
    source /opt/conda/etc/profile.d/conda.sh
    conda env list | grep  $PWD/venv || conda create -y --prefix $PWD/venv python=3.11 pip ipykernel
    conda activate $PWD/venv

    pip install -U -r requirements.txt
    test -d segment-anything-2 || git clone https://github.com/facebookresearch/segment-anything-2.git
    cd segment-anything-2
    pip install -e ".[demo]"

    mkdir -p checkpoints
    test -f checkpoints/sam2_hiera_tiny.pt || wget --directory-prefix checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt 

    

download:
    #!/bin/bash
    set -exo pipefail


prepare:
    #!/bin/bash
    set -exo pipefail        