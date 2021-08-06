## Installation 

Code tested with Python 3.7 and cuda 11.0

1. Clone this repository ([git installation required](https://git-scm.com/))
   ```
   cd $HOME # or another directory for this project/repo
   git clone https://github.com/naivelogic/dotatron.git
   cd dotatron
   ```
   
2. Install environment with [Anaconda](https://www.continuum.io/downloads): 
   
   ```
   conda create -n dotatron -y python=3.7
   conda activate dotatron
   pip install -r requirements.txt
   pip install -U 'git+https://github.com/facebookresearch/fvcore.git' 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' 
   python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

   python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html

  conda install cudatoolkit=11.0

   
   ## if using Jupyter Notebooks create custom jupyter kernel for dotatron
   python -m ipykernel install --user --name=dotatron
   ```

3. Install the [DOTA Development kit](https://github.com/CAPTAIN-WHU/DOTA_devkit.git) that enables:
    - loades and visualizes the data
    - evaluate the results
    - splits and merges the data

    ```
    cd datasets/dota_devkit

    sudo apt-get install swig
    swig -c++ -python polyiou.i
    python setup.py build_ext --inplace
    ```

