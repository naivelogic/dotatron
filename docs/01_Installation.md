## Installation 

1. Clone this repository ([git installation required](https://git-scm.com/))
   ```
   cd $HOME # or another directory for this project/repo
   git clone https://github.com/naivelogic/dotatron.git
   cd dotatron
   ```
2. Install environment with [Anaconda](https://www.continuum.io/downloads): 
   
   ```
   conda env create -f dotatron-env.yml
   conda activate dotatron
   
   ## if using Jupyter Notebooks create custom jupyter kernel for dotatron
   python -m ipykernel install --user --name=dotatron
   ```

3. Install the [DOTA Development kit](https://github.com/CAPTAIN-WHU/DOTA_devkit.git) that enables:
    - loades and visualizes the data
    - evaluate the results
    - splits and merges the data

    ```
    git submodule update --init --recursive

    sudo apt-get install swig
    cd DOTA_devkit
    swig -c++ -python polyiou.i
    python setup.py build_ext --inplac
    ```
