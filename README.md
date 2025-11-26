### Dataset generation for Antenna Design Optimization using Ansys
1. Download Ansys Electronics Desktop (AEDT) student version (above 2022) and set up the environment for scripting by opening HFSS in home page ribbon.
2. Setup the pyaedt console by HFSS->simulation->pyaedt console.
3. Use the provided scripts in the `automation` folder to generate datasets for different antenna
---------------------------------------------------------------------------------------------------------------------------------------------
1. For faster prototyping use MATLAB to generate datasets (based on MoM not FEM).

### Dataset overview
1. Generative.csv
2. Patch_data_insetfed.csv
3. Patch_data_coax.csv
4. Antenna_s11.csv - obtained from Mendely data repository https://data.mendeley.com/datasets/3gxr2vvd9n/2

### ML algorithms tested
#### Ensemble Methods - predict a target variable based on multiple input features
1. Adaboost
2. Random Forest
3. CatBoost - initially developed for gradient boosting on decision trees with categorical features support - skipped as dataset has no significant change in  categorical features, some changes could be introducing different feeds like apperture, using slot antennas,etc... 
---------------------------------------------------------------------------------------------------------------------------------------------
#### Neural Networks - model complex relationships between inputs and outputs and predict multiple target variables (desired)
4. Autoencoders
5. Variational autoencoders
6. Restricted Boltzmann Machines
---------------------------------------------------------------------------------------------------------------------------------------------------
The transition from 50 to 100 hidden nodes causes the RuntimeError due to a numerical instability issue amplified by the larger network.

    With 100 hidden nodes (W is 100×11), the computed activation v_dotW is larger in magnitude, potentially leading to floating-point overflow when fed into torch.sigmoid.

    If the sigmoid input is too large (positive or negative), the output prob_h can become extremely close to 1.0 or 0.0, but a tiny precision error (e.g., 1.0+ϵ) can make it numerically slightly outside the [0,1] range.

    The function torch.bernoulli() strictly requires its input to be in [0,1], hence the failure.
    Fix: Reduce the learning rate or the initial weight scale (currently * 0.01 in GaussianRBM.__init__) to manage the scale of activations or reduce it to 70.
---------------------------------------------------------------------------------------------------------------------------------------------------
7. Transformers

### Miscellaneous
1. Genetic Algorithms for patch antenna miniaturization - Antenna miniaturization techniques - optimizing material used but also comes with a caveat (manufacturing becomes difficult).
2. RAG chatbot for antenna design queries.
------------------------------------------------------------------------------------------------------------------------------------------------

### The requirements.txt file contains all the necessary libraries to run the code (generated using pip freeze hence many stray libraries may also get installed). Please install them using pip.

```bash
pip install -r requirements.txt
```