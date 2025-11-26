### Dataset generation for Antenna Design Optimization using Ansys
1. Download Ansys Electronics Desktop (AEDT) student version (above 2022) and set up the environment for scripting.
2. Setup the pyaedt console by HFSS->simulation->pyaedt console.
3. Use the provided scripts in the `automation` folder to generate datasets for different antenna
---------------------------------------------------------------------------------------------------------------------------------------------
1. For faster prototyping use MATLAB to generate datasets (based on MoM not FEM).

### Dataset overview
1. Generative.csv
2. Patch_data_insetfed.csv
3. Patch_data_coax.csv

### ML algorithms tested
#### Ensemble Methods - predict a target variable based on multiple input features
1. Adaboost
2. Random Forest
3. CatBoost
---------------------------------------------------------------------------------------------------------------------------------------------
#### Neural Networks - model complex relationships between inputs and outputs and predict multiple target variables (desired)
4. Autoencoders
5. Variational autoencoders
6. Restricted Boltzmann Machines
7. Transformers

### Miscellaneous
1. Genetic Algorithms for patch antenna miniaturization - Antenna miniaturization techniques - optimizing material used but also comes with a caveat (manufacturing becomes difficult).
2. RAG chatbot for antenna design queries.

