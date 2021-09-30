# Drug-Attrition-Oracle
Models for predicting wether a drug will be withdrawn from the market. Part of the AI4EU-MODRAI challenge. 

*This project has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement 825619.*

## Project structure
    .
    ├── bayes_opt                                           # Configuration and results of optimization for different models
    ├── complementary_model_results                         # Results of complementary models on the test set
    ├── cross_conformal                                     # Results of cross-conformal method for the primary DAO model
    ├── cross_conformal_descriptors                         # Configuration and results of optimization for different models
    ├── data                                                # Data used for training the models
    |   ├── processing_pipeline                             # Outputs from the semi-automated processing of the raw files
    |   |   ├── TDC_predictions                             # Outputs from the TDC subtask models
    |   |   ├── descriptors                                 # Molecular descriptors for the MasterDB
    |   |   ├── test                                        # Molecules for the test set
    |   |   ├── train                                       # Molecules for the training set
    |   |   ├── MasterDB_15Sep2021.csv                      # Contains all molecules and withdrawn labels processed
    |   |   ├── MasterDB_15Sep2021_standardized.csv         # MasterDB with standardized SMILES
    |   |   ├── chembl_atc_codes.csv                        # ATC codes for molecules in MasterDB from Chembl
    |   |   └── drug_indications.csv                        # Drug indications for molecules in MasterDB from Chembl
    |   └── raw                                             # Raw files used in modelling
    ├── production                                          # Folder for models used in production
    |   ├── egconv_production                               # Files necessary for deployment of the graph only model   
    |   ├── complementary_model                             # Files necessary for deployment of the complementary model
    |   └── descriptors_production                          # Files necessary for deployment of the model based on molecular descriptors
    ├── src                                                 # Code and scripts for modelling
    |   ├── TDC                                             # Training and inference on Therapeutics Data Commons  
    |   |   ├── TDC_EGConv_lightning.py                     # Pytorch lightning class for training of the models 
    |   |   ├── tdc_inference.py                            # Produce predictions (variables) for the complementary models
    |   |   ├── tdc_tasks_training.py                       # Run training across all subtasks
    |   |   ├── tdc_training.py                             # Training and optimization for each subtask
    |   |   ├── train_complementary.py                      # Training and inference for the complementary model 
    |   ├── descriptors                                     # Training and inference on models with molecular descriptors  
    |   |   ├── EGConv_escriptors.py                        # Architecture of the models
    |   |   ├── bayes_opt_descriptors.py                    # Bayes optimization of the descriptors model
    |   |   ├── descriptors_lightning.py                    # Pytorch lightning class for training of the models
    |   |   ├── train_gcn_desc                              # Run training and production for the descriptors models
    |   ├── smiles_only                                     # Files necessary for deployment of the graph only model 
    |   |   ├── EGConv.py                                   # Architecture of the graph only model
    |   |   ├── bayes_opt.py                                # Bayes optimization of the model
    |   |   ├── EGConv_lightning.py                         # Pytorch lightning class for training of the models
    |   |   ├── train_gcn.py                                # Run training and production for the models
    |   ├── utils                                           # Metrics, dataloaders and other files
    |   ├── dao.py                                          # Full DAO functionality
    |   ├── dao_descriptors.py                              # Full DAO functionality for the descriptors model
    |   ├── generate_descriptors.py                         # Generate RDKit descriptors
    |   ├── indications_atc.py                              # Download drug indications from chembl
    |   ├── preprocess_withdrawn.py                         # Download CHEMBL data on withdrawn drugs
    |   ├── standardize_dataset.py                          # Standardize smiles for a dataset
    |   ├── train_test_split.py                             # Make train-test split for a dataset
    └── Drug Attrition Oracle.ipynb                         # Notebook with examples of the Drug Attrition Oracle
    
## Installation
Create a conda environment from file
```
conda env export > environment_droplet.yml
```

## Using Drug Attrition Oracle
DAO can be used directly from the Drug Attrition Oracle notebook.

## Processing custom datasets
DAO offers functionality to standardise smiles strings of an input dataset. Navigate to the src folder and run
```
python standardize_dataset.py
```
You can also create train-test split based on molecualar scaffolds
```
python train_test_split.py
```

## Training DAO models
To train the simple GNN model for predicting withdrawn drugs with the conformal method. Navigate to the src/smiles_only folder and run
```
python train_gcn.py -production True
```
If you want to optimize the hyperparameters of the model you can run
```
python bayes_opt.py
```

## Training descriptors models
Navigate to the src/descriptors folder and run
```
python train_gcn_desc.py -production True
```

## Training complementary models
Training complementary models takes a long time, since 38 models are downloaded, optimized and trained. To run, navigate to src/TDC folder and run
```
python tdc_tasks_training.py
```
After the training is done you can use this to make predictions for your original datasets
```
python tdc_inference.py
```
And from these predictions, train a complementary model
```
python train_complementary.py
```


