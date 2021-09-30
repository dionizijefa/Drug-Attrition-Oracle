# Drug-Attrition-Oracle
Models for predicting wether a drug will be withdrawn from the market. Part of the AI4EU-MODRAI challenge. 

*This project has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement 825619.*

## Project structure
    .
    ├── bayes_opt                       # Configuration and results of optimization for different models
    ├── complementary_model_results     # Results of complementary models on the test set
    ├── cross_conformal                 # Results of cross-conformal method for the primary DAO model
    ├── cross_conformal_descriptors     # Configuration and results of optimization for different models
    ├── data                            # Data used for training the models
    │   ├── processing_pipeline         # Outputs from the semi-automated processing of the raw files
    │   │   ├── TDC_predictions         # Outputs from the TDC subtask models
    │   │   ├── descriptors             # Molecular descriptors for the MasterDB
    │   │   ├── test                    # Molecules for the test set
    │   │   └── train                   # Molecules for the training set
    │   └── raw                         # Raw files used in modelling
    └── ...
