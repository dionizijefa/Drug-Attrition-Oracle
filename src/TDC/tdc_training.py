import sys
sys.path.append("..")
from pathlib import Path
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real
from skopt.utils import use_named_args
from standardiser import standardise
from torch import Tensor
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import DataLoader
from TDC_EGConv_lightning import Conf, EGConvNet
from utils.data_func import smiles2graph, create_loader

root = Path(__file__).resolve().parents[2].absolute()


def tdc_training(task, data, seed, batch_size, epochs, gpu):
    split = data.get_split()
    train = split['train']
    valid = split['valid']
    test = split['test']

    # if the class of interest is the majority class or not heavily disbalanced optimize AUC
    # else optimize average precision
    # if regression optimize ap
    if not ((train.iloc[0]['Y'] == 0) or (train.iloc[0]['Y'] == 1)):
        problem = 'regression'
    else:
        if train['Y'].value_counts(normalize=True)[1] > 0.3:
            problem = 'auc'
        else:
            problem = 'ap'

    standardized = []
    for drug in train['Drug']:
        try:
            standardized.append(standardise.run(drug))
        except:
            standardized.append(0)
    train['standardized_smiles'] = standardized
    train = train.loc[train['standardized_smiles'] != 0]

    standardized = []
    for drug in test['Drug']:
        try:
            standardized.append(standardise.run(drug))
        except:
            standardized.append(0)
    test['standardized_smiles'] = standardized
    test = test.loc[test['standardized_smiles'] != 0]

    standardized = []
    for drug in valid['Drug']:
        try:
            standardized.append(standardise.run(drug))
        except:
            standardized.append(0)
    valid['standardized_smiles'] = standardized
    valid = valid.loc[valid['standardized_smiles'] != 0]

    # balanced sampling
    if problem != 'regression':
        ones = train['Y'].value_counts()[1]
        zeros = train['Y'].value_counts()[0]
        class_sample_count = [zeros, ones]
        weights = 1 / Tensor(class_sample_count)
        samples_weights = weights[train['Y'].values]
        sampler = WeightedRandomSampler(samples_weights,
                                        num_samples=len(samples_weights),
                                        replacement=True)
        train_data_list = []
        for index, row in train.iterrows():
            train_data_list.append(smiles2graph(row, 'Y'))
        train_loader = DataLoader(train_data_list, num_workers=0, batch_size=batch_size,
                                  sampler=sampler, drop_last=True)

    else:
        train_data_list = []
        for index, row in train.iterrows():
            train_data_list.append(smiles2graph(row, 'Y'))
        train_loader = DataLoader(train_data_list, num_workers=0, batch_size=batch_size, drop_last=True)


    valid_loader = create_loader(valid, withdrawn_col='Y', batch_size=batch_size)
    test_loader = create_loader(valid, withdrawn_col='Y', batch_size=batch_size)

    dim_1 = Categorical([128, 256, 512, 1024, 2048], name='hidden_channels')
    dim_2 = Integer(1, 8, name='num_layers')
    dim_3 = Categorical([2, 4, 8, 16], name='num_heads')
    dim_4 = Integer(1, 8, name='num_bases')
    dim_5 = Real(1e-5, 1e-3, name='lr')
    dimensions = [dim_1, dim_2, dim_3, dim_4, dim_5]

    @use_named_args(dimensions=dimensions)
    def inverse_ap(hidden_channels, num_layers, num_heads, num_bases, lr):
        conf = Conf(
            batch_size=batch_size,
            reduce_lr=True,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            num_bases=num_bases,
            lr=lr,
            seed=seed,
        )
        model = EGConvNet(
            problem,
            conf.to_hparams(),
            reduce_lr=conf.reduce_lr,
        )

        early_stop_callback = EarlyStopping(monitor='result',
                                            min_delta=0.00,
                                            mode=('min' if problem == 'regression' else 'max'),
                                            patience=10,
                                            verbose=False)

        print("Starting training")
        trainer = Trainer(
            max_epochs=epochs,
            gpus=[gpu],  # [0]  # load from checkpoint instead of resume
            weights_summary='top',
            callbacks=[early_stop_callback],
            logger=False,
            deterministic=True,
            auto_lr_find=False,
            num_sanity_val_steps=0,
            checkpoint_callback=False,
        )

        trainer.fit(model, train_loader, valid_loader)
        results = trainer.test(model, test_loader)

        if problem == 'ap':
            test_results = round(results[0]['test_ap'], 3)
            return 1/test_results
        elif problem == 'auc':
            test_results = round(results[0]['test_auc'], 3)
            return 1/test_results
        else:
            test_result = round(results[0]['test_mse'], 3)
            return test_result

    res = gp_minimize(inverse_ap,  # minimize the inverse of average precision
                      dimensions=dimensions,  # hyperparams
                      acq_func="EI",  # the acquisition function
                      n_calls=20,  # the number of evaluations of f
                      n_random_starts=5,  # the number of random initialization points
                      random_state=seed)  # the random seed

    print('Value of the minimum: {}'.format(res.fun))
    print('Res space: {}'.format(res.x))
    print('\n')

    # train a final model and evaluate on the test set
    conf = Conf(
        batch_size=batch_size,
        reduce_lr=True,
        hidden_channels=res.x[0],
        num_layers=res.x[1],
        num_heads=res.x[2],
        num_bases=res.x[3],
        lr=res.x[4],
        seed=seed,
    )
    model = EGConvNet(
        problem,
        conf.to_hparams(),
        reduce_lr=conf.reduce_lr,
    )

    early_stop_callback = EarlyStopping(monitor='result',
                                        min_delta=0.00,
                                        mode=('min' if problem == 'regression' else 'max'),
                                        patience=10,
                                        verbose=False)

    print("Starting training")
    trainer = Trainer(
        max_epochs=epochs,
        gpus=[gpu],  # [0]  # load from checkpoint instead of resume
        weights_summary='top',
        callbacks=[early_stop_callback],
        logger=False,
        deterministic=True,
        auto_lr_find=False,
        num_sanity_val_steps=0,
        checkpoint_callback=False,
    )

    trainer.fit(model, train_loader, valid_loader)
    results = trainer.test(model, test_loader)

    if problem == 'ap':
        test_ap = round(results[0]['test_ap'], 3)
        test_auc = round(results[0]['test_auc'], 3)
    elif problem == 'auc':
        test_ap = round(results[0]['test_ap'], 3)
        test_auc = round(results[0]['test_auc'], 3)
    else:
        test_mse = round(results[0]['test_mse'], 3)

    results_path = Path(root / 'bayes_opt/bayes_opt_TDC.txt')
    if not results_path.exists():
        with open(results_path, "w") as file:
            file.write("Bayes opt TDC - EGConv")
            file.write("\n")

    with open(results_path, "a") as file:
        print('Task: {}'.format(task), file=file)
        print('Hidden: {}'.format(res.x[0]), file=file)
        print('Layers: {}'.format(res.x[1]), file=file)
        print('Heads: {}'.format(res.x[2]), file=file)
        print('Bases: {}'.format(res.x[3]), file=file)
        print('Learning rate: {}'.format(res.x[4], file=file))
        print('Res space: {}'.format(res.space), file=file)
        if problem == 'ap':
            print('AP on the test set: {}'.format(test_ap), file=file)
            print('AUC on the test set: {}'.format(test_auc), file=file)
        elif problem == 'auc':
            print('AP on the test set: {}'.format(test_ap), file=file)
            print('AUC on the test set: {}'.format(test_auc), file=file)
        else:
            print('MSE on the test set: {}'.format(test_mse), file=file)
        file.write("\n")
        file.write("\n")

    # Use test data for validation, train final model for production
    train = pd.concat([train, test])

    if problem != 'regression':
        ones = train['Y'].value_counts()[1]
        zeros = train['Y'].value_counts()[0]
        class_sample_count = [zeros, ones]
        weights = 1 / Tensor(class_sample_count)
        samples_weights = weights[train['Y'].values]
        sampler = WeightedRandomSampler(samples_weights,
                                        num_samples=len(samples_weights),
                                        replacement=True)
        train_data_list = []
        for index, row in train.iterrows():
            train_data_list.append(smiles2graph(row, 'Y'))
        train_loader = DataLoader(train_data_list, num_workers=0, batch_size=batch_size,
                                  sampler=sampler, drop_last=True)

    else:
        train_data_list = []
        for index, row in train.iterrows():
            train_data_list.append(smiles2graph(row, 'Y'))
        train_loader = DataLoader(train_data_list, num_workers=0, batch_size=batch_size, drop_last=True)

    conf = Conf(
        batch_size=batch_size,
        reduce_lr=True,
        hidden_channels=res.x[0],
        num_layers=res.x[1],
        num_heads=res.x[2],
        num_bases=res.x[3],
        lr=res.x[4],
        seed=seed,
    )

    conf.save_dir = '{}/production/'.format(root)

    logger = TensorBoardLogger(
        conf.save_dir,
        name='TDC',
        version='{}'.format(task),
    )

    model_checkpoint = ModelCheckpoint(
        dirpath=(logger.log_dir + '/checkpoint/'),
        monitor='result',
        mode=('min' if problem == 'regression' else 'max'),
        save_top_k=1,
    )

    model = EGConvNet(
        problem,
        conf.to_hparams(),
        reduce_lr=conf.reduce_lr,
    )

    early_stop_callback = EarlyStopping(monitor='result',
                                        min_delta=0.00,
                                        mode=('min' if problem == 'regression' else 'max'),
                                        patience=10,
                                        verbose=False)

    print("Starting training")
    trainer = Trainer(
        max_epochs=epochs,
        gpus=[gpu],  # [0]  # load from checkpoint instead of resume
        weights_summary='top',
        callbacks=[early_stop_callback, model_checkpoint],
        deterministic=True,
        auto_lr_find=False,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, train_loader, test_loader)
