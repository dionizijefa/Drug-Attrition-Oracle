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
from src.TDC.TDC_EGConv_lightning import Conf, EGConvNet
from src.utils.data_func import smiles2graph, create_loader

root = Path(__file__).resolve().parents[2].absolute()


def tdc_training(task, data, seed, batch_size, epochs, gpu):
    split = data.get_split()
    train = split['train']
    valid = split['valid']
    test = split['test']

    # if the class of interest is the majority class or not heavily disbalanced optimize AUC
    # else optimize average precision
    # if regression optimize ap
    if (train.iloc[0]['Y'] == 0) or (train.iloc[0]['Y'] == 1):
        problem = 'regression'
    else:
        if train['Y'].value_counts(normalize=True)[1] > 0.3:
            problem = 'auc'
        else:
            problem = 'ap'
    datasets = [train, valid, test]
    for split in datasets:
        standardized = []
        for drug in split['Drug']:
            try:
                standardized.append(standardise.run(drug))
            except:
                standardized.append(0)
        split['standardized_smiles'] = standardized
        split = split.loc[split['standardized_smiles'] != 0]

    # balanced sampling
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
                              sampler=sampler)

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

        early_stop_callback = EarlyStopping(monitor='results_epoch',
                                            min_delta=0.00,
                                            mode='max',
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
                      n_calls=5,  # the number of evaluations of f
                      n_random_starts=2,  # the number of random initialization points
                      random_state=seed)  # the random seed

    print('Value of the minimum: {}'.format(res.fun))
    print('Res space: {}'.format(res.x))
    print('\n')

    results_path = Path(root / 'bayes_opt')

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

    early_stop_callback = EarlyStopping(monitor='results_epoch',
                                        min_delta=0.00,
                                        mode='max',
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

    if problem == 'ap' or 'auc':
        test_ap = round(results[0]['test_ap'], 3)
        test_auc = round(results[0]['test_auc'], 3)
    else:
        test_mse = round(results[0]['test_mse'], 3)

    if not results_path.exists():
        results_path.mkdir(exist_ok=True, parents=True)
        with open(results_path / "bayes_opt_TDC.txt", "w") as file:
            file.write("Bayes opt TDC - EGConv")
            file.write("\n")

    with open(results_path / "bayes_opt.txt", "a") as file:
        print('Task: {}'.format(task), file=file)
        print('Hidden: {}'.format(res.x[0]), file=file)
        print('Layers: {}'.format(res.x[1]), file=file)
        print('Heads: {}'.format(res.x[2]), file=file)
        print('Bases: {}'.format(res.x[3]), file=file)
        print('Learning rate: {}'.format(res.x[4], file=file))
        print('Res space: {}'.format(res.space), file=file)
        if problem == 'ap' or 'auc':
            print('AP on the test set: {}'.format(test_ap), file=file)
            print('AUC on the test set: {}'.format(test_auc), file=file)
        else:
            print('MSE on the test set: {}'.format(test_mse), file=file)
        file.write("\n")
        file.write("\n")

    # Use test data for validation, train final model for production
    train = pd.concat([train, test])

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
                              sampler=sampler)

    conf = Conf(
        batch_size=batch_size,
        reduce_lr=True,
        hidden_channels=res.x[0],
        num_layers=res.x[1],
        num_heads=res.x[2],
        num_bases=res.x[3],
        lr=res.x[4],
        seed=seed,
        save_dir='{}/TDC_production/'.format(root)
    )

    logger = TensorBoardLogger(
        conf.save_dir,
        name='TDC',
        version='{}'.format(task),
    )

    model_checkpoint = ModelCheckpoint(
        dirpath=(logger.log_dir + '/checkpoint/'),
        monitor='result_epoch',
        mode='max',
        save_top_k=1,
    )

    model = EGConvNet(
        problem,
        conf.to_hparams(),
        reduce_lr=conf.reduce_lr,
    )

    early_stop_callback = EarlyStopping(monitor='results_epoch',
                                        min_delta=0.00,
                                        mode='max',
                                        patience=10,
                                        verbose=False)

    print("Starting training")
    trainer = Trainer(
        max_epochs=epochs,
        gpus=[gpu],  # [0]  # load from checkpoint instead of resume
        weights_summary='top',
        callbacks=[early_stop_callback, model_checkpoint],
        logger=TensorBoardLogger,
        deterministic=True,
        auto_lr_find=False,
        num_sanity_val_steps=0,
        checkpoint_callback=False,
    )

    trainer.fit(model, train_loader, test_loader)
