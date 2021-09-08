from pytorch_lightning.callbacks import EarlyStopping




early_stop_callback = EarlyStopping(monitor='val_ap_epoch',
                                    min_delta=0.00,
                                    mode='max',
                                    patience=10,
                                    verbose=False)
