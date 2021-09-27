import sys
sys.path.append("..")
from pathlib import Path
from tdc.single_pred import ADME, Tox
import click
from tdc_training import tdc_training

root = Path(__file__).resolve().parents[2].absolute()

@click.command()
@click.option('-batch_size', default=36)
@click.option('-gpu', default=0)
@click.option('-seed', default=0)
@click.option('-epochs', default=100)
def main(batch_size, gpu, seed, epochs):
    adme = ['Caco2_Wang', 'HIA_Hou', 'Pgp_Broccatelli', 'Bioavailability_Ma',
            'Lipophilicity_AstraZeneca', 'Solubility_AqSolDB',
            'BBB_Martins', 'PPBR_AZ', 'VDss_Lombardo', 'CYP2C19_Veith', 'CYP2D6_Veith', 'CYP3A4_Veith',
            'CYP1A2_Veith', 'CYP2C9_Veith', 'CYP2C9_Substrate_CarbonMangels', 'CYP2D6_Substrate_CarbonMangels',
            'CYP3A4_Substrate_CarbonMangels', 'Half_Life_Obach', 'Clearance_Hepatocyte_AZ']
    tox = ['LD50_Zhu', 'hERG', 'AMES', 'DILI', 'Skin Reaction', 'Carcinogens_Languin', 'ClinTox']
    tox_21 = ['nr-ar', 'nr-ar-lbd', 'nr-ahr', 'nr-aromatase', 'nr-er', 'nr-er-lbd', 'nr-ppar-gamma', 'sr-are',
              'sr-atad5', 'sr-hse', 'sr-mmp', 'sr-p53']

    for task in adme:
        data = ADME(name=task)
        tdc_training(task, data, seed, batch_size, epochs, gpu)
    for task in tox:
        data = Tox(name=task)
        tdc_training(task, data, seed, batch_size, epochs, gpu)
    for task in tox_21:
        data = Tox(name='Tox21', label_name=task)
        tdc_training(task, data, seed, batch_size, epochs, gpu)


if __name__ == '__main__':
    main()
