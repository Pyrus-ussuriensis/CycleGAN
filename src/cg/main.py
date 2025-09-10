from cg.CycleGAN import CycleGAN
from lightning.pytorch.cli import LightningCLI
from cg.data.monet2photo import Monet2PhotoDM


def cli_main():
    cli = LightningCLI(model_class=CycleGAN, datamodule_class=Monet2PhotoDM, parser_kwargs={"default_config_files": ["configs/cg.yaml"]})


if __name__ == "__main__":
    cli_main()
        


