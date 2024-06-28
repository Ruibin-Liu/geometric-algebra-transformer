#!/usr/bin/env python3
# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
# Adapted by Ruibin Liu, ruibinliuphd@gmail.com

import hydra

from gatr.experiments.res_prop import ResPropExperiment


@hydra.main(config_path="../config", config_name="res_prop", version_base=None)
def main(cfg):
    """Entry point for res_prop experiment."""
    exp = ResPropExperiment(cfg)
    exp()


if __name__ == "__main__":
    main()
