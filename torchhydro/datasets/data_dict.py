"""
Author: Wenyu Ouyang
Date: 2021-12-31 11:08:29
LastEditTime: 2025-04-18 08:55:29
LastEditors: Wenyu Ouyang
Description: A dict used for data source and data loader
FilePath: /torchhydro/torchhydro/datasets/data_dict.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

from torchhydro.datasets.data_sets import (
    BaseDataset,
    ForecastDataset,
    HFDataset,
    BasinSingleFlowDataset,
    DplDataset,
    FlexibleDataset,
    HydroMeanDataset,
    # HydroMultiSourceDataset,
    PrecipitationFusionDataset,
    MopexPrecipitationGagesAttrFusionDataset,
    ObsForeDataset,
    Seq2SeqDataset,
    SeqForecastDataset,
    TransformerDataset,
    ReservoirsDataset,
    ReservoirDataset,
    ReservoirREGUDataset,
)

datasets_dict = {
    "StreamflowDataset": BaseDataset,
    "ForecastDataset": ForecastDataset,
    "HFDataset": HFDataset,
    "SingleflowDataset": BasinSingleFlowDataset,
    "DplDataset": DplDataset,
    "FlexDataset": FlexibleDataset,
    # "MultiSourceDataset": HydroMultiSourceDataset,
    "PrecipitationFusionDataset": PrecipitationFusionDataset,
    "MopexPrecipitationGagesAttrFusionDataset": MopexPrecipitationGagesAttrFusionDataset,
    "Seq2SeqDataset": Seq2SeqDataset,
    "TransformerDataset": TransformerDataset,
    "ReservoirsDataset": ReservoirsDataset,
    "ReservoirDataset": ReservoirDataset,
    "ReservoirREGUDataset": ReservoirREGUDataset,
    "SeqForecastDataset": SeqForecastDataset,
    "TransformerDataset": TransformerDataset,
    "ObsForeDataset": ObsForeDataset,
}
