"""El YAML es valido y el experimento corre de punta a punta (en pequeno)."""
import copy
import os
import tempfile

import pandas as pd
import pytest
from experimentos.parte1 import cargar_config, ejecutar_parte1, validar_config


def test_los_yaml_del_repo_son_validos():
    for ruta in ("configs/parte1.yaml", "configs/demo.yaml"):
        cfg = cargar_config(ruta)
        assert cfg["corridas"] >= 1 and cfg["presupuesto_evaluaciones"] > 0
        assert set(cfg["dimensiones"]) <= {2, 3} or all(isinstance(n, int) for n in cfg["dimensiones"])


def test_el_experimento_pedido_cumple_el_enunciado():
    cfg = cargar_config("configs/parte1.yaml")
    assert cfg["corridas"] >= 30
    assert sorted(cfg["dimensiones"]) == [2, 3]
    assert set(cfg["metodos"]) == {"gd", "ga", "pso", "de"}
    assert len(cfg["funciones"]) == 2


def test_validacion_detecta_errores_de_escritura():
    cfg = cargar_config("configs/demo.yaml")
    mala = copy.deepcopy(cfg)
    mala["metodos"] = ["gd", "pos"]
    with pytest.raises(ValueError):
        validar_config(mala)
    mala = copy.deepcopy(cfg)
    mala["funciones"] = ["rosenbrok"]
    with pytest.raises(ValueError):
        validar_config(mala)


def test_experimento_completo_en_pequeno():
    cfg = cargar_config("configs/demo.yaml")
    cfg["corridas"] = 3
    cfg["presupuesto_evaluaciones"] = 600
    with tempfile.TemporaryDirectory() as carpeta:
        cfg["carpeta_resultados"] = carpeta
        resumen = ejecutar_parte1(cfg)
        assert len(resumen) == 2 * 2 * len(cfg["metodos"])
        assert (resumen["total_eq_media"] <= 600).all()
        assert os.path.exists(os.path.join(carpeta, "resumen.csv"))
        corridas = pd.read_csv(os.path.join(carpeta, "corridas.csv"))
        assert len(corridas) == 2 * 2 * len(cfg["metodos"]) * 3
