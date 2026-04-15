from importlib import import_module

MODEL_REGISTRY = {
    "no_prolif_no_shed": "Model_structures.Model_class_no_prolif.SmolModel",
    "no_prolif_cst_shed": "Model_structures.Model_class_no_Prolif_cst_Shed.SmolModel",
    "no_prolif_SA_shed": "Model_structures.Model_class_no_Prolif_SA_Shed.SmolModel",
    "no_prolif_Cell_num_shed": "Model_structures.Model_class_no_Prolif_Cell_num_Shed.SmolModel",
    "cst_prolif_no_shed": "Model_structures.Model_class_cst_prolif.SmolModel",
    "cst_prolif_cst_shed": "Model_structures.Model_class_cst_Prolif_cst_Shed.SmolModel",
    "cst_prolif_SA_shed": "Model_structures.Model_class_cst_Prolif_SA_Shed.SmolModel",
    "cst_prolif_Cell_num_shed": "Model_structures.Model_class_cst_Prolif_Cell_num_Shed.SmolModel",
    "SA_prolif_no_shed": "Model_structures.Model_class_SA_prolif.SmolModel",
    "SA_prolif_cst_shed": "Model_structures.Model_class_SA_Prolif_cst_Shed.SmolModel",
    "SA_prolif_SA_shed": "Model_structures.Model_class_SA_Prolif_SA_Shed.SmolModel",
    "SA_prolif_Cell_num_shed": "Model_structures.Model_class_SA_Prolif_Cell_num_Shed.SmolModel",
    "Cell_num_prolif_no_shed": "Model_structures.Model_class_Cell_num_prolif.SmolModel",
    "Cell_num_prolif_cst_shed": "Model_structures.Model_class_Cell_num_Prolif_cst_Shed.SmolModel",
    "Cell_num_prolif_SA_shed": "Model_structures.Model_class_Cell_num_Prolif_SA_Shed.SmolModel",
    "Cell_num_prolif_Cell_num_shed": "Model_structures.Model_class_Cell_num_Prolif_Cell_num_Shed.SmolModel",
    # ... add extra models here
}

def load_model(model_key: str):
    if model_key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model key: {model_key}")

    module_path, class_name = MODEL_REGISTRY[model_key].rsplit(".", 1)
    mod = import_module(module_path)
    cls = getattr(mod, class_name)
    return cls