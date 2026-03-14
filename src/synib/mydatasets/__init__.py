from importlib import import_module


def _export_public(module_name):
    try:
        module = import_module(module_name, package=__name__)
    except Exception:
        return

    public = getattr(module, "__all__", None)
    if public is None:
        public = [name for name in vars(module) if not name.startswith("_")]
    globals().update({name: getattr(module, name) for name in public})


_export_public(".Irony_Cremad")
_export_public(".Factor_CL_Datasets")
_export_public(".ScienceQA.ScienceQA")
_export_public(".ScienceQA.ScienceQA_CB")
_export_public(".ScienceQA.ScienceQA_CB_mem")
_export_public(".ESNLI.ESNLIDataset")
_export_public(".ESNLI.ESNLI_CB")
_export_public(".MUStARD.MUStARD_Dataset")
