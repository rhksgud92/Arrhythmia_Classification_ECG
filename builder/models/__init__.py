import importlib


def get_model(args):

    model_module = importlib.import_module("builder.models." + args.model)
    model = getattr(model_module, args.model.upper())

    return model
