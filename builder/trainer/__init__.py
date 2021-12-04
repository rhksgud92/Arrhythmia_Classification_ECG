from .trainer import *

def get_trainer(args, iteration, x, y, model, logger, device, scheduler, optimizer, criterion, pat_info=None, flow_type="train"):
    if args.trainer == "binary_classification": 
        model, iter_loss = binary_classification(args, iteration, x, y, model, logger, device, scheduler, optimizer, criterion, pat_info, flow_type)
    else:
        print("Selected trainer is not prepared yet...")
        exit(1)

    return model, iter_loss