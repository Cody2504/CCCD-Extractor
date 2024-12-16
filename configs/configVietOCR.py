dataset_params = {
    'name':'CCCDOCR',
    'data_root':'./datasets/ocr_data',
    'train_annotation':'train_annotation.txt',
    'valid_annotation':'val_annotation.txt',
    'image_height':64
}

params = {
    'print_every':200,
    'valid_every':15*200,
    'iters':20000,
    'checkpoint':'./checkpoint/transformerocr_checkpoint.pth',    
    'export':'./weights/transformerocr.pth',
    'metrics': 10000,
    'batch_size': 8,
    
}