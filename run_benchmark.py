import shutil
import torch


from dataset import *
from utils import *
from settings_benchmark import *

from dataset import writer
from torch.utils.tensorboard import SummaryWriter

all_dataset = prepareDatasets()
print(f"Models: {[name for name in models]}")
print(f"Datasets: {[name for name in all_dataset]}")

# 自检：尝试加载每个模型一次，以确保每个模型都能加载
print("Trying to load each model...")
for name_model in models:
    model:nn.Module = models[name_model]()
    


root_result = "result"
if not os.path.exists(root_result):
    os.mkdir(root_result)

id_card = 0
# 手动选择显卡
count_card = torch.cuda.device_count()
if count_card > 1:
    while True:
        s = input(f"Please choose a video card number (0-{count_card-1}): ")
        if s.isdigit():
            id_card = int(s)
            if id_card >= 0 and id_card < count_card:
                break
        print("Invalid input!")
        continue
device_cuda = torch.device(f'cuda:{id_card}' if torch.cuda.is_available() else 'cpu')
print(f"\n\nVideo Card {id_card} will be used.")


        
for name_model in models:
    root_result_model = os.path.join(root_result, name_model)
    if not os.path.exists(root_result_model):
        os.mkdir(root_result_model)
    # foo = models[name_model]()
    # total = sum([param.nelement() for param in foo.parameters()])
    # print("Model:{}, Number of parameter: {:.3f}M".format(name_model, total/1e6))
    # continue
    # 在各个训练集上训练
    for name_dataset in all_dataset:
        dataset = all_dataset[name_dataset]
        
        trainLoader = DataLoader(dataset=dataset['train'],batch_size=2, shuffle=True, drop_last=False, num_workers=0)
        valLoader = DataLoader(dataset=dataset['val'])
        testLoader = DataLoader(dataset=dataset['test'])
        model:nn.Module = models[name_model]().to(device_cuda)
        
        
            
        root_result_model_dataset = os.path.join(root_result_model, name_dataset)
        path_flag = os.path.join(root_result_model_dataset, f"finished.flag")
        if os.path.exists(path_flag):
            continue
        if os.path.exists(root_result_model_dataset):
            shutil.rmtree(root_result_model_dataset)
        os.mkdir(root_result_model_dataset)
        
        
        print(f"\n\n\nCurrent Model:{name_model}, Current training dataset: {name_dataset}")
        

        log_section = f"{name_model}_{name_dataset}"
        


        funcLoss = DiceLoss() if 'loss' not in dataset else dataset['loss']
        thresh_value = None if 'thresh' not in dataset else dataset['thresh']
        # optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad ], lr=1e-3, weight_decay=1e-4)
        optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad ],
                                    lr=1e-4, weight_decay=0.001)
        NUM_MAX_EPOCH = 300
        bestResult = {"epoch":-1, "dice":-1}
        ls_best_result = []
        for epoch in range(NUM_MAX_EPOCH):
            torch.cuda.empty_cache()


            log_section_parent = f"{log_section}"
            result_train = traverseDataset(model=model, loader=trainLoader, 
                        thresh_value=thresh_value, 
                        log_section=f"{log_section_parent}_{epoch}_train",
                        log_writer=writer if epoch%5==0 else None,
                        description=f"Train Epoch {epoch}", device=device_cuda,
                        funcLoss=funcLoss, optimizer=optimizer)
            
            for key in result_train:
                writer.add_scalar(tag=f"{log_section}/{key}_train", 
                                scalar_value=result_train[key],  
                                global_step=epoch  
                                )

            # val
            result = traverseDataset(model=model, loader=valLoader, 
                        thresh_value=thresh_value, 
                        log_section=f"{log_section_parent}_{epoch}_val",
                        log_writer=writer if epoch%5==0 else None,
                        description=f"Val Epoch {epoch}", device=device_cuda,
                        funcLoss=funcLoss, optimizer=None)
            for key in result:
                writer.add_scalar(tag=f"{log_section}/{key}_val", 
                                scalar_value=result[key],  
                                global_step=epoch  
                                )
            


            dice = result['dice']
            print(f"val dice:{dice}. ({name_model} on {name_dataset})")
            if dice > bestResult['dice']:
                bestResult['dice'] = dice
                bestResult['epoch'] = epoch
                ls_best_result.append("epoch={}, val_dice={:.3f}".format(epoch, dice))
                print("best dice found. evaluating on testset...")

                result = traverseDataset(model=model, loader=testLoader, 
                        thresh_value=thresh_value, 
                        log_section=None,
                        log_writer=None,
                        description=f"Test Epoch {epoch}", device=device_cuda,
                        funcLoss=funcLoss, optimizer=None)
                ls_best_result.append(result)
                
                path_json = os.path.join(root_result_model_dataset, "best_result.json")
                with open(path_json, "w") as f:
                    json.dump(ls_best_result,f, indent=2)
                path_model = os.path.join(root_result_model_dataset, 'model_best.pth')
                torch.save(model.state_dict(), path_model)
            else:
                threshold = 100
                if epoch - bestResult['epoch'] >= threshold:
                    print(f"Precision didn't improve in recent {threshold} epoches, stop training.")
                    break

        with open(path_flag, "w") as f:
            f.write("training and testing finished.")
            
