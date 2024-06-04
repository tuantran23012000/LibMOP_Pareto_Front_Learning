import sys
import os
current_dir = os.getcwd()
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
import time
from tqdm import tqdm
import numpy as np
import torch
from utils import set_seed,visualize_3d,visualize_2d
import random
import torch.nn.functional as F
import torch
from models.MLP import MLP
from models.Transformer import Trans
from models.Transformer_MoEs import Trans_MoEs
import argparse
from problems.get_problems import Problem
from pymoo.indicators.hv import HV
import yaml
from losses.get_losses import LOG_func, Cheby_func, LS_func, SmoothCheby_func,EPO_func,PROD_func,UTILITY_func,COSINE_func
from pymoo.config import Config
Config.warnings['not_compiled'] = False
def sample_config(search_space_dict, reset_random_seed=False, seed=0):
    if reset_random_seed:
        random.seed(seed)
    
    config = dict()
    
    for key, value in search_space_dict.items():
        config[key] = random.choice(value)
        
    return config
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_epoch_disconnected(device, cfg, criterion, pb,pf,model_type):
    #model_type = "trans"
    set_seed(42)
    name = cfg['NAME']
    mode = cfg['MODE']
    ray_hidden_dim = cfg['TRAIN']['Ray_hidden_dim']
    out_dim = cfg['TRAIN']['Out_dim']
    n_tasks = cfg['TRAIN']['N_task']
    # num_hidden_layer = cfg['TRAIN']['Solver'][criterion]['Num_hidden_layer']
    # last_activation = cfg['TRAIN']['Solver'][criterion]['Last_activation']
    # ref_point = tuple(map(int, cfg['TRAIN']['Ref_point'].split(',')))
    lr = cfg['TRAIN']['OPTIMIZER']['Lr']
    wd = cfg['TRAIN']['OPTIMIZER']['WEIGHT_DECAY']
    type_opt = cfg['TRAIN']['OPTIMIZER']['TYPE']
    epochs = cfg['TRAIN']['Epoch']
    alpha_r = cfg['TRAIN']['Alpha']
    start = 0.
    if name == 'ZDT3':
        c_s = [[0.,0.81],[0.19,0.61],[0.41,0.41],[0.62,0.23],[0.825 ,0.1]] #ZDT3
        b_s = [[0.1,1],[0.25,0.8],[0.44,0.6],[0.64,0.4],[0.85,0.22]]
        idxs = [0,1,2,3,4]
        # expert_dim=[30,30,30,10,50]
        expert_dim=[30,30,30,10,50]
        # expert_dim=[10,10,10,10,10]
        lrs = [0.001,0.001,0.001,0.001,0.001]
    elif name == 'ZDT3_variant':
        c_s = [[0.8,0.62],[0.01,0.7]] #ZDT3_variant
        idxs = [0,1]
        expert_dim=[10,10]
        lrs = [0.001,0.001]
    elif name == 'DTLZ7':
        c_s = [[0.62,0.62,0.4],[0.01,0.62,0.5],[0.01,0.01,0.82],[0.62,0.01,0.6]]
        expert_dim=[10,10,10,10]
        lrs = [0.001,0.001,0.001,0.001]
        idxs = [0,1,2,3]

    hnet = Trans_MoEs(ray_hidden_dim, out_dim, expert_dim, len(expert_dim), n_tasks,2)
    hnet = hnet.to(device)
    param = count_parameters(hnet)
    print("Model size: ",param)
    print("Model type: ",model_type)
    sol = []
    if type_opt == 'adam':
        optimizer = torch.optim.Adam(hnet.parameters(), lr = lr, weight_decay=wd) 
    elif type_opt == 'adamw':
        optimizer = torch.optim.AdamW(hnet.parameters(), lr = lr, weight_decay=wd)
    start = time.time()
    for idx,c_ in enumerate(c_s):
        for epoch in tqdm(range(epochs)):
            #idx = random.choice(idxs)
            # c_ = c_s[idx]
            l_r = lrs[idx]
            optimizer.param_groups[0]['lr'] = l_r
            c = torch.tensor(c_)
            b = torch.tensor(b_s[idx])
            if n_tasks == 2:
                u1 = random.uniform(c[0], 1)
                u2 = random.uniform(c[1], 1)
                u = np.array([u1,u2])
                lda = (u/np.linalg.norm(u,1))
                ray = torch.from_numpy(
                        np.random.dirichlet((alpha_r, alpha_r), 1).astype(np.float32).flatten()
                        ).to(device)
            else:
                u1 = random.uniform(c[0], 1)
                u2 = random.uniform(c[1], 1)
                u3 = random.uniform(c[2], 1)
                u = np.array([u1,u2,u3])
                lda = (u/np.linalg.norm(u,1))
                ray = torch.from_numpy(
                        np.random.dirichlet((alpha_r, alpha_r,alpha_r), 1).astype(np.float32).flatten()
                        ).to(device)
            hnet.train()
            optimizer.zero_grad()
            if model_type == 'joint':
                output = hnet(ray,c)
            else:
                output = hnet(ray,idx)

            if cfg["NAME"] == "ex3":
                output = F.softmax(output,dim=2)
                output = torch.sqrt(output)
            else:
                output = F.sigmoid(output)
            if cfg["NAME"] == "ex2":
                output = 5*output
                    

            ray = ray.squeeze(0)
            
            obj_values = []
            objectives = pb.get_values(output)
            for i in range(len(objectives)):
                obj_values.append(objectives[i])
            losses = torch.stack(obj_values)
            loss = max(torch.abs((losses-c)/(b-c))*ray)
            loss.backward()
            optimizer.step()
            tmp = []
            for i in range(len(objectives)):
                tmp.append(objectives[i].cpu().detach().numpy().tolist())

            if epoch >= epochs-100: #5000: #9900:
                sol.append(tmp)

    end = time.time()
    time_training = end-start
    ind = HV(ref_point=np.ones(n_tasks))
    print(np.array(sol).shape,pf.shape)
    hv_loss_app = ind(np.array(sol))
    hv_loss = ind(pf)
    print("HV approximate: ",hv_loss_app)
    print("HV: ",hv_loss)
    print("SUb: ",hv_loss - hv_loss_app)
    torch.save(hnet.state_dict(),os.path.join(root_dir,"save_weights/best_weight_"+str(criterion)+"_"+str(mode)+"_"+str(name)+"_" + str(model_type)+".pt"))
    if n_tasks == 3:
        visualize_3d(sol,pf,cfg,criterion,pb,model_type)
    else:
        visualize_2d(sol,pf,cfg,criterion,pb,model_type)

def train_epoch(device, cfg, criterion, pb,pf,model_type):
    #model_type = "trans"
    set_seed(42)
    name = cfg['NAME']
    mode = cfg['MODE']
    ray_hidden_dim = cfg['TRAIN']['Ray_hidden_dim']
    out_dim = cfg['TRAIN']['Out_dim']
    n_tasks = cfg['TRAIN']['N_task']
    # num_hidden_layer = cfg['TRAIN']['Solver'][criterion]['Num_hidden_layer']
    # last_activation = cfg['TRAIN']['Solver'][criterion]['Last_activation']
    # ref_point = tuple(map(int, cfg['TRAIN']['Ref_point'].split(',')))
    lr = cfg['TRAIN']['OPTIMIZER']['Lr']
    wd = cfg['TRAIN']['OPTIMIZER']['WEIGHT_DECAY']
    type_opt = cfg['TRAIN']['OPTIMIZER']['TYPE']
    epochs = cfg['TRAIN']['Epoch']
    alpha_r = cfg['TRAIN']['Alpha']
    start = 0.
    if model_type == 'trans':
        hnet = Trans(ray_hidden_dim, out_dim, n_tasks,1)
    else:
        hnet = MLP(ray_hidden_dim, out_dim, n_tasks,6)
    hnet = hnet.to(device)
    param = count_parameters(hnet)
    print("Model size: ",param)
    print("Model type: ",model_type)
    sol,targets = [],[]
    if type_opt == 'adam':
        optimizer = torch.optim.Adam(hnet.parameters(), lr = lr, weight_decay=wd) 
    elif type_opt == 'adamw':
        optimizer = torch.optim.AdamW(hnet.parameters(), lr = lr, weight_decay=wd)
    start = time.time()

    for epoch in tqdm(range(epochs)):
        if n_tasks == 2:
            ray = torch.from_numpy(
                    np.random.dirichlet((alpha_r, alpha_r), 1).astype(np.float32).flatten()
                    ).to(device)
        else:
            ray = torch.from_numpy(
                    np.random.dirichlet((alpha_r, alpha_r,alpha_r), 1).astype(np.float32).flatten()
                    ).to(device)
        hnet.train()
        optimizer.zero_grad()
        output = hnet(ray)
        output = F.sigmoid(output)*5

        ray = ray.unsqueeze(0)
        
        obj_values = []
        objectives = pb.get_values(output)
        for i in range(len(objectives)):
            obj_values.append(objectives[i])
        losses = torch.stack(obj_values)
        if criterion == 'TCH':
            loss = Cheby_func(cfg).get_loss(losses,ray,None)
        elif criterion == 'STCH':
            loss = SmoothCheby_func(cfg).get_loss(losses,ray)
        elif criterion == 'EPO':
            loss = EPO_func(n_tasks,param).get_loss(losses,ray,list(hnet.parameters()))
        elif criterion == 'LS':
            loss = LS_func(cfg).get_loss(losses,ray)
        elif criterion == 'LOG':
            loss = LOG_func(cfg).get_loss(losses,ray)
        elif criterion == 'PROD':
            loss = PROD_func(cfg).get_loss(losses,ray)
        elif criterion == 'UTILITY':
            loss = UTILITY_func(cfg).get_loss(losses,ray,2.01)
        elif criterion == 'COSINE':
            loss = COSINE_func(cfg).get_loss(losses,ray)
        loss.backward()
        optimizer.step()
        tmp = []
        for i in range(len(objectives)):
            tmp.append(objectives[i].cpu().detach().numpy().tolist())

        # if epoch >= epochs-100: #5000: #9900:
        sol.append(tmp)

    end = time.time()
    time_training = end-start
    print("Time_training: ",time_training)
    ind = HV(ref_point=np.ones(n_tasks))
    print(np.array(sol).shape,pf.shape)
    def IGD(pf_truth, pf_approx):
        d_i = []
        for pf in pf_truth:
            d_i.append(np.min(np.sqrt(np.sum(np.square(pf-pf_approx),axis = 1))))
        igd = np.mean(np.array(d_i))
        return igd
    hv_loss_app = ind(np.array(sol))
    hv_loss = ind(pf)
    print("HV approximate: ",hv_loss_app)
    print("HV: ",hv_loss)
    print("SUb: ",hv_loss - hv_loss_app)
    # print("IGD: ",IGD(np.array(targets),np.array(sol)))
    # torch.save(hnet.state_dict(),os.path.join(root_dir,"save_weights/best_weight_"+str(criterion)+"_"+str(mode)+"_"+str(name)+"_" + str(model_type)+".pt"))
    # if n_tasks == 3:
    #     visualize_3d(sol,pf,cfg,criterion,pb,model_type)
    # else:
    #     visualize_2d(sol,pf,cfg,criterion,pb,model_type)

if __name__ == "__main__":
    device = torch.device(f"cuda:0" if torch.cuda.is_available() and not False else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver", type=str, choices=["LS", "KL","TCH","UTILITY","COSINE","Cauchy","PROD","LOG","AC","MC","EPO","STCH"],
        default="Cheby", help="solver"
    )
    parser.add_argument(
        "--problem", type=str, choices=["ZDT3","DTLZ7","ZDT3_variant"],
        default="CVX2", help="solver"
    )
    parser.add_argument(
        "--mode", type=str,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
        default="test"
    )
    parser.add_argument(
        "--model_type", type=str, choices=["mlp","trans"],                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
        default='trans'
    )
    args = parser.parse_args()
    criterion = args.solver 
    model_type = args.model_type
    print("model_type: ",model_type)
    print("Scalar funtion: ",criterion)
    problem = args.problem
    config_file = os.path.join(root_dir,"configs/"+problem+".yaml")
    with open(config_file) as stream:
        cfg = yaml.safe_load(stream)
    pb = Problem(problem, cfg['MODE'])
    pf = pb.get_pf()
    train_epoch(device,cfg,criterion,pb,pf,model_type)
    # train_epoch_framework(device,cfg,criterion,pb,pf,model_type)
    # predict_result(device,cfg,criterion,pb,pf,model_type)