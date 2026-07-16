# code come from https://github.com/mahmoodlab/PathomicFusion/blob/master/fusion.py
import torch
import torch.nn as nn
# from utils import init_max_weights
import math

def init_max_weights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()

class BilinearFusion(nn.Module):
    def __init__(self,  gate1=1, gate2=1, dim1=32, dim2=32, scale_dim1=1, scale_dim2=1, mmhid=1792, dropout_rate=0.25):
        super(BilinearFusion, self).__init__()
        # self.skip = bool(skip) 
        # self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2

        dim1_og, dim2_og, dim1, dim2 = dim1, dim2, dim1//scale_dim1, dim2//scale_dim2
        skip_dim = dim1+dim2+2 #if skip else 0

        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) #if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2) #if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(nn.Linear((dim1+1)*(dim2+1), mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.encoder2 = nn.Sequential(nn.Linear(mmhid+skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))    #skip_dim=66
        init_max_weights(self)

    def forward(self, vec1, vec2):
        ### Gated Multimodal Units
        # print("====================BilinearFusion starting!====================")
        if self.gate1:
            h1 = self.linear_h1(vec1)      #[8, 32]
            # print("h1 shape:",h1.shape)    
            z1 = self.linear_z1(vec1, vec2) #if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec2), dim=1))  #[8, 32]
            # print("z1 shape:",z1.shape)    
            o1 = self.linear_o1(nn.Sigmoid()(z1)*h1)  #[8, 32]
            # print("o1 shape:",o1.shape)            
        else:
            o1 = self.linear_o1(vec1)

        if self.gate2:
            h2 = self.linear_h2(vec2)      #[8, 32]
            # print("h2 shape:",h2.shape)
            z2 = self.linear_z2(vec1, vec2) #if self.use_bilinear else self.linear_z2(torch.cat((vec1, vec2), dim=1))  #[8, 32]
            # print("z2 shape:",z2.shape)
            o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)  #[8, 32]
            # print("o2 shape:",o2.shape)
        else:
            o2 = self.linear_o2(vec2)

        ### Fusion
        o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1).fill_(1)), 1)    #[8, 33]
        # print("o1 shape:",o1.shape)
        o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1).fill_(1)), 1)    #[8, 33]
        # print("o2 shape:",o2.shape)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1) # BATCH_SIZE X 1024   # [8,33,1]*[8,1,33]-->[8,33*33]-->[8,1089]
        # print("o12 shape:",o12.shape)
        out = self.post_fusion_dropout(o12)  #[8, 1089]
        # print("out shape:",out.shape)
        out = self.encoder1(out)          #[8, 64]
        # print("out shape:",out.shape)              
        out = torch.cat((out, o1, o2), 1)  #[8, 130]
        # print("out shape:",out.shape)
        out = self.encoder2(out)         #[8, 64]
        # print("out shape:",out.shape)
        return out


class TrilinearFusion_A(nn.Module):
    def __init__(self, skip=1, use_bilinear=1, gate1=1, gate2=1, gate3=1, dim1=32, dim2=32, dim3=32, scale_dim1=1, scale_dim2=1, scale_dim3=1, mmhid=96, dropout_rate=0.25):
        super(TrilinearFusion_A, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2
        self.gate3 = gate3

        dim1_og, dim2_og, dim3_og, dim1, dim2, dim3 = dim1, dim2, dim3, dim1//scale_dim1, dim2//scale_dim2, dim3//scale_dim3
        skip_dim = dim1+dim2+dim3+3 if skip else 0

        ### Path
        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim3_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim3_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        ### Graph
        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim2_og, dim3_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim2_og+dim3_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        ### Omic
        self.linear_h3 = nn.Sequential(nn.Linear(dim3_og, dim3), nn.ReLU())
        self.linear_z3 = nn.Bilinear(dim1_og, dim3_og, dim3) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim3_og, dim3))
        self.linear_o3 = nn.Sequential(nn.Linear(dim3, dim3), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=0.25)
        self.encoder1 = nn.Sequential(nn.Linear((dim1+1)*(dim2+1)*(dim3+1), mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))   #mmhid=96
        self.encoder2 = nn.Sequential(nn.Linear(mmhid+skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))
        
        init_max_weights(self)

    def forward(self, vec1, vec2, vec3):
        ### Gated Multimodal Units
        print("====================TrilinearFusion_A starting!====================")
        if self.gate1:
            h1 = self.linear_h1(vec1)
            print("h1 shape:",h1.shape)              
            z1 = self.linear_z1(vec1, vec3) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec3), dim=1)) # Gate Path with Omic
            print("z1 shape:",z1.shape)    
            o1 = self.linear_o1(nn.Sigmoid()(z1)*h1)
            print("o1 shape:",o1.shape)
        else:
            o1 = self.linear_o1(vec1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            print("h2 shape:",h2.shape)  
            z2 = self.linear_z2(vec2, vec3) if self.use_bilinear else self.linear_z2(torch.cat((vec2, vec3), dim=1)) # Gate Graph with Omic
            print("z2 shape:",z2.shape)    
            o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)
            print("o2 shape:",o2.shape)
        else:
            o2 = self.linear_o2(vec2)

        if self.gate3:
            h3 = self.linear_h3(vec3)
            print("h3 shape:",h3.shape)  
            z3 = self.linear_z3(vec1, vec3) if self.use_bilinear else self.linear_z3(torch.cat((vec1, vec3), dim=1)) # Gate Omic With Path
            print("z3 shape:",z3.shape)    
            o3 = self.linear_o3(nn.Sigmoid()(z3)*h3)
            print("o3 shape:",o3.shape)
        else:
            o3 = self.linear_o3(vec3)

        ### Fusion
        o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
        print("o1 shape:",o1.shape)
        o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
        print("o2 shape:",o2.shape)
        o3 = torch.cat((o3, torch.cuda.FloatTensor(o3.shape[0], 1).fill_(1)), 1)
        print("o3 shape:",o3.shape)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)
        print("o12 shape:",o12.shape)
        o123 = torch.bmm(o12.unsqueeze(2), o3.unsqueeze(1)).flatten(start_dim=1)
        print("o123 shape:",o123.shape)
        out = self.post_fusion_dropout(o123)
        print("out shape:",out.shape)        
        out = self.encoder1(out)
        print("out shape:",out.shape)
        if self.skip: out = torch.cat((out, o1, o2, o3), 1)  #[8,96+99]-->[8, 195]
        print("out shape:",out.shape)
        out = self.encoder2(out)
        print("out shape:",out.shape)
        return out


class TrilinearFusion_B(nn.Module):
    def __init__(self, skip=1, use_bilinear=1, gate1=1, gate2=1, gate3=1, dim1=32, dim2=32, dim3=32, scale_dim1=1, scale_dim2=1, scale_dim3=1, mmhid=96, dropout_rate=0.25):
        super(TrilinearFusion_B, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2
        self.gate3 = gate3

        dim1_og, dim2_og, dim3_og, dim1, dim2, dim3 = dim1, dim2, dim3, dim1//scale_dim1, dim2//scale_dim2, dim3//scale_dim3
        skip_dim = dim1+dim2+dim3+3 if skip else 0

        ### Path
        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim3_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim3_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        ### Graph
        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim2_og, dim1_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim2_og+dim1_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        ### Omic
        self.linear_h3 = nn.Sequential(nn.Linear(dim3_og, dim3), nn.ReLU())
        self.linear_z3 = nn.Bilinear(dim1_og, dim3_og, dim3) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim3_og, dim3))
        self.linear_o3 = nn.Sequential(nn.Linear(dim3, dim3), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=0.25)
        self.encoder1 = nn.Sequential(nn.Linear((dim1+1)*(dim2+1)*(dim3+1), mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.encoder2 = nn.Sequential(nn.Linear(mmhid+skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))
        
        init_max_weights(self)

    def forward(self, vec1, vec2, vec3):
        print("====================TrilinearFusion_B starting!====================")
        ### Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec3) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec3), dim=1)) # Gate Path with Omic
            o1 = self.linear_o1(nn.Sigmoid()(z1)*h1)
        else:
            o1 = self.linear_o1(vec1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec2, vec1) if self.use_bilinear else self.linear_z2(torch.cat((vec2, vec1), dim=1)) # Gate Graph with Path
            o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)
        else:
            o2 = self.linear_o2(vec2)

        if self.gate3:
            h3 = self.linear_h3(vec3)
            z3 = self.linear_z3(vec1, vec3) if self.use_bilinear else self.linear_z3(torch.cat((vec1, vec3), dim=1)) # Gate Omic With Path
            o3 = self.linear_o3(nn.Sigmoid()(z3)*h3)
        else:
            o3 = self.linear_o3(vec3)

        ### Fusion
        o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
        o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
        o3 = torch.cat((o3, torch.cuda.FloatTensor(o3.shape[0], 1).fill_(1)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)
        o123 = torch.bmm(o12.unsqueeze(2), o3.unsqueeze(1)).flatten(start_dim=1)
        out = self.post_fusion_dropout(o123)
        out = self.encoder1(out)
        if self.skip: out = torch.cat((out, o1, o2, o3), 1)
        out = self.encoder2(out)
        return out

def main():
    torch.manual_seed(42)

    batch_size = 8
    # dim1 = 1024
    # dim2 = 768
    # dim3 = 1280


    # dim1 = 32
    # dim2 = 32
    # dim3 = 32


    dim1 = 1024
    dim2 = 768
    # dim3 = 384


    vec1 = torch.randn(batch_size,dim1)  
    vec2 = torch.randn(batch_size,dim2) 
    # vec3 = torch.randn(batch_size, dim3)  

    bilinear_model = BilinearFusion(dim1=dim1, dim2=dim2)
    # trilinear_model_A = TrilinearFusion_A(dim1=dim1, dim2=dim2, dim3=dim3)
    # trilinear_model_B = TrilinearFusion_B(dim1=dim1, dim2=dim2, dim3=dim3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bilinear_model.to(device)
    # trilinear_model_A.to(device)
    # trilinear_model_B.to(device)

    vec1 = vec1.to(device)
    vec2 = vec2.to(device)
    # vec3 = vec3.to(device)

    with torch.no_grad():  
        bilinear_output = bilinear_model(vec1, vec2)
        print("BilinearFusion Output Shape:", bilinear_output.shape)

        # trilinear_output_A = trilinear_model_A(vec1, vec2, vec3)
        # print("TrilinearFusion_A Output Shape:", trilinear_output_A.shape)

        # trilinear_output_B = trilinear_model_B(vec1, vec2, vec3)
        # print("TrilinearFusion_B Output Shape:", trilinear_output_B.shape)

if __name__ == "__main__":
    main()