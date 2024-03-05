import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f

# Torch Model Creation
class AuxNet(nn.Module):
    
    def __init__(self, no_of_base_layers, no_of_aux_layers, no_of_end_layers, nodes_in_each_layer, 
                 b=0.99, s=0.2, input_shape=1, output_shape=2):
        super(AuxNet, self).__init__()

        self.beta = b
        self.lamda = s
        
        self.no_of_base_layers = no_of_base_layers
        self.no_of_aux_layers = no_of_aux_layers
        self.no_of_end_layers = no_of_end_layers
        self.nodes_in_each_layer = nodes_in_each_layer
        
        self.num_layers = no_of_base_layers + no_of_aux_layers + no_of_end_layers + 1
        self.layer_weights = torch.tensor([1.0 / self.num_layers for _ in range(self.num_layers)])
        # Layer weights = [...base layers... , ...aux layers... ,  ...end layers... ,  middle layer]
        
        # ================================= Base Layers =================================================
        base_layers = [nn.Sequential(nn.Linear(input_shape, nodes_in_each_layer), nn.ReLU(inplace=True))]
        base_out_layers = [nn.Sequential(nn.Linear(nodes_in_each_layer, output_shape), nn.Softmax(dim=1))]
        
        base_layers = base_layers + [nn.Sequential(nn.Linear(nodes_in_each_layer, nodes_in_each_layer),
                                                             nn.ReLU(inplace=True)) 
                                               for _ in range(1, no_of_base_layers)]
        base_out_layers = base_out_layers + [nn.Sequential(nn.Linear(nodes_in_each_layer, output_shape),
                                                                     nn.Softmax(dim=1)) 
                                                       for _ in range(1, no_of_base_layers)]
        
        self.base_layers = nn.ModuleList(base_layers)
        self.base_out_layers = nn.ModuleList(base_out_layers)
        
        # ================================= Middle Layers =================================================
        self.middle_layer = nn.Sequential(nn.Linear(nodes_in_each_layer, nodes_in_each_layer),
                                          nn.ReLU(inplace=True))
        self.middle_out_layer = nn.Sequential(nn.Linear(nodes_in_each_layer, output_shape),
                                              nn.Softmax(dim=1))

        # ================================= Aux Layers =================================================
        if no_of_aux_layers != 0:
            aux_layers = [nn.Sequential(nn.Linear(1, nodes_in_each_layer),
                                        nn.ReLU(inplace=True))
                          for _ in range(no_of_aux_layers)]
            
            aux_out_layers = [nn.Sequential(nn.Linear(nodes_in_each_layer, output_shape),
                                            nn.Softmax(dim=1))
                              for _ in range(no_of_aux_layers)]
            
            self.aux_layers = nn.ModuleList(aux_layers)
            self.aux_out_layers = nn.ModuleList(aux_out_layers)
        
        
        # ================================= End Layers =================================================
        end_layers = [nn.Sequential(nn.Linear((no_of_aux_layers+1)*nodes_in_each_layer, nodes_in_each_layer), nn.ReLU(inplace=True))]
        end_out_layers = [nn.Sequential(nn.Linear(nodes_in_each_layer, output_shape), nn.Softmax(dim=1))]
        
        end_layers = end_layers + [nn.Sequential(nn.Linear(nodes_in_each_layer, nodes_in_each_layer),
                                                 nn.ReLU(inplace=True)) 
                                   for _ in range(1, no_of_end_layers)]
        end_out_layers = end_out_layers + [nn.Sequential(nn.Linear(nodes_in_each_layer, output_shape),
                                                         nn.Softmax(dim=1)) 
                                           for _ in range(1, no_of_end_layers)]
        
        self.end_layers = nn.ModuleList(end_layers)
        self.end_out_layers = nn.ModuleList(end_out_layers)
    
    def forward(self, X, X_mask, base_feature):
        x = base_feature.view(1, -1)
        aux_input = X.view(-1)
        present_features = torch.where(X_mask)[0]
        out = []
        
        # ================================= Base Layers =================================================
        for i in range(self.no_of_base_layers):
            x = self.base_layers[i](x)
            out.append(self.base_out_layers[i](x))
        
        # ================================= Middle Layers =================================================
        middle = self.middle_layer(x)
        middle_out = self.middle_out_layer(middle)
        
        # ================================= Aux Layers =================================================
        
        aux_out = []
        for i in present_features:
            aux_out.append(self.aux_layers[i](aux_input[i].view(1, -1)))
            out.append(self.aux_out_layers[i](aux_out[-1]))
        # x = torch.concatenate(aux_out, dim=0)
        
        end_input = torch.zeros(((self.no_of_aux_layers+1), self.nodes_in_each_layer))
        
        if len(present_features):
            x = torch.concatenate(aux_out, dim=0)
            end_input[present_features] = x
        
        end_input[-1] = middle
        
        # taking layer weights of Aux layers and middle layer weights
        gamma = torch.concat([self.layer_weights[self.no_of_base_layers:self.no_of_base_layers+self.no_of_aux_layers],
                              self.layer_weights[-1].view(-1)])
        gamma_mask = torch.zeros_like(gamma, dtype=torch.float)
        gamma_mask[present_features] = 1.0 # Aux layers
        gamma_mask[-1] = 1.0 # Middle layer
        gamma = gamma * gamma_mask
        gamma = gamma / torch.sum(gamma)
        
        end_input = torch.multiply(end_input, gamma.view(-1, 1))
        x = end_input.view(1, -1)
        
        # ================================= Out Layers =================================================
        end_out = []
        for i in range(self.no_of_end_layers):
            end_out.append(self.end_layers[i](x))
            out.append(self.end_out_layers[i](end_out[-1]))
            x = end_out[-1]
            
        logits = torch.concatenate(out+[middle_out], dim=0)
        
        idx =   list(range(self.no_of_base_layers)) + \
                [self.no_of_base_layers + i.item() for i in present_features] + \
                list(range(self.no_of_base_layers+self.no_of_aux_layers, self.num_layers))
        
        weights = self.layer_weights[idx]
        weights = weights / torch.sum(weights)
        
        logits = torch.multiply(logits, weights.view(-1, 1))
        logit = torch.sum(logits, dim=0)
        
        self.layer_weights[idx] = weights
        
        return logit, logits
    
    
    def update_layer_weights(self, losses, mask):
        with torch.no_grad():
            present_features = np.where(mask)[0]
            j = 0            
            
            # Updating Base Layer Weights
            for i in range(self.no_of_base_layers):
                self.layer_weights[i] *= torch.pow(self.beta, losses[j])
                self.layer_weights[i] = max(self.layer_weights[i].item(), self.lamda / self.num_layers)
                j+=1
            
            # Updating Aux Layer Weights
            for i in present_features:
                self.layer_weights[self.no_of_base_layers+i] *= torch.pow(self.beta, losses[j])
                self.layer_weights[self.no_of_base_layers+i] = max(self.layer_weights[self.no_of_base_layers+i].item(), self.lamda / self.num_layers)
                j+=1
            
            # Upadting End and Middle Layer Weights
            for i in range(self.no_of_base_layers+self.no_of_aux_layers, self.num_layers):
                self.layer_weights[i] *= torch.pow(self.beta, losses[j])
                self.layer_weights[i] = max(self.layer_weights[i].item(), self.lamda / self.num_layers)
                j+=1
                
            self.layer_weights = self.layer_weights / torch.sum(self.layer_weights)