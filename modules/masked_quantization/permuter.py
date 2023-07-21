import torch
import torch.nn as nn 

# return value_indices and position_indices into raster-scan
class raster_scan_permuter(nn.Module):    
    def forward(self, indices, position_indices, **ignore_kwargs):
        sorted_position_indices, sorted_order = torch.sort(position_indices, descending=False, dim=-1)
        sorted_indices = indices.gather(1, sorted_order)
        return sorted_indices, sorted_position_indices

class identity_permuter(nn.Module):
    def forward(self, indices, position_indices, **ignore_kwargs):
        return indices, position_indices