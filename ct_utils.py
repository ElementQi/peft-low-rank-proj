import torch.nn as nn
import torch
import torch.nn.functional as F





def apply_to_innermost_layers(module, operation, prefix=''):
    """
    Recursively traverse all submodules of a model and apply a specific operation
    to the innermost layers, passing the full name of the layer.
    """
    has_children = False
    for name, layer in module.named_children():
        has_children = True
        new_prefix = f"{prefix}.{name}" if prefix else name
        apply_to_innermost_layers(layer, operation, new_prefix)
    
    # If the module has no children, it's an innermost layer
    if not has_children:
        
        # operation(module, prefix)
        # here we can get the name info
        inner_module = prefix.split(".")[-1] # like delta_theta, delta_A
        before_module = ".".join(prefix.split(".")[:-1])
        layer_num = prefix.split(".")[3]
        layer_prefix = ".".join(prefix.split(".")[:3])
        attention_name = prefix.split(".")[-2]
        the_layer = eval( f"self.{layer_prefix[layer_num]}.{attention_name}" )

        print(prefix.split(".")[-1])
        
        # no, since this is iterable
        return the_layer








def init_layers_with_active_block(module, active_block, prefix='', root_module=None):
    """
    Recursively traverse all submodules of a model and apply a specific operation
    to the innermost layers, passing the full name of the layer.
    """
    # Initialize root_module on the first call
    if root_module is None:
        root_module = module

    has_children = False
    for name, layer in module.named_children():
        has_children = True
        new_prefix = f"{prefix}.{name}" if prefix else name
        init_layers_with_active_block(layer, active_block, new_prefix, root_module)
    
    # If the module has no children, it's an innermost layer
    if not has_children:
        inner_module = prefix.split(".")[-1]  # like delta_theta, delta_A
        before_module = ".".join(prefix.split(".")[:-1])
        if inner_module == "delta_theta":
            layer_num = prefix.split(".")[3]
            # this is without the first `model` name
            layer_prefix = ".".join(prefix.split(".")[:3])
            attention_name = ".".join(prefix.split(".")[-3:-1])

            # model.model.layers.0.self_attn.q_proj.base_layer
            the_layer = eval(f"root_module.{layer_prefix}[{layer_num}].{attention_name}")
            
            # init that prefix
            if any(p in prefix for p in active_block):
                the_layer.spawn_delta_matrix("default")



def del_and_create_with_active_block(module, active_block, prefix='', root_module=None):
    """
    Recursively traverse all submodules of a model and apply a specific operation
    to the innermost layers, passing the full name of the layer.
    """
    # Initialize root_module on the first call
    if root_module is None:
        root_module = module

    has_children = False
    for name, layer in module.named_children():
        has_children = True
        new_prefix = f"{prefix}.{name}" if prefix else name
        del_and_create_with_active_block(layer, active_block, new_prefix, root_module)
    
    # If the module has no children, it's an innermost layer
    if not has_children:
        inner_module = prefix.split(".")[-1]  # like delta_theta, delta_A
        before_module = ".".join(prefix.split(".")[:-1])
        if (inner_module == "delta_A") or (inner_module == "delta_B"):
            layer_num = prefix.split(".")[3]
            # this is without the first `model` name
            layer_prefix = ".".join(prefix.split(".")[:3])
            attention_name = ".".join(prefix.split(".")[-3:-1])

            # model.model.layers.0.self_attn.q_proj.base_layer
            the_layer = eval(f"root_module.{layer_prefix}[{layer_num}].{attention_name}")
            
            # init that prefix
            if any(p in prefix for p in active_block):
                # value = the_layer.del_delta_create_AB("default")
                the_layer.del_delta_create_AB("default")





# def del_and_create_with_active_block(module, active_block, prefix='', root_module=None, collected_values=None):
#     """
#     Recursively traverse all submodules of a model and apply a specific operation
#     to the innermost layers, passing the full name of the layer.
#     Collect values returned by del_delta_create_AB and calculate their average.
#     """
#     # Initialize root_module and collected_values on the first call
#     if root_module is None:
#         root_module = module
#     if collected_values is None:
#         collected_values = []

#     has_children = False
#     for name, layer in module.named_children():
#         has_children = True
#         new_prefix = f"{prefix}.{name}" if prefix else name
#         del_and_create_with_active_block(layer, active_block, new_prefix, root_module, collected_values)
    
#     # If the module has no children, it's an innermost layer
#     if not has_children:
#         inner_module = prefix.split(".")[-1]  # like delta_theta, delta_A
#         before_module = ".".join(prefix.split(".")[:-1])
#         if inner_module == "delta_theta":
#             layer_num = prefix.split(".")[3]
#             # this is without the first `model` name
#             layer_prefix = ".".join(prefix.split(".")[:3])
#             attention_name = ".".join(prefix.split(".")[-3:-1])

#             # model.model.layers.0.self_attn.q_proj.base_layer
#             the_layer = eval(f"root_module.{layer_prefix}[{layer_num}].{attention_name}")
            
#             # init that prefix
#             if any(p in prefix for p in active_block):
#                 value = the_layer.del_delta_create_AB("default")
#                 collected_values.append(value)

#     # On the initial call, return the average of collected values
#     if root_module is module:
#         if collected_values:
#             avg_value = sum(collected_values) / len(collected_values)
#             return avg_value
#         else:
#             return None



# def low_rank_proj(delta_theta, r):
#     U, S, V = torch.svd(delta_theta)

#     # choose first r singular value
#     U_r = U[:, :r]
#     S_r = S[:r]
#     V_r = V[:, :r]

#     A = U_r @ torch.diag(S_r)
#     B = V_r.t()

#     reconstructed = A @ B
#     loss = F.mse_loss(delta_theta, reconstructed)

#     return A, B, loss


# def low_rank_proj(delta_theta, r):
#     U, S, Vh = torch.svd(delta_theta)

#     # choose first r singular value
#     U = U[:, :r]
#     S = S[:r]
#     U = U @ torch.diag(S)
#     Vh = Vh[:r, :]

#     reconstructed = Vh @ U
#     loss = F.mse_loss(delta_theta, reconstructed)

#     return Vh, U, loss


# def low_rank_proj(delta_theta, r):
#     U, S, Vh = torch.svd(delta_theta)

#     # choose first r singular value
#     U = U[:, :r]
#     S = S[:r]
#     U = U @ torch.diag(S)
#     Vh = Vh[:r, :]

#     # print(f"U:{U.shape}, S:{S.shape}, Vh:{Vh.shape}")
#     reconstructed = U @ Vh
#     reconstructed = Vh @ U
#     loss = F.mse_loss(delta_theta, reconstructed)

#     return Vh, U, loss




def low_rank_proj(delta_theta, r):
    U, S, Vh = torch.svd(delta_theta)
    # print("before projection")
    # print(f"U:{U.shape}, S:{S.shape}, Vh:{Vh.shape}")

    # choose first r singular value
    U = U[:, :r]
    S = S[:r]
    U = U @ torch.diag(S)
    Vh = Vh.T[:r, :]

    # print("after projection")
    # print(f"U:{U.shape}, S:{S.shape}, Vh:{Vh.shape}")
    reconstructed = U @ Vh
    loss = F.mse_loss(delta_theta, reconstructed)

    return U, Vh, loss



"""
delta_weight = delta_weight.T
U, S, Vh = torch.linalg.svd(delta_weight, full_matrices=full_matrices, driver=driver)
U = U[:, :new_rank]
S = S[:new_rank]
U = U @ torch.diag(S)
Vh = Vh[:new_rank, :]
if clamp is not None:
    dist = torch.cat([U.flatten(), Vh.flatten()])
    hi_val = torch.quantile(dist, clamp)
    low_val = -hi_val
    U = U.clamp(low_val, hi_val)
    Vh = Vh.clamp(low_val, hi_val)
if conv2d:
    U = U.reshape(target_lora_B.data.shape)
    Vh = Vh.reshape(target_lora_A.data.shape)
return Vh, U

target_lora_A.data, target_lora_B.data = self._svd_generalized_task_arithmetic_weighted_adapter
"""



# def init_layers_with_active_block(module, active_block, prefix=''):
#     """
#     Recursively traverse all submodules of a model and apply a specific operation
#     to the innermost layers, passing the full name of the layer.
#     """
#     # `module` is `self.model`
#     # `active_block` is `self.active_param_prefixs`
#     has_children = False
#     for name, layer in module.named_children():
#         has_children = True
#         new_prefix = f"{prefix}.{name}" if prefix else name
#         init_layers_with_active_block(layer, active_block, new_prefix)
    
#     # If the module has no children, it's an innermost layer
#     # model.model.layers.0.self_attn.q_proj.base_layer
#     if not has_children:
        
#         # operation(module, prefix)
#         # here we can get the name info
#         inner_module = prefix.split(".")[-1] # like delta_theta, delta_A
#         before_module = ".".join(prefix.split(".")[:-1])
#         if inner_module == "delta_theta":
#             layer_num = prefix.split(".")[3]
#             layer_prefix = ".".join(prefix.split(".")[:3])
#             # .self_attn.q_proj
#             attention_name = ".".join(prefix.split(".")[-3:-1])
#             # make sure there is a `self`
#             the_layer = eval( f"self.{layer_prefix}[{layer_num}].{attention_name}" )
            
#             # init that prefix
#             if any(p in prefix for p in active_block):
#                 the_layer.spawn_delta_matrix("default")




# def apply_to_innermost_layers(module, operation, depth=0):
#     """
#     递归遍历模型的所有子模块，并对最内层的模块应用特定操作。
#     """
#     has_children = False
#     for name, layer in module.named_children():
#         has_children = True
#         apply_to_innermost_layers(layer, operation, depth + 1)
    
#     # 如果当前模块没有子模块，则认为它是最内层模块
#     if not has_children:
#         operation(module, depth)

# # 定义你想对每层执行的操作
# def my_operation(layer, depth):
#     print("  " * depth + f"Applying operation to innermost layer: {type(layer)}")
#     # 这里可以执行你想对每层进行的操作，例如打印层的权重
#     if isinstance(layer, nn.Linear):
#         print("  " * depth + f"Layer weights: {layer.weight}")



# 应用操作到每一层
if __name__ == "__main__":
    # apply_to_layers(delta_model, my_operation)
    pass
