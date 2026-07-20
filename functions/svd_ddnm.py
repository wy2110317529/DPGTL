import torch
from tqdm import tqdm
import torchvision.utils as tvu
import torchvision
import os
import kornia.filters as F_korn
import kornia.color as C_korn
class_num = 951
from torchvision.utils import save_image


# --- 【关键修正结束】 ---
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def inverse_data_transform(x):
    x = (x + 1.0) / 2.0
    return torch.clamp(x, 0.0, 1.0)

def ddnm_diffusion(x, model, b, eta, A_funcs, y, cls_fn=None, classes=None, config=None, class_num=None):
    # 我们将在函数内部的no_grad上下文中进行操作，因为该最终方案不计算外部梯度
    with torch.no_grad():
        # --- 【新的引导超参数】 ---
        # 控制和谐化步骤的强度。数值越大，代表加的噪声越多，和谐化力度越强。
        # 这是一个全新的、需要调试的关键参数。建议从一个较小的值开始。
        harmonization_strength_steps = 5
        # --- 【方法二：自适应引导融合超参数】 ---
        # 是否启用“自适应引导融合”。True为启用，False为使用原始DDNM方式。
        use_adaptive_fusion = True
        
        skip = config.diffusion.num_diffusion_timesteps // config.time_travel.T_sampling
        n = x.size(0)
        x0_preds = []
        xs = [x]
        times = get_schedule_jump(config.time_travel.T_sampling,
                                  config.time_travel.travel_length,
                                  config.time_travel.travel_repeat,
                                 )
        time_pairs = list(zip(times[:-1], times[1:]))
        
        debug_folder = "debug_images" 
        os.makedirs(debug_folder, exist_ok=True) # 确保文件夹存在
        
        # --- 预计算范围空间和基础零空间 (只计算一次) ---
        # A_dagger * y
        range_space_y = A_funcs.A_pinv(y.reshape(n, -1)).reshape(x.size())
    
        for i, j in tqdm(time_pairs):
            i, j = i * skip, j * skip
            if j < 0: j = -1

            if j < i:  # normal sampling
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                xt = xs[-1].to('cuda')
                
                # 1. 原始噪声预测
                if cls_fn == None:
                    et = model(xt, t)
                else:
                    classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
                    et = model(xt, t, classes)
                    et = et[:, :3]
                    et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

                if et.size(1) == 6:
                    et = et[:, :3]
            
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                # --- 【核心融合逻辑：方法二 “自适应引导融合”】 ---


                # 如果不使用自适应融合，则退化为原始DDNM的数据一致性校正
                x0_t_h = x0_t - A_funcs.A_pinv(
                    A_funcs.A(x0_t.reshape(n, -1)) - y.reshape(n, -1)
                ).reshape(x0_t.size())
                # --- 【自适应引导融合逻辑结束】 ---
                progress = 1.0 - (i / config.diffusion.num_diffusion_timesteps)
                #if progress >=0.99:
                #    use_adaptive_fusion = False 
                if use_adaptive_fusion:
                    c = 0.2
                    s = 30.0
                    w_t = 1 / (1 + torch.exp(-torch.tensor(s * (progress - c))))
                    ns_generative = x0_t_h - A_funcs.A_pinv(A_funcs.A(x0_t.reshape(n, -1))).reshape(x.size())
                    ns_fused =  w_t * ns_generative

                    x0_t_draft = range_space_y + ns_fused
                else:
                    x0_t_draft = x0_t_h
                # 3. 进行和谐化处理
                # a. 模拟加噪: 将草稿图加入少量噪声
                # 确保时间步不越界
                #if progress >=0.99:
                #    harmonization_strength_steps = 0
                if harmonization_strength_steps > 0:
                    harmonization_t_step = min(t.item() + harmonization_strength_steps, config.diffusion.num_diffusion_timesteps - 1)
                    harmonization_t = (torch.ones(n) * harmonization_t_step).to(x.device)
                    at_harmonization = compute_alpha(b, harmonization_t.long())
                
                    xt_draft = at_harmonization.sqrt() * x0_t_draft + (1 - at_harmonization).sqrt() * torch.randn_like(x0_t_draft)
                
                    # b. 再次去噪得到和谐化的版本
                    et_harmonized = model(xt_draft, harmonization_t)
                    if et_harmonized.size(1) == 6: et_harmonized = et_harmonized[:,:3]
                    x0_t_harmonized = (xt_draft - et_harmonized * (1 - at_harmonization).sqrt()) / at_harmonization.sqrt()

                    x0_t_hat = x0_t_harmonized - A_funcs.A_pinv(
                        A_funcs.A(x0_t_harmonized.reshape(n, -1)) - y.reshape(n, -1)
                    ).reshape(x0_t_harmonized.size())
                else:
                    x0_t_hat = x0_t_draft
                
                    
                #et_hat = (xt - x0_t_hat * at.sqrt()) / (1 - at).sqrt()
                x0_preds.append(x0_t_hat.to('cpu'))

                c1 = (1 - at_next).sqrt() * eta
                c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
                xt_next = at_next.sqrt() * x0_t_hat + c1 * torch.randn_like(x0_t) + c2 * et
                xs.append(xt_next.to('cpu'))
            
            else:  # time-travel back
                next_t = (torch.ones(n) * j).to(x.device)
                at_next = compute_alpha(b, next_t.long())
                x0_t = x0_preds[-1].to('cuda')
                xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()
                xs.append(xt_next.to('cpu'))

    # 【已修正】确保return的缩进正确
    return [xs[-1]], [x0_preds[-1]] 

def ddnm_plus_diffusion(x, model, b, eta, A_funcs, y, sigma_y, cls_fn=None, classes=None, config=None):
    with torch.no_grad():
        # 是否启用“自适应引导融合”。True为启用，False为使用原始DDNM方式。
        use_adaptive_fusion = True
        harmonization_strength_steps = 5
        # setup iteration variables
        skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
        n = x.size(0)
        x0_preds = []
        xs = [x]

        # generate time schedule
        times = get_schedule_jump(config.time_travel.T_sampling, 
                               config.time_travel.travel_length, 
                               config.time_travel.travel_repeat,
                              )
        time_pairs = list(zip(times[:-1], times[1:]))        
        range_space_y = A_funcs.A_pinv(y.reshape(n, -1)).reshape(x.size())
        # reverse diffusion sampling
        for i, j in tqdm(time_pairs):
            i, j = i*skip, j*skip
            if j<0: j=-1 

            if j < i: # normal sampling 
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                xt = xs[-1].to('cuda')
                if cls_fn == None:
                    et = model(xt, t)
                else:
                    classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
                    et = model(xt, t, classes)
                    et = et[:, :3]
                    et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

                if et.size(1) == 6:
                    et = et[:, :3]

                # Eq. 12
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                sigma_t = (1 - at_next).sqrt()[0, 0, 0, 0]

                # Eq. 17
                x0_t_h = x0_t - A_funcs.Lambda(A_funcs.A_pinv(
                    A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
                ).reshape(x0_t.size(0), -1), at_next.sqrt()[0, 0, 0, 0], sigma_y, sigma_t, eta).reshape(*x0_t.size())
                #
                progress = 1.0 - (i / config.diffusion.num_diffusion_timesteps)
                if progress >=0.93:
                    use_adaptive_fusion = False 
                if use_adaptive_fusion:
                    c = 0.2
                    s = 30.0
                    w_t = 1 / (1 + torch.exp(-torch.tensor(s * (progress - c))))
                    ns_generative = x0_t_h - A_funcs.A_pinv(A_funcs.A(x0_t.reshape(n, -1))).reshape(x.size())
                    #g = A_funcs.A_pinv(y.reshape(n, -1)).reshape(x.size())- A_funcs.A_pinv(A_funcs.A(x0_t.reshape(n, -1))).reshape(x.size())
                    ns_fused =  w_t * ns_generative    #+ (1-w_t)*ns_base
                    x0_t_draft = range_space_y + ns_fused
                else:
                    x0_t_draft = x0_t_h
                #-----------------------------
                # 3. 进行和谐化处理
                # a. 模拟加噪: 将草稿图加入少量噪声
                # 确保时间步不越界
                #if progress >=0.95:
                #    harmonization_strength_steps = 0
                if harmonization_strength_steps > 0:
                    harmonization_t_step = min(t.item() + harmonization_strength_steps, config.diffusion.num_diffusion_timesteps - 1)
                    harmonization_t = (torch.ones(n) * harmonization_t_step).to(x.device)
                    at_harmonization = compute_alpha(b, harmonization_t.long())
                
                    xt_draft = at_harmonization.sqrt() * x0_t_draft + (1 - at_harmonization).sqrt() * torch.randn_like(x0_t_draft)

                    et_harmonized = model(xt_draft, harmonization_t)
                    if et_harmonized.size(1) == 6: et_harmonized = et_harmonized[:,:3]
                    x0_t_harmonized = (xt_draft - et_harmonized * (1 - at_harmonization).sqrt()) / at_harmonization.sqrt()

                    correction_term_harmonized = A_funcs.A_pinv(
                        A_funcs.A(x0_t_harmonized.reshape(n, -1)) - y.reshape(n, -1)
                    ).reshape(x0_t_harmonized.size(0), -1)
                    x0_t_hat = x0_t_harmonized - A_funcs.Lambda(
                        correction_term_harmonized,
                        at_next.sqrt()[0, 0, 0, 0],
                        sigma_y,
                        sigma_t,
                        eta
                    ).reshape(*x0_t_harmonized.size())
                    # --- 【核心融合逻辑结束】 ---
                else:
                    x0_t_hat = x0_t_draft

                xt_next = at_next.sqrt() * x0_t_hat + A_funcs.Lambda_noise(
                    torch.randn_like(x0_t).reshape(x0_t.size(0), -1), 
                    at_next.sqrt()[0, 0, 0, 0], sigma_y, sigma_t, eta, et.reshape(et.size(0), -1)).reshape(*x0_t.size())

                x0_preds.append(x0_t.to('cpu'))
                xs.append(xt_next.to('cpu'))
            else: # time-travel back
                next_t = (torch.ones(n) * j).to(x.device)
                at_next = compute_alpha(b, next_t.long())
                x0_t = x0_preds[-1].to('cuda')
                
                xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

                xs.append(xt_next.to('cpu'))
                
#             #ablation
#             if i%50==0:
#                 os.makedirs('/userhome/wyh/ddnm/debug/x0t', exist_ok=True)
#                 tvu.save_image(
#                     inverse_data_transform(x0_t[0]),
#                     os.path.join('/userhome/wyh/ddnm/debug/x0t', f"x0_t_{i}.png")
#                 )
                
#                 os.makedirs('/userhome/wyh/ddnm/debug/x0_t_hat', exist_ok=True)
#                 tvu.save_image(
#                     inverse_data_transform(x0_t_hat[0]),
#                     os.path.join('/userhome/wyh/ddnm/debug/x0_t_hat', f"x0_t_hat_{i}.png")
#                 )
                
#                 os.makedirs('/userhome/wyh/ddnm/debug/xt_next', exist_ok=True)
#                 tvu.save_image(
#                     inverse_data_transform(xt_next[0]),
#                     os.path.join('/userhome/wyh/ddnm/debug/xt_next', f"xt_next_{i}.png")
#                 )

    return [xs[-1]], [x0_preds[-1]]

# form RePaint
def get_schedule_jump(T_sampling, travel_length, travel_repeat):

    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = T_sampling
    ts = []

    while t >= 1:
        t = t-1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)

    _check_times(ts, -1, T_sampling)

    return ts

def _check_times(times, t_0, T_sampling):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= T_sampling, (t, T_sampling)