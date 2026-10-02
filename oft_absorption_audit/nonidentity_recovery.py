"""
非恒等恢复验证 (Non-identity Recovery Verification)
====================================================
对应审稿建议: 正式实验必须独立重载, 证明"跑实验时确实用了这些非恒等权重",
而不是靠手动改回或内存里残留的状态。

本脚本是**独立的正式实验进程**(与 identity_control.py 分开跑)。它从原始底座 +
磁盘 adapter 重新加载, 完成以下可追溯证据链:

  来源明确的 checkpoint
    → 加载参数与文件逐元素一致 (同 dtype)
    → 有效旋转确实非恒等 (计算 ‖R-I‖_F), adapter 确实参与计算 (未 merge, 走 OFT 路径)
    → 在此状态下完成 replay 与精度实验

严格遵循的流程 (用户给定 5 步):
  1. 独立进程: identity control 另跑 (见 identity_control.py), 本脚本是另一全新进程。
  2. 从原始底座 + 磁盘 adapter 重新加载, 不沿用任何外部模型对象。
  3. replay/merge 前记录每层实际状态:
       (a) 96 个 OFT 参数加载后与 checkpoint 文件逐元素核对 (同 dtype)
       (b) 用实际前向参数化计算有效 R, 记录 ‖R-I‖_F
            —— 有效 R 由"权重反推"得到 (参数化无关, 最忠实):
               unmerged 权重 W0, merged 权重 Wm=Rᵀ·W0
               => R_block_i = (Wm[i,i] · pinv(W0[i,i]))ᵀ
       (c) adapter 已启用, 未 merge, forward 确实经过 OFT 路径 (disable_adapter 对比)
       (d) 额外状态 (coft/eps/block_share/use_cayley_neumann) 一并核对
  4. unmerged 与 merged 模型各自独立加载, 只对 merged 那份执行 merge_and_unload;
     不保存一个模型引用再原地 merge 后当两个比。
  5. 同一份固定输入做局部 replay + 端到端比较, 每精度设置重新从原始 checkpoint 加载,
     保存输入指纹 / 参数指纹 / 实际旋转偏离 / 误差。

用法:
  python nonidentity_recovery.py --checkpoint-dir outputs/oft_adapter_checkpoint
  python nonidentity_recovery.py --checkpoint-dir outputs/oft_adapter_checkpoint --precisions fp32 fp64
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

_sdir = os.path.dirname(os.path.abspath(__file__))
if _sdir not in sys.path:
    sys.path.insert(0, _sdir)

from config import OFTAuditConfig


# ----------------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------------

def _set_deterministic(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tensor_fingerprint(t: torch.Tensor) -> str:
    """固定输入指纹: 对张量字节做 sha256。"""
    return hashlib.sha256(
        t.detach().cpu().numpy().tobytes()
    ).hexdigest()


def _load_safetensors(path: str) -> dict:
    try:
        from safetensors.torch import load_file
        return load_file(path)
    except ImportError:
        raise RuntimeError(
            "需要 safetensors 读取 adapter_model.safetensors, 请 pip install safetensors"
        )


def _canonical(name: str) -> str:
    for p in ("base_model.model.", "base_model."):
        if name.startswith(p):
            return name[len(p):]
    return name


def _load_base_model(cfg: OFTAuditConfig):
    from transformers import (
        AutoImageProcessor,
        AutoModelForImageClassification,
    )
    base = AutoModelForImageClassification.from_pretrained(
        cfg.base_model,
        num_labels=cfg.num_labels,
        ignore_mismatched_sizes=True,
        trust_remote_code=True,
    )
    processor = AutoImageProcessor.from_pretrained(cfg.base_model)
    return base, processor


def _build_oft_config(ckpt_dir: str):
    """读保存的 adapter_config, 强制精确 Cayley, 返回 (OFTConfig, raw_cfg_dict)。"""
    import json as _json
    from peft import OFTConfig
    with open(os.path.join(ckpt_dir, "adapter_config.json")) as f:
        raw = _json.load(f)
    r = int(raw.get("r", 8))
    target = raw.get("target_modules")
    if isinstance(target, (set, list)):
        target = list(target)
    # 强制精确 Cayley, 与训练语义 (PEFT 0.13.2 唯一模式) 对齐
    kwargs = dict(r=r, target_modules=target, module_dropout=float(raw.get("module_dropout", 0.0)))
    try:
        kwargs["use_cayley_neumann"] = False
        peft_config = OFTConfig(**kwargs)
        forced = True
    except TypeError:
        # 老版本 (如 0.13.2) 无该字段 -> 精确 Cayley 由构造保证
        peft_config = OFTConfig(**{k: v for k, v in kwargs.items() if k != "use_cayley_neumann"})
        forced = False
    return peft_config, raw, forced


# ----------------------------------------------------------------------------
# 核心: 单次精度下的完整验证
# ----------------------------------------------------------------------------

def _effective_R_from_weights(W0: torch.Tensor, Wm: torch.Tensor, block: int):
    """由未合并权重 W0 与合并后权重 Wm 反推有效旋转 R (块对角)。

    参数化无关: 直接反映前向实际使用的变换。
    Wm = Rᵀ · W0  (逐对角块) => R_block = (Wm_block · pinv(W0_block))ᵀ
    """
    d = W0.shape[0]
    assert W0.shape == Wm.shape and d % block == 0
    dev = W0.device
    Rs = []
    orth_errs = []
    for i in range(d // block):
        a, b = i * block, (i + 1) * block
        W0b = W0[a:b, a:b].to(torch.float64)
        Wmb = Wm[a:b, a:b].to(torch.float64)
        Rb = (Wmb @ torch.linalg.pinv(W0b)).T
        Rs.append(Rb.to(W0.dtype))
        I64 = torch.eye(block, dtype=torch.float64, device=dev)
        orth_errs.append((Rb.T @ Rb - I64).abs().max().item())
    R = torch.block_diag(*Rs) if Rs else torch.eye(d, dtype=W0.dtype, device=dev)
    R_full = R.to(W0.dtype)
    I = torch.eye(d, dtype=W0.dtype, device=dev)
    diff = (R_full - I).abs()
    return R_full, diff.max().item(), float(np.max(orth_errs))


def _cayley_blockdiag_from_oft_r(oft_r: torch.Tensor, dtype=torch.float64):
    """把整层 oft_r [r,b,b] 变成块对角正交矩阵 R [r*b, r*b] (与 PEFT _block_diagonal 等价)。

    默认在 float64 中组装, 避免使用 fp32 参数路径时把已舍入的旋转当双精度。
    调用方显式传入 dtype 才会在最后一步降到目标精度 (单次舍入, 而非中间降精度)。
    """
    blocks = [_cayley_R_from_param(oft_r[i]) for i in range(oft_r.shape[0])]
    R = torch.block_diag(*blocks)
    return R.to(dtype if dtype is not None else torch.float64)


def _gamma(n: int, dtype: torch.dtype) -> float:
    """Higham 常数 γ_n = n·u / (1 − n·u), u 为单位舍入 = finfo(dtype).eps / 2。"""
    u = float(torch.finfo(dtype).eps) / 2.0
    nu = n * u
    return nu / (1.0 - nu) if nu < 1.0 else float("inf")


def _local_forward_fp_bound(x, W0, Wm, R_full, dtype):
    """两条前向路径差值的**浮点一阶误差上界**(Higham, Accuracy and Stability Ch.2)。

    unmerged 路径 (输出侧旋转):  y_u = ((x W0ᵀ) R) + b   —— 两级矩阵乘
    merged   路径:              y_m = (x Wmᵀ) + b,  Wm = Rᵀ W0 —— 一级矩阵乘 + 合并生成

    fl(Σ a_i w_i) 的误差 ≤ γ_n · |A| @ |Wᵀ|。逐项:
      - 第一级 (xW0ᵀ) 舍入界 γ_in·|x||W0|, 经输出侧旋转 R 传播 => γ_in·(|x||W0|)|R|
      - 第二级 (tR) 舍入界 γ_out·|t||R|   (t = xW0ᵀ)
      - 合并权重的生成舍入: Wm = RᵀW0 在 fp 下计算有 δ_merge, 界 γ_out·|R||W0|, 再经 x 传播
      - 偏置 b 两条路径相同, 在差值中抵消 (不计入)

    输出侧旋转用 |R| (非 |R|ᵀ)。本界采用 dtype 的单位舍入 (fp32/fp64),
    仅对关闭 TF32 的档成立; TF32 下 matmul 有效单位舍入更大, 该界失效, 不应据此判定通过。

    返回整层所有元素的最大值(float, 在 float64 中计算以避免界本身被舍入污染)。
    不满足形状前提时返回 nan。
    """
    if x.shape[-1] != W0.shape[1] or Wm.shape != W0.shape or R_full.shape[0] != W0.shape[0]:
        return float("nan")
    xd = x.detach().reshape(-1, x.shape[-1]).to(torch.float64)
    W0d = W0.detach().to(torch.float64)
    Wmd = Wm.detach().to(torch.float64)
    Rd = R_full.detach().to(torch.float64)
    g_in = _gamma(xd.shape[-1], dtype)        # 第一级 xW0ᵀ 累加长度 (in)
    g_out = _gamma(Rd.shape[-1], dtype)       # 第二级 tR / xWmᵀ 累加长度 (out)
    Ra = Rd.abs()                             # 输出侧旋转用 |R| (非 |R|ᵀ)
    abs_T1 = xd.abs() @ W0d.abs().T           # |x||W0|  (batch, out)
    T1 = (xd @ W0d.T).abs()                    # |t|, t = xW0ᵀ (batch, out)
    # unmerged 两级: 第一级舍入经旋转 R 传播 + 第二级自身舍入
    Bu = (g_in * abs_T1) @ Ra + g_out * (T1 @ Ra)
    # merged 路径: 先算 Wm 的合并生成舍入 δ_merge, 再 x Wmᵀ 自身舍入
    g_merge = _gamma(W0d.shape[0], dtype)     # Wm=RᵀW0 累加长度 (out)
    delta_merge_abs_T = g_merge * (W0d.abs().T @ Ra)   # |x·δ_mergeᵀ| 的传播因子 (in,out)
    Bm = g_out * (xd.abs() @ Wmd.abs().T) + xd.abs() @ delta_merge_abs_T
    return float((Bu + Bm).max().item())


def _get_oft_r_tensor(layer):
    """取出该 OFT 层的 oft_r 参数张量, 返回 (tensor, adapter_name)。

    PEFT 0.13.2 里 `layer.oft_r` 是 `nn.ParameterDict({adapter_name: Parameter[r, b, b]})`,
    不是张量本身 —— 直接 `.detach()` 会 AttributeError。这里兼容两种形态。
    """
    od = layer.oft_r
    if isinstance(od, torch.Tensor):
        return od.detach(), None
    keys = list(od.keys())
    if not keys:
        return None, None
    key = "default" if "default" in keys else keys[0]
    return od[key].detach(), key


def _cayley_R_from_param(oft_r_block: torch.Tensor):
    """用 PEFT 实际 Cayley 公式从 oft_r 单块反推 Q, 在 float64 中计算并保持高精度返回。

    注意: 不在此处把结果降回 oft_r 的 dtype。若先降到 fp32 再被调用方升回 fp64,
    会得到"舍入后的 fp32 旋转转双精度"而非完整双精度结果。最终精度由
    _cayley_blockdiag_from_oft_r 的 dtype 参数在组装后一次性决定。
    """
    A = oft_r_block.to(torch.float64)
    skew = 0.5 * (A - A.transpose(0, 1))
    I = torch.eye(A.shape[0], dtype=torch.float64, device=A.device)
    Q = (I - skew) @ torch.linalg.inv(I + skew)
    return Q


def run_recovery_once(cfg, ckpt_dir, precision, fixed_input, input_fp, base_model_id,
                      adapter_sha, peft_ver, tf_ver, torch_ver, use_cayley_forced,
                      base=None):
    from peft import PeftModel

    dtype = torch.float64 if precision == "fp64" else torch.float32
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # TF32 控制: 仅 fp32_tf32 档开启, 其余 (fp32/fp64) 关闭。
    # TF32 是 FP32 下额外的精度损失来源, 必须显式固定才能做公平对照。
    allow_tf32 = (precision == "fp32_tf32")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # --- 步骤 2/4: 固定完整初始模型 (含随机分类头), 各精度从它 deepcopy 恢复 ---
    # 关键 1: 底座只加载一次 (main 传入 base), 每个精度 deepcopy 出独立副本, 保证
    #         跨精度的分类头 / 底座参数完全一致 ("只改变精度, 不改变模型")。
    # 关键 2: 只能调用一次 from_pretrained 拿 adapter, 再 deepcopy 出第二份 merged 分支,
    #         避免两分支随机初始化不同导致端到端差异 O(1)。
    peft_config, raw_cfg, forced = _build_oft_config(ckpt_dir)
    if base is None:
        base_u, _ = _load_base_model(cfg)
    else:
        import copy as _copy
        base_u = _copy.deepcopy(base)
    base_u.eval()
    base_u = base_u.to(device).to(dtype)

    import copy as _copy

    base_m = _copy.deepcopy(base_u)  # 与 base_u 逐位相同 (含随机分类头)
    base_m.eval()

    model_unmerged = PeftModel.from_pretrained(base_u, ckpt_dir, peft_config=peft_config)
    model_unmerged = model_unmerged.to(dtype).to(device)
    model_unmerged.eval()

    model_merged_src = PeftModel.from_pretrained(base_m, ckpt_dir, peft_config=peft_config)
    model_merged_src = model_merged_src.to(dtype).to(device)
    model_merged_src.eval()
    model_merged = model_merged_src.merge_and_unload()
    model_merged = model_merged.to(dtype).to(device)
    model_merged.eval()

    # 自洽: 两份分支的非注入参数必须逐位一致 (尤其随即初始化的分类头,
    #       它不在 checkpoint 里, 两次独立加载会得到不同随机值)
    def _classifier_params(model):
        out = {}
        for n, p in model.named_parameters():
            if "classifier" in n:
                out[_canonical(n).replace(".base_layer", "")] = p.detach()
        return out

    pu, pm = _classifier_params(model_unmerged), _classifier_params(model_merged)
    common = sorted(set(pu) & set(pm))
    head_max = 0.0
    for k in common:
        if pu[k].shape == pm[k].shape:
            head_max = max(head_max, (pu[k] - pm[k]).abs().max().item())
    print(f"  [自洽] 分类头等训练外参数一致: max={head_max:.3e} (比对 {len(common)} 个张量)")
    if head_max > 0:
        print("  [!] 两分支的非注入参数不同 => 端到端差异会被该差异主导, 不作 FOT 结论。")

    # --- 步骤 3(a): 加载参数与 checkpoint 文件逐元素核对 ---
    st = _load_safetensors(os.path.join(ckpt_dir, "adapter_model.safetensors"))
    param_file_diffs = {}
    for n, p in model_unmerged.named_parameters():
        if "oft_r" not in n:
            continue
        can = _canonical(n)
        # 键名对齐: 模型里 ParameterDict 使参数名为 "...oft_r.<adapter>",
        # 而 safetensors 文件里存的是 "...oft_r" (不带 adapter 后缀)。
        # 二者都必须尝试, 否则会全部漏判 => 表现为 FILE_KEY_MISSING。
        key = None
        can_stripped = can.split(".oft_r.")[0] + ".oft_r" if ".oft_r." in can else can
        cands = [
            can,
            "base_model.model." + can,
            can_stripped,
            "base_model.model." + can_stripped,
            n,
        ]
        for cand in cands:
            if cand in st:
                key = cand
                break
        if key is None:
            param_file_diffs[can] = "FILE_KEY_MISSING"
            continue
        tf = st[key].to(p.dtype)
        if tf.shape != p.shape:
            param_file_diffs[can] = f"SHAPE {tuple(tf.shape)} vs {tuple(p.shape)}"
            continue
        d = (tf - p.detach().cpu()).abs().max().item()
        param_file_diffs[can] = d

    # --- 步骤 3(c): adapter 启用/未 merge/走 OFT 路径 ---
    oft_layers = {}
    for n, m in model_unmerged.named_modules():
        if hasattr(m, "oft_r"):
            oft_layers[_canonical(n)] = m
    n_oft = len(oft_layers)
    # 确认未合并模型确实未合并 (adapter 未 merge): 读取 PEFT 各层的 merged 状态,
    # 兼容 dict / bool / 缺失等多种形态。"读不出来" 属于**未知**, 必须显式标记,
    # 不能伪装成 False (False 表示"确认未合并", 与"无法判定"语义不同)。
    def _merged_state(layer):
        m = getattr(layer, "merged", None)
        if isinstance(m, dict):
            return any(bool(x) for x in m.values())
        if isinstance(m, bool):
            return m
        return None  # 未知: 该 PEFT 版本无可识别的 merged 字段
    states = []
    merged_state_unknown = False
    for v in oft_layers.values():
        st = _merged_state(v)
        if st is None:
            merged_state_unknown = True
        else:
            states.append(st)
    any_merged_flag = bool(states) and any(states)
    # disable_adapter 对比: 有 adapter 时 logits 必须与 base-only 不同
    with torch.inference_mode():
        xb = fixed_input.to(dtype).to(device)
        logits_oft = model_unmerged(pixel_values=xb, return_dict=True).logits
        with model_unmerged.disable_adapter():
            logits_base = model_unmerged(pixel_values=xb, return_dict=True).logits
    participates_max = (logits_oft - logits_base).abs().max().item()

    # --- 步骤 3(b): 有效 R (权重反推) + Cayley 交叉验证 ---
    # 捕获 unmerged 每层输入 x (pre-hook)
    captured_x = {}
    handles = []
    for name, layer in oft_layers.items():
        def _make(name):
            def _pre(module, inp):
                captured_x[name] = inp[0].detach().clone()
            return _pre
        handles.append(layer.register_forward_pre_hook(_make(name)))

    with torch.inference_mode():
        _ = model_unmerged(pixel_values=xb, return_dict=True)

    # 取 merged 模型对应模块 (merge 后为普通 Linear, 权重即 W_merged)
    merged_modules = {_canonical(n): m for n, m in model_merged.named_modules()}

    per_layer = {}
    R_devs = []
    cayley_cross = []
    merge_impls = []
    w0_conds = []
    for name, layer in oft_layers.items():
        # oft_r 先取出来, 用它的首维作为真实块数 r (不要硬编码 8)
        oft_r, adapter_key = _get_oft_r_tensor(layer)
        W0 = layer.base_layer.weight.data  # [out,in]
        if name not in merged_modules:
            per_layer[name] = {"status": "MERGED_MODULE_MISSING"}
            continue
        Wm = merged_modules[name].weight.data
        if W0.shape[0] != W0.shape[1]:
            per_layer[name] = {"status": "NON_SQUARE", "shape": list(W0.shape)}
            continue
        r_blocks = oft_r.shape[0] if oft_r is not None else 8
        if oft_r is None or W0.shape[0] % r_blocks != 0:
            per_layer[name] = {"status": "INDIVISIBLE", "shape": list(W0.shape), "r": r_blocks}
            continue
        block = W0.shape[0] // r_blocks
        if oft_r is not None and oft_r.shape[1] != block:
            per_layer[name] = {
                "status": "BLOCK_SIZE_MISMATCH",
                "oft_r_shape": list(oft_r.shape),
                "expected_block": block,
            }
            continue
        # 实际旋转从 oft_r 经 Cayley 参数化得到 (不经 pinv 反推, 不受 W0 块条件数影响)。
        # ‖R-I‖_F、正交性、误差界均用这个真实旋转, 而非从权重反推的 R。
        if oft_r is None:
            R_full = torch.eye(W0.shape[0], dtype=W0.dtype, device=W0.device)
            cay_R = R_full.to(torch.float64)
            dev_val = 0.0
            orth_max = 0.0
            R_max = 0.0
            cross = float("nan")
            merge_impl = float("nan")
            cond_max = float("nan")
        else:
            R_full, _, _ = _effective_R_from_weights(W0, Wm, block)  # 仅作权重反推交叉验证
            cay_R = _cayley_blockdiag_from_oft_r(oft_r, dtype=torch.float64).to(W0.device)
            I_cay = torch.eye(cay_R.shape[0], dtype=cay_R.dtype, device=cay_R.device)
            dev_val = (cay_R - I_cay).abs().norm().item()            # ‖R-I‖_F 用实际 Cayley R
            orth_max = (cay_R.T @ cay_R - I_cay).abs().max().item()  # 实际旋转正交性 (应 ~1e-14)
            R_max = cay_R.abs().max().item()
            cross = (cay_R - R_full.to(torch.float64)).abs().max().item()  # Cayley vs 权重反推
            Wm_pred = cay_R.T @ W0.to(torch.float64)                 # W_merged = Q^T W0 (输出侧)
            merge_impl = (Wm_pred - Wm.to(torch.float64)).abs().max().item()
            cond_max = max(
                torch.linalg.cond(W0[a:a + block, a:a + block].to(torch.float64)).item()
                for a in range(0, W0.shape[0], block)
            )
        R_devs.append(dev_val)
        cayley_cross.append(cross)
        merge_impls.append(merge_impl)
        w0_conds.append(cond_max)
        # shape_note: 记录块数信息, 便于追溯
        shape_note = f"r={r_blocks}, block={block}, oft_r={tuple(oft_r.shape)}"

        # 局部 replay: 同一固定输入 x 喂 unmerged 层 与 merged 层
        if name not in captured_x:
            per_layer[name] = {"status": "NO_CAPTURED_INPUT"}
            continue
        x = captured_x[name].to(dtype).to(device)
        with torch.inference_mode():
            out_u = layer(x)                       # unmerged: base_layer(x) @ R
            out_m = merged_modules[name](x)        # merged: x @ W_merged^T
        if isinstance(out_u, (tuple, list)):
            out_u = out_u[0]
        if isinstance(out_m, (tuple, list)):
            out_m = out_m[0]
        out_u = out_u.to(dtype)
        out_m = out_m.to(dtype)
        local_err = (out_u - out_m).abs().max().item()

        # 相对误差 (按该层输出尺度归一) 与浮点一阶误差上界
        out_scale = float(out_u.abs().max().item())
        local_rel = local_err / max(out_scale, 1e-30)
        fp_bound = _local_forward_fp_bound(x, W0, Wm, cay_R.to(dtype), dtype)

        per_layer[name] = {
            "status": "OK",
            "R_dev_fro": dev_val,
            "R_max_abs": R_max,
            "R_orth_err": orth_max,
            "cayley_cross_max": cross,
            "merge_impl_max": merge_impl,      # Q^T W0 vs 实际合并权重 (不经 pinv)
            "W0_block_cond_max": cond_max,     # 解释 (A) 的精度上限
            "oft_r_adapter_key": adapter_key,
            "oft_r_shape": (list(oft_r.shape) if oft_r is not None else None),
            "shape_note": shape_note,
            "local_absorb_max": local_err,
            "local_absorb_rel": local_rel,     # 相对误差 = local_err / max|out|
            "local_fp_bound_max": fp_bound,    # Higham 一阶界, dtype 自适应
            "local_within_fp_bound": bool(local_err <= fp_bound) if fp_bound == fp_bound else False,
            "out_scale_max": out_scale,
            "file_param_maxdiff": next(
                (v for k, v in param_file_diffs.items() if k.startswith(name + ".oft_r")),
                "N/A",
            ),
        }

    for h in handles:
        h.remove()

    # --- 步骤 5: 端到端 (同一固定输入) ---
    with torch.inference_mode():
        logits_u = model_unmerged(pixel_values=xb, return_dict=True).logits
        logits_m = model_merged(pixel_values=xb, return_dict=True).logits
    e2e_max = (logits_u - logits_m).abs().max().item()
    e2e_mse = ((logits_u - logits_m) ** 2).mean().item()

    R_devs = np.array(R_devs) if R_devs else np.array([0.0])
    cayley_cross = np.array(cayley_cross) if cayley_cross else np.array([0.0])
    file_diffs = [v for v in param_file_diffs.values() if isinstance(v, float)]
    n_missing = sum(1 for v in param_file_diffs.values()
                    if isinstance(v, str) and "MISSING" in v)
    n_shape = sum(1 for v in param_file_diffs.values()
                  if isinstance(v, str) and v.startswith("SHAPE"))
    # 没匹配到任何张量时不要用 0.0 冒充成功 —— 那是默认值, 会被误读为"逐元素一致"
    file_max = max(file_diffs) if file_diffs else float("nan")
    file_all_match = (
        len(file_diffs) == len(param_file_diffs)
        and bool(file_diffs)
        and all(v < 1e-6 for v in file_diffs)
    )

    # 只统计 status=OK 的层, 否则失败层会被当成 0 而虚报成功 (与上面 param_file 同一类陷阱)
    ok_rows = [(k, v) for k, v in per_layer.items()
               if isinstance(v, dict) and v.get("status") == "OK"]

    def _fin(field):
        """取该字段的有限 float 列表 (剔除 None / nan)。"""
        out_vals = []
        for _, v in ok_rows:
            x = v.get(field)
            if isinstance(x, (int, float)) and x == x:   # 剔除 nan
                out_vals.append(float(x))
        return out_vals

    def _agg(field):
        vals = _fin(field)
        return max(vals) if vals else float("nan")

    n_ok = len(ok_rows)
    mi = np.array([v for v in merge_impls if v == v]) if merge_impls else np.array([float("nan")])
    cd = np.array([v for v in w0_conds if v == v]) if w0_conds else np.array([float("nan")])

    # 最差的层 (按局部吸收绝对误差), 便于追溯是哪一层主导了数值残差
    local_vals = {k: v.get("local_absorb_max") for k, v in ok_rows
                  if isinstance(v.get("local_absorb_max"), float)}
    worst_layer = max(local_vals, key=local_vals.get) if local_vals else "N/A"
    within_flags = [bool(v.get("local_within_fp_bound")) for _, v in ok_rows]
    logits_scale = float(logits_u.abs().max().item())

    summary = {
        "precision": precision,
        "dtype": str(dtype),
        "n_oft_layers": n_oft,
        "n_layers_ok": n_ok,
        "param_file_n_compared": len(param_file_diffs),
        "param_file_n_missing": n_missing,
        "param_file_n_shape_mismatch": n_shape,
        "param_file_maxdiff": file_max,
        "param_file_all_match": file_all_match,
        "adapter_participates_maxdiff": participates_max,
        "adapter_active": participates_max > 1e-2,
        "any_merged_flag": any_merged_flag,
        "merged_state_unknown": merged_state_unknown,
        "R_dev_fro_min": float(R_devs.min()),
        "R_dev_fro_mean": float(R_devs.mean()),
        "R_dev_fro_max": float(R_devs.max()),
        "cayley_cross_max": float(cayley_cross.max()),
        "merge_impl_max": float(np.nanmax(mi)),   # Q^T W0 vs 实际合并权重 (不经 pinv)
        "W0_block_cond_max": float(np.nanmax(cd)),
        "local_absorb_max": float(_agg("local_absorb_max")),
        "local_absorb_rel_max": float(_agg("local_absorb_rel")),
        "local_absorb_worst_layer": worst_layer,
        "local_fp_bound_max": float(_agg("local_fp_bound_max")),
        "local_within_fp_bound_all": bool(within_flags) and all(within_flags),
        "local_fp_bound_ratio_max": float(
            max((v.get("local_absorb_max", 0.0) / v.get("local_fp_bound_max", float("nan"))
                 for _, v in ok_rows
                 if isinstance(v.get("local_fp_bound_max"), float)
                 and v.get("local_fp_bound_max") > 0), default=float("nan"))
        ),
        "out_scale_max": float(_agg("out_scale_max")),
        "e2e_max_err": e2e_max,
        "e2e_mse": e2e_mse,
        "e2e_rel_max": e2e_max / max(logits_scale, 1e-30),
        "logits_scale_max": logits_scale,
        "unit_roundoff": float(torch.finfo(dtype).eps) / 2.0,
        # 执行配置: FP32 那一档的一阶界只在 TF32 关闭时才成立
        "tf32_matmul_enabled": bool(torch.backends.cuda.matmul.allow_tf32) if torch.cuda.is_available() else False,
        "tf32_cudnn_enabled": bool(torch.backends.cudnn.allow_tf32) if torch.cuda.is_available() else False,
        "use_cayley_neumann_forced": use_cayley_forced,
        "input_fingerprint": input_fp,
    }
    return summary, per_layer


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OFT 非恒等恢复验证")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--precisions", nargs="+", default=["fp32", "fp32_tf32", "fp64"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = OFTAuditConfig()
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.device:
        cfg.device = args.device
    cfg.seed = args.seed

    ckpt_dir = args.checkpoint_dir
    if ckpt_dir is None:
        ckpt_dir = os.path.join(cfg.output_dir, "adapter_checkpoint")

    os.makedirs(cfg.output_dir, exist_ok=True)
    _set_deterministic(cfg.seed)

    # 固定完整初始模型一次 (含随机分类头), 各精度从它 deepcopy, 保证跨精度模型一致
    base_model, _ = _load_base_model(cfg)

    # 固定输入 (确定性)
    rng = torch.Generator().manual_seed(cfg.seed)
    fixed_input = torch.randn(8, 3, 224, 224, generator=rng)
    input_fp = _tensor_fingerprint(fixed_input)

    # 文件指纹 + 版本
    adapter_path = os.path.join(ckpt_dir, "adapter_model.safetensors")
    adapter_sha = _file_sha256(adapter_path) if os.path.isfile(adapter_path) else "MISSING"
    import transformers, peft  # torch 已在顶层导入, 不可在此重复 import (会变成局部变量)
    peft_ver = peft.__version__
    tf_ver = transformers.__version__
    torch_ver = torch.__version__
    # 强制精确 Cayley 的判定 (用于报告)
    _, _, use_cayley_forced = _build_oft_config(ckpt_dir)

    print("=" * 64)
    print("OFT 非恒等恢复验证 (独立进程, 从磁盘重新加载)")
    print("=" * 64)
    print(f"  base: {cfg.base_model}")
    print(f"  adapter SHA256: {adapter_sha}")
    print(f"  peft={peft_ver} transformers={tf_ver} torch={torch_ver}")
    print(f"  强制 use_cayley_neumann=False: {use_cayley_forced}")
    print(f"  固定输入指纹: {input_fp[:16]}...")

    summaries = {}
    per_layer_all = {}
    for prec in args.precisions:
        print(f"\n--- 精度={prec} (从固定初始模型 deepcopy 恢复) ---")
        summary, per_layer = run_recovery_once(
            cfg, ckpt_dir, prec, fixed_input, input_fp, cfg.base_model,
            adapter_sha, peft_ver, tf_ver, torch_ver, use_cayley_forced,
            base=base_model,
        )
        summaries[prec] = summary
        per_layer_all[prec] = per_layer
        ncmp = summary["param_file_n_compared"]
        print(f"  加载参数与文件逐元素一致: {summary['param_file_all_match']} "
              f"(比对 {ncmp} 个张量, max={summary['param_file_maxdiff']:.3e}, "
              f"missing={summary['param_file_n_missing']}, shape_mismatch={summary['param_file_n_shape_mismatch']})")
        print(f"  adapter 参与计算: {summary['adapter_active']} (disable vs enable max={summary['adapter_participates_maxdiff']:.3e})")
        print(f"  未合并状态可判定: {not summary['merged_state_unknown']}"
              f" (any_merged_flag={summary['any_merged_flag']})")
        print(f"  有效旋转 ‖R-I‖_F: min={summary['R_dev_fro_min']:.4e} mean={summary['R_dev_fro_mean']:.4e} max={summary['R_dev_fro_max']:.4e}")
        print(f"  合并实现校验 Wm vs Q^T·W0 (不经求逆) max={summary['merge_impl_max']:.3e}")
        print(f"  Cayley 交叉验证(经 pinv 反解) max={summary['cayley_cross_max']:.3e}  W0 块条件数 max={summary['W0_block_cond_max']:.3e}")
        print(f"  局部吸收 max={summary['local_absorb_max']:.3e} "
              f"(相对 {summary['local_absorb_rel_max']:.3e}, 层={summary['local_absorb_worst_layer']})")
        if prec == "fp32_tf32":
            print(f"    浮点一阶界(Higham, 基于 FP32 单位舍入)={summary['local_fp_bound_max']:.3e} "
                  f"实际/界={summary['local_fp_bound_ratio_max']:.3e} "
                  f"(TF32 下该界失效, 仅作参考, 不据此判定)")
        else:
            print(f"    浮点一阶界(Higham γ_n·|x||W|)={summary['local_fp_bound_max']:.3e} "
                  f"实际/界={summary['local_fp_bound_ratio_max']:.3e} "
                  f"=> {'在界内' if summary['local_within_fp_bound_all'] else '超出界!'}")
        print(f"  端到端 max={summary['e2e_max_err']:.3e} mse={summary['e2e_mse']:.3e} "
              f"(相对 {summary['e2e_rel_max']:.3e}, logits 尺度 {summary['logits_scale_max']:.3e})")
        if prec != "fp64" and (summary["tf32_matmul_enabled"] or summary["tf32_cudnn_enabled"]):
            print("  [!] 本档启用了 TF32, 上面基于 FP32 单位舍入的一阶界不再成立, "
                  "请关闭 TF32 后重跑 (脚本默认已关)。")
        if summary["R_dev_fro_max"] < 1e-4:
            print("  [!] 有效旋转偏离 < 1e-4 => 近似恒等, 无法证明非恒等恢复!")
        if summary["local_absorb_rel_max"] > summary["R_dev_fro_max"]:
            print("  [!] 相对吸收误差超过旋转偏离量, 量纲异常, 需排查。")

    report = {
        "experiment": "non_identity_recovery",
        "base_model": cfg.base_model,
        "adapter_sha256": adapter_sha,
        "adapter_config_sha256": _file_sha256(os.path.join(ckpt_dir, "adapter_config.json")),
        "peft_version": peft_ver,
        "transformers_version": tf_ver,
        "torch_version": torch_ver,
        "use_cayley_neumann_forced": use_cayley_forced,
        "input_fingerprint": input_fp,
        "summaries": summaries,
        "per_layer": per_layer_all,
    }
    out = os.path.join(cfg.output_dir, "nonidentity_recovery_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\n报告已保存: {out}")

    # ================= 证据链判定 =================
    # 分成两层独立声明, 避免把"权重是否被正确加载"和"吸收是否精确"混为一谈:
    #   (I) 恢复证据链   : checkpoint 来源 -> 参数 -> adapter 参与 -> 有效旋转非恒等
    #   (II) 吸收等价性  : 残差落在浮点一阶界内 (诊断性上界, 非完整证明), 且随精度下降。
    #        Higham 界只说明"残差不超过有限精度量级", 与代数等价定理 + 跨精度缩放三者
    #        共同支持"吸收在精确算术下等价"; 单凭界内不能断言"全部误差都来自有限精度"。
    ok_all = True
    for prec, s in summaries.items():
        cond = {
            "参数与文件逐元素一致": bool(s["param_file_all_match"]),
            "adapter 实际参与前向": bool(s["adapter_active"]),
            "有效旋转非恒等 (‖R-I‖_F > 1e-4)": bool(s["R_dev_fro_max"] > 1e-4),
            "未合并模型确认未合并 (状态可判定且非合并)": bool(
                not s["merged_state_unknown"] and not s["any_merged_flag"]),
        }
        # 浮点一阶界仅对关闭 TF32 的档成立 (TF32 下 matmul 有效单位舍入更大, 该界失效)。
        # 因此 fp32_tf32 档不把"局部吸收在界内"作为通过/失败条件, 避免用失效的界误判。
        if prec != "fp32_tf32":
            cond["局部吸收残差在浮点一阶界内"] = bool(s["local_within_fp_bound_all"])
        missing = [k for k, v in cond.items() if not v]
        if missing:
            ok_all = False
            print(f"  [{prec}] 未通过项: {', '.join(missing)}")
        else:
            print(f"  [{prec}] 全部通过")

    # 合并实现忠实性: 只在最高精度下判定 (低精度下该项被权重存储舍入主导, 不适合做判据)
    hi = "fp64" if "fp64" in summaries else max(
        summaries, key=lambda p: {"fp16": 0, "fp32": 1, "fp64": 2}.get(p, 1)
    )
    merge_faithful = bool(summaries[hi]["merge_impl_max"] < 1e-12)
    print(f"  合并实现忠实 (最高精度={hi}, Wm vs Q^T·W0 = {summaries[hi]['merge_impl_max']:.3e}): "
          f"{'PASS' if merge_faithful else 'FAIL'}")

    # 跨精度: 残差是否随精度下降 => 归因有限精度而非实现差异
    scaling_ok = True
    scaling_note = "仅单一精度, 未做跨精度对照"
    if "fp32" in summaries and "fp64" in summaries:
        l32 = max(summaries["fp32"]["local_absorb_max"], 1e-300)
        ratio = summaries["fp64"]["local_absorb_max"] / l32
        scaling_ok = bool(ratio < 1e-6)
        scaling_note = (f"局部误差 FP64/FP32 = {ratio:.3e} "
                        f"({'下降 >1e6 倍' if scaling_ok else '未下降, 需排查'})")
    print(f"  跨精度缩放: {scaling_note}")

    chain_ok = ok_all and merge_faithful
    print("=" * 64)
    print("(I) 非恒等恢复:", "PASS" if chain_ok else "INCOMPLETE — 见上方未通过项")
    print("(II) 吸收代数等价:",
          "PASS — 残差在有限精度一阶界内, 与吸收代数等价定理及跨精度缩放一致 "
          "(浮点界为诊断性上界, 不与定理叠加作为'误差全来自有限精度'的完整证明)"
          if (chain_ok and scaling_ok)
          else "INCOMPLETE — 见上方未通过项")
    print("=" * 64)


if __name__ == "__main__":
    main()
