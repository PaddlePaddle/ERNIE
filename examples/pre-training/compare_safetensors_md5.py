#!/usr/bin/env python3
import os
import sys
import glob
import hashlib
from pathlib import Path
import paddle
from safetensors.paddle import load_file
import re


SUFFIX_WHEN_NO_MODEL = (
    "beta1_pow_acc_0",
    "beta2_pow_acc_0",
    "moment1_0",
    "moment2_0",
    ".w_0",
)


def md5_tensor(t: paddle.Tensor) -> str:
    return t._md5sum()


def _transform_key_by_filename(file_path: str, key: str) -> str:
    """
    与 print_safetensors.py 对齐的键规范：
    - master* 开头：在键后追加 "_w0"
    - optimizer* 开头：将键内的 '/' 全部替换为 '.'
    - 其它（如 model*）：不做处理
    """
    name = os.path.basename(file_path)
    if name.startswith("master"):
        return key.replace("/", ".") + ".w_0"
    if name.startswith("optimizer"):
        return key.replace("/", ".")
    return key


def _add_entry(flat: dict, key: str, val):
    """
    如果 val 是 Tensor，直接加入；否则按 dict 递归展开为扁平 key（用点号拼接）。
    """
    if isinstance(val, paddle.Tensor):
        flat[key] = val
        return
    if isinstance(val, dict):
        for sk, sv in val.items():
            _add_entry(flat, f"{key}.{sk}", sv)
        return
    try:
        flat[key] = paddle.to_tensor(val)
    except Exception as e:
        raise TypeError(f"无法将当前值转换为Tensor，key={key}，val类型={type(val)}，错误={e}")


def _dir_has_model(dir_path: str) -> bool:
    return len(glob.glob(os.path.join(dir_path, "**", "model-*.safetensors"), recursive=True)) > 0


def _iter_safetensors_files(dir_path: str):
    return sorted(glob.glob(os.path.join(dir_path, "**", "*.safetensors"), recursive=True))


def load_dir_normalized(dir_path: str) -> dict:
    """
    加载目录下所有 .safetensors，并按文件名规范转换 key。
    返回：规范化后的 扁平化 {key: paddle.Tensor}
    """
    d = {}
    for fp in _iter_safetensors_files(dir_path):
        for k, v in load_file(fp).items():
            nk = _transform_key_by_filename(fp, k)
            _add_entry(d, nk, v)
    return d


_EXPERTS_PATTERN = re.compile(r"(\.)experts\.(\d+)(?=\.|$)")


def _extract_expert_index(key: str):
    m = _EXPERTS_PATTERN.search(key)
    if not m:
        return None
    return int(m.group(2))


def _extract_signature_and_index(key: str):
    """返回 (signature, index)；signature 为将首次出现的 .experts.<idx> 的 <idx> 抹去后的模式。
    若不存在 experts 段，返回 (None, None)。
    例：
      key = a.b.experts.3.w -> signature=a.b.experts.{X}.w, index=3
    """
    m = _EXPERTS_PATTERN.search(key)
    if not m:
        return None, None
    idx = int(m.group(2))
    start_num, end_num = m.span(2)
    signature = key[:start_num] + "{X}" + key[end_num:]
    return signature, idx


def load_dir_normalized_with_expert_replicas(dir_path: str) -> dict:
    """
    加载并规范化 safetensors，同时处理目录内出现多组完全重复的 experts.0..K 的情况：
    - 统计每个 experts 索引在目录内出现的次数，若最大次数 > 1，则认为存在多组副本
    - 记 group_size = (目录中 experts 最大索引 + 1)
    - 第 c 次(从 0 开始)出现 experts.i 时，将其重映射为 experts.(i + c*group_size)
    这样避免不同组之间 key 冲突造成的覆盖。
    """
    # 第一遍：读取并缓存文件内容，仅统计每个“签名”的 experts 索引分布
    file_to_items = {}
    signature_to_idx_count = {}
    signature_to_max_idx = {}
    for fp in _iter_safetensors_files(dir_path):
        items = load_file(fp)
        file_to_items[fp] = items
        for k in items.keys():
            nk = _transform_key_by_filename(fp, k)
            sig, idx = _extract_signature_and_index(nk)
            if sig is None:
                continue
            dcnt = signature_to_idx_count.setdefault(sig, {})
            dcnt[idx] = dcnt.get(idx, 0) + 1
            prev_max = signature_to_max_idx.get(sig, -1)
            if idx > prev_max:
                signature_to_max_idx[sig] = idx

    # 计算需要重映射的签名及各自的 group_size 和副本次数
    signature_to_replicas = {}
    for sig, idx_counts in signature_to_idx_count.items():
        replicas = max(idx_counts.values()) if idx_counts else 1
        if replicas > 1:
            signature_to_replicas[sig] = replicas

    # 若没有任何签名存在重复，直接按常规路径加载（避免额外处理）
    if not signature_to_replicas:
        return load_dir_normalized(dir_path)

    signature_to_group_size = {sig: signature_to_max_idx[sig] + 1 for sig in signature_to_replicas.keys()}

    # 第二遍：按签名+原索引的出现顺序进行分组重映射
    expert_seen_counts = {sig: {} for sig in signature_to_replicas.keys()}
    out = {}

    def _remap_key(nk: str) -> str:
        m = _EXPERTS_PATTERN.search(nk)
        if not m:
            return nk
        # 计算签名
        start_num, end_num = m.span(2)
        signature = nk[:start_num] + "{X}" + nk[end_num:]
        if signature not in signature_to_replicas:
            return nk
        group_size = signature_to_group_size.get(signature, 0)
        if group_size <= 0:
            return nk
        orig_idx = int(m.group(2))
        seen_map = expert_seen_counts[signature]
        seen = seen_map.get(orig_idx, 0)
        new_idx = orig_idx + seen * group_size
        seen_map[orig_idx] = seen + 1
        # 构造替换后的 key（仅替换第一次匹配的数字段）
        return nk[:start_num] + str(new_idx) + nk[end_num:]

    for fp in _iter_safetensors_files(dir_path):
        items = file_to_items[fp]
        for k, v in items.items():
            nk = _transform_key_by_filename(fp, k)
            rnk = _remap_key(nk)
            _add_entry(out, rnk, v)

    return out


def filter_keys_when_no_model(keys):
    """当目录无 model*.safetensors 时，只保留后缀在白名单内的键。"""
    out = []
    for k in keys:
        if any(k.endswith(suf) for suf in SUFFIX_WHEN_NO_MODEL):
            out.append(k)
    return out


def compare_dirs(dir1: str, dir2: str):
    has_model_1 = _dir_has_model(dir1)
    has_model_2 = _dir_has_model(dir2)

    d1 = load_dir_normalized(dir1)
    # 目录2使用带 experts 重映射的加载，自动处理重复 experts 分组
    d2 = load_dir_normalized_with_expert_replicas(dir2)


    keys1 = set(d1.keys())
    keys2 = set(d2.keys())

    # 如果目录2没有 model*.safetensors，则仅在“only/both”比较阶段筛选指定后缀键
    #（满足你的要求：文件2只有 master/optimizer 时，只比较白名单后缀的键）
    if not has_model_2:
        keys1 = set(filter_keys_when_no_model(keys1))
        keys2 = set(filter_keys_when_no_model(keys2))

    # 计算集合并进行比较：如果存在 only keys，则忽略它们，只比较交集键
    only1 = sorted(keys1 - keys2)
    only2 = sorted(keys2 - keys1)
    both = sorted(keys1 & keys2)

    # 仍打印 only key 列表
    mism = []
    if only1:
        print("\n[ONLY DIR1] keys:", len(only1))
        for k in only1:
            print(k)
    if only2:
        print("\n[ONLY DIR2] keys:", len(only2))
        for k in only2:
            print(k)
    if not (only1 or only2):
        # 若没有 only，继续做 md5 对比再决定是否 OK
        pass
    for k in both:
        try:
            m1 = md5_tensor(d1[k])
            m2 = md5_tensor(d2[k])
        except Exception as e:
            mism.append((k, f"读取失败: {e}", None))
            continue
        if m1 != m2:
            # 打印具体的 tensor 内容
            print(f"\n[MISMATCH] {k}:")
            print(f"  dir1 shape: {d1[k].shape}, dtype: {d1[k].dtype}")
            print(f"  dir2 shape: {d2[k].shape}, dtype: {d2[k].dtype}")
            
            # 打印 tensor 的前几个值
            print(f"  dir1 values (first 10): {d1[k].flatten()[:10].numpy()}")
            print(f"  dir2 values (first 10): {d2[k].flatten()[:10].numpy()}")
            
            # 如果形状相同，计算差异
            if d1[k].shape == d2[k].shape:
                diff = paddle.abs(d1[k] - d2[k])
                print(f"  max diff: {paddle.max(diff).item()}")
                print(f"  mean diff: {paddle.mean(diff).item()}")
            
            mism.append((k, d1[k].shape, d2[k].shape))

    if mism:
        print("\n[MISMATCH] keys:", len(mism))
        for k, s1, s2 in mism:
            print(f"{k} | dir1:{s1} != dir2:{s2}")
        print("\nmd5比较失败")
    else:
        if not (only1 or only2):
            print("\n[OK] 两侧完全一致")
        print("\nmd5比较成功")


def main():
    if len(sys.argv) != 3:
        print("用法: python compare_safetensors_dirs.py <dir1> <dir2>")
        print("  说明: 会按文件名规则规范化键；若两侧均无 model*.safetensors，则仅比较优化器/主权重相关后缀键")
        sys.exit(1)
    compare_dirs(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()


