"""重复口播片段检测核心逻辑（纯标准库，不依赖 FastAPI，便于单测）。

以每个词的 (text, speaker) 组成符号序列，查找两处词序列完全相同的连续
口播片段：两处词索引区间不重叠，且两处首词的 start_ms 至少相隔
min_gap_ms。单词文本不拆分——匹配以整词为最小单位。只返回词数最多的
一对；同长度时按两处起始词索引的二元组 (i, j)（i < j）字典序裁决。
最佳词数不足 min_words 时返回 None。

算法（沿对角线递推 LCP），O(n²) 时间、O(1) 额外空间（n ≤ 2000）：

* 对角线 d = j - i 上，lcp(i, i+d) 由 lcp(i+1, i+1+d) 加 1 递推（符号
  不同则归零），从右向左单趟扫出；
* 起点对 (i, j)（i < j）的可用长度 usable = min(lcp(i,j), j-i)：区间
  不重叠要求片段长 L ≤ j-i；另需首词时间间隔
  start_ms[j] - start_ms[i] ≥ min_gap_ms；
* usable ≤ d，故 d < 当前最优长度时更小的对角线可整体剪掉；d 从大到小
  枚举，长重复会尽早抬高最优值；
* 全程维护 (最优长度, 字典序最小起点对)，最后校验最优长度 ≥ min_words。
"""

from .segmentation import validate_words


def find_duplicate_passages(words, min_words, min_gap_ms):
    """查找词数最多的重复口播片段对，返回 dict 或 None。

    参数 words 中每个对象需具备属性：start_ms / end_ms / text / speaker
    （沿用同一套词序与时间校验）。

    返回 dict：word_count / text / first / second；first/second 各含
    start_word / end_word（0-based、含端点）与 start_ms / end_ms，
    first 为起始词索引较小的一处；text 为共同文本（词 text 原样拼接，
    不插空格）。不存在词数 ≥ min_words 的合法片段对时返回 None。
    """
    validate_words(words)
    n = len(words)
    starts = [w.start_ms for w in words]
    # 快捷判空：词数不足以容纳两段不重叠的 min_words 片段，
    # 或全序列首末词的时间间隔都达不到 min_gap_ms。
    if n < 2 * min_words or starts[-1] - starts[0] < min_gap_ms:
        return None

    tokens = [(w.text, w.speaker) for w in words]
    best_len = 0
    best_pair = None  # (i, j)，i < j
    for d in range(n - 1, 0, -1):
        if d < best_len:
            break  # 之后对角线 usable <= d < best_len，无法更优或平局
        run = 0  # lcp(i, i+d)，随 i 递减递推
        for i in range(n - 1 - d, -1, -1):
            j = i + d
            if tokens[i] == tokens[j]:
                run += 1
            else:
                run = 0
                continue
            usable = run if run < d else d
            if usable < best_len:
                continue
            if starts[j] - starts[i] < min_gap_ms:
                continue
            if usable > best_len or (i, j) < best_pair:
                best_len = usable
                best_pair = (i, j)

    if best_pair is None or best_len < min_words:
        return None

    i, j = best_pair
    end_i = i + best_len - 1
    end_j = j + best_len - 1
    return {
        "word_count": best_len,
        "text": "".join(w.text for w in words[i : i + best_len]),
        "first": {
            "start_word": i,
            "end_word": end_i,
            "start_ms": starts[i],
            "end_ms": words[end_i].end_ms,
        },
        "second": {
            "start_word": j,
            "end_word": end_j,
            "start_ms": starts[j],
            "end_ms": words[end_j].end_ms,
        },
    }
