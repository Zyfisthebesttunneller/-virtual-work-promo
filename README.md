# 虚功原理：平衡之美

一部约 2 分 47 秒的力学科普短片，面向没有学过力学的普通观众。画面由 HTML Canvas 与 JavaScript 逐帧绘制，配乐与音效由 Python 合成。视频为 1920 × 1080、30 fps，配中英双语字幕。

风格参照 [abstract-algebra-promo](https://github.com/AnctyEnly453/abstract-algebra-promo)，沿用了它的整体做法：90 BPM 的小节时间线、冷开场闪切、金色粒子标题、罗马数字章节卡、放射光揭示、逐拍蒙太奇终章，以及管风琴、合唱、钢片琴与太鼓的配乐。

## 内容结构

| 段落 | 内容 |
| --- | --- |
| 冷开场 | 闪切预告；提问"它们凭什么一动不动？"；粒子拼成杠杆，并"想象地动一下" |
| 标题 | 虚功原理 · THE PRINCIPLE OF VIRTUAL WORK |
| I 平衡 | 传统做法：画出所有力，逐个列方程；新问法：假如它动一下。2 × 1 = 1 × 2，两块金色面积重合 |
| II 虚位移 | 想象中的、无穷小的、遵守约束的移动 δr；功 = 力 × 顺着力的位移 |
| III 原理 | 剪叉升降台配实时"虚功账本"：任意推动下 Σ δW = 0；约束力不做功，从方程里消失 |
| IV 机械 | 滑轮组 25 × 4 = 100 × 1；齿轮与液压机；力 × 距离的等面积曲线 |
| V 能量地形 | 平衡点即能量地形上的平坦处；稳定与不稳定；悬链线翻转成拱；高迪的倒挂链模型 |
| VI 从平衡到万物 | 达朗贝尔（1743）、拉格朗日《分析力学》（1788）、有限元法 |
| 终章 | 一根杠杆、一组滑轮、一条悬链、一座大桥 → 想象它动一下，就会明白它为什么不动 |

全片的颜色语言：橙色代表力，青色全息代表想象中的移动，金色面积代表功，洋红代表最终消失的约束力。

## 文件

| 文件 | 用途 |
| --- | --- |
| `promo.html` | 动画与播放器源码（播放时需与 `music.mp3` 放在同一目录） |
| `虚功原理宣传片.html` | 单文件播放器，配乐已内嵌，字体需联网加载 |
| `music.py`、`cues.json` | 配乐与音效合成；`cues.json` 为画面导出的音效时间点 |
| `export.py` | 用无头 Chromium 逐帧渲染，并交给 FFmpeg 编码 |
| `build.py` | 把源码和 MP3 打包成单文件播放器 |
| `music.mp3` | 最终配乐（−14 LUFS） |

## 重新生成

需要 Python 3（numpy、scipy、playwright 及其 Chromium）和 FFmpeg。离线导出时，`export.py` 会把 Google Fonts 换成本地字体：Latin 字体来自 npm 上的 `@fontsource/cinzel`、`@fontsource/cormorant-garamond` 和 `@fontsource/stix-two-text`，路径在脚本顶部的 `FONTS` 中设置；中文使用系统中的 Noto Serif CJK SC。

```bash
python export.py --cues cues.json          # 音效时间表
python music.py                            # -> music.wav, music.mp3
python build.py                            # -> 单文件播放器
python export.py --from 0 --to 30 --out seg/s0.mp4    # 分段渲染，再用 ffmpeg concat 拼接
python export.py --shots 19.5,80,124       # 只导出检查帧
```

画面全部使用 Canvas 2D，泛光、暗角和颗粒也在 Canvas 中完成，因此浏览器播放与离线导出的画面一致。
