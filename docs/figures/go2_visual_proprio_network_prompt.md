# 网络结构图生成提示词

使用方式：内置 image_gen 工具，非 CLI。最终图保留在同目录 PNG。

## 1. 最终底图生成

```text
Use case: infographic-diagram.
Create a high-resolution landscape 16:9 technical neural-network architecture infographic suitable for a Chinese robotics research PPT. Aim for 3840x2160 clarity. White background, flat vector-like boxes, subtle rounded corners, crisp dark text, thin orthogonal arrow routing, ample whitespace. Primary labels Simplified Chinese with standard English neural-network terms and tensor shapes. No photorealism, no decorative brain illustrations, no invented extra modules.
Title exactly: "Go2 视觉与本体感知融合网络"
Subtitle exactly: "学生策略结构设计｜输入 → 特征编码 → 拼接融合 → 12关节控制"
Small footer exactly: "设计稿 · 尚未训练验证；图中省略批量维 B"
Overall layout: two spacious horizontal color-coded network lanes occupying upper 75%, then a clear control flow strip across the bottom. Blue for proprioception, purple for visual perception, orange for fusion/action, green for control. Make network layers prominent and legible. Avoid crossed arrows. Every listed numerical shape must be correct. The diagram is specifically the student actor, do not add a critic or teacher.

TOP BLUE LANE, labelled "本体感知分支 · 50 Hz":
Input box "当前本体 p_t [45]" with compact items:
"角速度3 + 重力方向3 + 速度指令3"
"关节位置12 + 关节速度12 + 上一执行动作12"
Show three distinct routes:
(1) a direct arrow from p_t to the right fusion column, labelled "当前反馈 45".
(2) p_t into "10步历史 [10,45]" -> "Flatten 450" -> "本体 MLP" with exact layer sequence "450 → 256 → 128 → 64" and note "ELU；末层 Tanh". Output box "z_p [64]" -> fusion.
A branch ONLY from z_p to "线速度估计头 64 → 32 → 3" with note "ELU → Linear" -> output "v_est [3]" -> fusion.
(3) a thin, clearly labelled route from current p_t to the visual temporal fusion box, labelled "采图时 p_t [45]". It must not go into the CNN.

BOTTOM PURPLE LANE labelled "视觉分支 · 10 Hz":
A small grayscale staircase depth thumbnail labelled "深度示意 H×W=48×64".
Then "米制校验 / 无效值处理 / 固定归一化" -> "最近两帧 [2,48,64]" -> a substantial CNN card containing these rows, exactly:
"Conv1: 2→16, k5 s2 p2  → [16,24,32]"
"Conv2: 16→32, k3 s2 p1 → [32,12,16]"
"Conv3: 32→32, k3 s2 p1 → [32,6,8]"
"Flatten: 1536"
"FC: 1536→128→64"
"CNN隐藏层 ELU；末层 Tanh"
The CNN outputs "深度特征 [64]".
Arrow to a small "Concat 64+45=109" node receiving also the thin "采图时 p_t [45]" route from the top.
Then "GRUCell 输入109 / 隐状态128" with a small feedback arrow labelled "h_(k-1) [128] → h_k [128]".
Then "Linear 128→64 + Tanh" -> "z_v [64]" -> fusion.
Small note under GRU: "新深度帧更新；其余控制步保持特征；reset清零hidden".
A separate small neutral box "帧年龄 age [1] + 有效标志 valid [1]" routes to the fusion column and not to the CNN.

RIGHT FUSION COLUMN, five clearly labelled input partitions:
"p_t 45"
"z_p 64"
"z_v 64"
"v_est 3"
"age + valid 2"
All arrows go into one orange fusion box with exact text:
"Concat 拼接融合"
"45 + 64 + 64 + 3 + 2 = 178"
"沿特征维拼接，不做逐元素相加"
Then arrow to a visibly layered orange "动作 MLP" card:
"178 → 256 → 128 → 12"
"隐藏层 ELU；均值末层 Tanh"
Output exactly "μ [12]".
Near μ include small side box "log_std [12] 可训练" feeding the next decision only.
Decision/small bifurcation:
"训练：a_raw ~ Normal(μ,σ)"
"评估：a_raw = μ"
merge to bottom control strip.
If needed put this action column in the rightmost 25% with vertically stacked cards; don't omit any feature input.

BOTTOM GREEN CONTROL STRIP, large readable sequential boxes left-to-right:
"a_exec = clip(a_raw,-1,1)"
→ "q_target = q_home + 0.5 a_exec"
→ "按关节位置限位截断"
→ "PD · 500 Hz / kp=40, kd=2 / 力矩限幅"
→ small quadruped schematic "Go2 · 12关节".
Under the strip clearly state:
"FL / FR / RL / RR，每腿 hip / thigh / calf"
"动作头50 Hz；一个目标保持10个物理步"
Optionally a slim outside arrow from Go2 back to proprio input labelled "状态反馈", but do not clutter layer connections.
Critical invariants: fusion 178; CNN output 64; GRU input109 hidden128; velocity comes from proprio feature64; direct45 feedback must exist; no true simulated base velocity enters actor; no gait phase input, no critic/teacher; control output exactly12. Maintain accurate legible Chinese typography.

IMPORTANT ROUTING OVERRIDE: Draw a NEW, corrected version of the architecture. Use numbered connector ports instead of long inter-branch wires for the five final fusion inputs. This is essential because direct wires cross the lanes and can falsely point to the wrong feature.

Do NOT draw any long wire from p_t, z_p, z_v, v_est, or the frame metadata box into the rightmost input stack. Instead:
- The direct p_t output ends with a short blue rightward arrow to a circle "①".
- z_p output has a short blue rightward arrow to a circle "②", plus its local branch into the velocity estimation head.
- v_est output has a short blue rightward arrow to a circle "④".
- z_v output has a short purple rightward arrow to a circle "③".
- the age+valid metadata source has a short neutral rightward arrow to a circle "⑤".
On the RIGHT fusion panel, show exactly five independent rows in this exact order:
"① 当前本体 p_t [45]"
"② 本体历史 z_p [64]"
"③ 视觉特征 z_v [64]"
"④ 线速度估计 v_est [3]"
"⑤ 帧状态 age+valid [2]"
Each of these five rows has its own short rightward arrow into the orange "Concat 拼接融合" node.
A small legend MUST read "同编号端口相连，省略跨分支长连线".
Use matching circle numbers and matching colors to specify the connections. No wires in the gap between the main blue/purple lanes and this right fusion stack. Do not connect a blue feature to a purple feature.

Keep the ONE thin blue route from p_t to the visual "Concat 64+45=109", because it is a separate real input to the temporal encoder. That temporal Concat must have exactly TWO inputs: CNN64 from the left and captured p_t45 from above.
The age+valid source must NOT connect to that temporal Concat, to GRU, or to Gaussian sampling; it only ends at port⑤.
The Gaussian sampling box must receive ONLY μ and log_std. It must not receive any age/valid wire.
The bottom green control strip must have ONLY the left-to-right feedforward chain clipping -> q_target -> joint limits -> PD -> Go2. NO bottom feedback wire and NO arrow entering the clipping box from Go2. Add only a plain text footer "状态反馈进入下一周期本体输入；设计稿，尚未训练验证".
Every other parameter, layer shape and input field in the main specification remains unchanged. Make the numbered fusion rows as legible as the neural-network modules.
```

## 2. 动作输出连线修正

```text
Precisely edit ONLY ONE connection in this network architecture diagram. Preserve all five numbered ports, all network boxes, layer sizes, text, colors, images, and their positions.

There is an incorrect orange vertical arrow from the Gaussian sampling/evaluation box straight down to the Go2 robot area. Remove that arrow completely. Raw action must enter the clipping operation FIRST, never enter the robot directly.

Draw one new orange orthogonal connector from the BOTTOM CENTER of the Gaussian sampling box:
1. Go down into the narrow horizontal white gutter immediately ABOVE the green control panel.
2. Go LEFT across that white gutter.
3. Go down along the OUTSIDE LEFT EDGE of the green panel.
4. Turn RIGHT and end with an arrowhead entering the LEFT SIDE of the FIRST control box labelled "a_exec = clip(a_raw,-1,1)".
Label this orange connector "a_raw [12]" in a clear portion of the horizontal gutter.
If the white gutter is too narrow, add a little whitespace between the upper panels and lower green panel, preserving all other content.
This connector must have exactly ONE arrowhead, pointing INTO the CLIPPING box at bottom left. Do not point it at Go2, PD, joint limits, or q_target. Do not leave any residual vertical orange arrow above Go2.
Keep the green feedforward chain unchanged: clipping -> q_target -> joint limits -> PD -> Go2.
Keep every other existing correct diagram element unchanged.
```

## 3. 清理绕过采样框的多余线段

```text
Make a very small technical cleanup to this diagram. Keep EVERY box, every label, all five numbered ports, all other arrows, and the long orange a_raw route into bottom-left clipping unchanged.

The thin orange vertical line near x=1540 currently extends upward from the raw-action routing gutter into the action-network area. Remove ONLY its upper segment between the BOTTOM of the "动作 MLP" box and the BOTTOM edge of the dashed Gaussian training/evaluation box. There must be no bypass line running from the MLP straight to the control strip.

The long orange "a_raw [12]" route must START at the bottom border of the dashed "训练: a_raw ~ Normal(μ,σ) / 评估: a_raw = μ" box, then go down to the white gutter, left, and into the clipping box. Its only origin is the sampling/evaluation box. No orange line should extend above that box except the existing short μ and log_std arrows INTO it.

Do not alter any text or network parameter or layout. This is one line-segment removal and connection cleanup only.
```

