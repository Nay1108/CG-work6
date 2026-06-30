# 可微渲染实验报告
#### 王赛楠 202411998177 
## 一、实验概述

本实验通过Taichi框架实现了基于光线投射（Ray Casting）的可微渲染管线，并利用自动微分技术对三维场景中的光源位置进行反向优化。实验核心在于构建"渲染图像→计算误差→误差反传→更新参数"的完整闭环，并深入探究了可微渲染中的梯度消失问题及其解决方案。

## 二、实验原理分析

### 2.1 正向渲染管线

正向渲染管线针对屏幕上的每个像素，发射一条射线并计算其与场景中球体的交点。具体流程如下：

1. **射线-球体相交检测**：对于屏幕坐标$(x, y)$，计算该点对应的三维空间射线与球体的交点
2. **法线计算**：在交点处计算球面法向量 $\mathbf{n}$
3. **光照计算**：基于光源位置计算光照强度

### 2.2 可微渲染的梯度挑战

在标准Lambertian漫反射模型中：
$$I = \max(0, \mathbf{n} \cdot \mathbf{l})$$

当 $\mathbf{n} \cdot \mathbf{l} \le 0$ 时，光照强度被截断为0，梯度严格为0。这导致如果初始光源位于球体背面，优化过程将完全停滞——这就是典型的**梯度消失问题**。

### 2.3 Adam优化器

实验采用Adam优化器，其特点包括：
- **动量机制**：加速收敛，平滑更新轨迹
- **自适应学习率**：针对每个参数调整学习步长
- **偏差校正**：解决初期动量估计的偏差问题

更新公式：
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$
$$\hat{m}_t = m_t/(1-\beta_1^t), \quad \hat{v}_t = v_t/(1-\beta_2^t)$$
$$\theta_{t+1} = \theta_t - \eta \cdot \hat{m}_t/(\sqrt{\hat{v}_t} + \epsilon)$$

## 三、实验实现

### 3.1 场景配置

- **球体**：半径0.3，中心位置(0.5, 0.5, 0.5)
- **目标光源**：位置(0.8, 0.8, 0.2)
- **初始光源**：位置(0.2, 0.2, 0.8)（位于球体背面）
- **渲染分辨率**：256×256

### 3.2 核心代码实现

#### 目标图像生成
```python
@ti.kernel
def generate_target():
    for i, j in target_pixels:
        # 计算屏幕坐标对应的三维空间点
        x = (i + 0.5) / res
        y = (j + 0.5) / res
        # ... 相交检测和法线计算 ...
        # 使用标准Lambertian模型生成目标图像
        target_pixels[i, j] = ti.max(0.0, ti.min(1.0, dot_val))
```

#### 可微渲染与损失计算
```python
@ti.kernel
def render_and_compute_loss():
    for i, j in target_pixels:
        # ... 场景相交检测 ...
        # Leaky Lambertian光照模型
        intensity = ti.max(0.1 * dot_val, dot_val)
        # MSE损失计算（保留梯度）
        diff = intensity - target_pixels[i, j]
        loss[None] += (1.0 / (res * res)) * (diff ** 2)
```

#### 优化循环
```python
with ti.ad.Tape(loss=loss):
    render_and_compute_loss()

# Adam参数更新
for c in range(3):
    m[c] = beta1 * m[c] + (1 - beta1) * grad[c]
    v[c] = beta2 * v[c] + (1 - beta2) * grad[c] * grad[c]
    m_hat = m[c] / (1 - beta1**iter)
    v_hat = v[c] / (1 - beta2**iter)
    light_pos[None][c] -= lr * m_hat / (math.sqrt(v_hat) + eps)
```

### 3.3 可视化设计

GUI界面左右并排显示：
- **左侧**：固定不变的目标图像（Ground Truth）
- **右侧**：动态更新的当前渲染图像

这种设计直观展示了优化过程中光源位置的演变及其对渲染效果的影响。

## 四、实验结论

1. **可微渲染的成功实现**：通过Taichi的自动微分机制，成功构建了完整的可微渲染管线，实现了光源位置的端到端优化。

2. **梯度消失问题的有效解决**：Leaky Lambertian模型通过在背光面保留微弱梯度，有效解决了标准渲染模型中的梯度消失问题，使优化算法能够在全参数空间内有效搜索。

3. **Adam优化器的优势**：Adam的自适应学习率和动量机制显著加速了收敛过程，使得高维参数空间中的优化更加稳定高效。

4. **可视化验证**：实时可视化界面直观展示了优化过程，左侧目标图像与右侧当前渲染图像的逐渐重合，验证了算法的有效性。

## 五、运行录屏

<img width="800" height="450" alt="work6-ezgif com-video-to-gif-converter" src="https://github.com/user-attachments/assets/f59cea95-a097-4f2f-9f95-7aa44738fa3b" />


## 六、心得体会

通过本次实验，我深入理解了可微渲染的核心原理及其在逆渲染问题中的应用价值。特别是梯度消失问题的分析及其解决方案，让我认识到在可微编程中保持梯度连续传导的重要性。同时，Adam优化器的实际应用也加深了我对优化算法的理解。

实验过程中，Leaky机制的设计精巧而有效，展示了在传统计算机图形学与深度学习交叉领域中，如何通过微小的算法改进来解决根本性的梯度传播问题。这为我今后在可微图形学、逆渲染等领域的研究提供了宝贵经验。
