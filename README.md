# Translation-Based Attributed Network Embedding

TransANE的实现版本

## 使用方法

```python
cd Alg
python main.py -options
```

#### 基本参数选择介绍

- --data，输入数据集名称，以.mat形式存储，位置在./Alg/data下，初始值设为Haverford，原有数据集选项：[Swarthmore,Haverford]；

- --dimensions，调整重构非零元素的权重参数，初始值设置为256；

- --nodewalklen，节点随机游走的步长，初始值设为40；

- --nodewinsize，节点滑动窗口的大小，初始值设为5；

- --attrwalklen，属性值随机游走的步长，初始值设为40；

- --attrwinsize，属性值滑动窗口的大小，初始值设为5；

- --walknum，随机游走的次数，初始值设为10；

- --p，返回超参数；

- --q，进出超参数。


#### 输入数据格式

以文件夹形式存储，包含四个文件：attr_info.txt，doublelink.edgelist，group.txt，init.emb
- attr_info.txt：节点属性文件，格式为node_id 属性类别1_属性值 属性类别2_属性值 属性类别3_属性值 ...；
- doublelink.edgelist：连边文件，格式为head_id tail_id,以空格间隔；
- group.txt：节点label文件，每一行表示一个id的label；
- init.emb：节点embedding的初始化文件，以deepwalk pre-train产生的embedding作为初始化embedding。

#### 输出数据格式

以.mat格式存储，位于./Alg/emb下，以“数据集.emb.mat"命名，包含三个矩阵：node，attr，value

- node：节点的隐式表示矩阵，每一行对应一个节点的隐式向量，大小为节点个数*表示维度；
- attr：属性类别的隐式表示矩阵，每一行对应一个属性类别的隐式向量，大小为属性类别个数*表示维度；
- value：属性值的隐式表示矩阵，每一行对应一个属性值的隐式向量，大小为属性值个数*表示维度。

#### 主要源文件介绍

- main.py：主函数；
- graph.py：图数据预处理模块；
- train.py：训练函数与评估函数。

#### 评估函数

- node_classify()：节点分类实验；
- network_visualization(): 网络可视化实验。
