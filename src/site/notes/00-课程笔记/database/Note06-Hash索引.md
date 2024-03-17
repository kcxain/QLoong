---
{"dg-publish":true,"permalink":"/00-课程笔记/database/Note06-Hash索引/","title":"Note06-Hash 索引详解"}
---


# Note06-Hash 索引详解

相关概念：

- 桶
  - 有 M 个桶，每个桶是有相同容量的存储地（可以是内存页，也可以是磁盘块）
- 散列函数
  - h(k)，可以将键值 k 映射到 {0, 1, ... , M-1} 中的某个值
- 将具有键值 k 的记录 Record(k) 存储在对应 h(k) 编号的桶中

![](https://kkcx.oss-cn-beijing.aliyuncs.com/img/image-20230524144111560.png)

内存数据可以采用散列确定存储页，主文件可以采用散列确定存储块，索引亦可以采用散列确定索引项的存储块。

一个桶可以是一个存储块，也可以是若干个连续的存储块。

## 可扩展散列索引

这是一种动态散列索引。基本原理：

- 为桶引入一个间接层，即用一个指向块的指针数组来表示桶
- 指针数组能增长，其长度总是 2 的幂。因而数组每增长一次，桶的数目就加倍。
- h 为每个键计算出一个 K 位二进制序列。桶的总是使用从序列第一位或最后一位算起的 i 位，即当 i 是使用的位数时，桶数组将有 $2^i$ 项（从 $2^0,2^1...$ 逐渐增加到 $2^K$ 个）

### 举例

设 K=4

- 初始时，i=1，设当前格局为：

  ![](https://kkcx.oss-cn-beijing.aliyuncs.com/img/image-20230524145457319.png)

- 插入 1110，有空间，则存储

  ![](https://kkcx.oss-cn-beijing.aliyuncs.com/img/image-20230524145610956.png)

- 插入 1010，已满，需要扩展散列桶，进行分裂：i 增加 1，重新散列该块的数据到两个快中，其他不变

  ![](https://kkcx.oss-cn-beijing.aliyuncs.com/img/image-20230524145755356.png)

- 插入 0000

  ![](https://kkcx.oss-cn-beijing.aliyuncs.com/img/image-20230524145850146.png)

- 插入 0101，需要分裂桶

  ![](https://kkcx.oss-cn-beijing.aliyuncs.com/img/image-20230524145954022.png)

### 问题

- 当桶数组需要翻倍时，要做大量的工作
- 还有如块中存放记录很少，但是记录的前 20 位都一样，那么就需要 100 完个桶数组，并且它们全都指向同一个位置

## 线性散列索引

桶数 n 的选择总是使存储块的平均记录数与存储块所能容纳的记录总数成一个固定比例，超过此比例，则桶数增长一块，分裂。

用来做桶数组项序号的二进制位数是 $\lceil \text{log}_2n\rceil$，从位序列的右端开始取（低位）

假定散列函数值的 i 位正在用来给桶数组项编号，且有一个键值为 K 的记录想要插入到编号为 $a_1a_2\cdots a_i$ 的桶中，把它视作二进制整数，设值为 m

- 如果 $m < n$，正常插入
- 如果 $n\le m < 2^i$，那么桶还不存在，把记录存在 $m-2^{i-1}$，也就是把 $a_1$ 改为 0 时对应的桶。

### 举例

- 设当前格局为：

  ![](https://kkcx.oss-cn-beijing.aliyuncs.com/img/image-20230524153113792.png)

- 插入 0101，因为 i=1，取最右边一位：

  ![](https://kkcx.oss-cn-beijing.aliyuncs.com/img/image-20230524153344379.png)

- 此时 r/n=2>1.7，需要分裂，桶数增加 1，10 桶要由 00 桶分裂而来：

  ![](https://kkcx.oss-cn-beijing.aliyuncs.com/img/image-20230524153645027.png)

- 插入 0001，01 桶已满，加一个溢出块

  ![](https://kkcx.oss-cn-beijing.aliyuncs.com/img/image-20230524153748485.png)

- 插入 0111，后两位 11 并不存在，则将最高位的 1 变为 0，插入 01 桶：

  ![](https://kkcx.oss-cn-beijing.aliyuncs.com/img/image-20230524153847320.png)

- 此时 r/n=2 > 1.7，增加 11 桶，从 01 桶分裂而来

  ![](https://kkcx.oss-cn-beijing.aliyuncs.com/img/image-20230524153931864.png)
