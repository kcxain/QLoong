---
{"dg-publish":true,"permalink":"/00-课程笔记/Compilers/Note4-中间代码生成/","title":"Note4- 中间代码生成"}
---


# Note4- 中间代码生成

在这一部分，我们将学习各类语句的翻译，包括声明语句，赋值语句，控制语句，witch 语句，过程调用语句等。

## 声明语句

声明语句翻译的主要任务是收集标识符的类型等属性信息，并为每一个名字分配一个相对地址。

### 类型表达式

首先，基本类型，如 integer, real, char, boolean 等是**类型表达式**，**类型构造符**可以作用于类型表达式，其结果也是类型表达式

- 数据构造符 array
  - 若 T 是类型表达式，则 array(I, T) 是类型表达式
- 指针构造符 pointer
  - 若 T 是类型表达式，则 pointer(T) 是类型表达式，表示指向该类型的指针
- 笛卡尔积构造符 $\times$
  - 若 $T_1,T_2$ 是类型表达式，则 $T_1\times T_2$ 是类型表达式
- 函数构造符 $\rightarrow$
  - $T_1\times T_2 \times \cdots T_n \rightarrow R$ 是用来表示函数的类型表达式
- 记录构造符 $record$
  - 若有标识符 $N_1, N_2,\cdots ,N_n$ 与类型表达式 $T_1, T_2,\cdots,T_n$，则
    $record((N_1\times T_1)\times(N_2\times T_2)\times\cdots\times(N_n\times T_n))$ 是一个类型表达式，它可以表示结构体等

举例：

![](https://kkcx.oss-cn-beijing.aliyuncs.com/img/image-20230505111131074.png)

### 声明语句的翻译

声明语句的翻译就是解决开头提到的两个问题：

- 收集标识符的类型等属性信息
- 为每一个名字分配一个相对地址

我们从类型表达式可以知道该类型在运行时刻所需要的存储单元数量（类型的宽度）。在编译时刻，可以使用类型的宽度为每一个名字分配一个相对地址。

因此我们为每个符号在符号表中设置两个字段：类型和相对地址。

### 变量声明语句的 SDT

设置 enter 函数，enter(name, type, offset) 表示在符号表中为名字 name 创建记录，将 name 的类型设置为 type，相对地址设置为 offset。在 SDT 中，每到一个声明，就将 offset 加上这个类型的宽度。接下来，我详细举例。

设有如下 SDT：

![](https://kkcx.oss-cn-beijing.aliyuncs.com/img/image-20230506130906537.png)

我们要翻译 real x; int i;

1. 根据所有具有相同左部的产生式的 SELECT 集是否判断该文法是否为 LL(1) 文法。经验证，这是 LL(1) 文法。

2. 指针指向 real，首先根据第一个产生式替换 P

   ![](https://kkcx.oss-cn-beijing.aliyuncs.com/img/image-20230506131359459.png)

3. 此时语义动作 {offset=0} 在栈顶，我们执行这个语义动作，初始化了 offset，然后将语义动作 {a} 出栈

4. 此时栈顶为 D，根据第二个产生式替换 D

   ![](https://kkcx.oss-cn-beijing.aliyuncs.com/img/image-20230506131658624.png)

5. 进行若干步替换栈顶非终结符，得到树为：

   ![](https://kkcx.oss-cn-beijing.aliyuncs.com/img/image-20230506132225253.png)

6. 此时栈顶匹配成功，real 出栈，然后执行动作符号，计算出 B 的综合属性 B.type=read; B.width=8。然后执行第四条产生式中的语义动作，令 t=B.type, w=B.width。

7. 接下来，栈顶为 C，而指针指向 x，所以用第八个产生式替换，此时又要执行语义动作，将刚刚保存的 t, w 在赋值给 C

8. 接下来匹配到 x，和分号，然后要执行语义动作，这个语义动作就是在符号表中创建记录，并更新 offset

9. 后半部分同理，不再赘述，最终形成树和符号表插入条目为：

   ![](https://kkcx.oss-cn-beijing.aliyuncs.com/img/image-20230506132955670.png)

## 赋值语句

赋值语句翻译的主要任务就是生成对表达式求值的三地址码。

### 简单赋值语句的翻译

![](https://kkcx.oss-cn-beijing.aliyuncs.com/img/image-20230506133821653.png)

可以看到，这样的翻译进行了大量的 code 拼接，我们也可以不用这种方式，而是使用**增量翻译**，修改 gen() 函数，让 gen() 函数将生成的三地址指令拼接到至今为止已生成的指令序列之后。

### 数组引用的翻译

将数组引用翻译成三地址码时要解决的主要问题是确定数组元素的存放地址，也就是数组元素的寻址。

对于一个 $k$ 维数组：$array(n1,array(n2,\dots))$，数组元素 $a[i_1][i_2]\cdots[i_k]$ 的相对地址是 $base+i_1\times w_1+i_2\times w_2+\cdots+i_k \times w_k$，其中 $w_i$ 表示前 $i$ 维的大小。

而在数组引用基本文法为：

![](https://kkcx.oss-cn-beijing.aliyuncs.com/img/image-20230506164120963.png)

我们为 L 设置三个综合属性：

- L.type：L 生成的数组元素的类型
- L.offset：指示一个临时变量，该临时变量用于累加公式中的 $i_j\times w_j$ 项，从而计算数组元素的偏移值
- L.array：数组名在符号表的入口地址

## 控制流语句

控制流语句的基本文法：

![](https://kkcx.oss-cn-beijing.aliyuncs.com/img/image-20230506165323287.png)

### 控制流语句的代码结构

以 `S -> if B then S1 else S2` 为例，其代码结构为：

![](https://kkcx.oss-cn-beijing.aliyuncs.com/img/image-20230506165533809.png)

- B.true：用来存放当 B 为真时控制流转向的指令的标号
- B.false：用来存放当 B 为假时控制流转向的指令的标号
- S.next：用来存放紧跟在 S 代码之后的指令标号

布尔表达式 B 被翻译为由跳转指令构成的跳转代码。需要解决的问题是如何确定跳转指令中目标地址：

- 第一种方法：为跳转目标预分配标号并通过继承属性传递到标注位置。需要两遍扫描。
- 第二种方法：回填技术。一遍扫描就能完成。

### 控制流语句的 SDT

![](https://kkcx.oss-cn-beijing.aliyuncs.com/img/image-20230506170049647.png)

让我们尝试编写 if-then-else 语句的 SDT：

![](https://kkcx.oss-cn-beijing.aliyuncs.com/img/image-20230506205914003.png)

while-do 语句的 SDT：

![](https://kkcx.oss-cn-beijing.aliyuncs.com/img/image-20230506210039880.png)

## 布尔表达式

布尔表达式的值是通过代码序列中的跳转位置来表示的。如：

![](https://kkcx.oss-cn-beijing.aliyuncs.com/img/image-20230506210627895.png)

### 布尔表达式的 SDT

![](https://kkcx.oss-cn-beijing.aliyuncs.com/img/image-20230506210734462.png)

![](https://kkcx.oss-cn-beijing.aliyuncs.com/img/image-20230506210751764.png)

![](https://kkcx.oss-cn-beijing.aliyuncs.com/img/image-20230506210806369.png)

## 回填

生成一个跳转指令时，暂时不指定该跳转指令的目标标号。这样的指令都被放入由跳转指令组成的列表中。同一个列表中的所有跳转指令具有相同的目标标号。等到能够确定正确的目标标号时，才去填充这些指令的目标标号。

对非终结符 B 维护如下综合属性：

- B.truelist：指向一个包含**跳转指令**的列表，这些指令最终获得的目标标号就是当 B 为真时控制流应该转向的指令的标号
- B.falselist：同上，只不过是为真时跳转的标号

定义如下函数：

- makelist(i)：创建一个只包含 i 的列表，i 时跳转指令的标号，函数返回新创建的列表的指针
- merge(p1, p2)：将列表合并
- backpatch(p, i)：将 i 作为目标标号插入到 p 所指列表中的各跳转指令中

### 布尔表达式的回填

![](https://kkcx.oss-cn-beijing.aliyuncs.com/img/image-20230506215724064.png)

这里 M 的作用就是获得 B2 的第一条指令的标号

### 控制流语句的回填

![](https://kkcx.oss-cn-beijing.aliyuncs.com/img/image-20230506220232500.png)

![](https://kkcx.oss-cn-beijing.aliyuncs.com/img/image-20230506220744984.png)

## switch 语句的翻译

switch 语句可以翻译成这种代码结构：

![](https://kkcx.oss-cn-beijing.aliyuncs.com/img/image-20230506221643412.png)

也可以这样，将分支测试指令集中在一起：

![](https://kkcx.oss-cn-beijing.aliyuncs.com/img/image-20230506221723244.png)

## 过程调用的翻译

![](https://kkcx.oss-cn-beijing.aliyuncs.com/img/image-20230506221930142.png)
