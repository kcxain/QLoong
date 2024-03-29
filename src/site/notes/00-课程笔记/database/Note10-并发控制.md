---
{"dg-publish":true,"permalink":"/00-课程笔记/database/Note10-并发控制/","title":"Note10- 并发控制"}
---


# Note10- 并发控制

## 事务

事务是用户定义的一个数据库操作序列，这些操作要么全做，要么全不做，是一个不可分割的工作单位。

事务的特性：

- 原子性：即事务完全执行或完全不执行
- 一致性：事务执行的结果必须是使数据库从一个一致性状态变到另一个一致性状态
- 隔离性：表面看起来，每个事务都是在没有其它事务同时执行的情况下执行的
- 持久性：一个事务一旦提交，它对数据库中数据的改变就应该是永久性的

## 并发执行和调度

调度就是一个或多个事务的重要操作按时间排序的一个序列。如果不管数据库初始状态如何，一个调度对数据库状态的影响都和某个串行调度相同，则我们说这个调度是可串行化的。

### 优先关系

已知调度 S，其中涉及事务 T1 和 T2，可能还有其他事务。我们说 T1 优先于 T2，记作 T1 < T2，如果有 T1 的动作 A1 和 T2 的动作 A2，满足：

- 在 S 中 A1 在 A2 前
- A1 和 A2 都涉及同一数据库元素，并且 A1 和 A2 中至少有一个动作是写

因此，在任何冲突等价于 S 的调度中，A1 将出现在 A2 前。所以，如果这些调度中有一个是串行调度，那么该调度必然使 T1 在 T2 前。

### 优先图

优先图的结点是调度 S 中的事务。当这些事务是具有不同的 i 的 Ti 时，我们将仅用整数 i 来表示 Ti 的结点。如果 Ti < Tj，则有一条从结点 i 到结点 j 的弧。

如，假设有一个事务序列为：r2(A); r1(B); w2(A); r3(A); w1(B); w3(A); r2(B); w2(B);

事务 T1 不对 A 操作，而它最先对 B 操作，显然 T1 优先级最高，调度图如下：

![](https://kkcx.oss-cn-beijing.aliyuncs.com/img/image-20230529140413639.png)

### 冲突可串行性判定

就是看优先图中有没有环。

## 并发控制协议

### 基于锁的协议

锁是数据项上的并发控制标志。锁可以分为两种类型：

- 共享锁。如果事务 T 得到了数据项 Q 上的共享锁，则 T 可以读这个数据项，但不能写这个数据项。共享锁表示为 S
- 互斥锁。如果事务 T 得到了数据项 Q 上的互斥锁，则 T 既可以都这个数据项，也可以写这个数据项。互斥锁表示为 X

给定一个各种类型锁的集合，如下定义这个锁集合上的相容关系：

- 令 A 和 B 表示任意类型的锁。设事务 Ti 在数据项 Q 要求一个 A 型锁，事务 Tj 已经在 Q 上有一个 B 型锁。如果事务 Ti 能够获得 Q 上的 A 型锁，则说明 A 型锁和 B 型锁是相容的。

#### 死锁

产生死锁的原因是两个或多个事务都已封锁了一些数据对象，然后又都请求对已为其他事务封锁的数据对象加锁，从而出现死等待。

如何预防死锁呢？

- 一次封锁法。要求每个事务必须一次将所有要使用的数据全部加锁
  - 降低系统并发度，且难于事先精确确定封锁对象
- 顺序封锁法。预先对数据对象规定一个封锁顺序，所有事务都按这个顺序实行封锁。
  - 事务的封锁请求可以随着事务的执行而动态地决定，很难事先确定每一个事务要封锁哪些对象，因此也就很难按照规定地顺序去施加封锁

#### 等待图

事务等待图是一个有向图，结点表示正运行地事务，每条边表示事务地等待情况，若 T1 等待 T2，则 T1,    T2 之间划一条有向边

并发控制子系统周期性地生成等待图。如果发现图中存在回路，则表示系统中出现了死锁。

#### 死锁的恢复

- 选择牺牲者：必须决定回滚哪一个事务以打破死锁，应使事务回滚代价最小
  - 事务已计算了多久，还将计算多长时间
  - 该事务已使用了多少数据项
  - 为完成事务还需使用多少数据项
  - 回滚使将牵扯多少事务
- 饿死
  - 有可能同一事务总是被选为牺牲者，发生饿死
  - 因而必须保证一个事务被选为牺牲者的次数有限

#### 两段锁协议

两阶段锁协议要求每个事务分两个阶段进行数据项的加锁和解锁，每个事务中所有封锁请求先于任何一个解锁请求。

### 基于时间戳的协议

另一种决定事务可串行化次序的方法是事先选定事务的次序，最常见的方法是时间戳排序机制。

对于系统中每个事务 Ti，我们把一个唯一的固定时间戳和它联系起来，记为 TS(Ti)，如有一新事物 Tj 进入系统，则 TS(Ti) < TS(Tj)

每个数据项 Q 需要与两个时间戳值关联

- W-timestamp(Q)：表示成功执行 write(Q) 的所有事务的最大时间戳
- R-timestamp(Q)：表示成功执行 read(Q) 的所有事务的最大时间戳

时间戳协议

- 假设事务 Ti 发出 read(Q)
  - 若 TS(Ti) < W-timestamp(Q)，则 Ti 需要读入的 Q 值已被覆盖。因此 read 操作被拒绝，Ti 回滚
  - 若 TS(Ti) >= W-timestamp(Q)，则执行 read 操作，Rtimestamp(Q) 被设置为 max(TS(Ti), R-timestamp(Q))
- 假设事务 Ti 发出 write(Q)
  - 若 TS(Ti) < R-timestamp(Q)，则 Ti 产生的 Q 值是是先前所需要的值，且系统已假定该值不会产生。因此拒绝，Ti 回滚
  - 若 TS(Ti) < W-timestamp(Q)，则 Ti 试图写入的 Q 值已过时。拒绝，Ti 回滚
  - 其他情况，则执行，且 W-timestamp(Q) 被设置为 TS(Ti)

#### Thomas 写规则

- 假设事务 Ti 发出 read(Q)，处理方法与时间戳排序协议相同
- 假设事务 Ti 发出 write(Q)
  - 若 TS(Ti) < R-timestamp(Q)，则 Ti 产生的 Q 值是先前所需要的值，且系统已假定该值不会产生。因此 write 操作被拒绝，Ti 回滚
  - 若 TS(Ti) < W-timestamp(Q)，则 Ti 试图写入的 Q 值已过时。因此这个 write 操作可忽略。
  - 其他情况，则执行 write 操作，W-timestamp(Q) 被设置为 TS(Ti )
