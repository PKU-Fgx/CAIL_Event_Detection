# CAIL_Event_Detection
&emsp;&emsp;事件信息是法律案情的核心，法律事件检测旨在识别出法律案件中的事件触发词及其对应的事件类型，从而完成案件事实的快速重构，帮助机器和人类更好地理解法律案件。

## 1. Brach：`OfficalBaseline`

&emsp;&emsp;这是一个官方给的 `BERT+CRF` 的我自己的精简版本：
1. 还没有加多卡训练；
2. 把 Test 写到了外面去；
3. 精简了部分内容，比如把恒定长度512改为了动态长度来节省内存；
4. 目前训练还有小BUG，我估计是因为`input_ids`长度不对的原因，但是我不知道为什么前三个Epoch都是对的，然后跑到第四个Epoch的时候出现了错误；

## 1. Brach：`myCodeV1`

&emsp;&emsp;把代码改成了我自己熟悉的样子；

## 2. Brach：`myCodeV2`

&emsp;&emsp;保证基础可运行版本；

## 3. Brach：`magic_reform_v1`

&emsp;&emsp;开始魔改；

## 4. Brach：`acc_ver_1`

&emsp;&emsp;这里加上了多卡并行运行，但是有个BUG就是如果加CRF的同时使用fp16进行加速的话，loss会变成nan；(别加梯度剪枝就行了)

## 5. Brach：`myCodeALL`

&emsp;&emsp;加了很多可以使用的Trick版本；

## 6. Brach：`mlm`

&emsp;&emsp;预训练模型的代码；
