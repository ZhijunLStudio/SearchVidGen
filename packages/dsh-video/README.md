# @zhijunlstudio/dsh-video

视频生成能力缝（Service Definition）：生成器/裁判注册表（`ctx.video`）、
capabilities 声明与按能力路由（fail-loud，逐候选原因）、评测结算协议
（`computeVerdict`，权重/阈值归消费者）。

三角色：SD=本包；Provider=@zhijunlstudio/dsh-video-provider；
Consumer=@zhijunlstudio/dsh-video-tool。

新模型接入零代码：在 provider 包配置里加一行 remote/local 即可。
