# Geth: Generalized Elastic Training Hub

## Note

GCCL卡死，卡死在copy128b上，暂时做法是线程数从1024调成512之后，跑了100次没问题；猜测是可能是资源不够导致了一些奇妙的冲突

ragdoll的assert被注释掉了，后续对比实验的时候需要注意这一点

ogbn-arxiv在分成7/5份的时候会让metis死掉，我暂且蒙古，现在的数据直接用RECUR_MINI的
