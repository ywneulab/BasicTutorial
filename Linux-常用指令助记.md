---
title: Linux-常用指令助记
sitemap: true
categories: 其他
date: 2018-10-05 20:35:56
tags:
- Linux
---

1.查看磁盘空间

 df -hl 查看磁盘剩余空间
 df -h 查看每个根路径的分区大小

2.查看文件/文件夹大小

查看指定文件/文件夹大小：du -hs <文件名或文件夹名>

查看当前文件夹下所有文件大小（包括子文件夹）：du -sh 

3.查看文件数量

统计当前目录下文件的个数（不包括目录）

ls -l | grep "^-" | wc -l
---------------------
作者：spectre7
来源：CSDN
原文：https://blog.csdn.net/weixin_41278720/article/details/83591027
版权声明：本文为博主原创文章，转载请附上博文链接！

wget:

curl
```
curl -O [url]
```

# tar

# cat

# grep

# ls

## 统计文件个数


# 更换国内镜像源


清华源:
```
https://mirrors.tuna.tsinghua.edu.cn/
```

中科大源:

```
http://mirrors.ustc.edu.cn/help/homebrew-bottles.html
```

阿里源:
```
```

# apt

# pip

# brew

中科大
```
cd "$(brew --repo)"
git remote set-url origin https://mirrors.ustc.edu.cn/brew.git
```

# npm

https://npm.taobao.org/

```
npm config set registry http://registry.npm.taobao.org
```

# docker

从Registry中拉取镜像
```
sudo docker pull registry.cn-hangzhou.aliyuncs.com/zerozone/non-target-baseline-local:[镜像版本号]
sudo docker pull registry.cn-hangzhou.aliyuncs.com/zerozone/non-target-baseline-local:[镜像版本号]
```

将镜像推送到Registry
```
$ sudo docker login --username=ksws840662044 registry.cn-hangzhou.aliyuncs.com
$ sudo docker tag [ImageId] registry.cn-hangzhou.aliyuncs.com/zerozone/non-target-baseline-local:[镜像版本号]
$ sudo docker push registry.cn-hangzhou.aliyuncs.com/zerozone/non-target-baseline-local:[镜像版本号]
```

把容器打包成镜像
```
docker commit -m  ""   -a  ""   [CONTAINER ID]  [给新的镜像命名]
```

查看镜像
```
docker images
```

将指定的imageid的镜像重命名
```
$ sudo docker tag [ImageId] registry.cn-hangzhou.aliyuncs.com/zerozone/non-target-baseline-local:[镜像版本号]
```

推送
```
sudo docker push registry.cn-hangzhou.aliyuncs.com/zerozone/non-target-baseline-local:1.0.2
```


从本地push到镜像
```
sudo docker login --username=${uname} registry.cn-shanghai.aliyuncs.com
cd ${code_path} # where there is a Dockerfile for build this docker image
docker build -t ${docker_image_path}:${tag_version} .
docker push ${docker_image_path}:${tag_version}
```

查看容器:
```
docker ps -a
```

启动容器
```
# CONTAINER ID = a364c4f5ab8a
sudo nvidia-docker start a364c4f5ab8a
sudo nvidia-docker attach a364c4f5ab8a
```

容器和本地之间的文件拷贝:
将主机`/www/runoob`目录拷贝到容器 96f7f14e99ab(Container ID) 中 `/www` 目录下
```
docker cp /www/runoob 96f7f14e99ab:/www/
```
反之:
```
docker cp 96f7f14e99ab:/www/ /www/runoob
```

# 命令行光标移动快捷键

ctrl+a/e: 跳至行首/尾
ctrl+f/b: 左/右移一个字符
alt+f/b: 左/右一个单词
ctrl+h/d: 左/右删除
ctrl+u/k: 删除剩余(u不好使? 会删除所有), 带有剪切功能
ctrl+y: 复制
ctrl + p/n: 上一条/下一条历史记录
alt + p/n: 上一条/下一条最相近的历史记录
