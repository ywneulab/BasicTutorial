---
title: 乱七八糟
sitemap: true
categories: 其他
date: 2018-07-05 22:44:08
tags:
---

# ubuntu 输入法失效

点击 fcitx 没有反应, 尝试重新安装

```sh
sudo apt remove fcitx*
#rm -rf ~/.config/fcitx
sudo apt install fcitx
```

# gcc/g++ 降级升级

【链接】ubuntu16.04LTS降级安装gcc4.8
https://www.cnblogs.com/in4ight/p/6626708.html

# Jupyter

报错: `ImportError: cannot import name 'create_prompt_application'`

原因:https://github.com/jupyter/jupyter_console/issues/158

解决方案
```py
pip install 'prompt-toolkit==1.0.15'
```

# 无法挂载 D 盘 (windows 未完全关闭导致)

Unable to access “WinD”
```
Error mounting /dev/sda6 at /media/ubuntu/Media Center: Command-line `mount -t "ntfs" -o "uhelper=udisks2,nodev,nosuid,uid=1000,gid=1000,dmask=0077,fmask=0177" "/dev/sda6" "/media/rolindroy/Media Center"' exited with non-zero exit status 14: The disk contains an unclean file system (0, 0).
Metadata kept in Windows cache, refused to mount.
Failed to mount '/dev/sda6': Operation not permitted
The NTFS partition is in an unsafe state. Please resume and shutdown
Windows fully (no hibernation or fast restarting), or mount the volume
read-only with the 'ro' mount option
```
https://askubuntu.com/questions/462381/cant-mount-ntfs-drive-the-disk-contains-an-unclean-file-system

# 谷歌浏览器不能手势双击

有可能是没有打开这项功能, 需要在触控板设置里看一下

如果打开了, 那就是某个插件与手势可能有冲突, 目前已知的有 Mouse什么的 (鼠标手势插件)

# 如何在英文版 Chrome 登录印象笔记剪藏插件？

Chrome 商店上的最新版 https://chrome.google.com/webstore/detail/evernote-web-clipper/pioclpoplcdbaefihamjohnefbikjilc 已经可以切换到国内的印象笔记，方法是在选项界面用键盘输入一遍 上上下下左右左右BA ，不用考虑大小写，代码里就是判断的就是 ↑ ↑ ↓ ↓ ← → ← → B A。然后会出个新Tab: Developer Options

勾选上模拟简体中文，再点击“Save and reload clipper”后，就可以登录印象笔记了。（评论中有人遇到仍然不能登录的情况，此时请在上图的“Override Service URL”下方填写 app.yinxiang.com ，然后再“Save”试试。）

# windows 将 ctrl 和 caps clock 键互换

创建文件`caps_ctrl.reg`, 输入如下内容:
```
Windows Registry Editor Version 5.00
[HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Keyboard Layout]
"Scancode Map"=hex:00,00,00,00,00,00,00,00,03,00,00,00,1D,00,3A,00,3A,00,1D,00,00,00,00,00
```
双击运行, 然后查看是否修改完成

![](https://wx2.sinaimg.cn/large/d7b90c85ly1g15kd4bi01j20z10l9q50.jpg)

若成功修改, 重启即可.

# 双系统 windows 方法 ubuntu 文件
下载 ext2explore工具     

地址：https://sourceforge.net/projects/ext2read/files/latest/download

# 更改windows分区导致出现 grub error: unknown filesystem

```py
ls
set
#以(hd0,msdos1)为例，分别输入：
set root=hd0,msdos1
set prefix=(hd0,msdos1)/boot/grub

# 检查是否正确, 如果出现unknown filesystem则错误, 如果没有返回则正确
insmod normal

# 输入 normal, 引导界面正确呈现
normal


# 选择ubuntu，进入之后启动终端，输入如下命令
sudo  update-grub
sudo grub-install /dev/sda
```

# 修复引导

Boot Repair

# Ubuntu 无损扩容

1。先在windows里面划分出一个未分配的空间
2。用linux live creater或者其他linux livecd制作软件  制作带有ubuntu镜像的u盘
3。在bios里面用u盘启动ubuntu（选择 try ubuntu without installing）
4。在u盘启动的ubuntu里打开GParted
5。在GParted中，将未分配的空间移动到你想要合并的分区的附近（遇到swap要swapoff，遇到/ /home等要进行unmount，不过貌似u盘启动的ubuntu默认就是unmunt的）
6。提交移动/调整大小操作，根据操作涉及的空间大小需要等待一段时间
7。操作完成后，重启机器，无损扩容完成！

# Ubuntu 互换 ctrl 和 caps 键位

即刻生效
```py
setxkbmap -option "ctrl:swapcaps"
```

重启生效
```py
# 在/etc/default/keyboard文件中添加
XKBOPTIONS="ctrl:nocaps"
```

# pip 安装速度慢
**Linux系统pip、conda等包管理程序下载速度慢的主要原因是默认的下载镜像源是国外的，而解决方法是修改镜像源到国内即可，具体如下**：

##pip
**目前可用源：**
http://pypi.douban.com/ 豆瓣
http://pypi.hustunique.com/ 华中理工大学
http://pypi.sdutlinux.org/ 山东理工大学
http://pypi.mirrors.ustc.edu.cn/ 中国科学技术大学
http://mirrors.aliyun.com/pypi/simple/ 阿里云
https://pypi.tuna.tsinghua.edu.cn/simple/ 清华大学

###1.临时使用：

可以使用pip时添加参数 -i[url]，如：

```
pip install -i http://pypi.douban.com/simple/ gevent
```

###2.永久生效：
修改`~/.pip/pip.conf`，修改index-url至相应源
（如果没有该文件或文件夹，就先在home下创建`.pip`文件夹，再在`.pip`文件夹里面创建`.pip.conf`文件）
```
[global]
index-url = http://mirrors.aliyun.com/pypi/simple
trusted-host =  mirrors.aliyun.com #对应阿里云的host，其他的可以自己查一下，也可以不写这一句
```

##conda

**可用源**
清华
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
中科大
http://mirrors.ustc.edu.cn/anaconda/pkgs/free/

```
$ conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
```
然后可以输入下面的指令查看当前的源：
```
$ conda config --show
```
![这里写图片描述](http://img.blog.csdn.net/20180310204357838?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQva3N3czAyOTI3NTY=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

或者修改成.condarc文件，在文件中输入下面的指令。

```
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - defaults
show_channel_urls: true
```

# 更换 apt 源
备份系统自带源
```sh
mv /etc/apt/sources.list /etc/apt/sources.list.bak
```
修改/etc/apt/sources.list文件
```sh
/etc/apt/sources.list  
```

阿里源: https://opsx.alibaba.com/mirror



若更新了源以后仍然不好使, 尝试删除`sources.list.d`(将其重命名)

若 apt-get update时卡在 waiting for headers, 则删除下面的文件
```sh
# rm /var/lib/apt/lists/*
rm /var/lib/apt/lists/partial/*
apt-get update
```


# pyenv 安装速度慢


# atom 安装插件慢

**方法一:** 翻墙

**方法二:** 利用 git 克隆到本地安装

先进入插件安装的目录
```
cd /Users/zerozone/.atom/packages
```
将需要的仓库 clone 下来，再`npm install`
```
git clone git@github.com:Glavin001/atom-beautify.git
cd atom-beautify
npm install
```
最后重启下 atom 就可以看到插件安装成功

# opencv python2.7 缺少文件

```
python2.7  from .cv2 import *ImportError: libSM.so.6: cannot open shared object file: No such file or directory
```
方法一: 升级到python3

方法二: 安装指定版本的opencv-python(其他版本貌似对应的是python3):
```
pip install opencv-python==3.2.0.8
```
