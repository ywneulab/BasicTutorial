---
title: Git 手册
sitemap: true
categories: 其他
date: 2018-07-11 21:45:32
tags:
- Git
---

# git rebase

回到某个提交状态, 删除之前的
```
git rebase --onto master~3 master
```

# 删除 add 的文件

如果要删除文件，最好用 git rm file_name，而不应该直接在工作区直接 rm file_name。

如果一个文件已经add到暂存区，还没有 commit，此时如果不想要这个文件了，有两种方法：

1，用版本库内容清空暂存区，git reset HEAD

2，只把特定文件从暂存区删除，git rm --cached xxx

# 配置用户密码不用每次输入

## 显式配置(不安全)
```sh
git config --global user.name "xxxx"
git config --global user.email "xxxx@foxmail.com"
git config --global credential.helper store
```
上述指令输完后, 会生成`~/.gitconfig`文件, 内容如下:
```
[user]
	name = hellozhaozheng
	email = hellozhaozheng@foxmail.com
[credential]
	helper = store
```
之后, 再次输入密码, 密码和账号会显式存储在`~/.git-credentials`文件中.(显式存储, 不安全)

## 秘钥配置 ssh

step 1: 生成公钥
```
ssh-keygen -t rsa -C "hellozhaozheng@foxmail.com"
# Generating public/private rsa key pair...
# 三次回车即可生成 ssh key
```
step 2: 查看已生成的公钥
```
cat ~/.ssh/id_rsa.pub
# ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC6eNtGpNGwstc....
```
step3: 复制已生成的公钥添加到git服务器
step4: 使用ssh协议clone远程仓库 or 如果已经用https协议clone到本地了，那么就重新设置远程仓库
```
git remote set-url origin git@xxx.com:xxx/xxx.git
```


# 常用指令
初始化当前本地文件夹为仓库

git init

添加文件:

单个: git add readme.md
全部: git add -A  / git add ./

提交修改: git commit -m "must write commit"

查看状态: git status

查看日志: git log

版本回退:

回退一个: git reset -hard HRAD^

回退两个: git reset -hard HARD^^

回退多个: git reset -hard HEAD~100

首次连接: git remote add origin https//www.github...

提交: git push origin master


更新本地仓库:

git fetch origin master  获取

git merge origin/master  合并




如果将别人的仓库拉到了自己的仓库里,  为了push成功:

1. 删除.git相关
2. git rm rf --cached ./themes/next
3. git add ./themes/next
4. git commit -m "next"
5. git push origin master

子模块: 可以独立提交


# 从远程库更新本地的单个文件

https://stackoverflow.com/questions/3334475/git-how-to-update-checkout-a-single-file-from-remote-origin-master

# 只显示更改的文件的名字

https://segmentfault.com/q/1010000006760132

## git
git checkout misc: 切换分支
git stash: 暂存到缓存里, 新建另一个编辑
git stash pop: 从缓存理加载出刚刚的分支, 这里可以和chechout结合使用来切换分支同时将刚刚修改的操作添加到当前分支中

若要提交新的commit:
先切换分支: git checkout misc
再查看状态: git status
添加修改: git add .
提交: git commit -m ".."
推送: git push origin misc
查看提交结果: git log --oneline --decorate

更新master到本地分支:
```
//查询当前远程的版本
$ git remote -v
//获取最新代码到本地(本地当前分支为[branch]，获取的远端的分支为[origin/branch])
$ git fetch origin master  [示例1：获取远端的origin/master分支]
$ git fetch origin dev [示例2：获取远端的origin/dev分支]
//查看版本差异
$ git log -p master..origin/master [示例1：查看本地master与远端origin/master的版本差异]
$ git log -p dev..origin/dev   [示例2：查看本地dev与远端origin/dev的版本差异]
//合并最新代码到本地分支
$ git merge origin/master  [示例1：合并远端分支origin/master到当前分支]
$ git merge origin/dev [示例2：合并远端分支origin/dev到当前分支]
---------------------

git merge orgin vino
```



**解决冲突:**
git mergetool
:diffg RE
:diffg LO

vimdiff3


忘记切换分支, 提交到了错误的分支:

git reset HEAD~1
git stash
git checkout target_branch
git stash pop
git add .
git commit -m "..."
git push origin target_branch


比较本地分支和远端分支的区别
(先要更新本地的远程分支?)
git diff vino origin/vino
git diff vino origin/master

用某一个分支的文件覆盖另一个分支的文件
用git checkout branch -- filename
如： 分支test上有一个文件A，你在test1分支上， 此时如果想用test分支上的A文件替换test1分支上的文件的话，可以使用git checkout test1, 然后git checkout test -- A
