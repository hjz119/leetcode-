#### 二维数组中的查找

```python
# -*- coding:utf-8 -*-
class Solution:
    # array 二维列表
    def Find(self, target, array):
        if array==[[]]:
            return False
        row=len(array)-1
        col=0
        while row>=0 and col<len(array[0]):
            if target==array[row][col]:
                return True
            if target>array[row][col]:
                col+=1
                continue
            if target<array[row][col]:
                row-=1
                continue
        return False
        # write code here
```

#### 重建二叉树

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回构造的TreeNode根节点
    def reConstructBinaryTree(self, pre, tin):
        # write code here
        if pre==[]:
            return None
        root=TreeNode(pre[0])
        index=tin.index(root.val)
        son_tin_left=tin[0:index]
        son_tin_right=tin[index+1:]
        son_pre_left=[x for x in pre if x in son_tin_left]
        son_pre_right = [x for x in pre if x in son_tin_right]
        root.left=self.reConstructBinaryTree(son_pre_left,son_tin_left)
        root.right = self.reConstructBinaryTree(son_pre_right, son_tin_right)
        return root
```

关键在根，前序给根，中序根据根给列表分成两半（前中都要分），得到子树序列从而递归

#### 用两个栈实现队列

```python
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.A=[]
        self.B=[]
    def push(self, node):
        self.A.append(node)
        
        # write code here
    def pop(self):
        if not self.B:
            while self.A:
                self.B.append(self.A[-1])
                self.A.pop()
        return self.B.pop()
        # return xx
```

#### 旋转数组的最小数字

二分法夹逼，找断层。mid>high，最小元素在后面递增子数组；mid<high，后面

```python
class Solution:
    def minNumberInRotateArray(self, rotateArray):
        low=0
        high=len(rotateArray)-1
        if len(rotateArray)==0:
            return 0
        while high-low>1:
            mid=int((low+high)/2)
            if rotateArray[mid]>rotateArray[high]:
                low=mid
            elif rotateArray[mid]<rotateArray[high]:
                high=mid
            else:
                high-=1
        return rotateArray[high]
```

实际上是查找两个数字所以是while high-low>1

#### 斐波那契数列

```python
class Solution:
    def Fibonacci(self, n):
        if n==0:
            return 0
        if n==1:
            return 1
        temp1=0
        temp2=1
        for i in range(n-1):
            result=temp1+temp2
            temp1=temp2
            temp2=result
        return result
```

#### 跳台阶

```python
class Solution:
    def jumpFloor(self, number):
        # write code here
        a = 1
        b = 1
        for i in range(number):
            a,b = b,a+b
        return a
```

```python
# -*- coding:utf-8 -*-
class Solution:
    def jumpFloor(self, number):
        if number<=1:
            return 1
        tmp1=1
        tmp2=1
        for i in range(number-1):
            res=tmp1+tmp2
            tmp1=tmp2
            tmp2=res
        return res
        # write code here
```

#### 变态跳台阶

```python
# -*- coding:utf-8 -*-
class Solution:
    def jumpFloorII(self, number):
        if number==1 or number==0:
            return 1
        return pow(2,number-1)
        
        # write code here
```

反正跳0阶就是一种

#### 矩形覆盖

```python
# -*- coding:utf-8 -*-
class Solution:
    def rectCover(self, number):
        if number==0:
            return 0
        if number==1:
            return 1
        if number==2:
            return 2
        tmp1=1
        tmp2=1
        for i in range(number-1):#实际从3开始，要更好满足规律所以tmp1取1
            res=tmp1+tmp2
            tmp1=tmp2
            tmp2=res
        return res
        # write code here
```

还是变相斐波那契(适用于f(n)=f(n-1)+f(n-2))

#### 二进制中1的个数

```python
class Solution:
    def NumberOf1(self, n):
        if n<0:
            count=1
            n=n&0x7fffffff
        else:
            count=0
        while n!=0:
            count+=n&1
            n=n>>1
        return count
        # write code here
```

核心是与1相与右移，负数注意处理开头的1

#### 数值的整数次方

```python
class Solution:
    def Power(self, base, exponent):
        if base==0:
            return 0
        if exponent==0:
            return 1
        result=1
        exponent1=abs(exponent)
        for i in range(exponent1):
            result*=base
        if exponent>0:
            return result
        else:
            return 1/result
        # write code here
```

按数学规则来就好，注意单独考虑base=0或者exponent=0，还有负指数的情况

#### 调整数组顺序使奇数位于偶数前面

```python
# -*- coding:utf-8 -*-
class Solution:
    def reOrderArray(self, array):
        for i in range(len(array)):
            if array[i]%2==0:
                for j in range(i+1,len(array)):
                    if array[j]%2!=0:
                        temp=array[j]
                        array.pop(j)
                        array.insert(i,temp)
                        break
        return array
        # write code here
```

找到偶数，找到其后的第一个奇数，删除奇数，插入到偶数前面。break是因为只要第一个奇数，否则后面的奇数会一直往前插，相对位置改变。

#### 链表中倒数第k个结点

```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def FindKthToTail(self, head, k):
        l=[]
        p=head
        if not p or k==0:
            return None
        
        while p:
            l.append(p)
            p=p.next
        if k>len(l):
            return None
        return l[-k]
            
        # write code here
```

遍历，所有节点储存在列表中，直接输出-k，注意k>len(l)以及空的一些情况。

#### 反转链表

```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 返回ListNode
    def ReverseList(self, pHead):
        if not pHead:
            return None
        l=[]
        while pHead:
            l.append(pHead)
            pHead=pHead.next
        for i in range(len(l)-1,0,-1):
            l[i].next=l[i-1]
        l[0].next=None
        return l[-1]
        # write code here
```

遍历，所有节点存储在列表里后再做反转操作

#### 合并两个排序的链表

```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 返回合并后列表
    def Merge(self, pHead1, pHead2):
        if pHead1==None and pHead2==None:
            return None
        l=[]
        while pHead1 and pHead2:
            if pHead1.val<=pHead2.val:
                l.append(pHead1)
                pHead1=pHead1.next
                
            else:
                l.append(pHead2)
                pHead2=pHead2.next
                
        while pHead1:
            l.append(pHead1)
            pHead1=pHead1.next
        while pHead2:
            l.append(pHead2)
            pHead2=pHead2.next
        for i in range(len(l)-1):
            l[i].next=l[i+1]
        
        return l[0]
        # write code here
```

还是靠列表来辅助，两个链表节点先按升序存入列表，列表再把链表接起来

#### 树的子结构

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def HasSubtree(self, pRoot1, pRoot2):
        if not pRoot1 or not pRoot2:
            return False
        return self.IsSubtree(pRoot1,pRoot2) or self.HasSubtree(pRoot1.left,pRoot2) or self.HasSubtree(pRoot1.right,pRoot2)
    def IsSubtree(self,A,B):
        if not B:
            return True
        if not A or A.val!=B.val:
            return False
        return self.IsSubtree(A.left,B.left) and self.IsSubtree(A.right,B.right)
        
        # write code here
```

两个函数都是后是不是前的子结构。区别：第一个后的根不停去贴前的节点，贴完前的根找不到子结构就贴左孩子，贴右孩子。要完成贴的操作就调用函数一。第二个是贴上去以后起点相同判断是不是子结构。

对于第一个，pRoot1空则肯定不存在子结构，pRoot2空不存在是因为空树不是任何树的子结构。

第二个两个 if 判断语句不能颠倒顺序。因为如果颠倒了顺序，会先判断pRoot1 是否为None, 其实这个时候，pRoot1 的节点已经遍历完成确认相等了，但是这个时候会返回 False，判断错误。B还没空A先空了才是False。

#### 二叉树的镜像

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回镜像树的根节点
    def Mirror(self, root):
        if root:
            root.left,root.right=root.right,root.left
            self.Mirror(root.left)
            self.Mirror(root.right)
        else:return 
        # write code here
```

#### 顺时针打印矩阵

```python
class Solution:
    # matrix类型为二维列表，需要返回列表
    def printMatrix(self, matrix):
        result=[]
        row=len(matrix)
        col=len(matrix[0])
        row_start=0
        col_start=0
        number=row*col
        while(len(result)!=number):
            for j in range(col_start,col):
                result.append(matrix[row_start][j])
            if len(result)==number:#应对只有一行，否则会index out of range
                return result
            for i in range(row_start+1,row):
                result.append(matrix[i][col-1])
            if len(result)==number:#应对只有一列
                return result
            for j in range(col-2,col_start-1,-1):
                result.append(matrix[row-1][j])
            for i in range(row-2,row_start,-1):
                result.append(matrix[i][col_start])
            row_start+=1
            col_start+=1
            row-=1
            col-=1
        return result

```

按自己想法一圈一圈打印（自己画个4*4的矩阵看着写），带入开始结束的变量，打印完一圈开始和结束也要更新。

#### 包含min函数的栈

```python
class Solution:
    def __init__(self):
        self.l=[]
        self.min_node=[]
    def push(self, node):
        self.l.append(node)
        if self.min_node==[]:
            self.min_node.append(node)
        else:
            if node<self.min_node[-1]:
                self.min_node.append(node)
            else: pass#可有可无
        # write code here
    def pop(self):
        if self.min_node[-1]==self.l[-1]:
            self.min_node.pop()
        self.l.pop()
        # write code here
    def top(self):
        return self.l[-1]
        # write code here
    def min(self):
        return self.min_node[-1]
```

辅助栈基本只起记录作用，压入弹出操作还是在主栈上操作。

push 最小栈为空直接push，更小的压入最小栈。 弹出时先看最小栈的-1，与l[-1]相等则弹出

#### 栈的压入、弹出序列

```python
# -*- coding:utf-8 -*-
class Solution:
    def IsPopOrder(self, pushV, popV):
        # write code here
        if not pushV or len(pushV) != len(popV):
            return False
        stack = []
        for i in pushV:
            stack.append(i)
            while len(stack) and stack[-1] == popV[0]:
                stack.pop()
                popV.pop(0)
        if len(stack):
            return False
        return True
```

此题的pushV和popV是列表。思路：逐个压入，遇到和弹出序列头相等的都弹出。

#### 从上往下打印二叉树（层序遍历）

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回从上到下每个节点值列表，例：[1,2,3]
    def PrintFromTopToBottom(self, root):
        if not root:
            return []
        q=[root]
        l=[]
        while q:
            l.append(q[0].val)
            if q[0].left:
                q.append(q[0].left)
            if q[0].right:
                q.append(q[0].right)
            q.pop(0)
        return l
        # write code here
```

#### 二叉搜索树的后序遍历序列！

```python
class Solution:
    def VerifySquenceOfBST(self, sequence):
        if sequence==[]:
            return False
        root = sequence[-1]
        length = len(sequence)
        for i in range(0, length):
            if sequence[i] > root:
                #left_root_pos = i - 1
                break   #左子树[0:i]右子树[i:]
        for j in range(i, length):
            if sequence[j] < root:
                return False
        # 当前这棵树没问题，开始检查左右子树
        if i==0:#左子树为空,底层
            left_OK=True
        else:
            left_OK=self.VerifySquenceOfBST(sequence[0:i])
            
        if len(sequence[i:-1])==0:#右子树为空
            right_OK=True
        else:
            right_OK=self.VerifySquenceOfBST(sequence[i:-1])
        return left_OK and right_OK    
        

```

#### 二叉树中和为某一值的路径

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回二维列表，内部每个列表表示找到的路径
    def FindPath(self, root, expectNumber):
        if not root:
            return []
        if root and not root.left and not root.right and root.val==expectNumber: 
            return [[root.val]] #底层
        res=[]
        left_res=self.FindPath(root.left,expectNumber-root.val)#[[5,7]]
        right_res=self.FindPath(root.right,expectNumber-root.val)#[[12]]
        for i in left_res+right_res: #i是列表，一个结果[[5,7],[12]]
            res.append([root.val]+i)
        return res
        # write code here
```

#### 复杂链表的复制

<img src="C:\Users\81481\Documents\image-20200211201641047.png" alt="image-20200211201641047" style="zoom:80%;" />

```python
# -*- coding:utf-8 -*-
# class RandomListNode:
#     def __init__(self, x):
#         self.label = x
#         self.next = None
#         self.random = None
class Solution:
    # 返回 RandomListNode
    def Clone(self, pHead):
        if not pHead:
            return None
        #第一步
        dummy=pHead
        while dummy:
            copynode=RandomListNode(dummy.label)
            copynode.next=dummy.next
            dummy.next=copynode
            dummy=copynode.next
        #第二步 
        dummy=pHead
        while dummy:
            copynode=dummy.next
            if dummy.random:
                copynode.random=dummy.random.next
            dummy=copynode.next
        #第三步  
        dummy=pHead
        copyHead=pHead.next
        while dummy:
            copynode=dummy.next
            dummynext=copynode.next
            dummy.next=dummynext
            if dummynext: #加条件判断是为了处理到尾巴的情况，否则none.next要报错
                copynode.next=dummynext.next
            else:
                copynode.next=None
            dummy=dummynext
        return copyHead
        # write code here
```

copynode基本都要，dummynext第三步要，用于辅助。

#### 二叉搜索树与双向链表

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def Convert(self, pRootOfTree):
        if pRootOfTree==None:
            return None
        l=self.middle_trans(pRootOfTree)
        for i in range(0,len(l)-1):
            l[i].right=l[i+1]
        for i in range(1,len(l)):
            l[i].left=l[i-1]
        return l[0]
        
        
        # write code here
        
    def middle_trans(self,pRootOfTree):
        if pRootOfTree==None:
            return []
        res=[]
        res+=self.middle_trans(pRootOfTree.left)
        res.append(pRootOfTree)
        res+=self.middle_trans(pRootOfTree.right)
        return res
        
```

二叉搜索树中序遍历就是升序。

#### 字符串的排列

```python
# -*- coding:utf-8 -*-
import itertools
class Solution:
    def Permutation(self, ss):
        if len(ss)<=1:
            return ss
        res=set()
        for i in range(len(ss)): #每个字符轮流当头，加上剩下字符的全排列
            for j in self.Permutation(ss[:i]+ss[i+1:]): #除了底层都返回的是列表
                res.add(ss[i]+j)
        return sorted(res)
        
        # write code here
```

#### 数组中出现次数超过一半的数字

```python
# -*- coding:utf-8 -*-
import collections
class Solution:
    def MoreThanHalfNum_Solution(self, numbers):
        ha=collections.Counter(numbers)
        tmp=len(numbers)/2
        for number,count in ha.items():
            if count>tmp:
                return number
        return 0
        # write code here
```

```python
# -*- coding:utf-8 -*-
class Solution:
    def MoreThanHalfNum_Solution(self, numbers):
        hash_table=dict()
        for i in numbers:
            if i not in hash_table.keys():
                hash_table[i]=1
            else:
                hash_table[i]+=1
            if hash_table[i]>len(numbers)/2:
                return i
        return 0
        # write code here
        #自己写的哈希版本
```

#### 最小的k个数

```python
# -*- coding:utf-8 -*-
class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        if k>len(tinput):
            return []
        l=sorted(tinput)
        return l[:k]
        # write code here
```

排序

#### 连续子数组的最大和

```python
# -*- coding:utf-8 -*-
import itertools
class Solution:
    def FindGreatestSumOfSubArray(self, array):
        dp=array
        for i in range(1,len(array)):
            dp[i]=max(dp[i],dp[i-1]+dp[i])
        return max(dp)
        
        # write code here
```

动态规划，每个位置的最大f(n)=max{f(n),f(n-1)+f(n)}

#### 整数中1出现的次数

```python
# -*- coding:utf-8 -*-
import collections
class Solution:
    def NumberOf1Between1AndN_Solution(self, n):
        count=0
        base=1
        round=n
        while round>0:
            weight=round%10
            round/=10
            if weight==0:
                count+=round*base
            elif weight==1:
                count+=round*base+n%base+1
            else:
                count+=round*base+base
            base*=10
            
        return count
        # write code here
```

#### 把数组排成最小的数

```python
# -*- coding:utf-8 -*-
class Solution:
    def PrintMinNumber(self, numbers):
        lmb=lambda n1,n2:int(str(n1)+str(n2))-int(str(n2)+str(n1))
        arr=sorted(numbers,cmp=lmb)
        return ''.join([str(i) for i in arr])
        # write code here
```

#### 丑数

```python
# -*- coding:utf-8 -*-
class Solution:
    def GetUglyNumber_Solution(self, index):
        if index==0:
            return 0
        res=[1]
        t2,t3,t5=0,0,0
        for i in range(1,index):
            wait_push=min(res[t2]*2,res[t3]*3,res[t5]*5)
            res.append(wait_push)
            if wait_push==res[t2]*2:
                t2+=1
            if wait_push==res[t3]*3:
                t3+=1
            if wait_push==res[t5]*5:
                t5+=1
        return res[-1]
        # write code here
```

丑数序列可分为3部分。

#### 第一个只出现一次的字符

哈希

```python
# -*- coding:utf-8 -*-
class Solution:
    def FirstNotRepeatingChar(self, s):
        if s=='':
            return -1
        str_hash={}
        for i in range(len(s)):
            if s[i] in str_hash.keys():
                str_hash[s[i]]+=1
            else:
                str_hash[s[i]]=1
        for i in range(len(s)):
            if str_hash[s[i]]==1:
                return i
        return -1
            
        # write code here
```

#### 数组中的逆序对

```python
count = 0
class Solution:
    def InversePairs(self, data):
        global count
        def MergeSort(lists):
            global count
            if len(lists) <= 1:
                return lists
            num = int( len(lists)/2 )
            left = MergeSort(lists[:num])
            right = MergeSort(lists[num:])
            r, l=0, 0
            result=[]
            while l<len(left) and r<len(right):
                if left[l] < right[r]:
                    result.append(left[l])
                    l += 1
                else:
                    result.append(right[r])
                    r += 1
                    count += len(left)-l #左边已经是升序，一个大后面的都大，都是逆序对
            
            result += left[l:]
            result += right[r:]
            return result
        MergeSort(data)
        return count%1000000007
```

```python
# -*- coding:utf-8 -*-
class Solution:
    count=0
    def InversePairs(self, data):
        self.MergeSort(data)
        return self.count%1000000007
    def MergeSort(self,data):
        if len(data)<=1:
            return data
        mid=int(len(data)/2)
        left=self.MergeSort(data[:mid])
        right=self.MergeSort(data[mid:])
        l,r=0,0
        res=[]
        while l<len(left) and r<len(right):
            if left[l]<=right[r]:
                res.append(left[l])
                l+=1
            else:
                res.append(right[r])
                self.count+=len(left)-l
                r+=1
        res+=left[l:]
        res+=right[r:]
        return res
        # write code here
```

归并排序：先得到左右有序对，再合并成一个有序对

#### 两个链表的第一个公共节点

```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def FindFirstCommonNode(self, pHead1, pHead2):
        p1=pHead1
        p2=pHead2
        l=[]
        while p1:
            l.append(p1)
            p1=p1.next
        while p2:
            if p2 in l:
                return p2
            p2=p2.next
        return None
        # write code here
```

#### 数字在排序数组中出现的次数

```python
# -*- coding:utf-8 -*-
class Solution:
    def GetNumberOfK(self, data, k):
        if data==[]:
            return 0
        def GetFirstK(start, end, l, k):
            while start < end: #找一个数字所以是start<end
                mid = int((start + end) / 2) #注意是加
                if k <= l[mid]: #因为要找的是首次出现的位置，所以=的时候end前移，想想1,2,2,2,3
                    end = mid
                else:
                    start = mid + 1
            return start
        i=GetFirstK(0,len(data)-1,data,k)
        count=0
        while data[i]==k:
            count+=1
            i+=1
            if i==len(data):
                break
        return count
            
        # write code here
```

先二分查找找到这个数字第一次出现的位置，再往后统计。

#### 二叉树的深度

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def TreeDepth(self, pRoot):
        if not pRoot:
            return 0
        return max(self.TreeDepth(pRoot.left),self.TreeDepth(pRoot.right))+1
        # write code here
```

#### 平衡二叉树！

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def IsBalanced_Solution(self, pRoot):
        if not pRoot:
            return True
        return self.maxDepth(pRoot)!=-1
    def maxDepth(self,pRoot):
        if not pRoot:             #底层深度，也是空树
            return 0
        left=self.maxDepth(pRoot.left)  #左右子树均为平衡二叉树，都有深度
        if left==-1:
            return -1
        right=self.maxDepth(pRoot.right)
        if right==-1:
            return -1
        if abs(left-right)<=1:       #左右子树高度差绝对值不超过1
            return max(left+1,right+1)
        else:
            return -1
        
        
        # write code here
```

修改深度函数，是则返回深度，不是则返回-1

#### 数组中只出现一次的数字

```python
# -*- coding:utf-8 -*-
class Solution:
    # 返回[a,b] 其中ab是出现一次的两个数字
    def FindNumsAppearOnce(self, array):
        if array==[]:
            return []
        tmp=0
        for i in array:
            tmp^=i
        #看看tmp从低位开始第几位是1
        idx=0 #右边从第0位开始
        while tmp&1==0:
            tmp>>=1
            idx+=1
        a,b=0,0
        for i in array:
            if self.IsBit(i,idx):
                a^=i
            else:
                b^=i
        return [a,b]
    #判断一个数字第idx位是不是1，将数组分为两类
    def IsBit(self,num,idx):
        num>>=idx
        return num&1
        # write code here
```

这两个数字一定有一位不同。0异或数还是那个数。先得到idx（不同的那一位），再根据idx将数组分为两部分

#### 和为S的连续整数序列

```python
# -*- coding:utf-8 -*-
class Solution:
    def FindContinuousSequence(self, tsum):
        res=[]
        low,high=1,2
        while high>low:
            tmp_l=list(range(low,high+1))
            tmp_sum=sum(tmp_l)
            if tmp_sum==tsum:
                res.append(tmp_l)
                low+=1
            elif tmp_sum<tsum:
                high+=1
            else:
                low+=1
        return res
        # write code here
```

滑窗,只加不减

#### 和为S的两个数字

```python
# -*- coding:utf-8 -*-
class Solution:
    def FindNumbersWithSum(self, array, tsum):
        low,high=0,len(array)-1
        while high>low:
            if array[low]+array[high]==tsum:
                return [array[low],array[high]]
            elif array[low]+array[high]>tsum:
                high-=1
            else:
                low+=1
        return []
        
        # write code here
```

夹逼

#### 左旋转字符串

```python
# -*- coding:utf-8 -*-
class Solution:
    def LeftRotateString(self, s, n):
        return s[n:]+s[:n]
        # write code here
```

#### 翻转单词顺序列

```python
# -*- coding:utf-8 -*-
class Solution:
    def ReverseSentence(self, s):
        l=s.split(' ')
        return ' '.join(list(reversed(l)))
        # write code here
```

#### 扑克牌顺子

```python
# -*- coding:utf-8 -*-
import collections
class Solution:
    def IsContinuous(self, numbers):
        if len(numbers)!=5:
            return False
        l=[x for x in numbers if x!=0] #非0挑出来
        c=collections.Counter(l)
        for key,value in c.items():
            if value>1: #非0有重复直接False
                return False
        if max(l)-min(l)<5: #关键，实际上满足这个就可
            return True
        else:
            return False
        # write code here
```

#### 孩子们的游戏

```python
# -*- coding:utf-8 -*-
class Solution:
    def LastRemaining_Solution(self, n, m):
        if not n:
            return -1
        l=range(n)
        i=0
        while len(l)>1:
            i=(m-1+i)%len(l)
            l.pop(i)
        return l[0]
            
        # write code here
```

#### 求1+2+3+...+n（逻辑与得短路特性）

```python
# -*- coding:utf-8 -*-
class Solution:
    def Sum_Solution(self, n):
        return n and (n+self.Sum_Solution(n-1))
        # write code here
```

#### 不用加减乘除做加法

```python
# -*- coding:utf-8 -*-
class Solution:
    def Add(self, num1, num2):
        l=[]
        l.append(num1)
        l.append(num2)
        return sum(l)
        
        # write code here
```

#### 把字符串转换成整数

```python
# -*- coding:utf-8 -*-
class Solution:
    def StrToInt(self, s):
        l=['0','1','2','3','4','5','6','7','8','9']
        if not s or (len(s)==1 and s[0] not in l): #'+'，只有一个必须是数值
            return 0
        num=0
        if s[0]=="+" or s[0]=='-':
            for i in range(1,len(s)):
                if s[i] in l:
                    num=num*10+l.index(s[i])
                else:
                    return 0
            if s[0]=='+':
                return num
            else:
                return -num
        for i in range(0,len(s)):
            if s[i] in l:
                num=num*10+l.index(s[i])
            else:
                return 0
        return num
            
            
        # write code here
```

巧用index。先处理带正负号的，再处理不带的。

#### 数组中重复的数字

```python
# -*- coding:utf-8 -*-
class Solution:
    # 这里要特别注意~找到任意重复的一个值并赋值到duplication[0]
    # 函数返回True/False
    def duplicate(self, numbers, duplication):
        l=[]
        for i in numbers:
            if i not in l:
                l.append(i)
            else:
                duplication[0]=i
                return True
        return False
                
        # write code here
```

#### 构建乘积数组

```python
# -*- coding:utf-8 -*-
class Solution:
    def multiply(self, A):
        B=[None]*len(A)
        B[0]=1
        for i in range(1,len(A)): #下三角
            B[i]=B[i-1]*A[i-1]
        tmp=1
        for i in range(len(A)-2,-1,-1): #上三角
            tmp*=A[i+1]
            B[i]*=tmp
        return B
                # write code here
```

#### 正则表达式匹配

```python
# -*- coding:utf-8 -*-
class Solution:
    # s, pattern都是字符串
    def match(self, s, pattern):
        if s=='' and pattern=='':
            return True
        elif s!='' and pattern=='':
            return False
        elif s=='' and pattern!='':
            if len(pattern)>1 and pattern[1]=='*':
                return self.match(s,pattern[2:])
            else:
                return False
        else:
            if len(pattern)>1 and pattern[1]=='*':
                if s[0]!=pattern[0] and pattern[0]!='.':
                    return self.match(s,pattern[2:])
                else:
                    return self.match(s,pattern[2:]) or self.match(s[1:],pattern[2:]) or self.match(s[1:],pattern) #对应匹配0,1，多位
            else:
                if s[0]!=pattern[0] and pattern[0]!='.':
                    return False
                else:
                    return self.match(s[1:],pattern[1:])
        # write code here
```

#### 表示数值的字符串

```python
# -*- coding:utf-8 -*-
class Solution:
    # s字符串
    def isNumeric(self, s):
        try:
            p=float(s)
            return True
        except:
            return False
        # write code here
```

#### 字符流中第一个不重复的字符

```python
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.s=''
        self.dict1={}
    # 返回对应char
    def FirstAppearingOnce(self):
        for i in self.s:
            if self.dict1[i]==1:
                return i
        return '#'
        # write code here
    def Insert(self, char):
        self.s+=char
        if char in self.dict1.keys():
            self.dict1[char]+=1
        else:
            self.dict1[char]=1
        # write code here
```

#### 链表中环的入口结点

```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def EntryNodeOfLoop(self, pHead):
        l=[]
        p=pHead
        while p:
            if p in l:
                return p
            else:
                l.append(p)
                p=p.next
        return None
        # write code here
```

遍历，第一个重复的就是入口结点

#### 删除链表中重复的结点

```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def deleteDuplication(self, pHead):
        if pHead==None or pHead.next==None:
            return pHead
        first=ListNode(-1)
        first.next=pHead
        last=first
        while pHead and pHead.next:
            if pHead.val==pHead.next.val:
                val=pHead.val
                while pHead and pHead.val==val:
                    pHead=pHead.next
                last.next=pHead
            else:
                last=pHead
                pHead=pHead.next
        return first.next
                    
        # write code here
```

#### 二叉树的下一个结点

```python
# -*- coding:utf-8 -*-
# class TreeLinkNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.next = None
class Solution:
    def GetNext(self, pNode):
        if pNode==None:
            return None
        if pNode.right:
            pNode=pNode.right
            while pNode.left:
                pNode=pNode.left
            return pNode
        while pNode.next:
            if pNode.next.left==pNode:
                return pNode.next
            pNode=pNode.next
        return None
        # write code here
```

按有无右子树分为两部分

#### 对称的二叉树！

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def isSymmetrical(self, pRoot):
        def isSame(p1,p2):
            if not p1 and not p2: #底层
                return True
            if p1 and p2 and p1.val==p2.val: #如果当前节点对称
                return isSame(p1.left,p2.right) and isSame(p1.right,p2.left)
            return False
        if not pRoot:
            return True
        if not pRoot.left and not pRoot.right:
            return True
        if pRoot.left and pRoot.right:
            return isSame(pRoot.left,pRoot.right)
        return False
        # write code here
```

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def isSymmetrical(self, pRoot):
        if not pRoot:
            return True
        if pRoot and not pRoot.left and not pRoot.right:
            return True
        if pRoot.left and pRoot.right:
            return self.issame(pRoot.left,pRoot.right)
        return False
    def issame(self,p1,p2):
        if not p1 and not p2:
            return True
        if p1 and p2 and p1.val==p2.val:
            return self.issame(p1.left,p2.right) and self.issame(p1.right,p2.left)
        return False
        
        
        
        # write code here
```

空，一个节点，有左右子树

#### 按之字形顺序打印二叉树

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def Print(self, pRoot):
        if not pRoot:
            return []
        res=[]
        curNodes=[pRoot]
        isEvenLayer=True
        while curNodes:
            curValues=[]
            nextNodes=[]
            isEvenLayer=not isEvenLayer
            for node in curNodes:
                curValues.insert(0,node.val) if isEvenLayer else curValues.append(node.val)
                if node.left:
                    nextNodes.append(node.left)
                if node.right:
                    nextNodes.append(node.right)
            res.append(curValues)
            curNodes=nextNodes
        return res
            
        # write code here
```

#### 把二叉树打印成多行

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回二维列表[[1,2],[4,5]]
    def Print(self, pRoot):
        if not pRoot:
            return []
        res=[]
        curNodes=[pRoot]
        while curNodes:
            curValues=[]
            nextNodes=[]
            for node in curNodes:
                curValues.append(node.val)
                if node.left:
                    nextNodes.append(node.left)
                if node.right:
                    nextNodes.append(node.right)
            res.append(curValues)
            curNodes=nextNodes
        return res
        # write code here
```

与上题类似，也相当于层序遍历，偶数层不用翻转

#### 序列化二叉树！

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    flag=-1
    def Serialize(self, root):
        if not root:
            return '#'
        return str(root.val)+','+self.Serialize(root.left)+','+self.Serialize(root.right)
        # write code here
    def Deserialize(self, s):
        self.flag+=1
        l=s.split(',')
        if self.flag>=len(s):
            return None
        if l[self.flag]!='#':
            root=TreeNode(int(l[self.flag]))
            root.left=self.Deserialize(s)
            root.right=self.Deserialize(s)
            return root
        return None
        # write code here
```

用flag来取字符串中元素

#### 二叉搜索树的第k个结点

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回对应节点TreeNode
    def KthNode(self, pRoot, k):
        res=self.midtrans(pRoot)
        if k<=0 or len(res)<k:
            return None
        return res[k-1]
    
    
    def midtrans(self,root):
        if not root:
            return []
        res=[]
        res+=self.midtrans(root.left)
        res.append(root)
        res+=self.midtrans(root.right)
        return res
        # write code here
```

[]跟None不是一个概念

#### 数据流中的中位数

```python
import math
import heapq

class Solution:
    nums = []
    def Insert(self, num):
        heapq.heappush(self.nums, num)

    def GetMedian(self):
        mid = math.ceil(len(self.nums)/2) #一定要向上取整，因为下面是返回mid个，例如1,2,3,4,5，向下取整取不到
        return (heapq.nlargest(mid, self.nums)[-1] + heapq.nsmallest(mid, self.nums)[-1])/2.0 #大中最小，小中最大一定是中间两个

```

（牛客有点问题）

#### 滑动窗口的最大值

```python
# -*- coding:utf-8 -*-
class Solution:
    def maxInWindows(self, num, size):
        if size==0:
            return []
        window,res=[],[]
        for i,x in enumerate(num):
            if window and i-size>=window[0]: #第一步控制窗口大小
                window.pop(0) 
            while window and x>=num[window[-1]]: #第二步总是把大的放在头
                window.pop()
            window.append(i)
            if i>=size-1: #第三步头加到res里面
                res.append(num[window[0]])
        return res
        
        
        # write code here
```

双端队列，其实就三步

#### 矩阵中的路径

```python
class Solution:
    def hasPath(self, matrix, rows, cols, path):
        # write code here
        for i in range(rows):
            for j in range(cols):
                if self.find(matrix,rows,cols,path,i,j,[False]*rows*cols):
                    return True
        return False
    def find(self,matrix,rows,cols,path,i,j,flag): #当前点是否能找到路径
        if not path:
            return True
        index=i*cols+j
        if i<0 or j<0 or i>=rows or j>=cols or matrix[index]!=path[0] or flag[index]==True:
            return False
        
        flag[index]=True
        if self.find(matrix,rows,cols,path[1:],i-1,j,flag) or self.find(matrix,rows,cols,path[1:],i+1,j,flag) or self.find(matrix,rows,cols,path[1:],i,j-1,flag) or self.find(matrix,rows,cols,path[1:],i,j+1,flag):
            return True
        flag[index]=False
        return False
```

#### 机器人的运动范围

```python
# -*- coding:utf-8 -*-
class Solution:
    count=0
    def movingCount(self, threshold, rows, cols):
        arr=[[0 for i in range(cols)] for j in range(rows)]
        self.find(arr,0,0,threshold)
        return self.count
    def find(self,arr,i,j,threshold):
        if i<0 or j<0 or i>=len(arr) or j>=len(arr[0]):
            return
        s=sum(map(int,str(i)+str(j)))
        if s>threshold or arr[i][j]==1:
            return
        arr[i][j]=1
        self.count+=1
        self.find(arr,i+1,j,threshold)
        self.find(arr,i,j+1,threshold)
        # write code here
```

#### 剪绳子

```python
# -*- coding:utf-8 -*-
class Solution:
    def cutRope(self, number):
        if number==2:
            return 1
        if number==3:
            return 2
        dp=[0,1,2,3]
        for i in range(4,number+1):
            maxi=0
            for j in range(1,i/2+1):
                tmp=dp[j]*dp[i-j]
                if tmp>maxi:
                    maxi=tmp
            dp.append(maxi)
        return dp[number]
        
        # write code here
```

#### 冒泡排序

```python
def sort(arr):
    for i in range(len(arr)): #冒泡那么多轮
        for j in range(0,len(arr)-1-i):
            if arr[j]>arr[j+1]:
                arr[j],arr[j+1]=arr[j+1],arr[j]
    return arr

arr=[12, 11, 13, 5, 6]
print(sort(arr))
```

每轮上浮出最大值，前面无序的位置不合适就交换

#### 选择排序

```python
def sort(arr):
    for i in range(len(arr)):
        min_idx=i #未排序的首下标
        for j in range(i+1,len(arr)):
            if arr[j]<arr[min_idx]:
                min_idx=j
        arr[i],arr[min_idx]=arr[min_idx],arr[i]
    return arr
arr=[12, 11, 13, 5, 6]
print(sort(arr))
```

每次都从未排好序的序列里选择最小的push到已经排好序的尾部

#### 插入排序

```python
def sort(arr):
    for i in range(1,len(arr)):
        key=arr[i] #待插入元素
        j=i-1
        while j>=0 and key<arr[j]:
            arr[j+1]=arr[j] #后移
            j-=1
        arr[j+1]=key #key>arr[j]的时候跳出了
    return arr
arr=[12, 11, 13, 5, 6]
print(sort(arr))
```

将每一个元素插入前面已经排好序的序列

#### 快速排序

```python
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
arr=[12, 11, 13, 5, 6]
print(quicksort(arr))
```

选取一个基准，比基准小的放左边，大的放右边，再递归子序列。

#### 堆排序

```python
def heapify(arr, n, i): #长度为n的数组里使arr[i]变成根节点
    largest = i  
    l = 2 * i + 1     # left = 2*i + 1 
    r = 2 * i + 2     # right = 2*i + 2 
  
    if l < n and arr[i] < arr[l]: 
        largest = l 
  
    if r < n and arr[largest] < arr[r]: 
        largest = r 
  
    if largest != i: #如果要发生交换
        arr[i],arr[largest] = arr[largest],arr[i]  # 交换
  
        heapify(arr, n, largest) #实际上传入的是l或者r，不断往下走看需不需要继续交换
  
def heapSort(arr): 
    n = len(arr) 
  
    # Build a maxheap. 
    for i in range(n, -1, -1):  #从后往前不断进数调整
        heapify(arr, n, i) 
  
    # 一个个交换元素
    for i in range(n-1, 0, -1): 
        arr[i], arr[0] = arr[0], arr[i]   #arr[0]是堆顶，每次得到最大的
        heapify(arr, i, 0) #调整前i个使arr[0]又是堆顶
  
arr = [ 12, 11, 13, 5, 6, 7] 
heapSort(arr) 
n = len(arr) 
print ("排序后") 
for i in range(n): 
    print ("%d" %arr[i]),
```

#### 计数排序（线性、O(n))

```python
def counting_sort(a, k):  # k = max(a)
    n = len(a)  # 计算a序列的长度
    b = [0 for i in range(n)]  # 设置输出序列并初始化为0
    c = [0 for i in range(k + 1)]  # 设置计数序列并初始化为0，
    for j in a:
        c[j] = c[j] + 1
    for i in range(1, len(c)): #加上前面多少数比它小，这样才能确定初始的插入位置
        c[i] = c[i] + c[i-1]
    for j in a:
        b[c[j] - 1] = j
        c[j] = c[j] - 1
    return b
```

#### 桶排序（计数排序是桶排序的特殊情况）复杂度O(M+N)

举个例子，排序一个数组[5,3,6,1,2,7,5,10]

值都在1-10之间，建立10个桶：

[0 0 0 0 0 0 0 0 0 0]   桶

[1 2 3 4 5 6 7 8 9 10]  桶代表的值

遍历数组，第一个数字5，第五个桶加1

[0 0 0 0 1 0 0 0 0 0]

第二个数字3，第三个桶加1

[0 0 1 0 1 0 0 0 0 0]

遍历后

[1 1 1 0 2 1 1 0 0 1]

输出

[1 2 3 5 5 6 7 10]



适应外部排序，即数据量比较大，但是数据范围比较小的排序