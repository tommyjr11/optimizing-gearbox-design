from tqdm import tqdm
import numpy as np
import torch
from torch.autograd import Variable
import random
import time
import matplotlib.pyplot as plt
def check(x):

    # g1 = -x1*x2^2*x3 + 27
    g1 = -x[0]*(x[1]**2)*x[2] + 27
    # g2 = -x1*x2^2*x3^2 + 397.5 
    g2 = -x[0]*x[1]**2*x[2]**2 + 397.5
    # g3 = -x2*x6^4*x3*x4^-3 + 1.93
    g3 = -x[1]*x[5]**4*x[2]*x[3]**-3 + 1.93
    # g4 = -x2*x7^4*x3*x5^-3 + 1.93
    g4 = -x[1]*x[6]**4*x[2]*x[4]**-3 + 1.93
    # g5 = 10*x6^-3*(16.91*10^6+(745*x4*x2^-1*x3^-1)^2)^0.5 - 1100
    g5 = 10*x[5]**-3*(16.91*10**6+(745*x[3]*x[1]**-1*x[2]**-1)**2)**0.5 - 1100
    # g6 = 10*x7^-3*(157.5*10^6+(745*x5*x2^-1*x3^-1)^2)^0.5 - 850
    g6 = 10*x[6]**-3*(157.5*10**6+(745*x[4]*x[1]**-1*x[2]**-1)**2)**0.5 - 850
    # g7 = x2*x3 - 40
    g7 = x[1]*x[2] - 40
    # g8 = -x1*x2^-1  + 5
    g8 = -x[0]*x[1]**-1 + 5
    # g9 = x1*x2^-1 - 12
    g9 = x[0]*x[1]**-1 - 12
    #  g10 = 1.5*x6 - x4 + 1.9
    g10 = 1.5*x[5] - x[3] + 1.9
    # g11 = 1.1*x7 - x5 + 1.9
    g11 = 1.1*x[6] - x[4] + 1.9
    if g1 <= 0 and g2 <= 0 and g3 <= 0 and g4 <= 0 and g5 <= 0 and g6 <= 0 and g7 <= 0 and g8 <= 0 and g9 <= 0 and g10 <= 0 and g11 <= 0:
        return True
    else:
        return False

class anVeriable:
    
        def __init__(self,limition):
            self.postion = np.random.uniform(limition[0],limition[1])
            self.oldpostion = self.postion
            self.limition = limition 
            self.gapbetween = self.limition[1] - self.limition[0]
        def movehead(self,speeddrop,learnspeed,jue2):
            if (speeddrop > 0 and (self.postion - 1/learnspeed*self.gapbetween) > self.limition[0]):
                self.postion = self.postion - 1/learnspeed*self.gapbetween
                jue2 = jue2 + 1
                return jue2
            if (speeddrop < 0 and (self.postion + 1/learnspeed*self.gapbetween) < self.limition[1]):
                self.postion = self.postion + 1/learnspeed*self.gapbetween
                jue2 = jue2 + 1
                return jue2
            self.postion = self.postion
            return jue2
           
        def updateoldpostion(self):
            self.oldpostion = self.postion
        
        def moveback(self):
          
            self.postion = self.oldpostion

def plot_line_chart(data):
    x = range(len(data))  # 生成横坐标的序列，从0到数组长度减1
    plt.plot(x, data)  # 绘制折线图
    plt.xlabel('X轴')  # 设置横坐标标签
    plt.ylabel('Y轴')  # 设置纵坐标标签
    plt.title('折线图')  # 设置标题
    plt.show()  # 显示图形

def Antfunction(x):
    f = 0.7854*x[1]**2*x[0]*(14.9334*x[2]-43.0934+3.3333*x[2]**2) + 0.7854*(x[4]*x[6]**2+x[3]*x[5]**2) - 1.508*x[0]*(x[6]**2+x[5]**2) + 7.477*(x[6]**3+x[5]**3)
    return f

def countrigthnow(antArrey):
    a = Variable(torch.tensor(antArrey),requires_grad=True)
    c = Antfunction(a)
    c.backward()
    
    return a.grad

def sort(lst,list2):
    for w in range (0,len(lst)):
        lst[w] = lst[w]*list2[w]
    indexed_lst = list(enumerate(lst))

    # 对列表进行排序，排序规则是按照元素的值排序
    sorted_indexed_lst = sorted(indexed_lst, key=lambda x: x[1])

    # 获取排序后的元素在原数组中的位置
    original_indices = [i for i, v in sorted_indexed_lst]

    # 获取排序后的数组
    sorted_lst = [v for i, v in sorted_indexed_lst]
    return sorted_lst,original_indices

def coutminvalue(learnspeed,learntime,changetime,changebate):
    start_time = time.time()
    num = 0
    limition = [ [2.9, 3.9] , [0.7,0.8], [17, 28], [7.3, 8.3], [7.3, 8.3], [2.9, 3.9], [5, 5.5]]
    dynamic = 0
    jue =2
    juenow = 0
    juenow2 = 0
    basicspeed = learnspeed
    valuelist = []
    while(True):
        Antgroups = []
        Antpostiongroup = []
        dropspeedgroup = []
        for limitions in (limition):
            Antgroups.append(anVeriable(limitions))
        Antpostiongroup = [Antgroups[0].postion,Antgroups[1].postion,Antgroups[2].postion,Antgroups[3].postion,Antgroups[4].postion,Antgroups[5].postion,Antgroups[6].postion]
        if(check(Antpostiongroup)):
            break
    #train now
    gap = [Antgroups[0].gapbetween,Antgroups[1].gapbetween,Antgroups[2].gapbetween,Antgroups[3].gapbetween,Antgroups[4].gapbetween,Antgroups[5].gapbetween,Antgroups[6].gapbetween]
    for i in tqdm(range(learntime)):
        dropspeedgroup = countrigthnow(Antpostiongroup)
        sorted_lst,original_indices = sort(countrigthnow(Antpostiongroup),gap)
        for j in range(0,7):
                j = 6- j
                juenow2 = Antgroups[original_indices[j]].movehead(dropspeedgroup[original_indices[j]],learnspeed,juenow2)
        Antpostiongroup = [Antgroups[0].postion,Antgroups[1].postion,Antgroups[2].postion,Antgroups[3].postion,Antgroups[4].postion,Antgroups[5].postion,Antgroups[6].postion]
        if(check(Antpostiongroup)): 
            jue = 2
            dynamic = 0
            juenow = 0
            learnspeed = basicspeed
            
            
          
            for j in range(0,7):
                Antgroups[j].updateoldpostion()
            continue
        else:
            # 创建一个从0到6的列表
            lst = list(range(7))
          
            #    从列表中随机选择6个不重复的元素
            random_lst = random.sample(lst, 7)
          
            for j in range(0,7):
                    
                    if(check(Antpostiongroup)):
                        break
                    Antgroups[random_lst[j]].moveback()
                    Antpostiongroup = [Antgroups[0].postion,Antgroups[1].postion,Antgroups[2].postion,Antgroups[3].postion,Antgroups[4].postion,Antgroups[5].postion,Antgroups[6].postion]
                    

            
            if (j>=(juenow2-1)):
                dynamic = dynamic + 1
                

            if (int(dynamic/changetime) >= 1):
                if(jue == 2):
                    jue = 3

                if( learnspeed < 2 and jue == 3):
                    
                    jue = 4

                if(jue == 3):
                    learnspeed = learnspeed/(changebate)
                    juenow = juenow  + 1
                    dynamic = 0

                if(jue == 4):
                    if(juenow != 0):
                        learnspeed = basicspeed
                    learnspeed = learnspeed*(changebate)
                    juenow = 0
                    jue = 4
                    dynamic = 0

               
            
            if(jue == 4 and j<(juenow2-1)):
                learnspeed = basicspeed
                dynamic = 0
                juenow = 0
                jue = 2   

            if(jue == 3 and j<(juenow2-1)):
                learnspeed = basicspeed
                dynamic = 0
                jue = 2

          
        

        for j in range(0,7):
            
            Antgroups[j].updateoldpostion()
        valuelist.append(Antfunction(Antpostiongroup))

        
        
        juenow2 = 0
    end_time = time.time()
    print("程序运行的时间是：", end_time - start_time, "秒")
    # print(Antfunction(Antpostiongroup))
    print(Antpostiongroup)
    print(Antfunction(Antpostiongroup))
    plot_line_chart(valuelist)
    return Antfunction(Antpostiongroup)

print(coutminvalue(learnspeed=7,learntime=5000,changetime=1,changebate= 1.1))

