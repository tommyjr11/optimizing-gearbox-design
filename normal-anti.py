import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# 设置信息量权重
a = 2;
# %设置启发量权重
b = 2;
# 设置信息素挥发因子
p = 0.8;
# 区间缩小因子
r = 0.8;
# 设置最大循环次数NC_max
NC_max = 100;
# 给定蚂蚁数量

# 设置维数
D = 7

# 根据第参数根据上下届生成其随机数
def randomnum(x):
    if(x == 1):
        return random.uniform(2.9, 3.9)
    elif(x == 2):
        return random.uniform(0.7,0.8)
    elif(x == 3):
        return random.uniform(17, 28)
    elif(x == 4):
        return random.uniform(7.3, 8.3)
    elif(x == 5):
        return random.uniform(7.3, 8.3)
    elif(x == 6):
        return random.uniform(2.9, 3.9)
    elif(x == 7):
        return random.uniform(5, 5.5)
    else:
        return print("error")




# 更新范围设定
def update_range(x,k,r,dirtction,value,muplite):
    # muplite = r**k
    if(x == 1 and dirtction == 1):
        return (random.uniform(0, (3.9-value)) * muplite)
    elif(x == 1 and dirtction == 0):
        return (random.uniform(0, (value-2.9)) * muplite)
    elif(x == 2 and dirtction == 1):
        return (random.uniform(0, (0.8-value)) * muplite)
    elif(x == 2 and dirtction == 0):
        return (random.uniform(0, (value-0.7)) * muplite)
    elif(x == 3 and dirtction == 1):
        return (random.uniform(0, (28-value)) * muplite)
    elif(x == 3 and dirtction == 0):
        return (random.uniform(0, (value-17)) * muplite)
    elif(x == 4 and dirtction == 1):
        return (random.uniform(0, (8.3-value)) * muplite)
    elif(x == 4 and dirtction == 0):
        return (random.uniform(0, (value-7.3)) * muplite)
    elif(x == 5 and dirtction == 1):
        return (random.uniform(0, (8.3-value)) * muplite)
    elif(x == 5 and dirtction == 0):
        return (random.uniform(0, (value-7.3)) * muplite)
    elif(x == 6 and dirtction == 1):
        return (random.uniform(0, (3.9-value)) * muplite)
    elif(x == 6 and dirtction == 0):
        return (random.uniform(0, (value-2.9)) * muplite)
    elif(x == 7 and dirtction == 1):
        return (random.uniform(0, (5.5-value)) * muplite)
    elif(x == 7 and dirtction == 0):
        return (random.uniform(0, (value-5)) * muplite)
    else:
        return print("error")


# 检查生成参数是否满足约束条件
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
    # if(g1>0):
    #     print("g1 = ",g1)
    # if(g2>0):
    #     print("g2 = ",g2)
    # if(g3>0):
    #     print("g3 = ",g3)
    # if(g4>0):
    #     print("g4 = ",g4)
    # if(g5>0):
    #     print("g5 = ",g5)
    # if(g6>0):
    #     print("g6 = ",g6)
    # if(g7>0):
    #     print("g7 = ",g7)
    # if(g8>0):
    #     print("g8 = ",g8)
    # if(g9>0):
    #     print("g9 = ",g9)
    # if(g10>0):
    #     print("g10 = ",g10)
    # if(g11>0):
    #     print("g11 = ",g11)
    if g1 <= 0 and g2 <= 0 and g3 <= 0 and g4 <= 0 and g5 <= 0 and g6 <= 0 and g7 <= 0 and g8 <= 0 and g9 <= 0 and g10 <= 0 and g11 <= 0:
        return True
    else:
        return False
def create_animation(data, filename='scatter_animation2.gif'):
    fig, ax = plt.subplots()

    # 设置x轴和y轴的限制
    ax.set_xlim(0, data.shape[0]-1)
    ax.set_ylim(np.min(data), np.max(data))

    # 初始化一个散点图对象
    scat = ax.scatter([], [])

    def animate(i):
        # 更新散点图的数据
        scat.set_offsets(np.c_[np.arange(data.shape[0]), data[:, i]])
        return scat,

    # 创建动画
    ani = FuncAnimation(fig, animate, frames=data.shape[1], interval=200, blit=True)

     # 保存为mp4视频
    ani.save(filename, writer='pillow', fps=120)
    plt.show()    
# 蚂蚁位置矩阵
def Antposition(Ant_Quantity):
    judge = []
    # 初始化蚂蚁位置矩阵
    Ant_position = [[0 for i in range(D)] for i in range(Ant_Quantity)]
    # 随机生成蚂蚁位置
    for i in range(Ant_Quantity):
        while True:
            for j in range(D):
                Ant_position[i][j] = randomnum(j+1)
            # 判断是否满足约束条件
            if check(Ant_position[i]) == True:
                break
    return Ant_position



# 更新每一只蚂蚁的位置，参数为蚂蚁位置矩阵，且在其领域内随机生成新的位置,k为迭代次数,向前估计。
def updatepositionpos(Ant_position,k,r,Ant_Quantity,max,muplite):
    num = 0
    # 初始化新的蚂蚁位置矩阵
    new_Ant_position = [[0 for i in range(D)] for i in range(Ant_Quantity)]
    for i in range(Ant_Quantity):
        while True:
            for j in range(D):
                # 随机生成新的位置
                randomtime = random.uniform(0,1)
                if(randomtime > 0.5):
                    new_Ant_position[i][j] = Ant_position[i][j] + update_range(j+1,k,r,1,Ant_position[i][j],muplite)
                else:
                    new_Ant_position[i][j] = Ant_position[i][j] - update_range(j+1,k,r,0,Ant_position[i][j],muplite)
            # 判断是否满足约束条件
            num = num + 1
            if (check(new_Ant_position[i]) == True and new_Ant_position[i] != Ant_position[i]) :
                num = 0
                break
            if(num == max):
                new_Ant_position[i] = Ant_position[i]
                num = 0
                break
    return new_Ant_position


# 更新每一只蚂蚁的位置，参数为蚂蚁位置矩阵，且在其领域内随机生成新的位置,k为迭代次数,向后估计。
def updatepositionneg(Ant_position,k,r,Ant_Quantity,max,muplite):
    num = 0
    # 初始化新的蚂蚁位置矩阵
    new_Ant_position = [[0 for i in range(D)] for i in range(Ant_Quantity)]
    for i in range(0,Ant_Quantity):
        while True:
            for j in range(0,D):
                # 随机生成新的位置
                randomtime = random.uniform(0,1)
                if(randomtime > 0.5):
                    new_Ant_position[i][j] = Ant_position[i][j] + update_range(j+1,k,r,1,Ant_position[i][j],muplite)
                else:
                    new_Ant_position[i][j] = Ant_position[i][j] - update_range(j+1,k,r,0,Ant_position[i][j],muplite)
            # 判断是否满足约束条件
            num = num + 1
            if (check(new_Ant_position[i]) == True and new_Ant_position[i] != Ant_position[i]):
                num = 0
                break
            if(num == max):
                new_Ant_position[i] = Ant_position[i]
                num = 0
                break
    return new_Ant_position



# 计算当前蚂蚁所在函数的值,f = 0.7854*x2^2*x1(14.9334*x3-43.0934+3.3333*x3^2) + 0.7854*(x5*x7^2+x4*x6^2) - 1.508*x1*(x7^2+x6^2) + 7.477(x7^3+x6^3)
def Antfunction(x):
    f = 0.7854*x[1]**2*x[0]*(14.9334*x[2]-43.0934+3.3333*x[2]**2) + 0.7854*(x[4]*x[6]**2+x[3]*x[5]**2) - 1.508*x[0]*(x[6]**2+x[5]**2) + 7.477*(x[6]**3+x[5]**3)
    return f



# 计算当前所有蚂蚁位置的信息量增量dt
def Antdt(Ant_position,new_Ant_position,Ant_Quantity):
    dt = np.zeros( Ant_Quantity)
    for i in range(0,Ant_Quantity):
        if((Antfunction(Ant_position[i]) - Antfunction(new_Ant_position[i]))>0):
            dt[i] = Antfunction(Ant_position[i]) - Antfunction(new_Ant_position[i])
        else:
            dt[i] = 0
    return dt



# 更新每只当前蚂蚁的信息量
def updatet(t,dt,p,Ant_Quantity):
    for i in range(0,Ant_Quantity):
        t[i] = p*t[i] + dt[i]
    return t

# 蚂蚁i与蚂蚁j所在空间点适应度期望值（启发量）为：
def Antexpectation(Ant_position,i,j):
    if(Antfunction(Ant_position[i]) - Antfunction(Ant_position[j])>0):
        E = Antfunction(Ant_position[i]) - Antfunction(Ant_position[j])
    else:
        E = 0
    
    return E

# 蚂蚁i依据概率移动到蚂蚁j所在的位置，移动概率为:
def Antmovep(t,Earray,j,Ant_Quantity):
    sum = 0
    
    for k in range(0,Ant_Quantity):
        sum = sum + t[k]*Earray[k]
    
    if(sum == 0):
            p = 0
    else:
        p = t[j]*Earray[j]/sum
    return p

def Antfindminvalue(Iteration,Ant_Quantity,pl,D,r,max):
    Earray = np.zeros(Ant_Quantity)
    p = np.zeros(Ant_Quantity)
    Ant_position = Antposition(Ant_Quantity)
    new_Ant_position = [[0 for i in range(D)] for i in range(Ant_Quantity)]
    t = np.ones(Ant_Quantity)
    dt = np.ones( Ant_Quantity)
    pos_sum = 0
    data = []
    value = []
    for m in range(Iteration):
        
        for i in range(0,Ant_Quantity):
            
            for j in range(0,Ant_Quantity):
                Earray[j] = Antexpectation(Ant_position,i,j)
            for j in range(0,Ant_Quantity):
                p[j] = Antmovep(t,Earray,j,Ant_Quantity)
            
            print("=----------------------------------------------------   ",i)
            print(Antfunction(Ant_position[i]),Ant_position[i])
            # 依据概率p[i]移动到蚂蚁j所在的位置,轮盘取值
            
            k = np.random.random()
            if(k==1):
                new_Ant_position[i] = Ant_position[Ant_Quantity-1]
            else:
                for h in range(0,Ant_Quantity):
                    pos_sum = pos_sum + p[h]
                    if(pos_sum>=k):
                        new_Ant_position[i] = Ant_position[h]
                        break
                if(p[h]==0):
                    new_Ant_position[i] = Ant_position[h]
                if (pos_sum ==0):
                    new_Ant_position[i] = Ant_position[i]
            pos_sum = 0
        # 更新信息量
        dt = Antdt(Ant_position, new_Ant_position,Ant_Quantity)
        t = updatet(t, dt, pl,Ant_Quantity)
        
        # 更新蚂蚁位置
        Ant_position = new_Ant_position
        new_Ant_position = [[0 for i in range(D)] for i in range(Ant_Quantity)]
        
        muplite = r**m
        new_Ant_positionpos = updatepositionpos(Ant_position,m,r,Ant_Quantity,max,muplite)
        new_Ant_positionneg = updatepositionneg(Ant_position,m,r,Ant_Quantity,max,muplite)
        

        for i in range(Ant_Quantity):
            
            if(Antfunction(new_Ant_positionpos[i])<Antfunction(new_Ant_positionneg[i])):
                new_Ant_position[i] = new_Ant_positionpos[i]
            else:
                new_Ant_position[i] = new_Ant_positionneg[i]
            value.append(Antfunction(new_Ant_position[i]))
        data.append(value)
        value = []
        # # 更新信息量
        # dt = Antdt(Ant_position, new_Ant_position,Ant_Quantity)
        # t = updatet(t, dt, pl,Ant_Quantity)
        Ant_position = new_Ant_position
       
        new_Ant_position = [[0 for i in range(D)] for i in range(Ant_Quantity)]
        new_Ant_positionpos = [[0 for i in range(D)] for i in range(Ant_Quantity)]
        new_Ant_positionneg = [[0 for i in range(D)] for i in range(Ant_Quantity)]
    # 找出最小值
    minvlue = Antfunction(Ant_position[0])
    for i in range(Ant_Quantity):
        
        if(Antfunction(Ant_position[i])<=minvlue):
            minvlue = Antfunction(Ant_position[i])
            choose = i
    create_animation(np.array(data).T)
    return minvlue ,Ant_position[choose]
print(Antfindminvalue(600,80,0.8,7,0.992,600))


        

        

