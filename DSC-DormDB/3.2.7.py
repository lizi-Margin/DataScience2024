#######################
#购物车实现：
#制定商品条目
#启动程序，让用户输入初始金额；
#用户可选择操作：0、退出 1、查看商品列表 2、加入购物车 3、结算购物车 4、查看余额 5、清空购物车及购买历史
#允许用户根据商品编号购买商品；
#用户选择结算购物车后检测余额是否足够，够就直接扣款，不够就提醒；
#用户可以一直购买商品，也可以直接退出；
#用文件保存购买历史、购物车历史以及商品列表
#######################
import numpy as np
products = {'Iphone8':6888,'MacPro':14800,'mi6':2499,'Coffee':31,'Book':80,'Nike shoes':799}
shopping_cart = {}
buy = {}

#从文件中读取shopping_cart、buy以及products三个字典，若没有就用默认的初始化情况
def initialize():
    try:
        f = open('shopping_cart.txt','r')
        a = f.read()
        global shopping_cart     #global 用于修改全局变量
        shopping_cart = eval(a)  # eval() 函数用来执行一个字符串表达式，并返回表达式的值。在这里即返回从文件中读取到的字典。
        f.close()
        f = open('buy.txt','r')
        a = f.read()
        global buy
        buy = eval(a)
        f.close()
        f = open('products.txt','r')
        a = f.read()
        global products
        products = eval(a)
        f.close()
    except FileNotFoundError:
        pass
    
#展示商品，当参数为1时，输出所有商品；当参数为3时，输出购物车中的商品
def show_item(content):
    print("##############################################")
    if content == 1:
        print("序号{:<10s}商品名称{:<10s}价格{:<10s}".format(" "," "," "))  #format 控制输出格式
        k = 0
        for i in products:
            print("{:<14d}{:<18s}{}".format(k,i,products[i]))
            k = k+1
    elif content == 3:
        print("购物车中有如下商品:")
        print("序号{:<10s}商品名称{:<10s}价格{:<10s}数量{:<10s}".format(" "," "," "," "))
        k = 0
        for i in shopping_cart:
            print("{:<14d}{:<18s}{:<14d}{}".format(k,i,products[i],shopping_cart[i]))
            k = k+1

#展示可进行的操作
def show_operation():
    print("##############################################")
    print("您可进行如下操作（选择对应序号即可）")
    print("0 退出")
    print("1 查看商品列表")
    print("2 加入购物车")
    print("3 结算购物车")
    print("4 查看余额")
    print("5 清空购物车及购买历史")
    choice = input('您选择的操作是:')
    return choice

#将商品加入购物车
def in_cart():
    show_item(1)
    print("您想加入购物车的是？")
    while True:
        choice = input('请输入所选择商品序号:')
        if choice.isdigit() :
            choice = int(choice)
            if 0<=choice<len(products) :
                break
            else:
                print("无该商品！")
        else:
            print("无该商品！")
    product = list(products)[choice]
    if product in shopping_cart:
        shopping_cart[product] +=1
    else:
        shopping_cart[product] = 1
    print("已帮您加入购物车")

#买家完成付款
def pay(money):
    show_item(3)
    list_pay = input("您想结算的商品是？")
    xlist = list_pay.split(",")
    xlist = [int(xlist[i]) for i in range(len(xlist)) if 0<=int(xlist[i])<len(shopping_cart)]
    c,s=np.unique(xlist,return_counts=True)    #np.unique 用于对list排序，当可选参数return_counts=True时，返回两个参数，第一个是去除数组中的重复数字后，进行排序的结果，第二个是第一个返回参数中每个元素在原list中的个数
    total = 0
    pay_item = [list(shopping_cart)[c[i]] for i in range(len(c))]
    for i in range(len(c)):
        total += products[pay_item[i]]*s[i]
    if total<money:
        for i in range(len(pay_item)):
            if pay_item[i] in buy:
                buy[pay_item[i]] +=s[i]
            else:
                buy[pay_item[i]] =1
            shopping_cart[pay_item[i]] -=1
            if shopping_cart[pay_item[i]] == 0:
                del shopping_cart[pay_item[i]]
        print("已经结算清！")
        return total
    else:
        print("余额不足！")
        return 0

#清空购买以及购物车历史，即清空对应的字典即可
def clean_history():
    global buy
    buy.clear()
    global shopping_cart
    shopping_cart.clear()
    
if __name__ == '__main__':
    initialize()
    money = int(input('请输入初始金额:'))
    while True:
        choice = show_operation()
        if choice.isdigit():
            choice = int(choice)
            if choice == 0:
                break
            elif choice == 1:
                show_item(1)
            elif choice == 2:
                in_cart()
            elif choice == 3:
                delta = pay(money)
                money -= delta
            elif choice == 4:
                print("当前余额： ",money)
            elif choice == 5:
                clean_history()
                print("已清空历史")
            else:
                print("操作不当！")
        else:
            print("操作不当！")
        
    #将购物车和已购买的字典存储
    f = open('shopping_cart.txt','w')
    f.write(str(shopping_cart))
    f.close()
    f = open('buy.txt','w')
    f.write(str(buy))
    f.close()
    f = open('products.txt','w')
    f.write(str(products))
    f.close()
    print("购物信息已经储存好，欢迎下次光临！")
